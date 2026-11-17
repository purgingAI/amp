"""The customer's REAL recent payments, from Mongo `walletLedger`.

"My last transaction" has to mean the last one they actually made. Before this the assistant
answered from a fixture list, so a customer who had just paid a merchant was told about an
older transfer to a person — which is the kind of wrong that destroys trust in everything
else the assistant says.

Three things the raw ledger makes you deal with:

* **Two rows per payment.** The principal and the fee are separate `walletLedger` documents
  sharing a `txnId`. Grouping and taking the larger is what gives you the payment the customer
  remembers making.
* **The counterparty is encrypted**, and decrypts to the registered legal name — "Magical
  Universe" for the merchant trading as MClothes. Where a `merchantId` is present we prefer
  the trading name the customer actually saw.
* **Status is the ledger's, not the transfer's.** Dev leaves these `PENDING` well after the
  money has moved, so PENDING is mapped to "in progress" and never to "failed".

Falls back to the fixtures in `data.py` when Mongo is unreachable, so the demo still runs
offline — and says which it used.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: Ledger status -> (our state, where the money is). Deliberately conservative: an unknown or
#: still-settling status is "in progress", never "failed", because telling a customer their
#: money failed when it did not is the worse error.
_STATUS = {
    "SUCCESS":   ("SUCCESS", "CREDITED"),
    "COMPLETED": ("SUCCESS", "CREDITED"),
    "SETTLED":   ("SUCCESS", "CREDITED"),
    "PENDING":   ("PENDING", "WITH_PARTNER"),
    "INITIATED": ("PENDING", "HELD_BY_US"),
    "PROCESSING": ("PENDING", "WITH_PARTNER"),
    "FAILED":    ("FAILED", "HELD_BY_US"),
    "REJECTED":  ("FAILED", "HELD_BY_US"),
    "REVERSED":  ("REVERSED", "RETURNED"),
    "REFUNDED":  ("REVERSED", "RETURNED"),
    "ON_HOLD":   ("ON_HOLD", "HELD_BY_US"),
    "UNDER_REVIEW": ("ON_HOLD", "HELD_BY_US"),
}

#: `held` is how many rows the cached fill actually went and got. Without it a caller asking
#: for 600 rows was served the 30 a previous caller had warmed the cache with, and simply got
#: a shorter list — which for the spend breakdown meant a total computed over a fraction of
#: the period and silently too small. A cache may return fewer rows than exist; it must never
#: return fewer rows than were ASKED for while pretending that is all there is.
#: Per CUSTOMER. One shared cache was fine with one configured sender; with logins it would
#: hand one customer's ledger to the next person who asked within the TTL.
_CACHES: dict = {}


def _cache() -> dict:
    import userctx
    return _CACHES.setdefault(userctx.user_id(), {"at": 0.0, "rows": None, "held": 0})
_TTL = 20.0


_CLIENT = None


def _mongo():
    """One MongoClient for the process, not one per call.

    This used to build a fresh client every time, and a MongoClient is not a connection —
    it is a connection POOL that opens up to 100 sockets and keeps them. `recent()` is called
    from the suggestions strip, the history handler, the split handler and `find_txn`, so a
    busy minute created hundreds of pools: 408 established sockets to Mongo, after which the
    server simply stopped answering and every request hung. Reported as "stuck on pay from
    wallet", which was just the first request unlucky enough to need the database.

    `maxPoolSize` is pinned low on purpose. This is a demo surface with a handful of
    concurrent readers; a pool of 100 is not headroom, it is a way for one leak to exhaust a
    shared dev database.
    """
    global _CLIENT
    import pymongo
    import breaker
    breaker.check()
    cfg = json.load(open(os.path.join(AMP, "openwallet", ".secrets.json")))
    if _CLIENT is None:
        # BOTH bounds, because the URI carries `minPoolSize=100&maxPoolSize=5000` and a
        # maxPoolSize below the URI's minPoolSize is rejected outright
        # ("minPoolSize must be smaller or equal to maxPoolSize"). It also explains the
        # original symptom: minPoolSize=100 means every client opens a hundred sockets
        # EAGERLY, so four leaked clients was four hundred connections. The app wants that
        # pool; a read-only assistant does not.
        _CLIENT = pymongo.MongoClient(cfg["mongo_uri"], serverSelectionTimeoutMS=6000,
                                      minPoolSize=1, maxPoolSize=8, connectTimeoutMS=6000)
    import userctx
    return _CLIENT["APP_DB"], userctx.user_id()


def _num(v, default: float = 0.0) -> float:
    """Mongo stores these as Decimal128, which float() will not take directly."""
    if v is None:
        return default
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return default


def _hours_ago(dt) -> float:
    if not isinstance(dt, datetime):
        return 0.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ref = dt.replace(tzinfo=None) if dt.tzinfo else dt
    return max(0.0, (now - ref).total_seconds() / 3600.0)


#: Always read this many from the ledger regardless of what the caller asked for, and slice
#: afterwards. The cache is keyed on time alone, so a `recent(8)` followed by a
#: `recent(20)` — which is exactly what "show me my last 20 transactions" does right after a
#: status question — used to return the cached eight and quietly answer the wrong question.
#: One generous read serves every caller and costs the same round trip.
_FETCH = 30


def recent(limit: int = 8, fresh: bool = False) -> list[dict] | None:
    """The customer's real recent payments, newest first. None if the ledger is unreachable.

    `fresh` skips the cache. It exists for one caller: re-checking a figure the customer has
    disputed. Answering "are you sure?" out of the same cached rows that produced the
    disputed number is not a check — it is the same arithmetic on the same input, and it
    would confirm a wrong answer as readily as a right one.
    """
    _C = _cache()
    if (not fresh and _C["rows"] is not None
            and time.time() - _C["at"] < _TTL
            and _C["held"] >= limit):
        return _C["rows"][:limit]
    want, limit = limit, max(limit, _FETCH)
    import breaker
    try:
        breaker.check()
        db, user_id = _mongo()
        import pii
        raw = list(db["walletLedger"].find(
            {"userId": user_id, "txnType": "DEBIT"},
            sort=[("createdDate", -1)], limit=limit * 4))
    except Exception as exc:
        breaker.record(exc)
        return None

    # One payment per txnId: the principal is the larger of the two rows (the other is the fee).
    best: dict[str, dict] = {}
    for r in raw:
        tid = str(r.get("txnId") or "")
        if not tid:
            continue
        amt = _num(r.get("amount"))
        if tid not in best or amt > _num(best[tid].get("amount")):
            best[tid] = r

    # The ledger's own status is not maintained for wallet transfers in dev — 282 PENDING
    # against 1 SUCCESS for this customer, including payments that demonstrably completed.
    # `userEvent` carries the real outcome ("Wallet transaction completed", eventStatus 200),
    # so it decides. Telling someone a finished payment is "in progress" is a factual error,
    # not a cosmetic one.
    completed: set[str] = set()
    try:
        for ev in db["userEvent"].find(
                {"txnId": {"$in": [str(t) for t in best]}},
                {"txnId": 1, "eventStatus": 1, "eventMessage": 1}):
            msg = str(ev.get("eventMessage") or "").lower()
            if str(ev.get("eventStatus")) == "200" and (
                    "complete" in msg or "success" in msg):
                completed.add(str(ev.get("txnId")))
    except Exception:
        pass

    import data as cx_data
    out = []
    for r in sorted(best.values(), key=lambda x: x.get("createdDate") or 0, reverse=True):
        tid = str(r.get("txnId"))
        if tid in completed:
            state, position = "SUCCESS", "CREDITED"
        else:
            state, position = _STATUS.get(str(r.get("transactionStatus") or "").upper(),
                                          ("PENDING", "WITH_PARTNER"))
        merchant_id = r.get("merchantId")
        payee = None
        if merchant_id:
            m = next((x for x in cx_data.MERCHANTS if x.get("userId") == merchant_id), None)
            payee = m["name"] if m else None          # the trading name they actually saw
        if not payee:
            payee = pii.decrypt(r.get("receiver")) or "someone"
        out.append({
            "ref": str(r.get("txnId")),
            "payee": payee,
            "amount": _num(r.get("amount")),
            "currency": r.get("txnCurrency") or "AED",
            "discount": _num(r.get("discountAmount")),
            "campaign": r.get("campaignLedgerReference"),
            "product": r.get("product") or "Payment",
            "is_merchant": bool(merchant_id),
            "state": state, "position": position,
            "hours_ago": _hours_ago(r.get("createdDate")),
            "total": _num(r.get("totalAmount")),
        })
        if len(out) >= limit:
            break

    _C.update(at=time.time(), rows=out, held=limit)
    return out[:want]


if __name__ == "__main__":
    rows = recent()
    if rows is None:
        print("  ledger unreachable")
    else:
        print(f"  {len(rows)} real payments")
        for r in rows:
            d = f" (−{r['discount']:g} {r['campaign']})" if r["discount"] else ""
            print(f"   {r['ref']}  {r['payee'][:26]:28} {r['amount']:>8.2f} {r['currency']}"
                  f"{d}  {r['state']:8} {r['hours_ago']:.1f}h ago")


def find_ref(ref: str) -> dict | None:
    """One ledger row of THIS customer's, by transaction reference, whatever its type.

    `recent()` reads debits, because the question it answers is "what have I paid". That
    makes it the wrong place to look up a reference a customer has handed us: a refund is a
    CREDIT, so a screenshot of a refund matched nothing and was answered "the reference on
    it isn't one of the payments from your wallet" — about a refund the customer had
    genuinely received, to their own wallet, with the right number on the screen.

    Still scoped to the authenticated customer. Widening the TYPE is not widening the
    OWNER: a reference that belongs to somebody else returns None here exactly as before.
    """
    ref = (ref or "").strip()
    if not ref or not ref.isdigit():
        return None
    try:
        db, user_id = _mongo()
        rows = list(db["walletLedger"].find({"userId": user_id, "txnId": ref}))
    except Exception as exc:                                  # noqa: BLE001
        import breaker; breaker.record(exc)
        return None
    if not rows:
        return None
    # The principal is the larger of the rows sharing a txnId; the other is the fee.
    best = max(rows, key=lambda r: _num(r.get("amount")))
    return {
        "ref": ref,
        "txn_type": str(best.get("txnType") or "").upper(),
        "amount": _num(best.get("amount")),
        "state": str(best.get("status") or best.get("state") or "").upper(),
        "payee": (best.get("receiverName") or best.get("senderName")
                  or best.get("merchantName") or ""),
        "created": best.get("createdDate"),
        "raw_types": sorted({str(r.get("txnType") or "").upper() for r in rows}),
    }


def refunded_link(ref: str) -> dict | None:
    """Was the payment in this screenshot a REFUND of something the customer paid?

    A refund screenshot carries the refund's own reference, and that reference does not
    belong to the customer at all — the rows under it are the merchant's, plus a credit leg
    with no owner. Looking it up in their ledger therefore found nothing, and a customer
    holding a picture of their own refund was told "the reference on it isn't one of the
    payments from your wallet".

    What ties the two together is `originalTxnId`. So: find what this reference is a refund
    OF, then check whether THAT payment is theirs. If it is, we can speak about it — it is
    their money and their transaction.

    Nothing from the other party's rows is returned. The link is read from them; every
    figure in the answer comes from the customer's own row.
    """
    ref = (ref or "").strip()
    if not ref or not ref.isdigit():
        return None
    try:
        db, user_id = _mongo()
        legs = list(db["walletLedger"].find({"txnId": ref}, {"originalTxnId": 1,
                                                             "transactionStatus": 1}))
        states = {str(r.get("transactionStatus") or "").upper() for r in legs}
        # An ordinary payment also carries an originalTxnId — pointing at ITSELF. So the
        # presence of the field proves nothing; what marks a refund is a leg actually
        # marked REFUNDED, pointing at a DIFFERENT transaction.
        if "REFUNDED" not in states:
            return None
        originals = {str(r.get("originalTxnId")) for r in legs
                     if r.get("originalTxnId") and str(r.get("originalTxnId")) != ref}
        if not originals:
            return None
        for orig in originals:
            mine = list(db["walletLedger"].find({"userId": user_id, "txnId": orig}))
            if not mine:
                continue
            principal = max(mine, key=lambda r: _num(r.get("amount")))
            return {
                "refund_ref": ref,
                "original_ref": orig,
                "amount": _num(principal.get("amount")),
                "product": str(principal.get("product") or ""),
                "status": str(principal.get("transactionStatus") or "").upper(),
                "refund_status": sorted({str(r.get("transactionStatus") or "").upper()
                                         for r in legs}),
            }
    except Exception as exc:                                  # noqa: BLE001
        import breaker; breaker.record(exc)
        return None
    return None
