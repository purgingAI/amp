"""The customer's world: their recent payments, who they can pay, and what is around them.

Deliberately a fixture rather than a live query. Two reasons, and the second is the real one:

  1. The demo has to be able to show every cell of the refund matrix on cue, and real dev
     data does not contain a conveniently stuck-on-hold transfer when you need one.
  2. A customer-support surface reading live production money data is a thing you build
     behind an access review, not during a demo build. The seam is `transactions_for()` —
     swap its body for the real statement call and nothing above it changes.

The one live transaction (`LIVE_REF`) is real: it is the dev transfer the voice flow actually
settles, so the end-to-end run shows a genuine payment alongside the fixtures.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from engine.matrix import Evidence, Position, State
from taxonomy import CORRIDORS

CUSTOMER = {"id": "agt_sagar", "name": "Sagar Uniyal", "wallet": "7bddfbea",
            "currency": "AED", "kyc": "Successful",
            "limits": {"per_txn": 200, "per_day": 500, "per_month": 5000}}

#: Set by the server when a real payment settles, so "my last transaction" means the one the
#: customer just watched happen rather than a fixture.
LIVE_REF: str | None = None


@dataclass
class Txn:
    ref: str
    payee: str
    amount: float
    currency: str
    corridor: str
    state: State
    position: Position
    hours_ago: float
    note: str = ""
    refund_raised: bool = False
    demo: bool = True          # False for a payment read from the real ledger
    is_merchant: bool = False

    @property
    def sla_hours(self) -> float:
        return CORRIDORS.get(self.corridor, {}).get("sla_hours", 0.5)

    def evidence(self) -> Evidence:
        return Evidence(state=self.state, position=self.position,
                        hours_elapsed=self.hours_ago, sla_hours=self.sla_hours,
                        amount=self.amount, currency=self.currency,
                        corridor=self.corridor, txn_ref=self.ref)

    def public(self) -> dict:
        """What a customer may see. No internal ids, no partner names, no position —
        'where the money physically sits' is how WE decide, not something to show a
        customer as a status."""
        # A merchant payment is not a "corridor"; calling it "Within UAE (wallet to wallet)"
        # describes the rails, not what the customer did.
        where = ("Payment to a merchant" if self.is_merchant
                 else CORRIDORS.get(self.corridor, {}).get("name", self.corridor))
        return {"ref": self.ref, "payee": self.payee, "amount": self.amount,
                "currency": self.currency,
                "corridor": where,
                "status": PUBLIC_STATUS[self.state],
                "when": _ago(self.hours_ago), "refund_raised": self.refund_raised}


PUBLIC_STATUS = {
    State.SUCCESS: "Delivered",
    State.FAILED: "Failed",
    State.PENDING: "In progress",
    State.ON_HOLD: "Being reviewed",
    State.REVERSED: "Refunded",
    State.UNKNOWN: "Unknown",
}


def _ago(hours: float) -> str:
    if hours < 1:
        return f"{int(hours * 60)} min ago"
    if hours < 24:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


#: One transaction per interesting matrix cell, so the demo can walk the whole policy.
TRANSACTIONS: list[Txn] = [
    Txn("1202461007105029", "Alex Morgan", 20, "AED", "AE-AE",
        State.SUCCESS, Position.CREDITED, 0.3, "the real dev transfer"),
    Txn("1202461006221144", "Alex Morgan", 250, "AED", "AE-PK",
        State.FAILED, Position.HELD_BY_US, 1.5,
        "debited, bounced before leaving us — the classic 'money gone' complaint"),
    Txn("1202461005338821", "Imran Sheikh", 500, "AED", "AE-PK",
        State.FAILED, Position.WITH_PARTNER, 26,
        "pushed to the payout partner then failed — a claim, not an instant reversal"),
    Txn("1202461004117705", "Rahul Verma", 300, "AED", "AE-IN",
        State.PENDING, Position.WITH_PARTNER, 19,
        "past its corridor SLA — 'please wait' stopped being true"),
    Txn("1202461007009912", "Fatima Noor", 120, "AED", "AE-BD",
        State.PENDING, Position.WITH_PARTNER, 0.4, "still comfortably inside SLA"),
    Txn("1202461003884412", "Ahmed Khan", 900, "AED", "AE-PK",
        State.ON_HOLD, Position.HELD_BY_US, 5,
        "stopped by a compliance check — never ours to release"),
    Txn("1202461002771130", "Alex Morgan", 75, "AED", "AE-AE",
        State.REVERSED, Position.RETURNED, 72, "already refunded", refund_raised=True),
    Txn("1202461001665508", "Jordan Rao", 40, "AED", "AE-AE",
        State.SUCCESS, Position.CREDITED, 96, ""),
]


#: True when the list below came from the real ledger rather than the fixtures.
LIVE_HISTORY = False


def transactions_for(_customer_id: str = CUSTOMER["id"], limit: int = 8) -> list[Txn]:
    """The customer's recent payments — REAL ones from the ledger where we can read it.

    "My last transaction" has to mean the last one they actually made. Falling back to the
    fixtures when the ledger is unreachable keeps the demo working offline, but a real
    payment must never be hidden behind a made-up one.

    **Fixtures are never mixed with live rows.** They used to be appended for the states the
    ledger lacked, so the refund matrix stayed demonstrable — and the result was that the
    assistant told a customer their last payment was a FAILED AED 250 transfer to Pakistan
    when their real last payment was a successful AED 85 to a merchant. The fixture carried a
    1.5h timestamp, younger than every real row, so it won `latest()` and `latest_problem()`
    outright. Inventing a failed payment on a money surface is not a cosmetic bug: the
    customer went looking for money that had never gone missing.

    So: if the ledger answers, these are the real rows and nothing else. The fixtures remain
    as the OFFLINE fallback, and `CX_DEMO_TXNS=1` puts them back deliberately for showing the
    refund matrix — which is the only thing that ever needed them.
    """
    global LIVE_HISTORY
    try:
        import history
        rows = history.recent(limit=limit)
    except Exception:
        rows = None
    if not rows:
        LIVE_HISTORY = False
        return TRANSACTIONS
    LIVE_HISTORY = True
    live = [Txn(ref=r["ref"], payee=r["payee"], amount=r["amount"], currency=r["currency"],
                corridor="AE-AE", state=State(r["state"]), position=Position(r["position"]),
                hours_ago=r["hours_ago"],
                note=(f"{r['campaign']} −{r['discount']:g}" if r["discount"] else ""),
                demo=False, is_merchant=r["is_merchant"])
            for r in rows]
    if os.environ.get("CX_DEMO_TXNS") == "1":
        have = {t.state for t in live}
        extra = [t for t in TRANSACTIONS
                 if t.state in (State.FAILED, State.ON_HOLD, State.REVERSED)
                 and t.state not in have]
        return sorted(live + extra, key=lambda t: t.hours_ago)
    return sorted(live, key=lambda t: t.hours_ago)


#: How far back to look when the customer names a SPECIFIC payment. Wider than the default
#: listing window on purpose: `transactions_for()` returns the most recent 8, which is the
#: right size for "show me my recent payments" and the wrong size for "status of
#: 1171261008114504". Looking only at the recent 8 meant a reference the assistant had itself
#: just quoted — a split's own target, a receipt from an hour ago — came back "I can't find
#: that payment".
FIND_WINDOW = 50


def find_txn(ref: str | None) -> Txn | None:
    if not ref:
        return None
    needle = str(ref).strip().lower()
    for t in transactions_for(limit=FIND_WINDOW):
        if t.ref.lower() == needle or needle in t.ref.lower():
            return t
    return None


def find_txn_by_name(who: str | None) -> Txn | None:
    """The most recent payment to this person or shop, or None.

    Token-subset both ways, like every other name comparison here: the ledger holds
    "Alex Umashankar Morgan Umashankar Gaurishankar" and the customer says "alex".
    Returning None is a real answer — the caller must say it could not find a payment to
    that person rather than reaching for a different one.
    """
    import re as _re
    want = {t for t in _re.split(r"[^\w\u0900-\u097f]+", (who or "").lower())
            if len(t) > 2}
    if not want:
        return None
    best = None
    for t in transactions_for(limit=FIND_WINDOW):
        have = {x for x in _re.split(r"[^\w\u0900-\u097f]+", (t.payee or "").lower())
                if len(x) > 2}
        if have and (want <= have or have <= want or len(want & have) >= 2):
            if best is None or t.hours_ago < best.hours_ago:
                best = t
    return best


def latest() -> Txn | None:
    txns = transactions_for()
    return min(txns, key=lambda t: t.hours_ago) if txns else None


#: How bad a state is, for picking which payment an unreferenced complaint is about.
#: A customer saying "my money is gone" means the FAILED one, not the pending one that is
#: four minutes old and behaving perfectly — recency alone picks the wrong payment.
_SEVERITY = {State.FAILED: 0, State.ON_HOLD: 1, State.PENDING: 2,
             State.REVERSED: 3, State.SUCCESS: 4, State.UNKNOWN: 5}


def latest_problem(intent: str | None = None) -> Txn | None:
    """The transaction an unreferenced complaint is most likely about.

    Ranked by how serious the state is first and recency second. For "debited but not
    credited" we go further and require the money to have actually left the wallet — that
    is the literal complaint, and a payment where nothing was debited cannot be it.
    """
    txns = transactions_for()
    pool = [t for t in txns if t.state in (State.FAILED, State.PENDING, State.ON_HOLD)]
    # When the customer has already told us the state ("stuck on hold", "it failed"), that
    # is better evidence than our severity ranking. Honour it when such a payment exists.
    said_state = {"txn_on_hold": State.ON_HOLD, "txn_failed": State.FAILED}.get(intent or "")
    if said_state:
        matching = [t for t in pool if t.state is said_state]
        if matching:
            return min(matching, key=lambda t: t.hours_ago)
    if intent == "debited_not_credited":
        debited = [t for t in pool if t.position in (Position.HELD_BY_US,
                                                     Position.WITH_PARTNER)]
        pool = debited or pool
    if not pool:
        return latest()
    return min(pool, key=lambda t: (_SEVERITY[t.state], t.hours_ago))


# ---------------------------------------------------------------------------------------
# Beneficiaries
# ---------------------------------------------------------------------------------------

BENEFICIARIES = [
    {"name": "Alex Morgan", "agent": "agt_alex", "wallet": "***908",
     "corridor": "AE-AE", "last_sent": 20.0, "mandate": True},
    {"name": "Imran Sheikh", "agent": "agt_imran", "wallet": "***441",
     "corridor": "AE-PK", "last_sent": 500.0, "mandate": False},
    {"name": "Rahul Verma", "agent": "agt_rahul", "wallet": "***772",
     "corridor": "AE-IN", "last_sent": 300.0, "mandate": False},
    {"name": "Fatima Noor", "agent": "agt_fatima", "wallet": "***115",
     "corridor": "AE-BD", "last_sent": 120.0, "mandate": False},
]


def suggestions(session: str | None = None) -> list[dict]:
    """The prompts above the input box, rebuilt on EVERY turn from what just happened.

    Two sources, and the order matters. First the conversation: what the customer just asked
    determines what they are most likely to ask next, and a strip that ignores the last turn
    is indistinguishable from a static row of buttons. Then the ledger: a payment that landed
    a minute ago, something stuck, something failed.

    This used to be session-blind and the page only ever called it once, at greeting — so the
    same four chips sat there for the whole conversation however much changed underneath.
    """
    from engine import conversation          # local: data is imported by the engine

    txns = transactions_for()
    out: list[dict] = []
    seen: set[str] = set()

    def add(label: str, text: str) -> None:
        if text not in seen and len(out) < 6:
            seen.add(text)
            out.append({"label": label, "text": text})

    topic = conversation.topic(session) if session else None
    last_paid = conversation.last_payment(session) if session else None
    last = txns[0] if txns else None

    # ── 1. what they JUST did ───────────────────────────────────────────────────────────
    offer = conversation.split_offer(session) if session else None
    if offer:
        who = (offer["merchant"] or "").split()[0]
        add(f"Split that {offer['amount']:g} with someone",
            f"split my payment to {offer['merchant']} with")
        add(f"Who owes me?", "who owes me money")

    if last_paid:
        ref, payee = last_paid.get("ref"), (last_paid.get("payee") or "").split(" ")[0]
        add(f"Did it reach {payee}?" if payee else "Did it arrive?", f"status of {ref}")
        add("Get a receipt", f"status of {ref}")
        add("Refund that", f"refund {ref}")

    # ── 2. where the conversation is ────────────────────────────────────────────────────
    if topic in ("split_payment", "split_status"):
        add("Who owes me?", "who owes me money")
        try:
            from split import api as _split_api
            if _split_api.can_nudge():
                add("Remind them", "remind everyone about their share")
        except Exception:
            pass
        add("Split my last payment", "split my last payment")
    elif topic == "txn_history":
        add("Only the failed ones", "show me my failed transactions")
        add("This month", "show me my transactions this month")
        add("Show me more", "show me more transactions")
    elif topic in ("txn_status", "txn_failed", "debited_not_credited", "txn_on_hold"):
        add("All my transactions", "show me my last 5 transactions")
        add("I want a refund", "i want a refund")
    elif topic in ("refund_request", "refund_status"):
        add("How long will it take?", "how long does a refund take")
        add("My transactions", "show me my last 5 transactions")
    elif topic == "offers_nearby":
        add("Which is nearest?", "which is nearest")
        add("Which has the best offer?", "which has the best offer")
    elif topic in ("send_money", "pay_merchant"):
        add("What's the fee?", "what are the fees")
        add("What are my limits?", "what is my transfer limit")
    elif topic == "fees_query":
        add("How long does it take?", "how long does it take")
    elif topic == "limits_query":
        add("Send money", "i want to send money")

    # ── 3. what the ledger says needs attention ─────────────────────────────────────────
    if last and last.hours_ago < 0.5 and not last_paid:
        p = last.public()
        add(f"Did my {p['amount']:g} to {p['payee'].split()[0]} arrive?",
            f"status of {last.ref}")
    stuck = next((t for t in txns if t.state in (State.ON_HOLD, State.PENDING)
                  and t.hours_ago > 1), None)
    if stuck:
        add(f"Why is my {stuck.amount:g} still pending?", f"why is {stuck.ref} on hold")
    broke = next((t for t in txns if t.state is State.FAILED), None)
    if broke:
        add("Money debited but not received", "money was debited but not received")

    # ── 4. the standing offers, last ────────────────────────────────────────────────────
    live = [m for m in MERCHANTS if m.get("offer")]
    if live:
        m = live[0]
        add(f"{m['offer']['label']} at {m['name'].split()[0]}", "any offers near me")
    add("Offers near me", "any offers near me")
    add("My transactions", "show me my last 5 transactions")
    add("Send money", "i want to send money")
    add("What are my limits?", "what is my transfer limit")
    return out[:6]


_LIMIT_CACHE: dict[str, dict] = {}


def payee_limits(name: str) -> dict | None:
    """The mandate limits for a payee, read from the signed mandate itself.

    Needed at PROPOSAL time. Without it the assistant cheerfully offers to send AED 10,000 to
    someone whose mandate caps at 200, the customer confirms, and it fails afterwards — the
    guardrail works but the conversation wasted their time and looked broken.
    """
    if logged_in():
        return None              # no signed mandates for app logins; the wallet's own limits apply
    if name in _LIMIT_CACHE:
        return _LIMIT_CACHE[name]
    out = None
    try:
        import os as _os
        import sys as _sys
        amp = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if amp not in _sys.path:
            _sys.path.insert(0, amp)
        from app import mandate_store as ms
        for b in BENEFICIARIES:
            if b["name"] != name:
                continue
            m = ms.find_active(CUSTOMER["id"], b["agent"])
            if m:
                out = {"per_txn": float(m.limits.per_txn),
                       "per_day": float(m.limits.per_day),
                       "green_cap": float(m.green_tier_cap),
                       "currency": m.currency}
            break
    except Exception:
        out = None
    _LIMIT_CACHE[name] = out
    return out


def logged_in() -> bool:
    """A real customer logged in through the app (not the single configured demo sender)."""
    try:
        import userctx
        return not userctx.is_configured_sender()
    except Exception:                                              # noqa: BLE001
        return False


_ROSTERS: dict = {}
_ROSTER_TTL = 300.0


def roster(fresh: bool = False) -> list[dict]:
    """The people THIS customer can send to: everyone they have paid wallet-to-wallet.

    Built from their own ledger — each DEBIT they made, matched to the CREDIT leg of the same
    transaction, which is the receiver's own row and so carries the receiver's real userId
    and (encrypted) mobile. Nobody they have not paid appears here, so "send money to Alex"
    for a customer who has never paid a Alex finds nobody — as it must.

    Mobiles are kept on the SERVER: payments are addressed by mobile, but the browser only
    ever sees names. Cached per customer for five minutes.
    """
    import userctx
    uid = userctx.user_id()
    hit = _ROSTERS.get(uid)
    if hit and not fresh and time.time() - hit[0] < _ROSTER_TTL:
        return hit[1]
    out: list[dict] = []
    try:
        import history
        import pii
        db, _ = history._mongo()
        debits = list(db["walletLedger"].find(
            {"userId": uid, "txnType": "DEBIT", "merchantId": None},
            {"txnId": 1, "amount": 1, "createdDate": 1}, sort=[("createdDate", -1)], limit=200))
        by_txn = {str(d.get("txnId")): d for d in debits if d.get("txnId")}
        seen: dict[str, dict] = {}
        if by_txn:
            for c in db["walletLedger"].find(
                    {"txnId": {"$in": list(by_txn)}, "txnType": "CREDIT", "userId": {"$ne": uid}},
                    {"txnId": 1, "userId": 1, "mobileNumber": 1, "receiver": 1, "createdDate": 1,
                     "merchantId": 1}):
                ruid = c.get("userId")
                if not ruid or c.get("merchantId") or ruid in seen:
                    continue
                if any(m.get("userId") == ruid for m in MERCHANTS):
                    continue                     # a merchant's wallet, not a person
                mob = pii.decrypt(c.get("mobileNumber"))
                name = pii.decrypt(c.get("receiver")) or ""
                if not (mob and pii.looks_like_mobile(mob) and name):
                    continue
                d = by_txn.get(str(c.get("txnId"))) or {}
                seen[ruid] = {"name": name.strip(), "userId": ruid, "mobile": mob,
                              "corridor": "AE-AE", "mandate": False, "agent": None,
                              "wallet": "***" + mob[-3:],
                              "last_sent": float(str(d.get("amount") or 0) or 0),
                              "at": d.get("createdDate")}
        out = sorted(seen.values(), key=lambda r: r.get("at") or 0, reverse=True)[:15]
    except Exception:                                              # noqa: BLE001
        out = hit[1] if hit else []
    _ROSTERS[uid] = (time.time(), out)
    return out


def roster_entry(name: str | None) -> dict | None:
    """The one person on THIS customer's roster a name refers to, or None. Exact full name
    first, then a unique first-name match — never a guess between two people."""
    if not name:
        return None
    want = " ".join(str(name).lower().split())
    people = roster()
    exact = [p for p in people if p["name"].lower() == want]
    if exact:
        return exact[0]
    first = want.split()[0]
    hits = [p for p in people if p["name"].lower().split()[0] == first]
    return hits[0] if len(hits) == 1 else None


def sendable() -> list[dict]:
    """Who we may OFFER to send to.

    Only saved beneficiaries and people already paid. The assistant never invents a
    recipient and never offers a name the customer has not already transacted with — the
    suggestion list is drawn from history, not from anything generative.

    A logged-in customer gets THEIR roster; the fixture list below belongs to the single
    configured demo sender and is never shown to anyone else.
    """
    if logged_in():
        return [dict(p, recent=True) for p in roster()]
    paid = {t.payee for t in transactions_for()}
    out = []
    for b in BENEFICIARIES:
        out.append(dict(b, recent=b["name"] in paid))
    out.sort(key=lambda b: (not b["mandate"], not b["recent"]))
    return out


# ---------------------------------------------------------------------------------------
# Nearby merchants and their live offers
# ---------------------------------------------------------------------------------------
# The offers mirror the Incentive Engine's real shapes — flat discount, percentage with a
# cap, and fee waiver — because the point of showing them is that the customer pays the
# discounted amount through the same rails, not that a number appears on a card.

MERCHANTS = [
    {"mid": "MID7741", "name": "Mocha & More", "category": "Cafe", "metres": 40,
     "offer": {"type": "PERCENT", "value": 15, "cap": 12, "min_spend": 20,
               "label": "15% off, up to AED 12", "campaign": "CAMP-2291"}},
    {"mid": "MID3310", "name": "Al Noor Pharmacy", "category": "Pharmacy", "metres": 85,
     "offer": {"type": "FLAT", "value": 10, "cap": 10, "min_spend": 50,
               "label": "AED 10 off over AED 50", "campaign": "CAMP-2304"}},
    {"mid": "MID9028", "name": "Gulf Mart", "category": "Grocery", "metres": 160,
     "offer": {"type": "PERCENT", "value": 8, "cap": 25, "min_spend": 75,
               "label": "8% back, up to AED 25", "campaign": "CAMP-2288"}},
    {"mid": "MID5512", "name": "Spice Route", "category": "Restaurant", "metres": 210,
     "offer": {"type": "FEE_WAIVER", "value": 0, "cap": 0, "min_spend": 0,
               "label": "No payment fee today", "campaign": "CAMP-2310"}},
    {"mid": "MID1177", "name": "City Barbers", "category": "Salon", "metres": 340,
     "offer": None},
]


#: Real dev merchants, when configured. Put a "merchants" array in openwallet/.secrets.json and
#: these replace the fixtures above:
#:   {"mid": "...", "name": "...", "category": "...", "metres": 40,
#:    "mobile": "<the merchant wallet's mobile>", "userId": "<their user id>"}
#: `mobile` is what the transaction resolves the receiver by, and `userId` is what the
#: Incentive Engine scopes campaigns to as receiverType=MERCHANT. Without them the demo still
#: runs, but merchant payments settle against the configured peer and offers are indicative.
def _load_real_merchants() -> list[dict] | None:
    import json as _json
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "openwallet", ".secrets.json")
    try:
        cfg = _json.load(open(path))
    except Exception:
        return None
    rows = cfg.get("merchants")
    if not rows:
        return None
    out = []
    for i, m in enumerate(rows):
        out.append({"mid": m.get("mid") or f"MID{i}", "name": m["name"],
                    "category": m.get("category", "Merchant"),
                    "metres": int(m.get("metres", 50 + i * 70)),
                    "mobile": m.get("mobile"), "userId": m.get("userId"),
                    "offer": None, "real": True})
    return out


_real = _load_real_merchants()
if _real:
    MERCHANTS = _real


_MOBILE_CACHE: dict[str, str | None] = {}
_PAYABLE_CACHE: dict[str, bool] = {}


#: Why the last bulk lookup failed, if it did. Read by the engine so an empty merchant list
#: can be explained ("dev is unreachable") instead of silently looking like "no merchants".
_PAYABLE_ERROR: str | None = None

#: When the last bulk attempt ran, and how long before another is worth making.
_PAYABLE_TRIED: float = 0.0
_PAYABLE_RETRY: float = 15.0


def payable_error() -> str | None:
    return _PAYABLE_ERROR


def merchant_payable(user_id: str | None) -> bool:
    """Can this merchant actually be paid?

    Needs BOTH a decryptable mobile (the receiver is resolved by hashing one) and an ACTIVE
    `wallet_master` row. Three of the ten dev merchants have no wallet at all and return
    "No wallet found" at transact time — offering them, or putting a campaign on them, means
    advertising something the customer cannot complete.
    """
    if not user_id:
        return False
    if user_id not in _PAYABLE_CACHE:
        _load_payable_bulk()
    return _PAYABLE_CACHE.get(user_id, False)


def _load_payable_bulk() -> None:
    """Resolve wallets and mobiles for EVERY merchant in one pass.

    One query and one decrypt loop instead of a database round trip per merchant — the
    per-merchant version re-authenticated to Vault each time and took tens of seconds.
    """
    global _PAYABLE_TRIED
    ids = [m["userId"] for m in MERCHANTS if m.get("userId")]
    if not ids:
        return
    # Throttle. With the cache deliberately left empty on failure, an unguarded retry means
    # every merchant_payable() in the loop re-attempts the whole lookup — ten probes for one
    # screen. One attempt per window is enough to recover quickly without that.
    if (time.time() - _PAYABLE_TRIED) < _PAYABLE_RETRY:
        return
    _PAYABLE_TRIED = time.time()
    try:
        from devdb import q
        import pii
        marks = ",".join(["%s"] * len(ids))
        rows = q(f"""SELECT u.id, u.mobile_number,
                            MAX(CASE WHEN w.status='ACTIVE' THEN 1 ELSE 0 END) has_wallet
                     FROM user u LEFT JOIN wallet_master w ON w.user_id = u.id
                     WHERE u.id IN ({marks}) GROUP BY u.id, u.mobile_number""",
                 ids, n=len(ids) + 5)
        for r in rows:
            mob = pii.decrypt(r["mobile_number"])
            mob = mob if pii.looks_like_mobile(mob) else None
            _MOBILE_CACHE[r["id"]] = mob
            _PAYABLE_CACHE[r["id"]] = bool(r["has_wallet"]) and bool(mob)
        globals()["_PAYABLE_ERROR"] = None
    except Exception as exc:
        # Do NOT cache a verdict here. Caching False on a failed lookup is how a VPN that
        # dropped for ten seconds became "no merchant on the platform can be paid" for the
        # rest of the process's life — the retry never happened, because the cache was warm
        # with an answer nobody had actually given. An outage leaves the cache empty so the
        # next call tries again, and the reason is recorded so the UI can say which it is.
        global _PAYABLE_ERROR
        _PAYABLE_ERROR = f"{type(exc).__name__}: {exc}"


def merchant_mobile(user_id: str | None) -> str | None:
    """The merchant's plaintext mobile, decrypted from dev on first use and cached in memory.

    A payment resolves its receiver by hashing a plaintext mobile — there is no userId or
    wallet-id path — so this is required to pay a merchant at all. It is resolved at runtime
    rather than written into `.secrets.json`: customer PII should not sit in a config file on
    a laptop to save a database round trip.
    """
    if not user_id:
        return None
    if user_id not in _MOBILE_CACHE:
        _load_payable_bulk()            # one pass fills mobiles and wallet status together
    return _MOBILE_CACHE.get(user_id)


def merchant_by_name(name: str) -> dict | None:
    return next((m for m in MERCHANTS if m["name"] == name), None)


def _apply_real_distance(rows: list[dict]) -> list[dict]:
    """Replace the placeholder distance with the real one where a store location exists.

    `distance_source` records which it is, so nothing downstream presents an invented
    distance as a measured one.
    """
    try:
        import stores
    except Exception:
        return rows
    for m in rows:
        near = stores.nearest(m.get("userId"))
        if near:
            m["metres"] = near["metres"]
            m["store_name"] = near.get("store_name")
            m["address"] = near.get("address")
            m["distance_source"] = "store_location"
        else:
            # No store on file: do NOT keep a flattering placeholder, or a merchant we cannot
            # locate outranks one we measured. Mark the distance unknown and let the ranker
            # treat it neutrally.
            m["metres"] = None
            m["distance_source"] = "unknown"
    return rows


def nearby(limit: int = 4, payable_only: bool = True) -> list[dict]:
    """Nearest first.

    Merchants without a live campaign still appear — hiding them would make the list an advert
    rather than a map of where you can pay. Merchants we CANNOT pay are excluded by default,
    because offering one is a dead end the customer only discovers at the till.
    """
    rows = _apply_real_distance(list(MERCHANTS))
    if payable_only:
        rows = [m for m in rows if not m.get("userId") or merchant_payable(m["userId"])]
    # Unknown distance sorts last, but is still offered.
    return sorted(rows, key=lambda m: (m["metres"] is None, m["metres"] or 0))[:limit]


def apply_offer(merchant: dict, amount: float) -> dict:
    """What the customer actually pays. Mirrors the Incentive Engine's rules: a minimum
    spend gate and a cap, with the discount never exceeding the bill."""
    offer = merchant.get("offer")
    base_fee = 0.0
    if not offer or amount < offer.get("min_spend", 0):
        return {"amount": amount, "discount": 0.0, "payable": amount, "fee": base_fee,
                "applied": None,
                "why": "no offer applies" if not offer else
                       f"spend at least AED {offer['min_spend']:g} to use this offer"}
    if offer["type"] == "FLAT":
        discount = min(offer["value"], amount)
    elif offer["type"] == "PERCENT":
        discount = min(amount * offer["value"] / 100.0, offer["cap"])
    else:                                  # FEE_WAIVER
        discount = 0.0
    discount = round(min(discount, amount), 2)
    return {"amount": amount, "discount": discount, "payable": round(amount - discount, 2),
            "fee": base_fee, "applied": offer["label"], "campaign": offer["campaign"],
            "why": f"{offer['label']} applied automatically"}
