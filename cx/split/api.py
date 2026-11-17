"""Real split-payment client against dev `/transaction/split-payment/*`.

Nothing here is simulated. Split payment lives in the global codebase and is deployed on
dev, so every call in this module hits the same endpoints the app calls, with the same
encrypted envelope and the same JWT.

## The envelope, and the one thing that is easy to get wrong

`@EncryptedRequest` on `SplitPaymentController` uses its DEFAULT key type,
`EncKeyType.ENC_KEY` — so the wrapping key is `globalWalletKeyMapper.encKey`, **not**
`staticEncKey`. `/transaction/ext/transact` uses `staticEncKey`, and the two are different
32-char values for the same user. Using the transact key here fails with `InvalidTag` and no
further explanation, which cost a round of debugging.

    K       = 32-char random
    encKey  = AESGCM(K,             key=encKey,  iv=encKey[:16])
    encData = AESGCM(json(payload), key=K,       iv=K[:16])

Responses come back wrapped the same way: unwrap `encKey` with the user's encKey, then open
`encData` with the result.

## What is NOT reachable

`GET /history` returns 403 "Access denied: insufficient permissions" from the gateway's RBAC
layer — there is no `@PreAuthorize` on the controller, so this is the endpoint allow-list for
this user rather than anything in the code. `/split-dashboard` and `/groups/{id}/history`
cover the same ground and do answer, so the assistant uses those.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
AMP = os.path.dirname(CX)
sys.path.insert(0, os.path.join(AMP, "openwallet"))

import openwallet_api as _api  # noqa: E402

BASE = os.environ.get("OPENWALLET_BASE", _api.BASE)
ROOT = "/transaction/split-payment"
TIMEOUT = 30

_ENC: str | None = None
_LAST_ERROR: str | None = None


def last_error() -> str | None:
    return _LAST_ERROR


def _enc_key() -> str:
    """The user's `encKey` from `globalWalletKeyMapper`, read once.

    Looked up by `entityKey`, which IS the user's UUID when a userId is present —
    `getUserKeyFromMapper(userId)` does `findByEntityKey(userId)`. There is no `userId`
    field on that collection, which is why an obvious query returns nothing.
    """
    global _ENC
    import userctx
    u = userctx.current()
    if not u.get("configured"):
        # An app login registered its OWN encKey with the wallet at /user/login; that is the
        # key the wallet now expects for this customer.
        if u.get("enc_key"):
            return u["enc_key"]
        raise RuntimeError("no session key for this login")
    if _ENC is None:
        from pymongo import MongoClient
        c = _api.cfg()
        db = MongoClient(c["mongo_uri"], serverSelectionTimeoutMS=15000,
                         minPoolSize=1, maxPoolSize=4).get_default_database()
        doc = db["globalWalletKeyMapper"].find_one({"entityKey": c["sender"]})
        if not doc or not doc.get("encKey"):
            raise RuntimeError("no encKey in globalWalletKeyMapper for this user")
        _ENC = doc["encKey"]
    return _ENC


def _token() -> str:
    return _api.token_for()            # the logged-in customer (or the configured sender)


def _headers() -> list[tuple[str, str]]:
    return [("Content-Type", "application/json"), ("Authorization", "Bearer " + _token()),
            ("x-latitude", "25.2048"), ("x-longitude", "55.2708"),
            ("x-device-id", "payi-laptop"), ("X-Device-Model", "PayI-Demo"),
            ("X-Manufacturer", "OpenWallet"), ("Accept", "application/json")]


def call(method: str, path: str, payload: dict | None = None) -> dict | None:
    """One encrypted round trip. Returns the decrypted `data`, or None on failure.

    Returning None rather than raising is deliberate: every caller is answering a customer,
    and "I can't reach that right now" is a better answer than a stack trace. The reason is
    kept in `last_error()` so the engine can say WHICH it was.
    """
    global _LAST_ERROR
    _LAST_ERROR = None
    try:
        enc = _enc_key()
        body = None
        if payload is not None:
            import base64
            import secrets
            k = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
            body = json.dumps({"encKey": _api.gcm_encrypt(k, enc),
                               "encData": _api.gcm_encrypt(json.dumps(payload), k)}).encode()
        req = urllib.request.Request(BASE + ROOT + path, data=body, method=method)
        for h, v in _headers():
            req.add_header(h, v)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = json.loads(e.read() or b"{}")
        if "encData" not in raw:
            # An unencrypted body is an error from in front of the controller — a gateway
            # rejection, usually RBAC. The message is the useful part.
            _LAST_ERROR = raw.get("message") or f"HTTP {raw.get('status')}"
            return None
        key = _api.gcm_decrypt(raw["encKey"], enc) if raw.get("encKey") else enc
        out = json.loads(_api.gcm_decrypt(raw["encData"], key))
        if out.get("status") not in (200, 201):
            _LAST_ERROR = out.get("message") or f"status {out.get('status')}"
            return None
        return out.get("data")
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None


# ── reads ────────────────────────────────────────────────────────────────────────────────

_PEOPLE_CACHE: tuple[list[dict], float] = ([], 0.0)
_PEOPLE_TTL = 20.0


def people(fresh: bool = False) -> list[dict]:
    """Who this customer has splits with, and the net balance with each.

    `netBalance` is signed from the CUSTOMER's point of view: negative means they owe.

    Cached for 20s because `resolve_user_id` consults it, and slot extraction resolves a
    participant on every utterance — without this, naming two people in one sentence meant
    two round trips before the sentence was even understood.
    """
    global _PEOPLE_CACHE
    rows, at = _PEOPLE_CACHE
    if not fresh and rows and (time.time() - at) < _PEOPLE_TTL:
        return rows
    got = call("GET", "/people")
    if got is None:
        return rows            # a failed refresh must not erase what we know
    _PEOPLE_CACHE = (got, time.time())
    return got


def dashboard() -> dict:
    return call("GET", "/split-dashboard") or {}


def groups() -> list[dict] | None:
    """None means the lookup FAILED; [] means there genuinely are none.

    `d or []` collapsed those two, so a transient failure looked exactly like "this customer
    has no groups" — the same mistake that made every merchant unpayable during a VPN drop.
    """
    d = call("GET", "/groups")
    if d is None:
        return None
    if isinstance(d, dict):
        return d.get("content") or []
    return d


def group_history(group_id: str) -> list[dict]:
    d = call("GET", f"/groups/{group_id}/history")
    if isinstance(d, dict):
        return d.get("content") or []
    return d or []


_SPLIT_TXNS: tuple[set, float] = (set(), 0.0)
_SPLIT_TXNS_TTL = 30.0


def already_split(fresh: bool = False) -> set[str]:
    """Transaction ids that already carry a split.

    The backend refuses a second split on the same payment ("A split expense already exists
    for this transaction"). Discovering that only after the customer has reviewed the shares
    and pressed confirm wastes their time and reads as broken — the same failure as offering
    a transfer the mandate will reject. So the split target is chosen knowing this.

    Built from every group's history, because `GET /history` is 403 for this user at the
    gateway. A failed read returns what we already knew rather than an empty set, so an
    outage cannot make everything look splittable.
    """
    global _SPLIT_TXNS
    known, at = _SPLIT_TXNS
    if not fresh and known and (time.time() - at) < _SPLIT_TXNS_TTL:
        return known
    gs = groups()
    if gs is None:
        return known
    found: set[str] = set()
    for g in gs:
        for row in group_history(g.get("groupId") or ""):
            tid = row.get("transactionId")
            if tid:
                found.add(str(tid))
    _SPLIT_TXNS = (found, time.time())
    return found


def detail(split_id: str) -> dict | None:
    return call("GET", f"/{split_id}")


def icons() -> list[dict]:
    return call("GET", "/groups/icons") or []


def resolve_user_id(name: str) -> str | None:
    """A display name -> a real OpenWallet userId, or None.

    None is a legitimate and important answer. The assistant's roster comes from the mandate
    config, which carries agent ids and masked wallets but no userId — so a name being on
    the roster does NOT mean there is a OpenWallet account behind it to split with. Three real
    sources are tried, newest-information-first, and if none of them knows the person the
    split is refused rather than built with a guessed id.
    """
    import re
    want = {t for t in re.split(r"[^\w\u0900-\u097f\u0600-\u06ff]+", (name or "").lower())
            if len(t) > 2}
    if not want:
        return None

    # 1. people this customer already splits with — authoritative, and already loaded
    for p in people():
        got = {t for t in re.split(r"\W+", (p.get("name") or "").lower()) if len(t) > 2}
        if want & got:
            return p.get("userId")

    # 2. an app login: ONLY people on this customer's own roster. Searching the whole user
    #    table for "a customer called Alex" would split a bill with a stranger.
    try:
        import data as _cxdata
        if _cxdata.logged_in():
            p = _cxdata.roster_entry(name)
            return p["userId"] if p else None
    except Exception:
        return None

    # 2b. the identities the demo is configured with (keyed BY userId)
    try:
        c = _api.cfg()
        for uid, v in (c.get("people") or {}).items():
            got = {t for t in re.split(r"\W+", str(v.get("name") or "").lower())
                   if len(t) > 2}
            if want & got:
                return uid
    except Exception:
        pass

    # 3. the real user table
    try:
        sys.path.insert(0, CX)
        from devdb import q
        like = sorted(want, key=len, reverse=True)[0]
        rows = q("SELECT id, first_name, last_name FROM user "
                 "WHERE user_type='CUSTOMER' AND (LOWER(first_name)=%s OR LOWER(last_name)=%s) "
                 "LIMIT 5", (like, like), n=5)
        # Exactly one match or nothing: two people called Alex must not be resolved by
        # picking the first row.
        if len(rows) == 1:
            return rows[0]["id"]
    except Exception:
        pass
    return None


# ── writes ───────────────────────────────────────────────────────────────────────────────

def create_post_txn(reference_txn_id: str, participants: list[dict], initiator_display: float,
                    split_type: str = "SHARE", bill_description: str = "",
                    initiator_share: float | None = None, currency: str = "AED",
                    group_id: str | None = None, save_as_group: bool = False,
                    group_name: str | None = None, group_icon: str | None = None) -> str | None:
    """Split a payment the customer has ALREADY made. Returns the split id (`SP-…`).

    `displayValue` on every participant and `initiatorDisplayValue` are not decoration: the
    backend recomputes the split and rejects the request if its own arithmetic disagrees with
    the figures we showed. So whatever the customer was told is what must be sent — which is
    the right design, and it means the assistant can never quietly round somebody's share.

    **`allocationValue` is ALWAYS required, whatever the docs say.** `split-payment-api.md`
    states it is "Ignored for EQUAL/SHARE split" and only needed for CUSTOM/AMOUNT/PERCENTAGE.
    `ParticipantShareDTO` says otherwise:

        @NotNull(message = "Allocation value is required",
                 groups = {PreTxnSplit.class, PostTxnSplit.class})
        @Positive(...)
        private BigDecimal allocationValue;

    Bean validation runs before any split-type logic, so a SHARE split without it is rejected
    with "Allocation value is required" — which is what happened on the first real call. For
    an equal split the allocation IS the computed share, so it is filled in from
    `displayValue` here rather than asking every caller to remember.
    """
    # For SHARE, `allocationValue` is a WEIGHT — a number of shares — not an amount:
    #
    #     share = billAmount * allocationValue / totalShares        (SplitCalculator:185)
    #
    # Sending the computed 42.50 made totalShares 42.5, so one participant's share came out
    # as the entire 85 and the initiator's as zero, and the request was rejected for a
    # display-value mismatch. One share each is what an equal split means. For AMOUNT the
    # allocation IS the amount, and for PERCENTAGE it is the percentage, so both are passed
    # through untouched.
    if split_type == "SHARE":
        participants = [dict(p, allocationValue=1) for p in participants]
        if initiator_share is None:
            initiator_share = 1              # the payer is one share too
    else:
        participants = [dict(p, allocationValue=p.get("allocationValue",
                                                      p.get("displayValue")))
                        for p in participants]

    # THE INITIATOR GOES IN THE PARTICIPANTS LIST. The documented top-level `initiatorShare`
    # and `initiatorDisplayValue` are not what a post-transaction split reads:
    #
    #     BigDecimal initiatorShareFromList = initiatorInList.map(getAllocationValue)...
    #     // "Initiator share always comes from the participants list (allocationValue);
    #     //  if not present, default to ZERO"
    #     requestDTO.setParticipants(... filter(p -> !p.getUserId().equals(initiatorUserId)))
    #
    # The service looks the payer up in the list, takes their allocation and display value
    # from it, and only then removes them. Omitting them makes `initiatorExcluded` true,
    # which silently switches to a split where the payer pays NOTHING — a different split
    # from the one the customer was shown, and the display-value check rejects it.
    import userctx
    me = userctx.user_id()
    if not any(p.get("userId") == me for p in participants):
        participants = participants + [{"userId": me,
                                        "allocationValue": initiator_share
                                        if initiator_share is not None else 1,
                                        "displayValue": round(initiator_display, 2)}]
    body: dict = {"splitType": split_type, "splitFlowType": "POST_TRANSACTION",
                  "referenceTxnId": reference_txn_id, "participants": participants,
                  "initiatorDisplayValue": round(initiator_display, 2), "currency": currency}
    if bill_description:
        body["billDescription"] = bill_description
    if initiator_share is not None:
        body["initiatorShare"] = (initiator_share if split_type == "SHARE"
                                  else round(initiator_share, 2))
    if group_id:
        body["groupId"] = group_id
    if save_as_group:
        body["saveAsGroup"] = True
        if group_name:
            body["groupName"] = group_name
        if group_icon:
            body["groupIcon"] = group_icon
    return call("POST", "/create", body)


def create_pre_txn(qr_token: str, participants: list[dict], initiator_display: float,
                   split_type: str = "EQUAL", bill_description: str = "",
                   initiator_share: float | None = None) -> str | None:
    """Split BEFORE the merchant is paid: participants pay the initiator, who then pays."""
    body: dict = {"qrToken": qr_token, "splitType": split_type,
                  "splitFlowType": "PRE_TRANSACTION", "participants": participants,
                  "initiatorDisplayValue": round(initiator_display, 2)}
    if bill_description:
        body["billDescription"] = bill_description
    if initiator_share is not None:
        body["initiatorShare"] = round(initiator_share, 2)
    return call("POST", "/create", body)


#: None = we have not tried; False = the gateway refused us. Reminders are blocked for this
#: user today ("Access denied: insufficient permissions" — RBAC at the gateway, not in the
#: controller, which carries no @PreAuthorize). Rather than hide the feature for ever or
#: leave a button that always fails, the first refusal is remembered and the offer stops.
#: Grant the permission and it starts working again with no code change.
_CAN_NUDGE: bool | None = None


def can_nudge() -> bool:
    """Whether to OFFER a reminder. Optimistic until proven otherwise — probing by actually
    sending one would mean messaging a real person just to test a permission."""
    return _CAN_NUDGE is not False


def nudge(split_id: str, user_ids: list[str] | None = None) -> bool:
    global _CAN_NUDGE
    body = {"participantUserIds": user_ids} if user_ids else {}
    ok = call("POST", f"/{split_id}/nudge", body) is not None
    if not ok and "permission" in (last_error() or "").lower():
        _CAN_NUDGE = False
    elif ok:
        _CAN_NUDGE = True
    return ok


if __name__ == "__main__":                                  # quick smoke
    print("people    :", json.dumps(people())[:300])
    print("dashboard :", json.dumps(dashboard())[:300])
    print("groups    :", json.dumps(groups())[:300])
    print("error     :", last_error())
