"""The real Incentive Engine, used for discovery and for the exact discount before paying.

Two endpoints, which are the two moments that matter to a customer:

  POST /incentive/campaign/customer/list   — "what can I save around here?"  (discovery)
  POST /incentive/campaign/customer/quote  — "what will I actually pay?"     (before the tap)

Discovery omits the amount, exactly as the app does on screen-load, so the customer sees
which campaigns exist. The quote is the binding one: it supplies the amount, the merchant as
`receiverType=MERCHANT`, and the engine enforces the amount range, the per-user cap and the
remaining budget before telling us what comes off.

**The quote is authoritative, not our arithmetic.** We do not compute the discount locally
and then hope the backend agrees — the transaction is sent with `isDiscounted` and the
campaign the engine quoted, and if the engine says a campaign no longer applies the customer
pays full price. Showing a discount the engine would refuse is how a demo becomes a
complaint.

If the service is unreachable the caller gets `None` and falls back to the local catalogue,
and the UI says the offers are indicative. It never fabricates a discount it cannot honour.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

AMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(AMP, "openwallet"))

BASE = os.environ.get("OPENWALLET_BASE", "https://api.example.com")
LIST_PATH = "/incentive/campaign/customer/list"
QUOTE_PATH = "/incentive/campaign/customer/quote"
TIMEOUT = float(os.environ.get("CX_INCENTIVE_TIMEOUT", "6"))

#: MERCHANT_PAYMENT (product 90), handled by WalletToWalletHandler alongside SEND_MONEY.
#: NOT SEND_MONEY_MERCHANT (62) — that code has no handler and falls back to wallet-to-bank.
PRODUCT_CODE = os.environ.get("CX_PRODUCT_CODE", "MERCHANT_PAYMENT")
#: Transfers abroad (product 1). Fee discounts for a corridor live under this code.
CROSS_BORDER_PRODUCT = "CROSS_BORDER_TRANSFER"

_last_error: str | None = None


def last_error() -> str | None:
    return _last_error


def _token() -> str | None:
    """Mint the same HS512 customer token the transaction call uses."""
    try:
        import openwallet_api
        return openwallet_api.token_for()           # the logged-in customer
    except Exception:
        return None


def _user_id() -> str | None:
    try:
        import userctx
        return userctx.user_id()
    except Exception:
        return None


def _post(path: str, payload: dict):
    global _last_error
    token = _token()
    if not token:
        _last_error = "no credentials configured"
        return None
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    for h, v in (("Content-Type", "application/json"),
                 ("Authorization", "Bearer " + token),
                 ("Accept", "application/json")):
        req.add_header(h, v)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except Exception:
            _last_error = f"HTTP {e.code}"
            return None
        _last_error = f"HTTP {e.code}: {body.get('message') or ''}"[:120]
        return None
    except Exception as e:
        _last_error = f"{type(e).__name__}"
        return None
    _last_error = None
    # ResponseWrapper: {status, message, data}
    return body.get("data") if isinstance(body, dict) else None


def discover(receiver_id: str | None = None, amount: float | None = None,
             corridor: str = "AE-AE", product_code: str | None = None) -> list[dict] | None:
    """Campaigns this customer can use. Amount omitted on screen-load, like the app.

    For a transfer abroad pass product_code=CROSS_BORDER_PRODUCT and the DESTINATION country
    code ("PK") as corridor — the engine compares it exactly with the code Transaction sends,
    so "AE-PK" matches nothing."""
    user = _user_id()
    if not user:
        return None
    query = {"productCode": product_code or PRODUCT_CODE, "userId": user, "corridor": corridor}
    if receiver_id:
        query["receiverId"] = receiver_id
        query["receiverType"] = "MERCHANT"
    if amount is not None:
        query["amount"] = amount
    out = _post(LIST_PATH, query)
    return out if isinstance(out, list) else None


def quote(amount: float, receiver_id: str | None = None, campaign_id=None,
          promo_code: str | None = None, transfer_fee: float | None = None,
          corridor: str = "AE-AE", reserve: bool = False,
          product_code: str | None = None) -> dict | None:
    """The binding number. `reserve=True` holds the budget until the payment lands."""
    user = _user_id()
    if not user:
        return None
    req = {"productCode": product_code or PRODUCT_CODE, "userId": user, "transactionAmount": amount,
           "currency": "AED", "corridor": corridor, "reserve": reserve}
    if transfer_fee is not None:
        req["transferFee"] = transfer_fee
    if receiver_id:
        req["receiverId"] = receiver_id
        req["receiverType"] = "MERCHANT"
    if campaign_id is not None:
        req["campaignId"] = campaign_id
    if promo_code:
        req["promoCode"] = promo_code
    out = _post(QUOTE_PATH, req)
    return out if isinstance(out, dict) else None


def as_offer(c: dict) -> dict:
    """One campaign in the shape the UI already renders, so the live and fallback paths look
    identical to the screen and differ only in the `source` badge."""
    benefit = (c.get("benefitType") or "").upper()
    value = c.get("discountValue")
    cap = c.get("maxDiscountCap")
    if benefit == "PERCENTAGE":
        label = f"{_num(value)}% off" + (f", up to AED {_num(cap)}" if cap else "")
    elif benefit == "FLAT":
        label = f"AED {_num(value)} off"
        if c.get("minTxnAmount"):
            label += f" over AED {_num(c['minTxnAmount'])}"
    else:
        label = c.get("displayText") or c.get("name") or "offer"
    return {
        "type": "PERCENT" if benefit == "PERCENTAGE" else "FLAT",
        "value": float(value or 0), "cap": float(cap or 0),
        "min_spend": float(c.get("minTxnAmount") or 0),
        "label": c.get("displayText") or label,
        "campaign": c.get("promoCode") or c.get("incentiveCode") or str(c.get("campaignId")),
        "campaign_id": c.get("campaignId"),
        "applies_on": c.get("appliesOn"),
        "incentive_type": c.get("incentiveType"),
        "apply_mode": c.get("applyMode"),
        "source": "live",
    }


def _num(v) -> str:
    try:
        f = float(v)
        return f"{f:g}"
    except (TypeError, ValueError):
        return str(v)
