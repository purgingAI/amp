"""Real PIN validation, delegated to the wallet's own `/pin-validate`.

This module deliberately knows nothing about whether a PIN is good. It forwards the digits
to `PinController.validatePin` and reports what the wallet said. The wallet owns the PIN,
the hash, the attempt counter and the lockout policy; a chat surface holding a second
opinion on any of those would be a second place to get them wrong.

The PIN is never logged, stored, cached or echoed — not into the decision trace, not into an
error message, not into the event stream the watch screen reads.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "openwallet"))

import openwallet_api as _api  # noqa: E402

BASE = os.environ.get("OPENWALLET_BASE", _api.BASE)
DEVICE = os.environ.get("CX_DEVICE_ID", "payi-laptop")


def _enc_key() -> str:
    from split import api as split_api
    return split_api._enc_key()


def _ctx():
    c = _api.cfg()
    return c, c["people"][c["sender"]]


def _headers(extra: dict | None = None) -> list[tuple[str, str]]:
    # The customer whose PIN this is: the app login, or the configured demo sender.
    who = _api.me()
    import userctx
    u = userctx.current()
    tok = u.get("access_token") if not u.get("configured") and u.get("access_token") else _api.token_for(who)
    device = DEVICE if u.get("configured") else os.environ.get("CX_LOGIN_DEVICE_ID", "openwallet-assistant-demo")
    h = [("Content-Type", "application/json"), ("Authorization", "Bearer " + tok),
         # APIConstants: HEADER_MOBILE = "x-mobile", HEADER_DEVICE_ID = "x-device-id".
         # Sending them as "mobileNumber"/"deviceId" gets a 400 naming the header it wanted.
         ("x-mobile", who["mobile"]), ("Accept-Language", "en"),
         ("x-latitude", "25.2048"), ("x-longitude", "55.2708"),
         ("x-device-id", device), ("X-Device-Model", "PayI-Demo"),
         ("X-Manufacturer", "OpenWallet"), ("Accept", "application/json")]
    return h + list((extra or {}).items())


def _call(method: str, path: str, payload: dict | None = None):
    enc = _enc_key()
    body = None
    if payload is not None:
        import base64
        import secrets
        k = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
        body = json.dumps({"encKey": _api.gcm_encrypt(k, enc),
                           "encData": _api.gcm_encrypt(json.dumps(payload), k)}).encode()
    req = urllib.request.Request(BASE + path, data=body, method=method)
    for h, v in _headers():
        req.add_header(h, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = json.loads(e.read() or b"{}")
    if "encData" not in raw:
        return raw
    key = _api.gcm_decrypt(raw["encKey"], enc) if raw.get("encKey") else enc
    return json.loads(_api.gcm_decrypt(raw["encData"], key))


def exists() -> bool | None:
    """Does this customer have a PIN set? Read-only — it does not touch the attempt counter.

    Used as the reachability probe precisely for that reason: validating a deliberately
    wrong PIN to see whether the endpoint answers would increment the wallet's own failure
    count and could lock a real account to prove a connection works.
    """
    out = _call("GET", "/user/pin-exists")
    return out.get("data") if isinstance(out, dict) else None


def verify(pin: str) -> tuple[bool, str]:
    """(ok, reason). The reason matters as much as the verdict.

    `PinServiceImpl.pinValidate` answers:
      200 + data=true   correct
      401 + data=false  wrong PIN            -> the user row's counter has just moved
      400 + data=false  account locked/closed
    Anything else is OUR problem, not a wrong PIN, and must not be treated as one: the wallet
    freezes the account after five wrong attempts and we cannot undo that.
    """
    out = _call("POST", "/user/pin-validate", {"pin": pin})
    if not isinstance(out, dict):
        return False, "UNREACHABLE"
    status, data = out.get("status"), out.get("data")
    msg = str(out.get("message") or "").lower()
    if status in (200, 201) and data is not False:
        return True, "OK"
    if status == 401:
        if "freez" in msg or "lock" in msg or "frozen" in msg:
            return False, "LOCKED"
        return False, "WRONG_PIN"
    if status == 400 and ("lock" in msg or "clos" in msg or "inactive" in msg):
        return False, "LOCKED"
    return False, f"UNEXPECTED:{status}:{out.get('message') or ''}"


if __name__ == "__main__":
    print("pin set for this customer:", exists())
