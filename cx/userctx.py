"""Who is logged in, for THIS request.

Before login existed the whole assistant served one configured customer: `openwallet/.secrets.json`
names a `sender`, and history, balance, payees, statements and payments all read it. A
management demo needs any real customer to log in and see their own wallet — so the
customer is now a per-request value:

    app logs in (mobile -> OTP -> PIN, against the wallet's own login API)
        -> server keeps the wallet's token and the user it belongs to, and hands the app an
           opaque app token (`auth`)
        -> every app request carries X-Cx-Auth: <auth>
        -> the server sets `current()` for the duration of that request
        -> every lookup asks `user_id()` instead of reading `cfg()["sender"]`

`current()` is a ContextVar, so concurrent requests on the threaded server cannot see each
other's customer. Work handed to another thread must carry it: use `run_as(user, fn)` or
`spawn(fn)` — a plain Thread starts with NO customer.

No login, no customer: when CX_REQUIRE_LOGIN=1 (the deployed demo) `user_id()` raises rather
than falling back to the configured sender. Silently answering with someone else's wallet is
the one failure a login exists to prevent. Locally, without that flag, the configured sender
remains the default so the existing tests and tools keep working.
"""
from __future__ import annotations

import contextvars
import json
import os
import secrets
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.dirname(HERE)

_CUR: contextvars.ContextVar[dict | None] = contextvars.ContextVar("cx_user", default=None)
_LOCK = threading.Lock()
#: app token -> logged-in user. In memory only: a restart logs everyone out, which for a demo
#: server is the right failure (no wallet tokens on disk).
_LOGINS: dict[str, dict] = {}
LOGIN_TTL = float(os.environ.get("CX_LOGIN_TTL", str(8 * 3600)))


class NotLoggedIn(Exception):
    pass


def require_login() -> bool:
    return os.environ.get("CX_REQUIRE_LOGIN", "0") == "1"


def _configured() -> dict:
    """The single configured sender — the pre-login behaviour, kept for local use."""
    try:
        c = json.load(open(os.path.join(AMP, "openwallet", ".secrets.json")))
        uid = c["sender"]
        p = (c.get("people") or {}).get(uid) or {}
        return {"user_id": uid, "name": p.get("name") or "Sagar Uniyal", "mobile": p.get("mobile"),
                "configured": True}
    except Exception:                                              # noqa: BLE001
        return {"user_id": None, "name": "", "configured": True}


def current() -> dict:
    u = _CUR.get()
    if u:
        return u
    if require_login():
        raise NotLoggedIn("no customer is logged in for this request")
    return _configured()


def user_id() -> str:
    uid = current().get("user_id")
    if not uid:
        raise NotLoggedIn("no customer id")
    return uid


def name() -> str:
    return current().get("name") or ""


def is_configured_sender() -> bool:
    """True when this request is the single configured demo customer — the only one whose
    demo fixtures (mandates, enrolled device key, roster names) are real."""
    return bool(current().get("configured"))


def set_current(user: dict | None):
    return _CUR.set(user)


def reset(token) -> None:
    _CUR.reset(token)


def run_as(user: dict | None, fn, *a, **kw):
    tok = _CUR.set(user)
    try:
        return fn(*a, **kw)
    finally:
        _CUR.reset(tok)


def spawn(fn, *a, **kw) -> threading.Thread:
    """A thread that carries the current customer with it."""
    ctx = contextvars.copy_context()
    t = threading.Thread(target=lambda: ctx.run(fn, *a, **kw), daemon=True)
    t.start()
    return t


def wrap(fn):
    """For executor.submit / map: bind the CURRENT customer into the callable. Captures the
    value, not the Context — one Context cannot be entered by several pool threads at once."""
    u = _CUR.get()
    return lambda *a, **kw: run_as(u, fn, *a, **kw)


# ── logins ───────────────────────────────────────────────────────────────────────────────
def remember(user: dict) -> str:
    """Store a completed login; return the opaque app token the browser will carry."""
    tok = "cxa_" + secrets.token_urlsafe(24)
    with _LOCK:
        _LOGINS[tok] = dict(user, at=time.time())
        # tidy expired ones
        for k in [k for k, v in _LOGINS.items() if time.time() - v["at"] > LOGIN_TTL]:
            _LOGINS.pop(k, None)
    return tok


def lookup(app_token: str | None) -> dict | None:
    if not app_token:
        return None
    with _LOCK:
        u = _LOGINS.get(app_token)
        if not u:
            return None
        if time.time() - u["at"] > LOGIN_TTL:
            _LOGINS.pop(app_token, None)
            return None
        return u


def forget(app_token: str | None) -> None:
    with _LOCK:
        _LOGINS.pop(app_token or "", None)


def public(u: dict) -> dict:
    """What the browser may see about its own login — never the wallet token or keys."""
    return {"name": u.get("name"), "mobile": (u.get("mobile") or "")[-4:].rjust(4, "•"),
            "user_id": u.get("user_id")}
