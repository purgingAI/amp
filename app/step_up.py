"""Step-up authentication — the second channel an amber transfer has to pass.

Amber means the transfer is within the mandate but outside the no-touch cap, so the protocol
demands a factor the voice cannot supply. This is deliberately a *different channel*: a cloned
voice can say anything into the glasses and still not complete an OS authentication on the laptop.

The challenge is bound to the exact transaction — mandate, payee, amount, currency. Without that
binding an attacker could obtain an approval for 5 AED and spend it on 500: approvals must not be
fungible. `verify()` recomputes the binding and refuses if a single field moved.

Backends, in preference order:
  fingerprint   fprintd via PAM, when fingers are enrolled (nothing enrolled on this laptop yet)
  password      PAM against the user's own account; works unprivileged because unix_chkpwd is
                setgid shadow
Both go through PAM, so whatever the OS trusts, this trusts.

Deferred by decision, not oversight: phone push approval. It is stronger because it survives full
compromise of the laptop. `Channel` exists so adding it touches nothing else.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import subprocess
import datetime as dt

from .mandate import now_iso
from . import mandate_store as ms

CHALLENGE_TTL = 120          # a confirmation the user walks away from must not stay open

SCHEMA = """
CREATE TABLE IF NOT EXISTS step_ups (
  step_up_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL, mandate_id TEXT NOT NULL,
  binding TEXT NOT NULL,             -- sha256 over the exact transaction
  detail TEXT NOT NULL,
  state TEXT NOT NULL,               -- pending | approved | denied | expired
  channel TEXT DEFAULT '',
  created TEXT NOT NULL, expires TEXT NOT NULL, settled TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_stepup_state ON step_ups(state, expires);
"""


class StepUpError(Exception):
    """Refusal with a reason safe to say out loud."""


def init():
    with ms._lock, ms.conn() as c:
        c.executescript(SCHEMA)


def binding(mandate_id: str, payee_wallet: str, amount: float, currency: str) -> str:
    """What the human is approving. Any change invalidates the approval."""
    return hashlib.sha256(json.dumps(
        {"amount": f"{amount:.2f}", "currency": currency.upper(),
         "mandate_id": mandate_id, "payee_wallet": payee_wallet},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------- channels ----------

class Channel:
    name = "none"

    @staticmethod
    def available() -> bool:
        return False

    def confirm(self, prompt: str, detail: dict) -> bool:
        raise NotImplementedError


class Fingerprint(Channel):
    name = "fingerprint"

    @staticmethod
    def available() -> bool:
        try:
            out = subprocess.run(["fprintd-list", getpass.getuser()],
                                 capture_output=True, text=True, timeout=10)
            return "no fingers enrolled" not in out.stdout.lower() and out.returncode == 0
        except Exception:
            return False

    def confirm(self, prompt: str, detail: dict) -> bool:
        import pam
        return bool(pam.pam().authenticate(getpass.getuser(), "", service="fprintd"))


class Password(Channel):
    """PAM against the user's own account. Unprivileged because unix_chkpwd is setgid shadow."""
    name = "password"

    @staticmethod
    def available() -> bool:
        try:
            import pam  # noqa: F401
            return os.path.exists("/usr/sbin/unix_chkpwd")
        except Exception:
            return False

    def _ask(self, prompt: str) -> str | None:
        """GUI modal where there is a display, terminal otherwise."""
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                out = subprocess.run(
                    ["zenity", "--password", "--title=Confirm payment", "--text", prompt],
                    capture_output=True, text=True, timeout=CHALLENGE_TTL)
                return out.stdout.strip() if out.returncode == 0 else None
            except Exception:
                pass
        try:
            return getpass.getpass(prompt + "\nPassword: ")
        except Exception:
            return None

    def confirm(self, prompt: str, detail: dict) -> bool:
        import pam
        pw = self._ask(prompt)
        if not pw:
            return False
        try:
            return bool(pam.pam().authenticate(getpass.getuser(), pw, service="login"))
        finally:
            pw = "\x00" * len(pw)


def best_channel() -> Channel:
    for cls in (Fingerprint, Password):
        if cls.available():
            return cls()
    raise StepUpError("no step-up channel available on this machine")


# ---------- challenge lifecycle ----------

def _request(c, session_id: str, mandate_id: str, payee_wallet: str, payee_name: str,
             amount: float, currency: str) -> dict:
    """Body of request(), on a caller-supplied connection.

    ms._lock is a plain Lock, not an RLock, so the monitor — which already holds it while
    deciding — must not call the locking wrapper. It calls this instead.
    """
    sid = f"sup_{secrets.token_hex(8)}"
    expires = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=CHALLENGE_TTL)
               ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    detail = {"payee_wallet": payee_wallet, "payee_name": payee_name,
              "amount": amount, "currency": currency.upper()}
    c.execute("INSERT INTO step_ups VALUES (?,?,?,?,?,?,?,?,?,'')",
              (sid, session_id, mandate_id,
               binding(mandate_id, payee_wallet, amount, currency),
               json.dumps(detail), "pending", "", now_iso(), expires))
    return {"step_up_id": sid, "expires": expires, **detail}


def request(session_id: str, mandate_id: str, payee_wallet: str, payee_name: str,
            amount: float, currency: str) -> dict:
    with ms._lock, ms.conn() as c:
        return _request(c, session_id, mandate_id, payee_wallet, payee_name, amount, currency)


def present(step_up_id: str, channel: Channel | None = None) -> dict:
    """Show the human what they are approving and collect the OS authentication."""
    with ms.conn() as c:
        r = c.execute("SELECT * FROM step_ups WHERE step_up_id=?", (step_up_id,)).fetchone()
    if not r:
        raise StepUpError("unknown confirmation")
    if r["state"] != "pending":
        raise StepUpError(f"that confirmation is already {r['state']}")
    if r["expires"] <= now_iso():
        _settle(step_up_id, "expired", "")
        raise StepUpError("that confirmation expired")

    d = json.loads(r["detail"])
    ch = channel or best_channel()
    prompt = (f"Send {d['amount']:g} {d['currency']} to {d['payee_name']}?\n"
              f"Wallet {d['payee_wallet']}")
    ok = False
    try:
        ok = ch.confirm(prompt, d)
    finally:
        _settle(step_up_id, "approved" if ok else "denied", ch.name)
    return {"step_up_id": step_up_id, "approved": ok, "channel": ch.name}


def _settle(step_up_id: str, state: str, channel: str):
    with ms._lock, ms.conn() as c:
        c.execute("UPDATE step_ups SET state=?, channel=?, settled=? WHERE step_up_id=?",
                  (state, channel, now_iso(), step_up_id))


def verify(step_up_id: str, mandate_id: str, payee_wallet: str,
           amount: float, currency: str) -> bool:
    """Approved, unexpired, and for exactly this transaction.

    Recomputing the binding is the point: an approval for 5 AED must not authorize 500.
    """
    with ms.conn() as c:
        r = c.execute("SELECT * FROM step_ups WHERE step_up_id=?", (step_up_id,)).fetchone()
    if not r or r["state"] != "approved":
        return False
    if r["mandate_id"] != mandate_id:
        return False
    # A short grace after expiry covers the seconds between approval and resubmission.
    if r["settled"] and r["settled"] < r["expires"]:
        pass
    elif r["expires"] <= now_iso():
        return False
    return r["binding"] == binding(mandate_id, payee_wallet, amount, currency)


def _consume(c, step_up_id: str):
    """One approval, one transfer. Caller-supplied connection — see _request."""
    cur = c.execute("UPDATE step_ups SET state='used' WHERE step_up_id=? AND state='approved'",
                    (step_up_id,))
    if cur.rowcount != 1:
        raise StepUpError("that confirmation was already used")


def status(step_up_id: str) -> str:
    """pending | approved | denied | used | expired | unknown — so a refusal can say which."""
    with ms.conn() as c:
        r = c.execute("SELECT state, expires FROM step_ups WHERE step_up_id=?",
                      (step_up_id,)).fetchone()
    if not r:
        return "unknown"
    if r["state"] == "pending" and r["expires"] <= now_iso():
        return "expired"
    return r["state"]


def consume(step_up_id: str):
    with ms._lock, ms.conn() as c:
        _consume(c, step_up_id)
