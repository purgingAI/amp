"""The monitor — session authority and the only component that may authorize a transfer.

Two agents never talk directly. Each opens a session with the monitor, which holds the session
keys, issues single-use nonces, enforces the mandate, verifies the device signature and writes an
append-only decision log the ledger can be checked against.

On the cryptography, and why none of it is ours:

  identity      Ed25519, enrolled once per agent (mandate_store.enrol_key)
  session keys  X25519 ECDH -> HKDF-SHA256, salted with both nonces AND the mandate_id, so a
                session's keys are valid for that mandate and no other
  payload       ChaCha20-Poly1305 AEAD. The design doc says XChaCha20; `cryptography` ships the
                IETF variant, which is the same cipher with a 96-bit nonce. Safe here because
                each session has its own key and nonces are a strict counter that never repeats
                within one — but it is a deviation from the doc and worth knowing.
  rotation      a symmetric ratchet, k_next = HKDF(k, "amp-rotate"). Old keys are discarded, so
                compromising one window does not read the previous one.
  session token 32 random bytes. It carries no claims because there is nothing in it to read —
                the monitor holds the state it maps to. That is the "only the monitor can decode
                it" property, achieved by holding the data rather than by hiding an algorithm.

A nonce is spent the moment it is presented, before any other work, so two racing replays cannot
both pass.
"""
from __future__ import annotations

import json
import os
import secrets
import datetime as dt

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from . import ledger, mandate_store as ms, device_key as dk, step_up, notify, trace
from .mandate import MandateError, now_iso

SESSION_TTL = 3600          # a session lives an hour
ROTATE_SECONDS = 300        # keys ratchet every 5 minutes
NONCE_TTL = 120             # a transfer nonce is short-lived on purpose

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  token TEXT NOT NULL UNIQUE,        -- opaque handle; no structure, no claims
  agent_id TEXT NOT NULL,
  mandate_id TEXT NOT NULL,
  send_key TEXT NOT NULL, recv_key TEXT NOT NULL,
  generation INTEGER NOT NULL DEFAULT 0,
  counter INTEGER NOT NULL DEFAULT 0,
  opened TEXT NOT NULL, rotated TEXT NOT NULL, expires TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS session_nonces (
  nonce TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  issued TEXT NOT NULL, expires TEXT NOT NULL,
  spent_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL, mandate_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  verdict TEXT NOT NULL,             -- green | amber | refused
  reason TEXT DEFAULT '',
  amount REAL, currency TEXT,
  transfer_id TEXT DEFAULT '',
  evidence TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_sessions_agent ON sessions(agent_id, state);
CREATE INDEX IF NOT EXISTS ix_decisions_mandate ON decisions(mandate_id, decision_id);
"""


class MonitorError(Exception):
    """Refusal with a reason safe to say out loud."""


def init():
    with ms._lock, ms.conn() as c:
        c.executescript(SCHEMA)


def _later(seconds: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _derive(shared: bytes, nonce_a: str, nonce_b: str, mandate_id: str) -> tuple[bytes, bytes]:
    """Two directional keys. Binding mandate_id into the salt is what stops a session opened
    for one mandate being reused to authorize against another."""
    material = HKDF(algorithm=hashes.SHA256(), length=64,
                    salt=f"{nonce_a}|{nonce_b}|{mandate_id}".encode(),
                    info=b"amp-monitor-v1").derive(shared)
    return material[:32], material[32:]


def _ratchet(key: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"amp-rotate").derive(key)


# ---------- session ----------

def open_session(agent_id: str, mandate_id: str, agent_eph_pub: bytes,
                 agent_nonce: str) -> dict:
    """Establish a session for one agent against one mandate.

    Returns the opaque token, the monitor's ephemeral public key and its nonce, so the agent can
    derive the identical keys on its side.
    """
    m = ms.get(mandate_id)
    if not m:
        raise MonitorError("no such mandate")
    if agent_id not in (m.payer.agent_id, m.payee.agent_id):
        raise MonitorError("that agent is not a party to this mandate")
    if not m.active:
        raise MonitorError("that mandate is not active")

    payer_pub, payee_pub = ms.public_key(m.payer.agent_id), ms.public_key(m.payee.agent_id)
    if not (payer_pub and payee_pub and m.verify(payer_pub, payee_pub)):
        raise MonitorError("that mandate does not carry two valid signatures")

    mon_eph = X25519PrivateKey.generate()
    shared = mon_eph.exchange(X25519PublicKey.from_public_bytes(agent_eph_pub))
    mon_nonce = secrets.token_hex(8)
    send, recv = _derive(shared, agent_nonce, mon_nonce, mandate_id)

    session_id = f"ses_{secrets.token_hex(8)}"
    token = secrets.token_urlsafe(32)          # opaque: nothing to parse out of it
    with ms._lock, ms.conn() as c:
        c.execute("INSERT INTO sessions (session_id,token,agent_id,mandate_id,send_key,recv_key,"
                  "generation,counter,opened,rotated,expires,state)"
                  " VALUES (?,?,?,?,?,?,0,0,?,?,?,'open')",
                  (session_id, token, agent_id, mandate_id, send.hex(), recv.hex(),
                   now_iso(), now_iso(), _later(SESSION_TTL)))

    trace.step("session:" + session_id, "session.open", mandate_id=mandate_id,
               detail={"agent": agent_id})
    return {"session_id": session_id, "token": token,
            "monitor_eph_pub": mon_eph.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
            "monitor_nonce": mon_nonce, "expires": _later(SESSION_TTL)}


def _session(c, token: str):
    s = c.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    if not s:
        raise MonitorError("unknown session")
    if s["state"] != "open":
        raise MonitorError("that session is closed")
    if s["expires"] <= now_iso():
        raise MonitorError("that session has expired")
    return s


def rotate(token: str) -> dict:
    """Advance the ratchet. Old keys are discarded, giving forward secrecy across windows."""
    with ms._lock, ms.conn() as c:
        s = _session(c, token)
        send, recv = _ratchet(bytes.fromhex(s["send_key"])), _ratchet(bytes.fromhex(s["recv_key"]))
        new_token = secrets.token_urlsafe(32)
        c.execute("UPDATE sessions SET send_key=?, recv_key=?, generation=generation+1,"
                  " counter=0, rotated=?, token=? WHERE session_id=?",
                  (send.hex(), recv.hex(), now_iso(), new_token, s["session_id"]))
        gen = s["generation"] + 1
    return {"token": new_token, "generation": gen, "rotated": now_iso()}


def seal(token: str, plaintext: bytes) -> dict:
    """AEAD under the session key. The counter is the nonce and never repeats within a
    generation; rotation resets it under a fresh key."""
    with ms._lock, ms.conn() as c:
        s = _session(c, token)
        ctr = s["counter"] + 1
        c.execute("UPDATE sessions SET counter=? WHERE session_id=?", (ctr, s["session_id"]))
        key = bytes.fromhex(s["send_key"])
    nonce = ctr.to_bytes(12, "big")
    return {"generation": s["generation"], "counter": ctr,
            "ciphertext": ChaCha20Poly1305(key).encrypt(nonce, plaintext, None).hex()}


def unseal(token: str, generation: int, counter: int, ciphertext_hex: str) -> bytes:
    with ms.conn() as c:
        s = _session(c, token)
    if generation != s["generation"]:
        raise MonitorError("that message was sealed under a rotated-out key")
    key = bytes.fromhex(s["send_key"])
    try:
        return ChaCha20Poly1305(key).decrypt(counter.to_bytes(12, "big"),
                                             bytes.fromhex(ciphertext_hex), None)
    except Exception:
        raise MonitorError("that message failed authentication")


# ---------- nonces ----------

def issue_nonce(token: str) -> dict:
    with ms._lock, ms.conn() as c:
        s = _session(c, token)
        nonce = f"n_{secrets.token_hex(8)}"
        c.execute("INSERT INTO session_nonces VALUES (?,?,?,?,'')",
                  (nonce, s["session_id"], now_iso(), _later(NONCE_TTL)))
        trace.begin(nonce)
        trace._step(c, nonce, "nonce.issue", mandate_id=s["mandate_id"])
    return {"nonce": nonce, "expires": _later(NONCE_TTL)}


def _spend_nonce(c, session_id: str, nonce: str):
    """Spend first, work second — two racing replays must not both pass."""
    r = c.execute("SELECT * FROM session_nonces WHERE nonce=?", (nonce,)).fetchone()
    if not r:
        raise MonitorError("unknown nonce")
    if r["session_id"] != session_id:
        raise MonitorError("that nonce belongs to another session")
    if r["spent_at"]:
        raise MonitorError("that authorization was already used")
    if r["expires"] <= now_iso():
        raise MonitorError("that authorization expired")
    cur = c.execute("UPDATE session_nonces SET spent_at=? WHERE nonce=? AND spent_at=''",
                    (now_iso(), nonce))
    if cur.rowcount != 1:
        raise MonitorError("that authorization was already used")


def _log(c, s, nonce, verdict, reason, amount, currency, transfer_id, evidence):
    c.execute("INSERT INTO decisions (ts,session_id,mandate_id,nonce,verdict,reason,amount,"
              "currency,transfer_id,evidence) VALUES (?,?,?,?,?,?,?,?,?,?)",
              (now_iso(), s["session_id"], s["mandate_id"], nonce, verdict, reason,
               amount, currency, transfer_id, json.dumps(evidence)))


# ---------- the enforcement point ----------

def authorize_transfer(token: str, nonce: str, payee_wallet: str, amount: float,
                       currency: str, timestamp: str, signature_hex: str,
                       evidence: dict | None = None, step_up_id: str = "") -> dict:
    """Mandate + device signature + ledger, or a refusal with a reason.

    Amber never moves money here: it returns requires_step_up and the caller must come back with
    a fresh nonce after the human has confirmed on a second channel.
    """
    evidence = dict(evidence or {})
    with ms._lock, ms.conn() as c:
        s = _session(c, token)
        _spend_nonce(c, s["session_id"], nonce)
        m = ms.get(s["mandate_id"])
        trace._step(c, nonce, "transfer.request", mandate_id=m.mandate_id,
                    detail={"amount": amount, "currency": currency,
                            "payee": m.payee.display_name or payee_wallet})

        def refuse(reason, stage="mandate.check"):
            trace._step(c, nonce, stage, "refused", m.mandate_id, {"reason": reason})
            _log(c, s, nonce, "refused", reason, amount, currency, "", evidence)
            return MonitorError(reason)

        if payee_wallet != m.payee.wallet:
            raise refuse("that payee is not the one this mandate covers")

        device_pub = ms.public_key(m.payer.agent_id)
        if not device_pub:
            raise refuse("no device key enrolled for the payer")
        if not dk.verify_transfer(device_pub, signature_hex, m.mandate_id, nonce,
                                  payee_wallet, amount, currency, timestamp):
            raise refuse("the device signature did not verify", "signature.verify")
        trace._step(c, nonce, "signature.verify", mandate_id=m.mandate_id,
                    detail={"algorithm": "Ed25519", "sig": signature_hex[:16] + "\u2026"})

        counters = ms.counters(m.mandate_id)
        try:
            tier = m.check(amount, currency, **counters)
        except MandateError as e:
            raise refuse(str(e))
        trace._step(c, nonce, "mandate.check", mandate_id=m.mandate_id,
                    detail={"per_txn": m.limits.per_txn, "spent_today": counters["spent_today"],
                            "per_day": m.limits.per_day, "cap": m.green_tier_cap})
        trace._step(c, nonce, "tier.decide", mandate_id=m.mandate_id, detail={"tier": tier})

        if tier == "amber":
            # An approval must not be fungible, so the challenge is bound to THIS transaction.
            # A step_up_id already approved for exactly these details lets it through.
            if step_up_id and step_up.verify(step_up_id, m.mandate_id, payee_wallet,
                                             amount, currency):
                try:
                    step_up._consume(c, step_up_id)
                except step_up.StepUpError as e:
                    raise refuse(str(e))
                evidence["step_up_id"] = step_up_id
                trace._step(c, nonce, "stepup.approve", mandate_id=m.mandate_id,
                            detail={"step_up_id": step_up_id})
            else:
                if step_up_id:
                    st = step_up.status(step_up_id)
                    raise refuse({
                        "used": "that confirmation was already used",
                        "denied": "you declined that confirmation",
                        "expired": "that confirmation expired",
                        "pending": "that confirmation has not been approved yet",
                        "unknown": "unknown confirmation",
                    }.get(st, "that confirmation does not match this payment"))
                ch = step_up._request(c, s["session_id"], m.mandate_id, payee_wallet,
                                      m.payee.display_name or m.payee.wallet, amount, currency)
                trace._step(c, nonce, "stepup.request", mandate_id=m.mandate_id,
                            detail={"step_up_id": ch["step_up_id"]})
                _log(c, s, nonce, "amber", "step-up required", amount, currency, "", evidence)
                return {"state": "requires_step_up", "tier": "amber", "amount": amount,
                        "currency": currency, "payee": m.payee.display_name or m.payee.wallet,
                        "step_up_id": ch["step_up_id"], "expires": ch["expires"]}

    # Ledger work outside the monitor's lock: it takes its own, and it re-checks the mandate
    # independently rather than trusting the verdict above.
    try:
        auth = ledger.authorize(m, amount, currency, nonce, counters,
                                detail={"session": s["session_id"], **evidence})
        trace.step(nonce, "ledger.authorize", mandate_id=m.mandate_id,
                   detail={"transfer_id": auth["transfer_id"], "held": amount})
        done = ledger.capture(auth["transfer_id"])
        trace.step(nonce, "ledger.capture", mandate_id=m.mandate_id,
                   detail={"txn_id": done.get("txn_id"),
                           "debit": f"-{amount:g} {currency}", "credit": f"+{amount:g} {currency}"})
    except (ledger.LedgerError, MandateError) as e:
        with ms._lock, ms.conn() as c:
            _log(c, s, nonce, "refused", str(e), amount, currency, "", evidence)
        raise MonitorError(str(e))

    with ms._lock, ms.conn() as c:
        _log(c, s, nonce, tier, "", amount, currency, auth["transfer_id"], evidence)
        # The payee learns about it here. Emitted after capture, so an event can only ever
        # describe money that has already moved.
        notify._emit(c, m.payee.agent_id, "credited", {
            "transfer_id": auth["transfer_id"], "amount": amount, "currency": currency,
            "from_name": m.payer.display_name or m.payer.agent_id,
            "from_wallet": m.payer.wallet, "to_wallet": m.payee.wallet,
            "tier": tier, "at": now_iso()})
        trace._step(c, nonce, "event.emit", mandate_id=m.mandate_id,
                    detail={"to": m.payee.agent_id, "amount": amount, "currency": currency})

    return {"state": "captured", "tier": tier, "transfer_id": auth["transfer_id"],
            "txn_id": done.get("txn_id"), "amount": amount, "currency": currency,
            "payee": m.payee.display_name or m.payee.wallet}


def decisions(mandate_id: str | None = None) -> list[dict]:
    with ms.conn() as c:
        rows = c.execute("SELECT * FROM decisions" + (" WHERE mandate_id=?" if mandate_id else "")
                         + " ORDER BY decision_id",
                         (mandate_id,) if mandate_id else ()).fetchall()
    return [dict(r) for r in rows]
