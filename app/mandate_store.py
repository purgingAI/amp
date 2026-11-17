"""Persistence for mandates, agent identity keys, and the spend counters mandates are checked against.

Separate from store.py so the payment path has its own database file and its own blast radius:
nothing in the agent-mesh tables can alter what a mandate permits.

The counters live here rather than in the mandate because they are facts about history, not about
authority. `Mandate.check()` is a pure function of (mandate, amount, counters) — which is what makes
it testable and what lets the monitor and the ledger agree on the same answer independently.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import datetime as dt
from contextlib import contextmanager
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.hazmat.primitives import serialization

from .mandate import Mandate, now_iso

DB = Path(__file__).resolve().parent.parent / "data" / "pay.db"
DB.parent.mkdir(exist_ok=True)
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_keys (
  agent_id TEXT PRIMARY KEY,
  public_key TEXT NOT NULL,          -- raw Ed25519, hex
  enrolled TEXT NOT NULL,
  note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS mandates (
  mandate_id TEXT PRIMARY KEY,
  payer_agent TEXT NOT NULL, payee_agent TEXT NOT NULL,
  currency TEXT NOT NULL,
  state TEXT NOT NULL,               -- proposed | active | revoked | expired
  doc TEXT NOT NULL,                 -- the whole mandate as JSON
  created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transfers (
  transfer_id TEXT PRIMARY KEY,
  mandate_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  amount REAL NOT NULL, currency TEXT NOT NULL,
  state TEXT NOT NULL,               -- authorized | captured | reversed | failed
  ts TEXT NOT NULL,
  detail TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_mandate_pair ON mandates(payer_agent, payee_agent, state);
CREATE INDEX IF NOT EXISTS ix_transfer_mandate ON transfers(mandate_id, ts);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=8000")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with _lock, conn() as c:
        c.executescript(SCHEMA)


# ---- agent identity keys ----

def enrol_key(agent_id: str, public_key: Ed25519PublicKey, note: str = "",
              rotate: bool = False) -> None:
    """Record an agent's identity key. Enrolment is the trust anchor: everything downstream
    assumes this moment was supervised and genuine."""
    raw = public_key.public_bytes(serialization.Encoding.Raw,
                                  serialization.PublicFormat.Raw).hex()
    with _lock, conn() as c:
        existing = c.execute("SELECT public_key FROM agent_keys WHERE agent_id=?",
                             (agent_id,)).fetchone()
        # Silently replacing a key would let anyone who reaches this function take over an
        # agent's identity. Rotation must be explicit and audited, so refuse here.
        if existing and existing["public_key"] != raw and not rotate:
            raise ValueError(f"{agent_id} already has a different key enrolled")
        if existing and existing["public_key"] != raw:
            # Rotation invalidates every mandate this agent signed, because they verify
            # against the OLD key. The caller must re-mint them; this only records that the
            # change was deliberate.
            c.execute("INSERT INTO agent_keys VALUES (?,?,?,?)"
                      .replace("INSERT INTO", "INSERT OR REPLACE INTO"),
                      (agent_id + "#rotated-" + now_iso(), existing["public_key"],
                       now_iso(), f"superseded: {note}"))
        c.execute("INSERT OR REPLACE INTO agent_keys VALUES (?,?,?,?)",
                  (agent_id, raw, now_iso(), note))


def public_key(agent_id: str) -> Ed25519PublicKey | None:
    with conn() as c:
        r = c.execute("SELECT public_key FROM agent_keys WHERE agent_id=?", (agent_id,)).fetchone()
    if not r:
        return None
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(r["public_key"]))


# ---- mandates ----

def save(m: Mandate, state: str = "active") -> Mandate:
    with _lock, conn() as c:
        c.execute("INSERT OR REPLACE INTO mandates VALUES (?,?,?,?,?,?,?,?)",
                  (m.mandate_id, m.payer.agent_id, m.payee.agent_id, m.currency, state,
                   json.dumps(m.to_dict()),
                   now_iso(), now_iso()))
    return m


def get(mandate_id: str) -> Mandate | None:
    with conn() as c:
        r = c.execute("SELECT doc FROM mandates WHERE mandate_id=?", (mandate_id,)).fetchone()
    return Mandate.from_dict(json.loads(r["doc"])) if r else None


def find_active(payer_agent: str, payee_agent: str) -> Mandate | None:
    """The live mandate for this exact pair, or None.

    Exact agent ids only — there is deliberately no fuzzy or 'nearest' lookup here. A mis-heard
    payee must fall through to a refusal, never to somebody else's mandate.
    """
    with conn() as c:
        rows = c.execute(
            "SELECT doc FROM mandates WHERE payer_agent=? AND payee_agent=? AND state='active'"
            " ORDER BY updated DESC", (payer_agent, payee_agent)).fetchall()
    for r in rows:
        m = Mandate.from_dict(json.loads(r["doc"]))
        if m.active:
            return m
    return None


def revoke(mandate_id: str, by: str) -> Mandate | None:
    m = get(mandate_id)
    if not m:
        return None
    m.revoke(by)
    with _lock, conn() as c:
        c.execute("UPDATE mandates SET state='revoked', doc=?, updated=? WHERE mandate_id=?",
                  (json.dumps(m.to_dict()), now_iso(), mandate_id))
    return m


# ---- counters ----

def counters(mandate_id: str, at: str | None = None) -> dict:
    """Spend and velocity for this mandate. Only captured transfers count against limits —
    an authorization that never captured moved no money."""
    now = dt.datetime.fromisoformat((at or now_iso()).replace("Z", "+00:00"))
    day = now.date().isoformat()
    month = now.strftime("%Y-%m")
    hour_ago = (now - dt.timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with conn() as c:
        rows = c.execute(
            "SELECT amount, ts FROM transfers WHERE mandate_id=? AND state='captured'",
            (mandate_id,)).fetchall()
    return {
        "spent_today": sum(r["amount"] for r in rows if r["ts"][:10] == day),
        "spent_month": sum(r["amount"] for r in rows if r["ts"][:7] == month),
        "txns_last_hour": sum(1 for r in rows if r["ts"] >= hour_ago),
    }
