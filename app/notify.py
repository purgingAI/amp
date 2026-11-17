"""Credited events — how the payee's agent learns money arrived.

An outbox rather than a direct call: the payee's laptop may be asleep, offline, or mid-reboot when
a transfer captures, and a payment that succeeded must not depend on the recipient being awake to
hear about it. Events persist until acknowledged.

Deliberately carries no authority. An event is a notification, not an instruction — nothing on the
receiving side can spend, approve or alter anything because of one. The money already moved in the
ledger before this row was written; if the outbox were lost entirely, the balances would still be
right.
"""
from __future__ import annotations

import json

from . import mandate_store as ms
from .mandate import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,            -- who it is for
  kind TEXT NOT NULL,                -- credited | revoked | mandate_proposed
  payload TEXT NOT NULL,
  ts TEXT NOT NULL,
  acked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_events_agent ON events(agent_id, event_id);
"""


def init():
    with ms._lock, ms.conn() as c:
        c.executescript(SCHEMA)


def _emit(c, agent_id: str, kind: str, payload: dict) -> int:
    """Caller-supplied connection — the monitor emits while holding ms._lock, which is a plain
    Lock, so it must not go through the locking wrapper (see step_up._request)."""
    cur = c.execute("INSERT INTO events (agent_id,kind,payload,ts,acked) VALUES (?,?,?,?,0)",
                    (agent_id, kind, json.dumps(payload), now_iso()))
    return cur.lastrowid


def emit(agent_id: str, kind: str, payload: dict) -> int:
    with ms._lock, ms.conn() as c:
        return _emit(c, agent_id, kind, payload)


def poll(agent_id: str, since: int = 0, limit: int = 50) -> list[dict]:
    with ms.conn() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE agent_id=? AND event_id>? ORDER BY event_id LIMIT ?",
            (agent_id, since, limit)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def ack(event_id: int):
    with ms._lock, ms.conn() as c:
        c.execute("UPDATE events SET acked=1 WHERE event_id=?", (event_id,))


def balance_of(agent_id: str) -> dict:
    """The payee's own wallet, derived from the ledger — never from the event stream."""
    from . import ledger
    with ms.conn() as c:
        r = c.execute("SELECT doc FROM mandates WHERE payee_agent=? ORDER BY updated DESC LIMIT 1",
                      (agent_id,)).fetchone()
    if not r:
        return {}
    from .mandate import Mandate
    m = Mandate.from_dict(json.loads(r["doc"]))
    return {"wallet": m.payee.wallet, "currency": m.currency,
            "balance": ledger.balance(m.payee.wallet)}
