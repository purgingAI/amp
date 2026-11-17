"""Step-level trace of a transfer, so the flow can be watched as it happens.

The decision log records *what was decided*. This records *how it got there* — every hop between
the agent, the monitor and the ledger, with timings. It exists for the observer screen: a person
watching should see the real backend sequence, not an animation loosely inspired by one.

Correlated by nonce. A nonce is unique per transfer attempt and already travels through every
stage, so it needs no new plumbing to thread a trace through the whole path.

Append-only and authority-free, like the outbox: nothing reads a trace to make a decision. If the
table were dropped, every balance and every verdict would be unchanged.
"""
from __future__ import annotations

import json
import time

from . import mandate_store as ms
from .mandate import now_iso

# The hops a transfer makes, in order. The observer draws its graph from this list, so a stage
# added here appears on screen without the UI knowing anything about payments.
STAGES = [
    ("session.open",      "agent_a", "monitor", "Session opened"),
    ("nonce.issue",       "monitor", "agent_a", "Nonce issued"),
    ("transfer.request",  "agent_a", "monitor", "Transfer requested"),
    ("signature.verify",  "monitor", "monitor", "Device signature verified"),
    ("mandate.check",     "monitor", "monitor", "Mandate and limits checked"),
    ("tier.decide",       "monitor", "monitor", "Risk tier decided"),
    ("stepup.request",    "monitor", "agent_a", "Step-up requested"),
    ("stepup.approve",    "agent_a", "monitor", "Step-up approved"),
    ("ledger.authorize",  "monitor", "ledger",  "Funds reserved"),
    ("ledger.capture",    "ledger",  "ledger",  "Entries posted"),
    ("event.emit",        "monitor", "agent_b", "Payee notified"),
]
STAGE_INFO = {s[0]: {"frm": s[1], "to": s[2], "label": s[3]} for s in STAGES}

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_seq INTEGER PRIMARY KEY AUTOINCREMENT,
  nonce TEXT NOT NULL,
  mandate_id TEXT DEFAULT '',
  stage TEXT NOT NULL,
  status TEXT NOT NULL,              -- ok | refused
  detail TEXT DEFAULT '{}',
  ts TEXT NOT NULL,
  elapsed_ms REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_traces_nonce ON traces(nonce, trace_seq);
"""

_started: dict[str, float] = {}


def init():
    with ms._lock, ms.conn() as c:
        c.executescript(SCHEMA)


def _step(c, nonce: str, stage: str, status: str = "ok",
          mandate_id: str = "", detail: dict | None = None) -> int:
    """Caller-supplied connection: the monitor traces while holding ms._lock, which is a plain
    Lock (see step_up._request for the same reason)."""
    t0 = _started.setdefault(nonce, time.monotonic())
    cur = c.execute(
        "INSERT INTO traces (nonce,mandate_id,stage,status,detail,ts,elapsed_ms)"
        " VALUES (?,?,?,?,?,?,?)",
        (nonce, mandate_id, stage, status, json.dumps(detail or {}), now_iso(),
         round((time.monotonic() - t0) * 1000, 1)))
    return cur.lastrowid


def step(nonce: str, stage: str, status: str = "ok",
         mandate_id: str = "", detail: dict | None = None) -> int:
    with ms._lock, ms.conn() as c:
        return _step(c, nonce, stage, status, mandate_id, detail)


def begin(nonce: str):
    """Mark t0 so elapsed_ms is measured from the first hop, not from process start."""
    _started[nonce] = time.monotonic()


def since(seq: int = 0, limit: int = 200) -> list[dict]:
    with ms.conn() as c:
        rows = c.execute("SELECT * FROM traces WHERE trace_seq>? ORDER BY trace_seq LIMIT ?",
                         (seq, limit)).fetchall()
    return [{**dict(r), "detail": json.loads(r["detail"]),
             **STAGE_INFO.get(r["stage"], {"frm": "monitor", "to": "monitor",
                                           "label": r["stage"]})} for r in rows]


def of(nonce: str) -> list[dict]:
    with ms.conn() as c:
        rows = c.execute("SELECT * FROM traces WHERE nonce=? ORDER BY trace_seq",
                         (nonce,)).fetchall()
    return [{**dict(r), "detail": json.loads(r["detail"]),
             **STAGE_INFO.get(r["stage"], {})} for r in rows]
