"""Bots, sessions and turns — who is talking to which agent, and everything that happened.

    login            -> bot_id      one per customer per agent ("bot_" + 12 hex), stable:
                                    logging in again returns the same bot
    open the agent   -> session_id  a new conversation under that bot ("ses_" + 16 hex)
    each turn        -> a row with the text, the speech-to-text details, the model's scores,
                       the engine's full decision trace and the timings

The CONVERSATION MEMORY itself (what "that one" or "and last month?" refers to) is the
engine's, keyed by session_id — exactly as before; this module adds identity and a durable
record. In production that memory moves to Redis so a session survives a restart and is
shared across instances; the ids here are already the keys it would use.

Never stored: the PIN. It travels only on /api/cx/act to the wallet's own validator, and
that endpoint does not write here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cx_sessions.db")
AGENT = "payment-agent"
_LOCK = threading.Lock()


@contextmanager
def _conn():
    """One connection per call, committed AND closed. `with sqlite3.connect(...)` only
    commits — it never closes — and the console's 2-second refresh leaked a handle per call
    until the engine hit its 1,024 open-file limit and could open nothing at all: chat
    turns stuck, speech requests failed. Closing is not optional."""
    c = _open()
    try:
        yield c
        c.commit()
    finally:
        c.close()


_SCHEMA_DONE = False


def _open():
    global _SCHEMA_DONE
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    if _SCHEMA_DONE:
        return c
    _SCHEMA_DONE = True
    c.executescript("""
        create table if not exists bots(bot_id text primary key, customer_id text, customer_name text,
            agent text, created real, last_login real);
        create table if not exists sessions(session_id text primary key, bot_id text, channel text,
            started real, last_at real, turns integer default 0);
        create table if not exists turns(id integer primary key autoincrement, session_id text, at real,
            text text, stt text, answer text, ms integer);
        create index if not exists turns_session on turns(session_id);
    """)
    return c


def login(customer_id: str, customer_name: str = "", agent: str = AGENT) -> dict:
    bot_id = "bot_" + hashlib.sha256(f"{customer_id}|{agent}".encode()).hexdigest()[:12]
    now = time.time()
    with _LOCK, _conn() as c:
        c.execute("insert into bots(bot_id, customer_id, customer_name, agent, created, last_login) "
                  "values(?,?,?,?,?,?) on conflict(bot_id) do update set last_login=excluded.last_login",
                  (bot_id, customer_id, customer_name, agent, now, now))
    return {"bot_id": bot_id, "agent": agent, "customer": customer_name}


def start(bot_id: str, channel: str = "app") -> dict | None:
    with _LOCK, _conn() as c:
        if not c.execute("select 1 from bots where bot_id=?", (bot_id,)).fetchone():
            return None
        sid = "ses_" + uuid.uuid4().hex[:16]
        now = time.time()
        c.execute("insert into sessions values(?,?,?,?,?,0)", (sid, bot_id, channel, now, now))
    return {"session_id": sid, "bot_id": bot_id, "started": now}


def known(session_id: str) -> bool:
    with _conn() as c:
        return bool(c.execute("select 1 from sessions where session_id=?", (session_id,)).fetchone())


def record(session_id: str, text: str, stt: dict | None, answer: dict, ms: int) -> int | None:
    if not session_id or not session_id.startswith("ses_"):
        return None
    with _LOCK, _conn() as c:
        cur = c.execute("insert into turns(session_id, at, text, stt, answer, ms) values(?,?,?,?,?,?)",
                        (session_id, time.time(), text, json.dumps(stt or {}), json.dumps(answer, default=str), ms))
        c.execute("update sessions set last_at=?, turns=turns+1 where session_id=?", (time.time(), session_id))
        return cur.lastrowid


def overview(limit: int = 30) -> dict:
    with _conn() as c:
        bots = [dict(r) for r in c.execute("select * from bots order by last_login desc limit 10")]
        sess = [dict(r) for r in c.execute(
            "select s.*, b.customer_name from sessions s join bots b using(bot_id) "
            "order by s.last_at desc limit ?", (limit,))]
    return {"bots": bots, "sessions": sess}


def turns(session_id: str) -> list[dict]:
    with _conn() as c:
        rows = [dict(r) for r in c.execute("select * from turns where session_id=? order by id", (session_id,))]
    for r in rows:
        r["stt"] = json.loads(r["stt"] or "{}")
        r["answer"] = json.loads(r["answer"] or "{}")
    return rows
