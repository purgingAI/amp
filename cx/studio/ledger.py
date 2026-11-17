"""The decision ledger: one row per message the Orchestrator routed — what it saw, how it decided,
what the agent answered, how long each part took, and what people said about it afterwards.

Everything the Studio learns from starts here (X-ray, golden sets, the Teach-me inbox, replays).
Text is MASKED before it is written: codes and card/transaction numbers (the engine's own
security.scrub), e-mails, phone numbers and IBANs. Rows older than 30 days are deleted.

    studio/data/ledger.db   (SQLite, owner-only)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time

import store

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
DB = os.path.join(store.DATA, "ledger.db")
RETENTION_DAYS = int(os.environ.get("STUDIO_LEDGER_DAYS", "30"))
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    os.makedirs(store.DATA, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript("""
      create table if not exists decisions(
        id integer primary key autoincrement, at real, source text, session text,
        text text, lang text, chosen text, by text, confidence real, margin real,
        trace text, answer text, mode text, ms_route integer, ms_total integer,
        feedback integer default 0, label text, labelled_by text, labelled_at real,
        engine_intent text, engine_agent text);
      create index if not exists d_at on decisions(at);
      create index if not exists d_src on decisions(source, at);
    """)
    cols = {r[1] for r in c.execute("pragma table_info(decisions)")}
    for col, typ in (("faith", "text"), ("faith_score", "real")):      # added in phase C
        if col not in cols:
            c.execute(f"alter table decisions add column {col} {typ}")
    try:
        os.chmod(DB, 0o600)
    except OSError:
        pass
    return c


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?971|00971|0)?\s?5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
_IBAN = re.compile(r"\bAE\d{2}\s?(?:\d{4}\s?){4}\d{3}\b", re.I)


def mask(text: str) -> str:
    """What is stored and shown: never a code, card, account, phone or e-mail."""
    t = _IBAN.sub("‹iban›", text or "")
    t = _EMAIL.sub("‹email›", t)
    t = _PHONE.sub("‹phone›", t)
    try:
        import security
        t = security.scrub(t)
    except Exception:                                       # noqa: BLE001
        t = re.sub(r"(?<!\d)\d{4,19}(?!\d)", "••••", t)
    return t


def _decision(trace: dict) -> tuple[float | None, float | None]:
    """The deciding brain's confidence, and its margin over the runner-up."""
    src = (trace.get("laya") or {}) if trace.get("by") == "laya" else (trace.get("reflex") or {})
    probs = sorted((src.get("probs") or {}).values(), reverse=True)
    conf = src.get("confidence")
    margin = (probs[0] - probs[1]) if len(probs) > 1 else None
    return conf, margin


def record(source: str, text: str, trace: dict, session: str = "", engine_intent: str | None = None,
           engine_agent: str | None = None) -> int:
    conf, margin = _decision(trace)
    safe = mask(text)
    tr = dict(trace, text=safe)
    with _lock:
        c = _db()
        try:
            with c:
                cur = c.execute("""insert into decisions(at, source, session, text, lang, chosen, by, confidence, margin, trace,
                                   ms_route, engine_intent, engine_agent) values (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (time.time(), source, session, safe, trace.get("language"), trace.get("chosen") or trace.get("blocked"),
                                 trace.get("by"), conf, margin, json.dumps(tr, ensure_ascii=False, default=str),
                                 trace.get("ms"), engine_intent, engine_agent))
                return int(cur.lastrowid)
        finally:
            c.close()


def finish(did: int, answer: str, mode: str | None, ms_total: int) -> None:
    with _lock:
        c = _db()
        try:
            with c:
                c.execute("update decisions set answer=?, mode=?, ms_total=? where id=?", (mask(answer)[:4000], mode, ms_total, did))
        finally:
            c.close()


def get(did: int) -> dict | None:
    c = _db()
    try:
        r = c.execute("select * from decisions where id=?", (did,)).fetchone()
        return _row(r) if r else None
    finally:
        c.close()


def recent(limit: int = 50, source: str | None = None) -> list[dict]:
    c = _db()
    try:
        q, a = "select * from decisions", []
        if source:
            q += " where source=?"
            a.append(source)
        q += " order by id desc limit ?"
        a.append(max(1, min(500, limit)))
        return [_row(r, light=True) for r in c.execute(q, a)]
    finally:
        c.close()


def set_faith(did: int, faith: dict, score: float | None) -> None:
    with _lock:
        c = _db()
        try:
            with c:
                c.execute("update decisions set faith=?, faith_score=? where id=?", (json.dumps(faith, ensure_ascii=False), score, did))
        finally:
            c.close()


def set_feedback(did: int, value: int) -> None:
    with _lock:
        c = _db()
        try:
            with c:
                c.execute("update decisions set feedback=? where id=?", (max(-1, min(1, int(value))), did))
        finally:
            c.close()


def set_label(did: int, agent: str, by: str) -> None:
    with _lock:
        c = _db()
        try:
            with c:
                c.execute("update decisions set label=?, labelled_by=?, labelled_at=? where id=?", (agent, by, time.time(), did))
        finally:
            c.close()


def stats(days: int = 7) -> dict:
    c = _db()
    try:
        since = time.time() - days * 86400
        rows = c.execute("select source, lang, chosen, by, feedback, ms_route from decisions where at>?", (since,)).fetchall()
    finally:
        c.close()
    from collections import Counter
    ms = sorted(r["ms_route"] for r in rows if r["ms_route"] is not None)
    pct = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))] if ms else None  # noqa: E731
    return {"n": len(rows), "by_source": Counter(r["source"] for r in rows), "by_lang": Counter(r["lang"] for r in rows),
            "by_agent": Counter(r["chosen"] for r in rows), "by_brain": Counter(r["by"] for r in rows),
            "thumbs_up": sum(r["feedback"] == 1 for r in rows), "thumbs_down": sum(r["feedback"] == -1 for r in rows),
            "ms_p50": pct(.5), "ms_p95": pct(.95)}


def purge() -> int:
    with _lock:
        c = _db()
        try:
            with c:
                return c.execute("delete from decisions where at<?", (time.time() - RETENTION_DAYS * 86400,)).rowcount
        finally:
            c.close()


def _row(r, light: bool = False) -> dict:
    d = dict(r)
    d["faith"] = json.loads(d["faith"]) if d.get("faith") else None
    if light:
        d.pop("trace", None)
        d.pop("faith", None)
        d["answer"] = (d.get("answer") or "")[:160]
    else:
        d["trace"] = json.loads(d["trace"] or "{}")
    return d
