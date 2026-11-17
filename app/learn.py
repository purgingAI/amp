"""Feedback-driven learning layer for Ask-the-company.

Human feedback fine-tunes the assistant WITHOUT touching the laptop-built read-only code graph
(which the hourly sync overwrites). Validated answers (👍, or a 👎→regenerated better answer) are
saved here and RECALLED as grounding for similar future questions — so the graph effectively
learns from feedback in real time. Stored in the operational SQLite (survives graph re-syncs).
"""
from __future__ import annotations
import re, time
from . import store

_STOP = {"the", "and", "for", "what", "who", "how", "why", "does", "our", "are", "was", "with",
         "this", "that", "from", "about", "into", "give", "detail", "explain", "being", "built"}


def _ensure():
    with store.conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, rating TEXT,
            reason TEXT, keywords TEXT, agent TEXT, validated INTEGER DEFAULT 0, ts TEXT)""")


try:
    _ensure()
except Exception:
    pass


def _keywords(text: str) -> str:
    terms = [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOP]
    seen = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return " ".join(seen[:20])


def add(question: str, answer: str, rating: str, reason: str = "", agent: str = "", validated: int = 0):
    try:
        with store.conn() as c:
            c.execute("INSERT INTO learnings (question,answer,rating,reason,keywords,agent,validated,ts) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      ((question or "")[:2000], (answer or "")[:9000], rating, (reason or "")[:800],
                       _keywords(question), agent, int(validated), time.strftime("%Y-%m-%dT%H:%M:%S")))
    except Exception:
        pass


def recall(query: str, limit: int = 3) -> list:
    """Human-validated answers whose question matches the query — reused to ground new answers."""
    terms = [t for t in re.split(r"[^a-z0-9]+", (query or "").lower()) if len(t) > 2 and t not in _STOP]
    if not terms:
        return []
    try:
        with store.conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT question,answer,reason,keywords FROM learnings WHERE validated=1 "
                "ORDER BY id DESC LIMIT 300").fetchall()]
    except Exception:
        return []
    scored = []
    for r in rows:
        hay = (r.get("keywords", "") + " " + (r.get("question", "") or "")).lower()
        s = sum(hay.count(t) for t in terms)
        if s:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]


def list_all(limit: int = 200) -> list:
    try:
        with store.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT id,question,answer,rating,reason,agent,validated,ts FROM learnings "
                "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()]
    except Exception:
        return []


def delete(learning_id: int) -> bool:
    try:
        with store.conn() as c:
            c.execute("DELETE FROM learnings WHERE id=?", (int(learning_id),))
        return True
    except Exception:
        return False


def for_module(module_name: str, limit: int = 3) -> list:
    """Validated learnings that mention a module — folded into that module's deep-dive.
    Split camelCase so 'StoreManagement' matches learnings about 'store management'."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", module_name or "")
    return recall(spaced, limit=limit)


def stats() -> dict:
    try:
        with store.conn() as c:
            up = c.execute("SELECT COUNT(*) n FROM learnings WHERE rating IN ('up','corrected')").fetchone()["n"]
            down = c.execute("SELECT COUNT(*) n FROM learnings WHERE rating='down'").fetchone()["n"]
            val = c.execute("SELECT COUNT(*) n FROM learnings WHERE validated=1").fetchone()["n"]
        return {"positive": up, "negative": down, "validated": val}
    except Exception:
        return {"positive": 0, "negative": 0, "validated": 0}
