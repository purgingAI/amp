"""A topic-scoped bundle of what one agent knows, in a form another agent can absorb.

The gesture moves a capsule. A capsule is everything this agent holds about one topic,
gathered from the three places that knowledge actually lives:

    learnings        amp.db        what the agent was told and rated
    notes            knowledge.db  written context about people and projects
    module_summaries knowledge.db  what the code does, per module

It is content-addressed. The id is a hash of the payload, so the same knowledge transferred
twice is the same capsule, and `apply` can recognise what it already holds instead of
duplicating it — which matters because the demo transfers the same topic repeatedly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
AMP_DB = DATA / "amp.db"
KNOW_DB = DATA / "knowledge.db"

CAPSULE_VERSION = 1


def _rows(db: Path, sql: str, params=()) -> list:
    """Query a store, tolerating one that has not been created yet.

    A fresh machine has no knowledge.db. That is a capsule with fewer sections, not an
    error: the receiving agent in a demo is usually the empty one.
    """
    if not db.exists():
        return []
    try:
        c = sqlite3.connect(db, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in c.execute(sql, params).fetchall()]
        finally:
            c.close()
    except sqlite3.Error:
        return []


def _like(topic: str) -> str:
    return f"%{(topic or '').strip().lower()}%"


def _digest(sections: dict) -> str:
    """Hash over the content, with keys sorted, so the id does not depend on row order."""
    blob = json.dumps(sections, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def build(topic: str, owner: str = "") -> dict:
    """Collect everything known about `topic` into one transferable capsule."""
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("a capsule needs a topic — transferring 'everything' is not a gesture")
    t = _like(topic)

    learnings = _rows(AMP_DB,
        "SELECT question, answer, rating, reason, keywords, agent, validated, ts "
        "FROM learnings WHERE lower(question) LIKE ? OR lower(answer) LIKE ? "
        "OR lower(keywords) LIKE ? ORDER BY ts DESC LIMIT 500", (t, t, t))

    notes = _rows(KNOW_DB,
        "SELECT * FROM notes WHERE lower(COALESCE(body,'')) LIKE ? "
        "OR lower(COALESCE(title,'')) LIKE ? LIMIT 300", (t, t))

    modules = _rows(KNOW_DB,
        "SELECT * FROM module_summaries WHERE lower(COALESCE(module,'')) LIKE ? "
        "OR lower(COALESCE(summary,'')) LIKE ? LIMIT 300", (t, t))

    sections = {"learnings": learnings, "notes": notes, "modules": modules}
    counts = {k: len(v) for k, v in sections.items()}
    return {
        "v": CAPSULE_VERSION,
        "topic": topic,
        "owner": owner,
        "created_at": time.time(),
        "id": _digest(sections),
        "counts": counts,
        "total": sum(counts.values()),
        "sections": sections,
    }


def verify_integrity(cap: dict) -> bool:
    """Recompute the content hash and compare it to the id the capsule claims.

    The signed envelope covers the MANIFEST — topic, counts, id — not the payload. That is
    deliberate (the payload can be large), but it means an edit that leaves the shape intact,
    such as rewriting one answer, changes nothing the signature protects. Recomputing the
    digest here closes that: the id is a hash of the sections, so any edit to them moves it.
    """
    return _digest(cap.get("sections") or {}) == cap.get("id")


def manifest(cap: dict) -> dict:
    """What the UI shows while the capsule is in the air — no payload, just shape."""
    return {"id": cap["id"], "topic": cap["topic"], "owner": cap.get("owner", ""),
            "counts": cap.get("counts", {}), "total": cap.get("total", 0)}


def _ensure_learnings(db: Path) -> None:
    c = sqlite3.connect(db, timeout=10)
    try:
        c.execute("""CREATE TABLE IF NOT EXISTS learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, rating TEXT,
            reason TEXT, keywords TEXT, agent TEXT, validated INTEGER DEFAULT 0, ts REAL)""")
        c.commit()
    finally:
        c.close()


def apply(cap: dict, receiver: str) -> dict:
    """Merge a capsule into the receiving agent's stores.

    Idempotent by content: a learning whose question and answer already exist is skipped
    rather than inserted again, so re-running the demo does not multiply rows. Returns what
    it actually took, which is what the receiving orb reports — claiming "47 items" while
    silently writing 3 would make the animation a lie.
    """
    if not isinstance(cap, dict) or cap.get("v") != CAPSULE_VERSION:
        raise ValueError(f"unsupported capsule version: {cap.get('v') if isinstance(cap, dict) else '?'}")

    sections = cap.get("sections") or {}
    learnings = sections.get("learnings") or []
    added = skipped = 0

    if learnings:
        _ensure_learnings(AMP_DB)
        c = sqlite3.connect(AMP_DB, timeout=10)
        try:
            for row in learnings:
                q, a = row.get("question"), row.get("answer")
                # Scoped to the receiver. Both laptops talk to the same hub and the same
                # amp.db — the learnings table separates agents by its `agent` column, not by
                # having a database each. Deduping globally would find the SENDER's rows and
                # skip every one, so the receiver would end up with nothing while the result
                # cheerfully reported success.
                dup = c.execute(
                    "SELECT 1 FROM learnings WHERE question IS ? AND answer IS ? AND agent IS ?",
                    (q, a, receiver)).fetchone()
                if dup:
                    skipped += 1
                    continue
                c.execute(
                    "INSERT INTO learnings (question,answer,rating,reason,keywords,agent,validated,ts) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (q, a, row.get("rating"), row.get("reason"), row.get("keywords"),
                     receiver, row.get("validated", 0), time.time()))
                added += 1
            c.commit()
        finally:
            c.close()

    # notes and module summaries travel in the capsule and are shown to the receiving agent,
    # but are not written into knowledge.db: that store is rebuilt from git history by
    # codegraph_build.py, and rows inserted by hand are silently dropped on its next run.
    # Writing them would look like it worked and quietly lose the data later.
    carried = len(sections.get("notes") or []) + len(sections.get("modules") or [])

    return {"capsule": cap["id"], "topic": cap.get("topic"), "receiver": receiver,
            "learnings_added": added, "learnings_skipped": skipped,
            "context_carried": carried,
            "note": "notes and module summaries are carried for context, not written to "
                    "knowledge.db, which is regenerated from git history"}
