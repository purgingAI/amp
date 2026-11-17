"""Shadow mode: real app messages go through the Orchestrator too — silently. Nothing is answered,
nothing reaches the customer; we only record what the Orchestrator WOULD have done next to what the
live engine did, so disagreements can be reviewed in the Teach-me inbox.

Source: the engine's own session store (~/amp/data/cx_sessions.db, table turns: text + the engine's
answer with intent, confidence and `model`). Only turns the intent MODEL classified are replayed —
replies read in context ("100 dirhams" answering "how much?", model = held-open / continuation /
chosen …) mean nothing on their own and would only add noise.

Everything is masked before it is stored (ledger.mask). Switch: learning settings → shadow.enabled.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import ledger
import learning
import router
import store

DB = os.environ.get("STUDIO_SESSIONS_DB", os.path.expanduser("~/amp/data/cx_sessions.db"))
STATE = "shadow.json"
BATCH = 25


def engine_agent(intent: str | None, model: str | None) -> str | None:
    if (model or "").startswith(("security", "abuse")):
        return "guardrails"
    if intent == "human_handoff":
        return "human"
    if intent == "out_of_scope":
        return "decline"
    return "action" if intent else None


def run_once() -> dict:
    if not os.path.exists(DB):
        return {"ok": False, "why": "no engine session store on this machine"}
    st = store.read(STATE, {}) or {}
    last = int(st.get("last_id", 0))
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    try:
        rows = c.execute("select id, session_id, at, text, answer from turns where id > ? order by id limit ?", (last, BATCH)).fetchall()
    finally:
        c.close()
    g = router.graph()
    seen = kept = 0
    for tid, session, at, text, answer in rows:
        last = tid
        seen += 1
        try:
            a = json.loads(answer or "{}")
        except ValueError:
            a = {}
        model = a.get("model") or ""
        if not text or not (model.startswith("openwallet-cx-intent") or model.startswith(("security", "abuse"))):
            continue                                        # context replies: meaningless on their own
        r = router.route(text, g)
        r["engine"] = {"intent": a.get("intent"), "confidence": a.get("confidence"), "model": model, "outcome": a.get("outcome")}
        did = ledger.record("shadow", text, r, session=session, engine_intent=a.get("intent"),
                            engine_agent=engine_agent(a.get("intent"), model))
        ledger.finish(did, a.get("say") or "", "engine", int(a.get("ms") or 0))
        kept += 1
    st.update(last_id=last, at=time.time(), seen=st.get("seen", 0) + seen, kept=st.get("kept", 0) + kept)
    store.write(STATE, st)
    return {"ok": True, "seen": seen, "kept": kept, "last_id": last}


def status() -> dict:
    st = store.read(STATE, {}) or {}
    total = None
    if os.path.exists(DB):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
            total = c.execute("select max(id) from turns").fetchone()[0]
            c.close()
        except sqlite3.Error:
            pass
    return {"enabled": learning.settings()["shadow"]["enabled"], "source": DB if os.path.exists(DB) else None,
            "last_id": st.get("last_id", 0), "latest_turn": total, "replayed": st.get("kept", 0), "read": st.get("seen", 0), "at": st.get("at")}


def worker() -> None:
    def loop():
        while True:
            try:
                if learning.settings()["shadow"]["enabled"] and router.laya_state()["state"] in ("ready", "unavailable"):
                    r = run_once()
                    if r.get("ok") and r["seen"] == BATCH:
                        continue                            # catching up: no pause
            except Exception:                               # noqa: BLE001 — shadowing must never take the studio down
                pass
            time.sleep(20)
    threading.Thread(target=loop, daemon=True, name="shadow").start()
