"""Provenance receipts — a reproducible, tamper-evident record of every decision the Orchestrator made.

Each routed turn becomes a receipt: the masked question (and its hash), the language, which brains decided
and how sure they were, the chosen agent, the exact model versions in force (reflex head, whether the
reasoning model was used, the live intent model, the knowledge base's retriever), whether calibrated
confidence was on, a snapshot of the live reflex's golden scores, and the passages the answer was built from
(by number + doc/page + a hash of their text). That "core" is canonicalised to JSON, hashed with SHA-256, and
chained to the receipt before it (``prev``), so the whole history is append-only and tamper-evident: change
one past receipt and every later hash stops matching.

    studio/data/receipts.jsonl    (append-only, owner-only; one JSON receipt per line)

``verify(hash)`` recomputes the chain from the start to catch tampering, and REPLAYS the routing decision
against the pinned reflex version to confirm the same agent is chosen today — proof the decision reproduces.

Faithfulness is NOT part of the integrity core (which protects the routing decision and its sources); it is a
quality signal joined from the ledger at display time, so this file never has to be rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import store

FILE = os.path.join(store.DATA, "receipts.jsonl")
GENESIS = "0" * 64
_lock = threading.Lock()


# ── hashing ───────────────────────────────────────────────────────────────────────────
def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _hash(prev: str, core: dict) -> str:
    return hashlib.sha256((prev + "\n").encode() + _canon(core)).hexdigest()


def _text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode()).hexdigest()[:16]


# ── building a receipt ──────────────────────────────────────────────────────────────────
def _versions(g: dict, chosen: str | None) -> dict:
    """The exact model versions a decision stood on."""
    v: dict = {"reflex_head": store.active("router")}
    try:
        import actions
        v["intent_model"] = actions.live_run()
    except Exception:                                       # noqa: BLE001
        v["intent_model"] = None
    try:
        import router
        node = router.node(g, chosen or "") or {}
        if node.get("kind") == "knowledge" and node.get("collection"):
            v["retriever"] = {node["collection"]: store.kb_active(node["collection"])}
    except Exception:                                       # noqa: BLE001
        pass
    return v


def _gate_snapshot() -> dict:
    """The live reflex head and the golden scores recorded when it was promoted."""
    try:
        rv = store.versions()["router"]
        act = rv.get("active")
        entry = next((x for x in rv.get("list", []) if x["id"] == act), {})
        return {"reflex_head": act, "scores": entry.get("scores") or {}}
    except Exception:                                       # noqa: BLE001
        return {}


def _core(did: int, g: dict, r: dict, masked: str, passages: list[dict], mode: str | None) -> dict:
    rx = r.get("reflex") or {}
    probs = sorted((rx.get("probs") or {}).values(), reverse=True)
    margin = round(probs[0] - probs[1], 4) if len(probs) > 1 else None
    chosen = r.get("chosen") or r.get("blocked")
    core = {
        "v": 1,
        "ledger": int(did),
        "at": round(time.time(), 3),
        "lang": r.get("language"),
        "q": masked,                                        # already masked — never raw PII
        "q_hash": _text_hash(masked),
        "by": r.get("by"),
        "chosen": chosen,
        "blocked": r.get("blocked"),
        "reflex": {"version": rx.get("version"), "confidence": round(rx.get("confidence"), 4) if rx.get("confidence") is not None else None, "margin": margin},
        "reasoning": {"used": r.get("by") == "laya", "confidence": round((r.get("laya") or {}).get("confidence", 0), 4)} if r.get("laya") else {"used": False},
        "versions": _versions(g, r.get("chosen")),
        "calibrated": _calibrated(),
        "gates": _gate_snapshot(),
        "mode": mode,
        "sources": [{"n": p.get("n"), "doc": p.get("doc"), "page": p.get("page"), "hash": _text_hash(p.get("text", ""))} for p in (passages or [])],
    }
    return core


def _calibrated() -> bool:
    try:
        import calibrate
        return bool(calibrate.get().get("applied"))
    except Exception:                                       # noqa: BLE001
        return False


# ── append / read ─────────────────────────────────────────────────────────────────────
def _tail_hash() -> tuple[str, int]:
    """The last receipt's hash and sequence number — what the next one chains onto."""
    prev, seq = GENESIS, 0
    if os.path.exists(FILE):
        last = None
        with open(FILE, "rb") as fh:
            for last in fh:                                 # cheap enough at this scale; one line kept
                pass
        if last:
            try:
                d = json.loads(last)
                prev, seq = d["hash"], int(d["seq"])
            except (ValueError, KeyError):
                pass
    return prev, seq


def record(did: int, g: dict, r: dict, text: str, passages: list[dict] | None = None, mode: str | None = None) -> str | None:
    """Write one receipt for a decision and return its hash. Never raises into the request path."""
    try:
        import ledger
        masked = ledger.mask(text or "")
    except Exception:                                       # noqa: BLE001
        masked = text or ""
    with _lock:
        try:
            prev, seq = _tail_hash()
            core = _core(did, g, r, masked, passages or [], mode)
            core["masked"] = masked != (text or "")         # PII was masked → replay runs on the masked form
            h = _hash(prev, core)
            os.makedirs(store.DATA, exist_ok=True)
            line = json.dumps({"seq": seq + 1, "hash": h, "prev": prev, "core": core}, ensure_ascii=False)
            with open(FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            try:
                os.chmod(FILE, 0o600)
            except OSError:
                pass
            return h
        except Exception:                                   # noqa: BLE001 — a receipt is never worth a failed answer
            return None


def _all() -> list[dict]:
    if not os.path.exists(FILE):
        return []
    out = []
    with open(FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _faith_of(did) -> tuple[float | None, int | None]:
    try:
        import ledger
        row = ledger.get(int(did))
        if row:
            f = row.get("faith") or {}
            cheap = (f.get("cheap") or {})
            return row.get("faith_score"), cheap.get("unsupported")
    except Exception:                                       # noqa: BLE001
        pass
    return None, None


def listing(limit: int = 60) -> dict:
    """Newest receipts first, light — for the Audit stream."""
    rows = _all()
    n = len(rows)
    out = []
    for d in reversed(rows[-max(1, min(400, limit)):]):
        c = d.get("core", {})
        out.append({"seq": d.get("seq"), "hash": d.get("hash"), "prev": d.get("prev"),
                    "at": c.get("at"), "ledger": c.get("ledger"), "q": c.get("q"), "lang": c.get("lang"),
                    "by": c.get("by"), "chosen": c.get("chosen"), "blocked": c.get("blocked"),
                    "mode": c.get("mode"), "sources": len(c.get("sources") or []),
                    "reflex": (c.get("reflex") or {}).get("version"), "calibrated": c.get("calibrated")})
    return {"receipts": out, "total": n}


def get(h: str) -> dict | None:
    """One receipt in full, with its chain neighbours and the faithfulness joined from the ledger."""
    rows = _all()
    for i, d in enumerate(rows):
        if d.get("hash") == h:
            score, unsupported = _faith_of(d.get("core", {}).get("ledger"))
            prevd = rows[i - 1] if i > 0 else None
            nextd = rows[i + 1] if i + 1 < len(rows) else None
            answer = None
            try:
                import ledger
                row = ledger.get(int(d["core"]["ledger"]))
                answer = (row or {}).get("answer")
            except Exception:                               # noqa: BLE001
                pass
            return {"receipt": d, "faith_score": score, "faith_unsupported": unsupported, "answer": answer,
                    "chain": {"prev": (prevd or {}).get("hash"), "next": (nextd or {}).get("hash"),
                              "seq": d.get("seq"), "total": len(rows)}}
    return None


# ── verification ───────────────────────────────────────────────────────────────────────
def _verify_chain(rows: list[dict], upto_hash: str | None = None) -> dict:
    """Recompute every hash from the start; report the first break."""
    prev = GENESIS
    for d in rows:
        core, want = d.get("core"), d.get("hash")
        if d.get("prev") != prev:
            return {"ok": False, "break_at": d.get("seq"), "why": "prev link does not match the receipt before it"}
        if _hash(prev, core) != want:
            return {"ok": False, "break_at": d.get("seq"), "why": "receipt contents do not match its hash (edited)"}
        prev = want
        if upto_hash and want == upto_hash:
            break
    return {"ok": True}


def verify(h: str) -> dict:
    """Chain integrity up to this receipt, plus a live replay of the routing decision."""
    rows = _all()
    d = next((x for x in rows if x.get("hash") == h), None)
    if not d:
        return {"found": False}
    chain = _verify_chain(rows, upto_hash=h)
    core = d.get("core", {})
    out = {"found": True, "chain_ok": chain["ok"], "chain": chain, "seq": d.get("seq"),
           "then_chosen": core.get("chosen"), "then_by": core.get("by")}
    # replay: route the (masked) question again through the pinned reflex version
    try:
        import router
        pinned = (core.get("versions") or {}).get("reflex_head")
        head_obj = None
        note = None
        if pinned and pinned != store.active("router"):
            path = os.path.join(store.DATA, "router", f"{pinned}.pkl")
            if os.path.exists(path):
                head_obj = router.load_head(pinned)
            else:
                note = f"pinned reflex {pinned} is no longer on disk — replayed against the live reflex"
        rr = router.route(core.get("q") or "", head_obj=head_obj)
        now = rr.get("chosen") or rr.get("blocked")
        same = now == core.get("chosen")
        if not same and core.get("masked"):
            # the question held personal data, so the stored text is masked — replaying the masked form can
            # route elsewhere than the original. Not a divergence we can judge, not a tamper.
            out.update(reproduced=None, now_chosen=now, now_by=rr.get("by"),
                       note="question contained personal data — replayed on the masked form, so the agent may differ")
        else:
            out.update(reproduced=same, now_chosen=now, now_by=rr.get("by"))
            if note:
                out["note"] = note
    except Exception as exc:                                # noqa: BLE001
        out.update(reproduced=None, replay_error=f"{type(exc).__name__}: {exc}")
    return out


def coverage(days: int = 7) -> dict:
    """What share of recent ledger decisions carry a receipt — shown on the Audit header."""
    try:
        import ledger
        rows = ledger.recent(500)
        recent_ids = {r["id"] for r in rows}
        have = {c.get("core", {}).get("ledger") for c in _all()}
        covered = len(recent_ids & have)
        return {"decisions": len(recent_ids), "with_receipt": covered,
                "pct": round(100 * covered / max(1, len(recent_ids)))}
    except Exception:                                       # noqa: BLE001
        return {"decisions": 0, "with_receipt": 0, "pct": 0}
