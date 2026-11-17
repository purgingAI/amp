"""Bar tuning: which reflex bar routes the labelled sentences best — measured, not guessed.

For every candidate bar the WHOLE Orchestrator decision runs (router.route, unchanged), on the golden
sentences and the human labels. Each model is asked once per sentence: reflex, reasoning model and
document searches are memoised for the sweep, so twelve bars cost about one pass.
Per bar: accuracy, and how many messages had to wait for the slower reasoning model. The
recommendation is the most accurate bar; among equals, the one that needs the reasoning model least.
Nothing changes until a person applies it (which saves a new flow version).
"""
from __future__ import annotations

import threading
import time

import store

FILE = "bar_tuning.json"
BARS = [round(0.25 + 0.05 * i, 2) for i in range(12)]          # 0.25 … 0.80
_lock = threading.Lock()


def run(by: str) -> dict:
    import calibrate
    import rag
    import router
    with _lock:
        g = router.graph()
        cands = router.candidates(g)
        ids = {c["id"] for c in cands}
        rows = [(t, a) for t, a in calibrate.labelled() if a in ids]
        store.write(FILE, dict(get(), state="running", started=time.time(), n=len(rows)))
        orig = (router.reflex, router.ask_laya, rag.search)
        memo: dict = {}

        def cached(fn, name):
            def w(*a, **k):
                key = (name, a[0], tuple(c["id"] for c in a[1]) if name != "search" else a[1], k.get("k"), id(a[2]) if name == "reflex" and len(a) > 2 else None)
                if key not in memo:
                    memo[key] = fn(*a, **k)
                return memo[key]
            return w
        router.reflex, router.ask_laya, rag.search = cached(orig[0], "reflex"), cached(orig[1], "laya"), cached(orig[2], "search")
        try:
            curve, cur = [], g["settings"]["reflex_threshold"]
            for bar in BARS:
                gg = dict(g, settings=dict(g["settings"], reflex_threshold=bar))
                ok = slow = 0
                miss = []
                for t, a in rows:
                    r = router.route(t, gg)
                    ok += r.get("chosen") == a
                    slow += bool(r.get("laya"))
                    if r.get("chosen") != a:
                        miss.append(t)
                curve.append({"bar": bar, "right": ok, "n": len(rows), "acc": round(ok / max(1, len(rows)), 4),
                              "reasoning": slow, "reasoning_rate": round(slow / max(1, len(rows)), 4), "misses": miss[:8]})
        finally:
            router.reflex, router.ask_laya, rag.search = orig
        # move only for MORE right answers — equal accuracy keeps today's bar (a one-sentence saving is noise)
        top = max(c["right"] for c in curve)
        now_pt = next((c for c in curve if abs(c["bar"] - cur) < 1e-9), None)
        best = now_pt if now_pt and now_pt["right"] == top else max(curve, key=lambda c: (c["right"], -c["reasoning"], -abs(c["bar"] - cur)))
        now = next((c for c in curve if abs(c["bar"] - cur) < 1e-9), None)
        out = {"state": "done", "at": time.time(), "by": by, "n": len(rows), "reflex": (router.head() or {}).get("version"),
               "current": cur, "current_point": now, "recommended": best["bar"], "curve": curve,
               "calibrated": bool(calibrate.get().get("applied"))}
        store.write(FILE, out)
        return out


def get() -> dict:
    return store.read(FILE) or {"state": "never"}


def start(by: str) -> None:
    s = get()
    if s.get("state") == "running" and time.time() - s.get("started", 0) < 900:
        raise RuntimeError("already running")
    threading.Thread(target=run, args=(by,), daemon=True).start()
