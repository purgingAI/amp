"""The Oracle: a pre-mortem for next week — and a forecaster that is scored on what really happened.

forecast() streams:
    traffic   the real customer requests of the last LOOKBACK days (ledger: app + shadow — not the Studio, not tests)
    shapes    those requests grouped by meaning (the live reflex's own encoder); each shape gets a PRESSURE — recent
              requests weigh more (half-life HALF_LIFE days) — and the top shapes are kept
    referee   for each shape: what the customers need and which agent SHOULD answer. People's labels win; otherwise a
              referee model reads the agents' descriptions and decides (marked so); without one, the live routing of the
              confident members is assumed (marked so — it can be wrong)
    tomorrow  for each shape, the customers of next week: new messages with the same need in the forms traffic drifts
              into — other languages, typos, code-mixing, shorter or angrier. Every one is routed by the live flow.
    weather   per shape: clear / cloudy / storm from the share the live flow would get wrong (or be unsure about)

Every forecast is stored. backtest() takes each forecast once real traffic has arrived after it: arrivals are assigned
to the nearest shape, and only PEOPLE's evidence counts (a label, a thumbs up or down): did the storms bring failures,
did the clear skies stay clear? The Oracle's own accuracy is shown next to its forecasts — or "waiting for evidence".
Nothing here changes a model, a label or a setting.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np

import router
import store

LOOKBACK = 14
HALF_LIFE = 3.0
SIM = 0.80                       # same shape: cosine on the live reflex's encoder
TOP = 7
HORIZON = 7 * 86400
SOURCES = ("app", "shadow")
LANGS = ("en", "hinglish", "hi", "ar")


def _enc():
    import encoder
    h = router.head() or {}
    return encoder.get(h.get("encoder")), h.get("version")


def _emb(texts: list[str]) -> np.ndarray:
    enc, _ = _enc()
    X = enc.encode(texts, max_len=64).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _traffic(since: float, until: float | None = None) -> list[dict]:
    import ledger
    c = ledger._db()
    try:
        q = ("select id, at, text, lang, chosen, confidence, label, feedback, by from decisions where source in (?, ?) and at >= ? "
             + ("and at < ? " if until else "") + "and text is not null and length(text) > 3 order by at")
        rows = [dict(r) for r in c.execute(q, (*SOURCES, since) + ((until,) if until else ()))]
    finally:
        c.close()
    seen, out = set(), []
    for r in rows:                                          # one per distinct sentence per day (the app logs some twice)
        k = (" ".join(r["text"].lower().split()), int(r["at"] // 86400))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _agents() -> list[dict]:
    g = router.graph()
    return [n for n in g["nodes"] if n["type"] == "agent" and n.get("enabled", True)]


# ── shapes ──────────────────────────────────────────────────────────────────────────────
def _shapes(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    X = _emb([r["text"] for r in rows])
    now = time.time()
    shapes: list[dict] = []
    for i, r in enumerate(rows):                            # leader clustering: join the closest shape, or start one
        best, bs = None, SIM
        for s in shapes:
            sim = float(X[i] @ s["c"])
            if sim >= bs:
                best, bs = s, sim
        if best is None:
            shapes.append({"c": X[i].copy(), "members": [i]})
        else:
            best["members"].append(i)
            m = X[best["members"]].mean(0)
            best["c"] = m / (np.linalg.norm(m) + 1e-9)
    out = []
    for s in shapes:
        ms = [rows[i] for i in s["members"]]
        pressure = sum(math.exp(-(now - m["at"]) / 86400 / HALF_LIFE * math.log(2)) for m in ms)
        recent = sum(1 for m in ms if now - m["at"] < 3.5 * 86400)
        older = len(ms) - recent
        rep = min(ms, key=lambda m: -float(X[rows.index(m)] @ s["c"]))
        out.append({"centroid": s["c"], "n": len(ms), "pressure": round(pressure, 3), "growth": round((recent - older) / max(1, older), 2),
                    "texts": list(dict.fromkeys(m["text"] for m in ms))[:8], "rep": rep["text"],
                    "langs": dict(Counter(m.get("lang") or router.language(m["text"]) for m in ms)),
                    "labels": [m["label"] for m in ms if m.get("label")], "routed": dict(Counter(m["chosen"] for m in ms if m.get("chosen"))),
                    "confident": [m["chosen"] for m in ms if m.get("chosen") and (m.get("confidence") or 0) >= 0.85],
                    "feedback": sum(int(m.get("feedback") or 0) for m in ms)})
    out.sort(key=lambda s: -s["pressure"])
    return out


# ── the referee + tomorrow's customers ─────────────────────────────────────────────────
def _referee(shape: dict) -> dict:
    """What these customers need, which agent should answer, and next week's versions of them."""
    import llm
    import judges
    agents = _agents()
    ids = {a["id"] for a in agents}
    desc = "\n".join(f"- {a['id']}: {a['name']} ({a.get('kind')}) — {(a.get('description') or '')[:160]}" for a in agents)
    out = llm.complete(
        "You forecast a payments-wallet assistant's customer messages. Reply in exactly this format, nothing else:\n"
        "NEED ||| what these customers want, in under 12 words\n"
        "AGENT ||| the id of the ONE agent that should handle them (from the list). " + judges.REFEREE_RULES + "\n"
        "WHY ||| one short reason\n"
        "MESSAGE ||| a new message a customer is likely to send NEXT week with the same need  (write 6 of these lines: "
        "two English, two Hinglish (Hindi in Latin letters), one Hindi in Devanagari, one Arabic; include realistic typos, "
        "code-mixing, shorthand or impatience the way real customers write)",
        f"Agents:\n{desc}\n\nRecent real messages of this kind:\n" + "\n".join(f"- {t}" for t in shape["texts"][:6]), max_tokens=700)
    need, agent, why, msgs = "", "", "", []
    for line in out.splitlines():
        if "|||" not in line:
            continue
        k, v = (x.strip() for x in line.split("|||", 1))
        k = k.upper()
        if k.startswith("NEED") and not need:
            need = v[:120]
        elif k.startswith("AGENT") and not agent:
            agent = v.strip(" `'\"").split()[0] if v.strip() else ""
        elif k.startswith("WHY") and not why:
            why = v[:200]
        elif k.startswith("MESS") and len(v) > 2:
            v = re.sub(r"^\(?(english|hinglish|hindi|arabic|devanagari)\)?\s*[:\-–]\s*", "", v.strip("\"'"), flags=re.I)
            msgs.append(v.strip("\"'")[:200])
    if agent not in ids:
        raise ValueError("referee named no known agent")
    return {"need": need, "agent": agent, "why": why, "messages": msgs[:6]}


_FILL = ["bhai ", "pls ", "yaar ", "sir ", "", ""]


def _perturb(texts: list[str], n: int = 6) -> list[str]:
    """No language model: next week's typing habits applied to this week's sentences (typos, fillers, no punctuation)."""
    rnd = random.Random(len("".join(texts)))
    out = []
    for i in range(n):
        t = texts[i % len(texts)].lower().rstrip("?.! ")
        words = t.split()
        if words and i % 2 == 0:                            # one dropped vowel — the most common typo
            j = rnd.randrange(len(words))
            w = words[j]
            vs = [k for k, ch in enumerate(w) if ch in "aeiou" and 0 < k < len(w) - 1]
            if vs:
                k = rnd.choice(vs)
                words[j] = w[:k] + w[k + 1:]
        t = _FILL[i % len(_FILL)] + " ".join(words) + ("??" if i % 3 == 0 else "")
        out.append(t)
    return list(dict.fromkeys(out))


def _expect(shape: dict, ref: dict | None) -> tuple[str | None, str]:
    if shape["labels"]:
        return Counter(shape["labels"]).most_common(1)[0][0], "people's labels"
    if ref:
        return ref["agent"], "the referee model"
    if shape["confident"]:
        return Counter(shape["confident"]).most_common(1)[0][0], "assumed from confident live routing"
    return None, "unknown"


def _weather(fail: float, unsure: float) -> str:
    risk = fail + 0.5 * unsure
    return "storm" if risk >= 0.4 else "cloudy" if risk >= 0.15 else "clear"


def forecast(by: str):
    t0 = time.time()
    rows = _traffic(t0 - LOOKBACK * 86400)
    _, ver = _enc()
    yield {"type": "traffic", "n": len(rows), "days": LOOKBACK, "langs": dict(Counter(r.get("lang") or "en" for r in rows)), "reflex": ver}
    if len(rows) < 5:
        yield {"type": "error", "error": f"only {len(rows)} real requests in {LOOKBACK} days — too few to forecast from"}
        return
    shapes = _shapes(rows)
    top = shapes[:TOP]
    yield {"type": "shapes", "n": len(shapes), "kept": len(top), "shapes": [{"i": i, "rep": s["rep"], "n": s["n"], "pressure": s["pressure"], "growth": s["growth"]} for i, s in enumerate(top)]}
    g = router.graph()
    names = {a["id"]: a["name"] for a in _agents()}
    kinds = {a["id"]: a.get("kind") for a in _agents()}
    out = []
    for i, s in enumerate(top):
        ref, how = None, "drafted from this week's sentences (no language model)"
        try:
            ref = _referee(s)
            how = "written by the referee model"
        except Exception:                                   # noqa: BLE001 — no model: perturb what we have
            ref = None
        expect, expect_src = _expect(s, ref)
        msgs = (ref["messages"] if ref and ref["messages"] else _perturb(s["texts"]))
        variants = []
        for m in msgs:
            r = router.route(m, g)
            conf = ((r.get("reflex") or {}).get("confidence"))
            chosen = r.get("chosen")
            stopped = r.get("by") == "guardrails"
            ok = None if expect is None else (chosen == expect or (stopped and kinds.get(expect) == "decline"))
            variants.append({"text": m, "lang": router.language(m), "chosen": chosen, "chosen_name": names.get(chosen, chosen or ("guardrails" if stopped else "—")),
                             "by": r.get("by"), "conf": round(conf, 3) if conf is not None else None, "ok": ok,
                             "unsure": conf is not None and conf < 0.6})
        judged = [v for v in variants if v["ok"] is not None]
        fail = sum(1 for v in judged if not v["ok"]) / max(1, len(judged))
        unsure = sum(1 for v in variants if v["unsure"]) / max(1, len(variants))
        try:
            import space
            pos = space.place(s["rep"])
            xy = [pos["x"], pos["y"]]
        except Exception:                                   # noqa: BLE001
            xy = None
        sh = {"i": i, "rep": s["rep"], "texts": s["texts"], "n": s["n"], "pressure": s["pressure"], "growth": s["growth"], "langs": s["langs"],
              "routed": {names.get(k, k): v for k, v in s["routed"].items()}, "feedback": s["feedback"],
              "need": (ref or {}).get("need") or "", "why": (ref or {}).get("why") or "", "expect": expect, "expect_name": names.get(expect, expect or "—"),
              "expect_src": expect_src, "variants_how": how, "variants": variants, "fail": round(fail, 3), "unsure": round(unsure, 3),
              "weather": _weather(fail, unsure), "xy": xy, "centroid": [round(float(x), 5) for x in s["centroid"]]}
        out.append(sh)
        yield {"type": "shape", **{k: v for k, v in sh.items() if k != "centroid"}}
    fc = {"id": f"fc-{int(t0)}", "at": t0, "by": by, "reflex": ver, "horizon": HORIZON, "traffic": len(rows), "lookback": LOOKBACK, "shapes": out,
          "storms": sum(1 for s in out if s["weather"] == "storm"), "ms": int((time.time() - t0) * 1000)}
    store.write(f"oracle/{fc['id']}.json", fc)
    yield {"type": "done", "forecast": _public(fc)}


def _public(fc: dict) -> dict:
    return fc | {"shapes": [{k: v for k, v in s.items() if k != "centroid"} for s in fc["shapes"]]}


# ── the Oracle, scored ─────────────────────────────────────────────────────────────────
def backtest(fc: dict) -> dict:
    """Real requests that arrived after the forecast, assigned to its shapes; only people's evidence is scored."""
    end = min(time.time(), fc["at"] + fc["horizon"])
    rows = _traffic(fc["at"], end)
    res = {"arrived": len(rows), "elapsed": round((end - fc["at"]) / fc["horizon"], 3), "shapes": [], "matched": 0}
    if not rows or not fc["shapes"]:
        return res | {"verdict": "waiting for real traffic"}
    C = np.array([s["centroid"] for s in fc["shapes"]], np.float32)
    X = _emb([r["text"] for r in rows])
    S = X @ C.T
    per = {i: [] for i in range(len(fc["shapes"]))}
    for k, r in enumerate(rows):
        j = int(S[k].argmax())
        if S[k, j] >= SIM:
            per[j].append(r)
    calls = {"storm_right": 0, "storm_wrong": 0, "clear_right": 0, "clear_wrong": 0}
    for i, s in enumerate(fc["shapes"]):
        arr = per[i]
        bad = sum(1 for r in arr if (r.get("label") and r["label"] != r.get("chosen")) or int(r.get("feedback") or 0) < 0)
        good = sum(1 for r in arr if (r.get("label") and r["label"] == r.get("chosen")) or int(r.get("feedback") or 0) > 0)
        if bad or good:
            risky = s["weather"] != "clear"
            if risky:
                calls["storm_right" if bad else "storm_wrong"] += 1
            else:
                calls["clear_right" if not bad else "clear_wrong"] += 1
        res["shapes"].append({"i": i, "arrived": len(arr), "evidence_bad": bad, "evidence_good": good, "weather": s["weather"]})
    res["matched"] = sum(len(v) for v in per.values())
    res["materialised"] = sum(1 for v in per.values() if v)
    judged = sum(calls.values())
    res["calls"] = calls
    res["accuracy"] = round((calls["storm_right"] + calls["clear_right"]) / judged, 3) if judged else None
    res["verdict"] = f"{calls['storm_right'] + calls['clear_right']} of {judged} calls confirmed by people" if judged else "waiting for evidence — no labels or thumbs on these shapes yet"
    return res


def history() -> list[dict]:
    d = store.path("oracle", "x")
    d = os.path.dirname(d)
    out = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        fc = store.read(f"oracle/{f}")
        if fc:
            out.append(fc)
    return out


def state() -> dict:
    hs = history()
    latest = hs[-1] if hs else None
    past = [{"id": fc["id"], "at": fc["at"], "storms": fc["storms"], "shapes": len(fc["shapes"]), "reflex": fc.get("reflex"), "score": backtest(fc)} for fc in hs[-12:]]
    judged = [p["score"] for p in past if p["score"].get("accuracy") is not None]
    return {"latest": _public(latest) if latest else None, "past": past[::-1],
            "track": {"forecasts": len(past), "scored": len(judged),
                      "accuracy": round(sum(p["accuracy"] for p in judged) / len(judged), 3) if judged else None}}
