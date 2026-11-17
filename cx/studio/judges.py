"""The judge court: every language-model judge is pinned, measured against people, and its numbers carry an interval.

A judge's verdict is not the truth. "Demystifying evals for AI agents" (Anthropic, Jan 2026) asks for judges that may
say UNKNOWN; Lee et al. (ICML 2026) show a judged rate must be corrected for the judge's own error — with a confidence
interval that includes the uncertainty of that correction; order-swap audits (arXiv 2607.08535) found ~15% of LLM
verdicts flip when the options are shown in another order. So for each judge:

  pin          judge id = name @ sha256(model + prompt)[:8]. Change either and it is a NEW judge: its old calibration no
               longer applies (shown as "changed since calibration").
  calibrate    run it on cases PEOPLE have already decided, twice — options in one order, then reversed:
                 accuracy (Wilson 95% interval) · how often it says UNKNOWN · how often the order flips the verdict
                 and, for the referee, how well it SPOTS the live router's real mistakes (sensitivity / specificity)
  correct      a rate the judge produced (e.g. the Oracle's "live would misroute X%") becomes
               (p + spec − 1) / (sens + spec − 1)  (Rogan–Gladen), with a bootstrap interval over both the calibration
               cases and the judged sample. Too few people-decided cases → no correction, and it says so.

Judges here:
  referee   which agent should handle a customer message (the Oracle's referee). Ground truth: the golden set and the
            labels people taught (labels.jsonl). Calibrates on its own — no labelling needed.
  faith     is each sentence of a knowledge answer supported by its passages (faith.py). Ground truth: sentences people
            mark in the calibration queue. The swap check re-judges recent answers with the passages reversed.
Calibration spends language-model calls only when an admin presses Calibrate. Nothing here changes a model or a label.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

import store

FILE = "judges.json"
LABELS = "judge_labels.jsonl"
MIN_CAL = 12                                                # fewer people-decided cases than this → no correction
BATCH = 20

# the Oracle's referee rules — oracle.py uses this exact text, so what is calibrated is what runs
REFEREE_RULES = ("A knowledge agent answers ONLY what its description says its documents cover — never pick it for anything "
                 "else. The action agent does things in the wallet and explains how wallet features work; human is for people "
                 "who ask for a person or are distressed; decline is for requests unrelated to the wallet")
REFEREE = ("You decide which agent of a payments-wallet assistant should handle each customer message. " + REFEREE_RULES +
           ". For EACH numbered message output one line: <number> ||| <agent id> — or <number> ||| UNKNOWN when it could "
           "reasonably belong to more than one agent or you cannot tell. Output only those lines.")


# ── pinning ─────────────────────────────────────────────────────────────────────────────
def _spec(name: str) -> dict:
    import llm
    if name == "referee":
        return {"name": "referee", "title": "Routing referee", "prompt": REFEREE, "model": llm.QUESTION_MODEL,
                "used_by": "Oracle — which agent SHOULD answer next week's customers", "truth": "golden set + labels people taught"}
    if name == "faith":
        import faith
        return {"name": "faith", "title": "Grounding judge", "prompt": faith.JUDGE, "model": llm.QUESTION_MODEL,
                "used_by": "every judged knowledge answer, and the verified answer cache", "truth": "sentences people mark below"}
    raise ValueError("no such judge")


def pin(name: str) -> dict:
    s = _spec(name)
    sha = hashlib.sha256((s["model"] + "\n" + s["prompt"]).encode()).hexdigest()[:8]
    return {k: v for k, v in s.items() if k != "prompt"} | {"id": f"{name}@{sha}", "sha": sha, "prompt_chars": len(s["prompt"])}


def _load() -> dict:
    return store.read(FILE, {"calibrations": {}})


# ── statistics ──────────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 4), round(min(1, c + h), 4)


def rogan_gladen(p: float, sens: float, spec: float) -> float | None:
    j = sens + spec - 1
    if j <= 0.05:                                           # a judge no better than chance cannot correct anything
        return None
    return min(1.0, max(0.0, (p + spec - 1) / j))


def corrected(k: int, n: int, cal: dict, draws: int = 2000, seed: int = 7) -> dict:
    """k of n judged positive (e.g. 'live gets it wrong'); cal: {tp, fn, tn, fp} from people-decided cases."""
    tp, fn, tn, fp = (cal.get(x, 0) for x in ("tp", "fn", "tn", "fp"))
    raw = {"observed": round(k / n, 4) if n else None, "n": n, "wilson": wilson(k, n)}
    if not n or tp + fn < 3 or tn + fp < 3 or tp + fn + tn + fp < MIN_CAL:
        return raw | {"estimate": None, "why": f"needs at least {MIN_CAL} people-decided cases with both outcomes — has {tp + fn} positive, {tn + fp} negative"}
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    est = rogan_gladen(k / n, sens, spec)
    if est is None:
        return raw | {"estimate": None, "why": "the judge is no better than chance on people-decided cases — its rate cannot be corrected"}
    rnd = random.Random(seed)
    sims = []
    for _ in range(draws):                                  # parametric bootstrap: judged sample, sensitivity, specificity
        p_ = sum(rnd.random() < k / n for _ in range(n)) / n
        se = sum(rnd.random() < sens for _ in range(tp + fn)) / (tp + fn)
        sp = sum(rnd.random() < spec for _ in range(tn + fp)) / (tn + fp)
        r = rogan_gladen(p_, se, sp)
        if r is not None:
            sims.append(r)
    sims.sort()
    lo, hi = (sims[int(.025 * len(sims))], sims[int(.975 * len(sims)) - 1]) if len(sims) > 50 else (None, None)
    return raw | {"estimate": round(est, 4), "lo": round(lo, 4) if lo is not None else None, "hi": round(hi, 4) if hi is not None else None,
                  "sens": round(sens, 3), "spec": round(spec, 3)}


# ── the referee ─────────────────────────────────────────────────────────────────────────
def _truth() -> list[dict]:
    """Messages people have decided: golden first, then taught labels; one per text."""
    import gates
    import router
    g = router.graph()
    seen, out = set(), []
    lab = []
    p = os.path.join(store.DATA, "labels.jsonl")
    if os.path.exists(p):
        lab = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for x in gates.golden() + lab:
        t, a = (x.get("text") or "").strip(), x.get("agent")
        if not t or not a or not router.node(g, a) or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append({"text": t, "agent": a, "lang": x.get("lang") or router.language(t), "src": "golden" if "id" in x and "key" not in x else "label"})
    return out


def _referee_batch(items: list[dict], agents: list[dict]) -> list[str | None]:
    import llm
    desc = "\n".join(f"- {a['id']}: {a['name']} ({a.get('kind')}) — {(a.get('description') or '')[:160]}" for a in agents)
    raw = llm.complete(REFEREE, f"Agents:\n{desc}\n\nMessages:\n" + "\n".join(f"{i + 1}. {x['text']}" for i, x in enumerate(items)), max_tokens=900)
    ids = {a["id"] for a in agents}
    got: list[str | None] = [None] * len(items)
    for line in raw.splitlines():
        parts = [x.strip() for x in line.split("|||")]
        if len(parts) >= 2 and parts[0].rstrip(".").isdigit():
            i, v = int(parts[0].rstrip(".")) - 1, parts[1].strip().strip("`")
            if 0 <= i < len(items):
                got[i] = "UNKNOWN" if v.upper() == "UNKNOWN" else v if v in ids else None
    return got


def calibrate_referee(by: str):
    """Events: phase, case {i, n, text, want, a, b, live} per message, then summary."""
    import router
    g = router.graph()
    agents = [n for n in g["nodes"] if n["type"] == "agent" and n.get("enabled", True) and any(e["to"] == n["id"] for e in g["edges"])]
    items = _truth()
    yield {"type": "phase", "text": f"{len(items)} messages people have decided · {len(agents)} agents · options shown in two orders",
           "order_a": [a["id"] for a in agents], "order_b": [a["id"] for a in agents][::-1]}
    rows = []
    for s in range(0, len(items), BATCH):
        chunk = items[s:s + BATCH]
        A = _referee_batch(chunk, agents)
        B = _referee_batch(chunk, agents[::-1])
        for j, it in enumerate(chunk):
            live = router.route(it["text"], g).get("chosen")
            r = {"i": s + j, "n": len(items), "text": it["text"], "lang": it["lang"], "want": it["agent"], "a": A[j], "b": B[j], "live": live}
            rows.append(r)
            yield {"type": "case", **r}
    res = _referee_metrics(rows)
    cal = _load()
    p = pin("referee")
    cal["calibrations"][p["id"]] = {"judge": "referee", "pin": p, "at": time.time(), "by": by, "metrics": res, "order": [a["id"] for a in agents],
                                    "rows": [{k: r[k] for k in ("text", "lang", "want", "a", "b", "live")} for r in rows]}
    store.write(FILE, cal)
    yield {"type": "summary", "judge": p["id"], **res}


def _referee_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if r["a"] and r["a"] != "UNKNOWN"]
    right = sum(r["a"] == r["want"] for r in answered)
    unknown = sum(r["a"] == "UNKNOWN" for r in rows)
    unparsed = sum(r["a"] is None for r in rows)
    both = [r for r in rows if r["a"] and r["b"]]
    flips = sum(r["a"] != r["b"] for r in both)
    # as a detector of the live router's mistakes (what the Oracle uses it for)
    det = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for r in answered:
        truly, says = r["live"] != r["want"], r["live"] != r["a"]
        det["tp" if truly and says else "fn" if truly else "fp" if says else "tn"] += 1
    return {"n": n, "answered": len(answered), "right": right, "accuracy": round(right / len(answered), 4) if answered else None,
            "accuracy_ci": wilson(right, len(answered)), "unknown": unknown, "unparsed": unparsed,
            "flips": flips, "swapped": len(both), "flip_rate": round(flips / len(both), 4) if both else None, "flip_ci": wilson(flips, len(both)),
            "detector": det}


# ── the grounding judge ─────────────────────────────────────────────────────────────────
def faith_queue(limit: int = 40) -> dict:
    """Judged answers, sentence by sentence, with what people have already said."""
    marks = _faith_marks()
    rows = _faith_rows(limit)
    out = []
    for r in rows:
        try:
            f = json.loads(r["faith"])
        except (TypeError, ValueError):
            continue
        j = f.get("judge") or {}
        sents = j.get("sentences") or []
        if not sents:
            continue
        out.append({"id": r["id"], "question": r["text"], "sentences": [{"i": i, "text": s["text"], "judge": s.get("verdict"), "why": s.get("why"),
                                                                          "person": marks.get(f"{r['id']}:{i}")} for i, s in enumerate(sents)],
                    "cheap": [s.get("verdict") for s in (f.get("cheap") or {}).get("sentences") or []]})
    return {"items": out, "marked": len(marks)}


def _faith_rows(limit: int) -> list[dict]:
    import sqlite3
    c = sqlite3.connect(os.path.join(store.DATA, "ledger.db"))
    c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute("select id, text, chosen, answer, faith from decisions where faith like '%\"judge\"%' order by id desc limit ?", (limit,))]
    finally:
        c.close()


def _faith_marks() -> dict:
    p = os.path.join(store.DATA, LABELS)
    out = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                m = json.loads(l)
                if m.get("judge") == "faith":
                    out[f"{m['did']}:{m['i']}"] = m["verdict"]          # newest wins
    return out


def mark_faith(did: int, i: int, verdict: str, by: str) -> dict:
    if verdict not in ("supported", "unsupported", "skip"):
        raise ValueError("verdict is supported, unsupported or skip")
    with open(store.path(LABELS), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"judge": "faith", "did": int(did), "i": int(i), "verdict": verdict, "by": by, "at": time.time()}) + "\n")
    return {"ok": True, "marked": len(_faith_marks())}


def faith_metrics() -> dict:
    q = faith_queue(400)
    det = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    agree = n = 0
    k_obs = n_obs = unknown = 0
    for it in q["items"]:
        for s in it["sentences"]:
            jv = s["judge"]
            if jv in ("supported", "partial", "unsupported"):
                n_obs += 1
                k_obs += jv == "unsupported"
            elif jv == "unknown":
                unknown += 1
            if s["person"] in ("supported", "unsupported") and jv in ("supported", "partial", "unsupported"):
                truly, says = s["person"] == "unsupported", jv == "unsupported"
                det["tp" if truly and says else "fn" if truly else "fp" if says else "tn"] += 1
                n += 1
                agree += truly == says
    return {"marked": n, "agree": agree, "accuracy": round(agree / n, 4) if n else None, "accuracy_ci": wilson(agree, n), "detector": det,
            "unknown": unknown, "unsupported_rate": corrected(k_obs, n_obs, det)}


def calibrate_faith(by: str, answers: int = 8):
    """The swap check: re-judge recent answers with the passages in reverse order; count verdicts that change."""
    import faith
    import rag
    import router
    g = router.graph()
    rows = _faith_rows(answers)
    yield {"type": "phase", "text": f"Re-judging {len(rows)} answers with their passages in reverse order…"}
    flips = n = 0
    for r in rows:
        try:
            f = json.loads(r["faith"])
            # passages are not stored with a decision: fetch them the way the knowledge agent did (rag.answer_stream),
            # keep each one's [n] so the answer's citations still point at the same text — only the ORDER changes
            col = (router.node(g, r["chosen"]) or {}).get("collection") or ""
            passages = [{"n": n, "text": h["text"][:1500]} for n, h in enumerate(rag.search(col, r["text"], k=6), 1)] if col else []
        except Exception:                                   # noqa: BLE001
            continue
        before = [s.get("verdict") for s in (f.get("judge") or {}).get("sentences") or []]
        if not passages or not before:
            continue
        again = faith.judge(r["answer"], list(reversed(passages))) or {}
        after = [s.get("verdict") for s in again.get("sentences") or []]
        pairs = [(a, b) for a, b in zip(before, after) if a in ("supported", "partial", "unsupported") and b in ("supported", "partial", "unsupported")]
        fl = sum(a != b for a, b in pairs)
        flips += fl
        n += len(pairs)
        yield {"type": "case", "id": r["id"], "question": r["text"], "before": before, "after": after, "flips": fl}
    m = faith_metrics() | {"flips": flips, "swapped": n, "flip_rate": round(flips / n, 4) if n else None, "flip_ci": wilson(flips, n)}
    cal = _load()
    p = pin("faith")
    cal["calibrations"][p["id"]] = {"judge": "faith", "pin": p, "at": time.time(), "by": by, "metrics": m}
    store.write(FILE, cal)
    yield {"type": "summary", "judge": p["id"], **m}


# ── what the court shows ────────────────────────────────────────────────────────────────
def oracle_correction(cal: dict | None) -> dict | None:
    """The latest Oracle forecast's 'live would misroute' share, corrected by the referee's own error."""
    try:
        import oracle
        latest = oracle.history()[-1]
    except Exception:                                       # noqa: BLE001
        return None
    k = n = 0
    for s in latest.get("shapes") or []:
        if s.get("expect_src") != "the referee model":          # people-labelled shapes need no correction
            continue
        vs = [v for v in s.get("variants") or [] if "ok" in v]
        n += len(vs)
        k += sum(1 for v in vs if not v["ok"])
    if not n:
        return None
    det = ((cal or {}).get("metrics") or {}).get("detector") or {}
    return {"forecast": latest.get("id"), "at": latest.get("at"), **corrected(k, n, det)}


def state() -> dict:
    cal = _load()["calibrations"]
    out = []
    for name in ("referee", "faith"):
        p = pin(name)
        mine = cal.get(p["id"])
        older = sorted((c for c in cal.values() if c["judge"] == name and c["pin"]["id"] != p["id"]), key=lambda c: c["at"])
        entry = {"pin": p, "status": "calibrated" if mine else "changed" if older else "uncalibrated",
                 "calibration": {k: v for k, v in mine.items() if k != "rows"} if mine else None,
                 "previous": older[-1]["pin"]["id"] if older else None}
        if name == "faith":
            entry["live"] = faith_metrics()
        if name == "referee":
            entry["oracle"] = oracle_correction(mine)
            rows = (mine or {}).get("rows") or []          # a few real cases for the swap theatre: every flip, then holds
            entry["examples"] = ([r for r in rows if r["a"] != r["b"]] + [r for r in rows if r["a"] == r["b"] and r["a"] not in (None, "UNKNOWN")][:4])[:7]
            entry["agents"] = (mine or {}).get("order") or None
        out.append(entry)
    return {"judges": out, "min_cal": MIN_CAL}
