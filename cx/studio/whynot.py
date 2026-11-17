"""Why-not: the smallest rewrite of a customer's sentence that would have sent it somewhere else.

An explanation people understand is a counterfactual: "had they written X instead of Y, it would have gone to Z".
search() finds it with the live reflex itself — no language model needed, and nothing is guessed:

    vocabulary   the words each agent's own training rows are known for (log-odds over the live version's recipe)
    edits        delete a word · insert one of the target's words · swap a word for one of them
    beam         breadth-first by number of edits, keeping the few sentences that moved closest to the target;
                 stops at the first sentence the reflex sends to the target — the fewest edits that flip it
    check        the counterfactual is routed through the FULL flow (guardrails, reasoning model, coverage), because
                 the reflex flipping is not the same as the customer reaching the target

Optionally (admins; it spends the referee model) a natural rewrite: the smallest change a person would make, verified
by routing it too. Read-only: an approved counterfactual becomes a lesson only through the golden set, by a person.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

import numpy as np

import router
import store

BEAM = 3
MAX_EDITS = 3
N_INSERT = 14
N_SWAP = 8
_VOCAB: dict = {}
_W = re.compile(r"[\wऀ-ॿ؀-ۿ']+", re.U)


def _tok(t: str) -> list[str]:
    return _W.findall(t.lower())


def _script(t: str) -> str:
    c = Counter("deva" if "\u0900" <= ch <= "\u097f" else "arab" if "\u0600" <= ch <= "\u06ff" else "latn" for ch in t if ch.isalpha())
    return c.most_common(1)[0][0] if c else "latn"


def _vocab(head_ver: str) -> dict:
    """Per agent: the words most typical of its training rows (smoothed log-odds against every other agent)."""
    if _VOCAB.get("ver") == head_ver:
        return _VOCAB["v"]
    import recipes
    rec = recipes.load(head_ver) or {}
    rows = rec.get("rows") or []
    per: dict = {}
    for r in rows:
        if len(r) >= 2:
            per.setdefault(r[1], Counter()).update(set(_tok(r[0])))
    allc = Counter()
    for c in per.values():
        allc.update(c)
    total = {a: sum(c.values()) for a, c in per.items()}
    T = sum(total.values()) or 1
    out = {}
    for a, c in per.items():
        sc = []
        for w, k in c.items():
            if len(w) < 2 or k < 2:
                continue
            rest = allc[w] - k
            lo = math.log((k + .5) / (total[a] - k + 1)) - math.log((rest + .5) / (T - total[a] - rest + 1))
            sc.append((lo * math.log(1 + k), w))
        out[a] = [w for _, w in sorted(sc, reverse=True)[:40]]
    _VOCAB.update(ver=head_ver, v=out)
    return out


def _scorer(h, known: list[str]):
    import encoder
    enc = encoder.get(h.get("encoder"))
    labels = list(h["clf"].classes_)
    idx = [labels.index(k) for k in known]

    def score(texts: list[str]) -> np.ndarray:
        P = h["clf"].predict_proba(enc.encode(texts, max_len=64))[:, idx]
        return P / (P.sum(1, keepdims=True) + 1e-12)
    return score


def _natural(text: str, target: dict) -> str | None:
    import llm
    try:
        out = llm.complete("You rewrite one customer message with the SMALLEST change that makes it clearly a request for the given "
                           "agent. Keep the customer's language, words and tone. Reply with the rewritten message only.",
                           f"Agent: {target['name']} — {(target.get('description') or target.get('kind') or '')[:200]}\nMessage: {text}", max_tokens=120)
    except Exception:                                       # noqa: BLE001
        return None
    out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
    return out[:300] or None


def search(text: str, target: str | None = None, natural: bool = False) -> dict:
    t0 = time.time()
    text = " ".join((text or "").split())[:300]
    if len(text) < 2:
        raise ValueError("type a sentence")
    g = router.graph()
    h = router.head()
    if not h:
        raise ValueError("no live reflex")
    cands = router.candidates(g)
    labels = list(h["clf"].classes_)
    known = [c["id"] for c in cands if c["id"] in labels]
    names = {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}
    r0 = router.route(text, g)
    score = _scorer(h, known)
    p0 = score([text])[0]
    chosen = known[int(p0.argmax())]
    if target is None:                                      # the runner-up: the "why not" people usually mean
        order = [known[i] for i in np.argsort(-p0)]
        target = next(a for a in order if a != chosen)
    if target not in known:
        raise ValueError("the reflex does not know that agent")
    ti = known.index(target)
    sc = _script(text)                                     # edits a customer could plausibly make: same script as the sentence
    vocab = [w for w in _vocab(h.get("version")).get(target, []) if w not in set(_tok(text)) and _script(w) == sc]
    base = {"text": text, "tokens": _tok(text) or [text.lower()], "edits": [], "p": float(p0[ti])}
    path_probs = {k: round(float(p0[i]), 4) for i, k in enumerate(known)}
    beam, seen, found, tried = [base], {" ".join(base["tokens"])}, None, 0
    if chosen == target:
        found = base
    for depth in range(1, MAX_EDITS + 1):
        if found:
            break
        nxt = []
        for st in beam:
            toks = st["tokens"]
            for i in range(len(toks)):                      # delete
                if len(toks) > 1:
                    nxt.append((toks[:i] + toks[i + 1:], {"op": "delete", "word": toks[i], "at": i}))
            for w in vocab[:N_INSERT]:                      # insert at the start or the end
                nxt.append(([w] + toks, {"op": "insert", "word": w, "at": 0}))
                nxt.append((toks + [w], {"op": "insert", "word": w, "at": len(toks)}))
            for i in range(len(toks)):                      # swap a word for one of the target's
                for w in vocab[:N_SWAP]:
                    nxt.append((toks[:i] + [w] + toks[i + 1:], {"op": "swap", "word": w, "was": toks[i], "at": i}))
        uniq = []
        for toks, e in nxt:
            k = " ".join(toks)
            if k not in seen:
                seen.add(k)
                uniq.append((toks, e))
        if not uniq:
            break
        tried += len(uniq)
        P = score([" ".join(t) for t, _ in uniq])
        cands_d = []
        for (toks, e), p in zip(uniq, P):
            par = next(b for b in beam if _parent(b["tokens"], toks, e))
            cands_d.append({"text": " ".join(toks), "tokens": toks, "edits": par["edits"] + [e], "p": float(p[ti]), "argmax": known[int(p.argmax())],
                            "probs": {k: round(float(p[i]), 4) for i, k in enumerate(known)}})
        flips = [c for c in cands_d if c["argmax"] == target]
        if flips:
            found = max(flips, key=lambda c: c["p"])
            break
        beam = sorted(cands_d, key=lambda c: -c["p"])[:BEAM]
    out = {"text": text, "chosen": r0.get("chosen"), "chosen_by": r0.get("by"), "blocked": r0.get("blocked"), "reflex_choice": chosen,
           "target": target, "names": names, "probs": path_probs, "vocab": vocab[:12], "tried": tried, "reflex": h.get("version")}
    if found and found is not base:
        steps, toks = [], base["tokens"]
        for e in found["edits"]:                            # replay the edits, scoring each intermediate sentence
            toks = _apply(toks, e)
            steps.append({"edit": e, "text": " ".join(toks)})
        Ps = score([s["text"] for s in steps])
        for s, p in zip(steps, Ps):
            s["probs"] = {k: round(float(p[i]), 4) for i, k in enumerate(known)}
        rf = router.route(found["text"], g)
        out |= {"found": True, "edits": len(found["edits"]), "counterfactual": found["text"], "steps": steps,
                "full": {"chosen": rf.get("chosen"), "by": rf.get("by"), "blocked": rf.get("blocked"), "agrees": rf.get("chosen") == target}}
    else:
        out |= {"found": chosen == target, "edits": 0, "counterfactual": None, "steps": [],
                "note": "already goes there" if chosen == target else f"no rewrite of up to {MAX_EDITS} word edits sends it there — the reflex is firm about this one"}
    if natural:
        node = router.node(g, target) or {}
        nt = _natural(text, node)
        if nt:
            rn = router.route(nt, g)
            same = " ".join(_tok(nt)) == " ".join(_tok(text))
            out["natural"] = {"text": nt, "chosen": rn.get("chosen"), "by": rn.get("by"), "ok": rn.get("chosen") == target, "already_asks": same}
    out["ms"] = int((time.time() - t0) * 1000)
    return out


def _apply(toks: list[str], e: dict) -> list[str]:
    i = e["at"]
    if e["op"] == "delete":
        return toks[:i] + toks[i + 1:]
    if e["op"] == "insert":
        return toks[:i] + [e["word"]] + toks[i:]
    return toks[:i] + [e["word"]] + toks[i + 1:]


def _parent(ptoks: list[str], toks: list[str], e: dict) -> bool:
    return _apply(ptoks, e) == toks
