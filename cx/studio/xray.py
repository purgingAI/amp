"""Decision X-ray: why the Orchestrator sent this message where it did — in a form people can learn from.

  words     how much each word pushed the reflex towards the chosen agent: drop the word, re-score,
            measure the fall (occlusion). All variants are embedded in ONE batch, so it is cheap on CPU.
  steps     the decision told as plain sentences, in the order the Orchestrator actually took it
  passage   for a knowledge agent: the passage that justified answering, words shared with the question
"""
from __future__ import annotations

import re

import encoder
import rag
import router

_TOKEN = re.compile(r"\S+")


def word_weights(text: str, target: str, h) -> list[dict]:
    words = _TOKEN.findall(text)[:40]
    if not h or not words or target not in list(h["clf"].classes_):
        return [{"w": w, "weight": 0.0} for w in words]
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    X = encoder.get(h.get("encoder")).encode(variants, max_len=64)
    col = list(h["clf"].classes_).index(target)
    P = h["clf"].predict_proba(X)[:, col]
    base = float(P[0])
    return [{"w": w, "weight": round(base - float(P[i + 1]), 4)} for i, w in enumerate(words)]


def _name(g, aid):
    n = router.node(g, aid) if aid else None
    return (n or {}).get("name", aid or "—")


def steps(tr: dict, g: dict) -> list[dict]:
    s, out = g["settings"], []
    lang = tr.get("language") or "en"
    out.append({"k": "language", "text": {"en": "Written in English.", "hinglish": "Written in Hinglish (romanised Hindi).",
                                          "hi": "Written in Hindi.", "ar": "Written in Arabic."}.get(lang, lang)})
    for st in tr.get("steps", []):
        if st.get("node") == "guard":
            v = st.get("verdict")
            out.append({"k": "guard", "ok": v == "clear", "text": "Guardrails: nothing unsafe — on to the Orchestrator." if v == "clear"
                        else f"Guardrails caught “{v}” — decided before any model ran."})
    for k in tr.get("skipped", []):
        out.append({"k": "skip", "text": f"{_name(g, k['agent'])} was not considered: {k['why']}."})
    rx = tr.get("reflex") or {}
    if rx.get("available"):
        sure = rx.get("confidence", 0) >= s["reflex_threshold"]
        out.append({"k": "reflex", "ok": sure, "text": f"Reflex {rx.get('version', '')}: {round(100 * rx.get('confidence', 0))}% sure it is "
                    f"{_name(g, rx.get('choice'))} — {'over' if sure else 'under'} its {round(100 * s['reflex_threshold'])}% bar."})
        if rx.get("untrained"):
            out.append({"k": "untrained", "text": f"It has never been trained on {', '.join(_name(g, a) for a in rx['untrained'])}, "
                        "so it cannot vote for " + ("them." if len(rx["untrained"]) > 1 else "it.")})
    cx = tr.get("context") or {}
    fu = cx.get("followup") or {}
    if cx.get("applied") or cx.get("rule") == "new-subject" or (fu.get("score", 0) >= 0.5 and (cx.get("turns") or cx.get("engine"))):
        if fu.get("cues"):
            out.append({"k": "context", "text": f"Read in its conversation ({cx.get('turns', 0)} earlier turn{'s' if cx.get('turns') != 1 else ''}): "
                        f"follow-up score {round(100 * fu.get('score', 0))}% — {', '.join(fu['cues'])}."})
        if cx.get("applied"):
            out.append({"k": "context", "ok": True, "text": f"Conversation: {cx.get('why')}."})
            if cx.get("standalone") and cx["standalone"] != tr.get("text"):
                out.append({"k": "context", "text": f"Searched the documents as: “{cx['standalone']}”."})
        elif cx.get("why"):
            out.append({"k": "context", "ok": False, "text": f"Conversation: {cx['why']} — judged on its own."})
    ly = tr.get("laya") or {}
    if ly.get("available"):
        model = "multilingual" if ly.get("model") == "multilingual" else "English"
        out.append({"k": "reasoning", "text": f"Reasoning model ({model}) read every agent's description: "
                    f"{_name(g, ly.get('choice'))}, {round(100 * ly.get('confidence', 0))}% sure."})
    elif ly:
        out.append({"k": "reasoning", "text": f"Reasoning model not used: {ly.get('reason', 'not needed')}."})
    by = tr.get("by")
    rule = {"reflex": "The reflex was sure, so it decided.",
            "reflex (unsure)": "Nobody was sure enough; the reflex's best guess was used.",
            "laya": "The reasoning model decided.", "fallback": "Nobody was sure; the fallback agent took it.",
            "guardrails": "The guardrails decided.", "handed over": "The first choice was handed over (below).",
            "only-agent": "Only one agent was connected.",
            "context": "The conversation decided — the earlier turns, not this sentence alone."}.get(by, by)
    if by:
        out.append({"k": "rule", "text": rule})
    for c in tr.get("coverage", []):
        ok = c["best"] >= c["need"]
        out.append({"k": "coverage", "ok": ok, "text": f"{_name(g, c['agent'])} checked its documents: best match {c['best']:.2f} "
                    f"(needs {c['need']:.2f}) — {'covered, it answers.' if ok else 'not covered.'}"})
    for h in tr.get("handoffs", []):
        out.append({"k": "handoff", "text": f"Handed from {_name(g, h['from'])} to {_name(g, h['to'])}."})
    if tr.get("blocked"):
        out.append({"k": "blocked", "text": "Stopped with a safe reply — no agent ran."})
    else:
        out.append({"k": "final", "text": f"→ {_name(g, tr.get('chosen'))}."})
    return out


def explain(text: str, tr: dict | None = None) -> dict:
    g = router.graph()
    tr = tr or router.route(text, g)
    h = router.head()
    target = tr.get("chosen")
    if not h or target not in list(h["clf"].classes_):
        target = ((tr.get("reflex") or {}).get("choice"))          # explain what the reflex leaned to
    out = {"text": text, "trace": tr, "target": target, "target_name": _name(g, target),
           "words": word_weights(text, target, h) if target else [], "steps": steps(tr, g), "english": None, "passage": None}
    en = rag.english(text)
    if en:
        out["english"] = en
    chosen = router.node(g, tr.get("chosen") or "") or {}
    cov = (tr.get("coverage") or [{}])[0]
    kb_agent = router.node(g, cov.get("agent")) if cov.get("agent") else (chosen if chosen.get("kind") == "knowledge" else None)
    if kb_agent and kb_agent.get("collection"):
        hits = rag.search(kb_agent["collection"], text, k=1)
        if hits:
            hp = hits[0]
            qw = {w for w in rag._words(text + " " + (en or "")) if len(w) > 2}
            out["passage"] = {"agent": kb_agent["id"], "doc": hp["doc"], "page": hp.get("page"), "heading": hp.get("heading"),
                              "text": hp["text"][:900], "dense": hp["dense"], "shared": sorted(qw & set(rag._words(hp["text"])))[:30]}
    return out
