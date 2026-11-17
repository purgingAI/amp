"""Genesis: a new company's assistant, built from its own documents in minutes — in the builder's SANDBOX.

The documents are uploaded into a fresh knowledge base first (the normal /collections endpoints). build() then
streams the construction as SSE events, step by step:

    read      what was uploaded: documents, passages, languages
    profile   a one-line purpose and example questions for the new agent, written from the passages
    wire      a sandbox flow: guardrails → orchestrator → the new knowledge agent + branded decline / human hand-off
    reason    the reasoning model is warmed (it is what routes to an agent the router was never trained on)
    train     a retriever trained from scratch on these documents (exam held back, scored before and after)
    shield    the same red-team attacks the main assistant faces, fired at the new flow
    smoke     its own example questions routed and answered from the documents
    done      a summary, and the assistant is ready to chat in the builder's sandbox

Nothing touches the live flow, the live router or real customers. remove() deletes the knowledge base and resets
the builder's sandbox to a copy of the live flow.
"""
from __future__ import annotations

import json
import re
import time

import router
import store

SAFETY_FAMILIES = ["prompt_injection", "roleplay_jailbreak", "credential_fishing", "harmful_illegal", "other_people_data"]


def _ev(step: str, state: str, **data) -> dict:
    return {"type": "step", "step": step, "state": state, **data}


def _headings(chunks: list[dict]) -> list[str]:
    """Section titles from the documents themselves (markdown headings, or the chunker's own heading)."""
    out = []
    for c in chunks:
        for h in [c.get("heading") or ""] + re.findall(r"(?m)^#{2,4}\s+(.+)$", c.get("text") or ""):
            h = re.sub(r"\s+", " ", h).strip(" #:-").strip()
            if 3 <= len(h) <= 60 and h.lower() not in {x.lower() for x in out}:
                out.append(h)
    return out


def _profile_from_headings(chunks: list[dict], company: str) -> dict:
    hs = [h for h in _headings(chunks) if company.lower() not in h.lower()]
    lead = hs[:3]
    topic = (", ".join(h.lower() for h in lead[:-1]) + " and " + lead[-1].lower()) if len(lead) > 1 else (lead[0].lower() if lead else "its documents")
    qs = []
    for h in hs[:8]:
        hl = h.lower()
        first = hl.split()[0] if hl.split() else ""
        plural = hl.endswith("s") and not hl.endswith("ss")
        qs.append(f"what happens with {hl}?" if " or " in hl else f"how do {hl} work?" if plural and " " not in hl
                  else f"what are the {hl}?" if plural else f"how does {hl} work?" if first.endswith("ing") else f"tell me about {hl}")
    own = company + ("'" if company.endswith("s") else "'s")
    return {"description": f"Answers customers' questions about {topic}, from {own} own help documents.", "topic": topic[:80], "examples": qs}


def _profile(cid: str, company: str, assistant: str) -> dict:
    import llm
    import rag
    chunks = rag.all_chunks(cid)
    sample = "\n\n".join(f"- {c.get('heading') or ''} {c['text'][:400]}" for c in chunks[:14])
    out = llm.complete(
        "You set up a customer assistant from a company's documents. Reply in exactly this format and nothing else:\n"
        "DESCRIPTION ||| one plain sentence saying what customers can ask this assistant about\n"
        "TOPIC ||| two to five words naming the subject (for example 'orders, delivery and returns')\n"
        "QUESTION ||| a question a real customer would ask that the documents answer   (write 8 of these lines; "
        "mostly English, two in Hinglish and one in Arabic)",
        f"Company: {company}\nAssistant name: {assistant}\nPassages:\n{sample}", max_tokens=1600)
    desc, topic, qs = "", "", []
    for line in out.splitlines():
        if "|||" not in line:
            continue
        k, v = (x.strip() for x in line.split("|||", 1))
        v = v.strip().strip('"')
        if k.upper().startswith("DESC") and not desc:
            desc = v[:300]
        elif k.upper().startswith("TOPIC") and not topic:
            topic = v[:80]
        elif k.upper().startswith("QUEST") and len(v) > 4:
            qs.append(v[:200])
    if not desc:
        return _profile_from_headings(chunks, company)
    return {"description": desc, "topic": topic or "its documents", "examples": qs[:8]}


def _graph(company: str, assistant: str, cid: str, prof: dict) -> dict:
    live = router.graph()
    s = dict(live.get("settings") or {})
    s.update(coverage_first=True, fallback="decline")
    aid = "gen_" + re.sub(r"[^a-z0-9]+", "_", cid)[:30]
    nodes = [
        {"id": "in", "type": "input", "name": "Customer message", "x": 70, "y": 330},
        {"id": "guard", "type": "guard", "name": "Guardrails", "x": 290, "y": 330, "enabled": True},
        {"id": "laya", "type": "router", "name": f"{company} Orchestrator", "x": 560, "y": 330},
        {"id": aid, "type": "agent", "kind": "knowledge", "name": f"{assistant} · {prof['topic']}", "x": 930, "y": 200, "enabled": True,
         "description": prof["description"], "examples": prof["examples"], "collection": cid, "genesis": True},
        {"id": "human", "type": "agent", "kind": "human", "name": "Human hand-off", "x": 930, "y": 400, "enabled": True,
         "reply": f"I'll connect you with a colleague from the {company} team now. They'll see this conversation, so you won't need to repeat yourself."},
        {"id": "decline", "type": "agent", "kind": "decline", "name": "Polite decline", "x": 930, "y": 540, "enabled": True,
         "reply": f"That's outside what I can help with — I'm {assistant}, {company}'s assistant for {prof['topic']}."},
        {"id": "out", "type": "output", "name": "Reply", "x": 1300, "y": 380},
    ]
    edges = [{"from": "in", "to": "guard"}, {"from": "guard", "to": "laya"}] + \
            [{"from": "laya", "to": n} for n in (aid, "human", "decline")] + [{"from": n, "to": "out"} for n in (aid, "human", "decline")]
    return {"nodes": nodes, "edges": edges, "settings": s, "genesis": {"company": company, "assistant": assistant, "collection": cid, "agent": aid, "at": time.time()}}


def build(cid: str, company: str, assistant: str, by: str, train: bool = True):
    import jobs
    import rag
    import redteam
    company = (company or "").strip()[:60] or "Acme"
    assistant = (assistant or "").strip()[:40] or "Nova"
    m = rag.meta(cid)
    if not m:
        yield {"type": "error", "error": "no such knowledge base"}
        return
    chunks = rag.all_chunks(cid)
    if not chunks:
        yield {"type": "error", "error": "upload at least one document first"}
        return
    # read
    langs = sorted({router.language(c["text"][:300]) for c in chunks[:60]})
    m["brand"] = company
    m["genesis"] = {"company": company, "assistant": assistant, "by": by, "at": time.time()}
    store.write(f"collections/{cid}/meta.json", m)
    yield _ev("read", "done", docs=len(m["docs"]), passages=len(chunks), languages=langs, doc_names=[d["name"] for d in m["docs"]][:8])
    # profile
    yield _ev("profile", "running")
    try:
        prof = _profile(cid, company, assistant)
    except Exception:                                       # noqa: BLE001 — no language model: read it from the documents' own headings
        prof = dict(_profile_from_headings(chunks, company), source="headings")
    yield _ev("profile", "done", **prof)
    # wire
    g = _graph(company, assistant, cid, prof)
    router.save_graph(json.loads(json.dumps(g)), by, sandbox_user=by)
    yield _ev("wire", "done", agent=g["genesis"]["agent"], nodes=[{"id": n["id"], "name": n["name"], "kind": n.get("kind") or n["type"]} for n in g["nodes"]])
    # reason
    yield _ev("reason", "running")
    router.warm_laya()
    t0 = time.time()
    while router.laya_state()["state"] == "loading" and time.time() - t0 < 120:
        time.sleep(2)
    st = router.laya_state()
    yield _ev("reason", "done" if st["state"] == "ready" else "skipped", model=st["state"],
              note="" if st["state"] == "ready" else "routing by document coverage until the reasoning model is ready")
    # train
    trained = None
    if train and len(chunks) < 8:
        yield _ev("train", "skipped", note=f"a retriever is fine-tuned from 8+ passages; these documents make {len(chunks)} — the general multilingual retriever answers")
    elif train and not jobs.running():
        yield _ev("train", "running")
        try:
            jid = jobs.start({"kind": "rag_tune", "collection": cid, "by": by, "from_scratch": True, "target": 85.0, "max_rounds": 1, "epochs": 1,
                              "questions_per_passage": 3, "auto_promote": True, "then_router": False, "threads": 4, "genesis": True})["id"]
            t0 = time.time()
            while time.time() - t0 < 900:
                s = jobs.state(jid) or {}
                if s.get("status") in ("done", "failed", "cancelled"):
                    break
                yield _ev("train", "running", progress=s.get("progress"), phase=s.get("phase"), job=jid)
                time.sleep(3)
            s = jobs.state(jid) or {}
            res = s.get("result") or {}
            trained = {"job": jid, "status": s.get("status"), "version": res.get("version"), "promoted": res.get("promoted"),
                       "before": (res.get("baseline") or {}).get("score"), "after": (res.get("best") or {}).get("score")}
            yield _ev("train", "done" if s.get("status") == "done" else "failed", **trained, error=s.get("error"))
        except Exception as exc:                            # noqa: BLE001
            yield _ev("train", "skipped", note=f"{type(exc).__name__}: {exc}")
    else:
        yield _ev("train", "skipped", note="another training job is running — the untrained retriever is used" if train else "skipped")
    # shield
    yield _ev("shield", "running")
    gs = router.sandbox_graph(by)
    attacks = [a for a in redteam.pick(12, SAFETY_FAMILIES)]
    held = 0
    shots = []
    for a in attacks:
        r = router.route(a["text"], gs)
        out, why = redteam.judge(a, r, gs)
        held += out != "breach"
        shots.append({"family": a["family"], "text": a["text"][:90], "outcome": out, "why": why, "blocked": r.get("blocked")})
        yield {"type": "attack", **shots[-1]}
    yield _ev("shield", "done", n=len(attacks), held=held)
    # smoke
    yield _ev("smoke", "running")
    agent = g["genesis"]["agent"]
    ok = 0
    tries = []
    for q in (prof["examples"] or [c["text"][:120] for c in chunks[:3]])[:4]:
        r = router.route(q, gs)
        cov = next((c for c in r.get("coverage", []) if c["agent"] == agent), None)
        hit = r.get("chosen") == agent
        ok += hit
        tries.append({"q": q, "chosen": r.get("chosen"), "by": r.get("by"), "coverage": cov and cov.get("best"), "ok": hit})
        yield {"type": "probe", **tries[-1]}
    yield _ev("smoke", "done", n=len(tries), ok=ok)
    yield {"type": "done", "company": company, "assistant": assistant, "collection": cid, "agent": agent, "topic": prof["topic"],
           "examples": prof["examples"], "trained": trained, "shield": {"n": len(attacks), "held": held}, "smoke": {"n": len(tries), "ok": ok}}


def current(by: str) -> dict | None:
    g = router.sandbox_graph(by)
    gen = g.get("genesis")
    if not gen:
        return None
    node = next((n for n in g.get("nodes", []) if n.get("id") == gen.get("agent")), {})
    topic = (node.get("name") or "").split(" · ", 1)[-1] if " · " in (node.get("name") or "") else ""
    import rag
    m = rag.meta(gen.get("collection") or "") or {}
    flow = [{"name": n["name"], "kind": n.get("kind") or n["type"]} for n in g.get("nodes", []) if n.get("type") not in ("input", "output")]
    return gen | {"topic": topic, "description": node.get("description", ""), "examples": node.get("examples", []),
                  "docs": [d["name"] for d in m.get("docs", [])][:12], "flow": flow}


def remove(by: str) -> dict:
    import rag
    gen = current(by)
    if not gen:
        return {"ok": True, "note": "no Genesis assistant in your sandbox"}
    try:
        rag.delete_collection(gen["collection"])
    except ValueError:
        pass
    store.write(router._sandbox_file(by), json.loads(json.dumps(router.graph())) | {"sandbox": by})
    return {"ok": True, "removed": gen}
