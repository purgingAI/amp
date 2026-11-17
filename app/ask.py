"""Ask-the-company: an agentic Q&A over the knowledge graph (knowledge.py).

Claude + KB tools, mirroring TeamLens's agent but self-contained in AMP. Used by:
  * /amp/ask  — "who is working on X", "what's happening", "who should I ask about Y"
  * auto-reply grounding — an agent answers on its owner's behalf FROM the graph
Prompt caching on the system + primer so repeat questions stay cheap.
"""
from __future__ import annotations
import json
from . import knowledge, codegraph, audit, learn
try:
    from . import orchestrator as _orc
    _client = _orc._client
except Exception:
    _client = None

MODEL = "claude-sonnet-4-6"

SYSTEM = ("You are the company assistant and living memory for '{company}'. Your PRIMARY source is a "
          "CODE-UNDERSTANDING knowledge graph: an LLM has read the actual code and distilled, per "
          "module, what it DOES — its purpose, concrete capabilities/features, endpoints, key classes "
          "and owners (re-tuned hourly on what changed). \n"
          "For ANY question about features, what/where/how something is built, what a module or area "
          "does, or the state of the system: use search_capabilities / get_module / list_modules — "
          "answer from this distilled understanding, NOT from raw commit messages. Use find_experts / "
          "get_person / get_team_updates for who-knows-X and who's-doing-what, and search_notes for the "
          "WHY behind decisions. GROUND every claim in tool data; never invent modules, features or "
          "names; if the graph doesn't have it, say so. Be crisp: short bullets, cite module names, "
          "capabilities and owners.\n"
          "EFFICIENCY: gather in as FEW turns as possible — issue ALL needed tool calls AT ONCE in the "
          "first turn (parallel tool_use), then after at most TWO rounds STOP and write the answer.")

TOOLS = [
    {"name": "search_capabilities", "description": "PRIMARY. Find which modules implement a feature/capability and what they do — search the code-understanding graph by concept (e.g. 'merchant onboarding', 'refund', 'budget reserve'). Returns modules with purpose, matched capabilities, owners.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_module", "description": "Deep dive on one module/area: full purpose, summary, all capabilities, endpoints, key classes, owners.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "module name e.g. IncentiveEngine, MerchantPortal, HBL"}}, "required": ["name"]}},
    {"name": "list_modules", "description": "The map of the system — all modules with their one-line purpose (optionally filter by repo).",
     "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}}}},
    {"name": "find_experts", "description": "Rank the best people to ask about a topic/tech/module.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
    {"name": "get_person", "description": "Profile of a person: tech areas, projects, current focus.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_team_updates", "description": "Latest status updates on what people are CURRENTLY working on.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "search_notes", "description": "Search institutional notes for the WHY/how behind work.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
]
def _get_module(name: str):
    """Module deep-dive, with any human-validated learnings for it folded in."""
    m = codegraph.get_module(name)
    if isinstance(m, dict) and not m.get("error"):
        notes = learn.for_module(name)
        if notes:
            m["human_validated_notes"] = [{"about": n["question"][:120], "note": n["answer"][:500],
                                           "correction_requested": n.get("reason") or None} for n in notes]
    return m


DISPATCH = {
    "search_capabilities": lambda a: codegraph.search_capabilities(a.get("query", "")),
    "get_module": lambda a: _get_module(a.get("name", "")),
    "list_modules": lambda a: codegraph.list_modules(repo=a.get("repo")),
    "find_experts": lambda a: knowledge.find_experts(a.get("topic", "")),
    "get_person": lambda a: knowledge.get_person(a.get("name", "")),
    "get_team_updates": lambda a: knowledge.search_updates(a.get("query", "")),
    "search_notes": lambda a: knowledge.search_notes(a.get("query", "")),
}


def _primer() -> str:
    parts = []
    if codegraph.enabled():
        ov = codegraph.overview(); fr = ov.get("freshness", {})
        mods = "; ".join(f"{m['module']}" for m in codegraph.list_modules(80))
        parts.append(f"Code graph: {ov.get('modules')} modules across {ov.get('repos')} repos, "
                     f"re-tuned {fr.get('age_minutes')} min ago.\nModules: {mods}")
    if knowledge.enabled():
        ppl = ", ".join(p["name"] for p in knowledge.list_people(40))
        parts.append("People: " + ppl)
    return "\n\n".join(parts) or "Knowledge graph not loaded."


def run(question: str, as_person: str | None = None, history: list | None = None, actor: str = "",
        improve_note: str | None = None):
    """Generator of {'type':'text'|'tool'|'done'|'error', ...}. Grounds answers in the graph.

    improve_note (not None) = the previous answer was 👎-rejected; produce a better one (the string
    may carry the user's reason). Human-validated answers for similar questions are recalled as
    grounding so feedback fine-tunes future responses."""
    if not _client:
        yield {"type": "error", "text": "No Anthropic key configured."}
        return
    if not (knowledge.enabled() or codegraph.enabled()):
        yield {"type": "error", "text": "Company knowledge graph not loaded yet."}
        return
    persona = (f"\nYou are answering AS {as_person} (an employee) on their behalf. Speak in first "
               f"person where natural, from what {as_person} actually works on, and defer to the "
               f"right expert if it's not your area.") if as_person else ""
    extra = ""
    recalled = learn.recall(question)
    if recalled:
        extra += ("\n\nHUMAN-VALIDATED ANSWERS (a person approved these for similar questions — reuse "
                  "what was confirmed correct and build on it):\n" + "\n".join(
                      f"- Q: {r['question'][:150]}\n  A: {r['answer'][:850]}"
                      + (f"\n  (correction the user asked for: {r['reason']})" if r.get('reason') else "")
                      for r in recalled))
    if improve_note is not None:
        extra += ("\n\nIMPORTANT: your previous answer to this exact question was REJECTED by the user "
                  "as not good enough"
                  + (f' — their reason: "{improve_note}"' if improve_note.strip() else "")
                  + ". Produce a clearly BETTER answer: more accurate, better structured, address the "
                    "gap, and go deeper into the actual modules/capabilities. Do NOT repeat the rejected answer.")
    system = [
        {"type": "text", "text": SYSTEM.format(company=knowledge.COMPANY) + persona + extra},
        {"type": "text", "text": "Current knowledge primer:\n" + _primer(), "cache_control": {"type": "ephemeral"}},
    ]
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    produced_text = False
    for _ in range(4):   # small budget; the forced synthesis below always yields a complete answer
        try:
            with _client.messages.stream(model=MODEL, max_tokens=1500, system=system, tools=TOOLS, messages=messages) as stream:
                for ev in stream:
                    if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                        produced_text = True
                        yield {"type": "text", "text": ev.delta.text}
                final = stream.get_final_message()
        except Exception as e:
            yield {"type": "error", "text": f"Model error: {e}"}
            return
        audit.llm(MODEL, final.usage, "ask", actor)
        messages.append({"role": "assistant", "content": final.content})
        if final.stop_reason != "tool_use":
            yield {"type": "done"}
            return
        results = []
        for blk in final.content:
            if blk.type == "tool_use":
                args = blk.input if isinstance(blk.input, dict) else {}
                yield {"type": "tool", "name": blk.name, "input": args}
                try:
                    out = DISPATCH[blk.name](args)
                except Exception as e:
                    out = {"error": str(e)}
                results.append({"type": "tool_result", "tool_use_id": blk.id, "content": json.dumps(out, default=str)[:9000]})
        messages.append({"role": "user", "content": results})

    # Tool-call budget exhausted without a final answer. Force ONE no-tools synthesis so the user
    # always gets a written answer from everything gathered (else broad queries return only narration).
    yield {"type": "reset"}   # tell the client to drop the interstitial narration and show only the final answer
    try:
        with _client.messages.stream(model=MODEL, max_tokens=2000, system=system, messages=messages) as stream:
            for ev in stream:
                if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                    produced_text = True
                    yield {"type": "text", "text": ev.delta.text}
            final = stream.get_final_message()
        audit.llm(MODEL, final.usage, "ask-final", actor)
    except Exception as e:
        if not produced_text:
            yield {"type": "error", "text": f"Model error: {e}"}
    yield {"type": "done"}


def answer_text(question: str, as_person: str | None = None, actor: str = "",
                history: list | None = None) -> str:
    """Non-streaming convenience: full answer as a string (used by auto-reply)."""
    return "".join(ev["text"] for ev in run(question, as_person, history=history, actor=actor)
                   if ev["type"] == "text").strip()
