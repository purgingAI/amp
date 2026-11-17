"""AMP Council mode — multiple agents deliberate with each other and converge on a conclusion.

Each persona is a Claude-backed AMP agent. They take turns reading the shared thread and posting
short messages (build on / challenge others), moving toward a decision. A facilitator detects
convergence and synthesizes the final conclusion. Every turn is a real AMP envelope in one thread,
so it also shows up in the normal client. Token-aware: short turns, capped rounds.
"""
from __future__ import annotations
import json
from . import store, protocol, orchestrator, audit

MODEL = orchestrator.MODEL

PERSONAS = {
    "architect":  ("🏛", "The Architect", "You obsess over clean design, module boundaries, and long-term maintainability."),
    "pragmatist": ("⚡", "The Pragmatist", "You ship the simplest thing that works; question scope and minimize effort."),
    "security":   ("🛡", "Security", "You threat-model everything: auth, data protection, secrets, abuse."),
    "qa":         ("🔬", "QA", "You hunt edge cases and failure modes; you define the test strategy."),
    "cost":       ("💰", "Cost & Perf", "You care about latency, token/$ cost, and infra footprint."),
}


def _turn_system(pid: str, topic: str) -> str:
    emoji, name, desc = PERSONAS[pid]
    return (f"You are **{name}** in a live multi-agent council deliberating a decision. {desc}\n"
            f"TOPIC: {topic}\n"
            "Read the transcript, then post ONE short turn (max 3 sentences). Explicitly build on or "
            "challenge specific other agents by name, add a concrete point they missed, and nudge toward "
            "a decision. Don't repeat yourself. No preamble — just your point.")


def _transcript_text(transcript) -> str:
    if not transcript:
        return "(you speak first — open the deliberation.)"
    return "\n".join(f"{PERSONAS[p][1]}: {t}" for p, t in transcript)


def run_council(topic: str, personas: list[str] | None = None, rounds: int = 2):
    """Generator yielding {type: 'turn'|'conclusion'|'error', ...} events."""
    if not orchestrator._client:
        yield {"type": "error", "text": "No ANTHROPIC_API_KEY."}
        return
    personas = [p for p in (personas or list(PERSONAS)) if p in PERSONAS] or list(PERSONAS)
    thread = protocol.new_id("thr")
    sec = store.get_agent("orchestrator")["secret"]
    transcript = []

    roster = [{"id": p, "emoji": PERSONAS[p][0], "name": PERSONAS[p][1]} for p in personas]
    _bcast(sec, thread, {"text": f"⚖ Council convened: {topic}",
                         "council": {"event": "start", "topic": topic, "roster": roster}})
    yield {"type": "start", "thread": thread, "topic": topic, "roster": roster}

    for r in range(rounds):
        for pid in personas:
            try:
                msg = orchestrator._client.messages.create(
                    model=MODEL, max_tokens=220, system=_turn_system(pid, topic),
                    messages=[{"role": "user", "content": "Transcript so far:\n" + _transcript_text(transcript)}])
                audit.llm(MODEL, msg.usage, f"council.{pid}")
                text = "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as e:
                text = f"(…{e})"
            transcript.append((pid, text))
            _bcast(sec, thread, {"text": text, "council": {"event": "turn", "round": r + 1,
                   "persona": pid, "emoji": PERSONAS[pid][0], "name": PERSONAS[pid][1]}},
                   frm=f"council:{pid}")
            yield {"type": "turn", "round": r + 1, "persona": pid,
                   "emoji": PERSONAS[pid][0], "name": PERSONAS[pid][1], "text": text}

    # facilitator synthesizes the conclusion
    try:
        conc = orchestrator._client.messages.create(
            model=MODEL, max_tokens=700,
            system=("You are the council FACILITATOR (the AMP Orchestrator). Synthesize the deliberation "
                    "into a crisp conclusion. Output Markdown with: **Decision** (one line), **Why** "
                    "(3 bullets citing which agents drove it), **Action items** (checklist), and "
                    "**Dissent / risks** (what to watch). Be decisive."),
            messages=[{"role": "user", "content": f"TOPIC: {topic}\n\nTRANSCRIPT:\n{_transcript_text(transcript)}"}])
        audit.llm(MODEL, conc.usage, "council.conclusion")
        text = "".join(b.text for b in conc.content if b.type == "text").strip()
    except Exception as e:
        text = f"(synthesis failed: {e})"
    _bcast(sec, thread, {"text": text, "council": {"event": "conclusion"}})
    yield {"type": "conclusion", "thread": thread, "text": text}


def _bcast(sec, thread, body, frm="orchestrator"):
    """Broadcast a council event to every connected agent (shared spectacle)."""
    env = protocol.make(frm, "broadcast", "council", body, sec, thread=thread)
    store.store_message(env)
