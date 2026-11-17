"""Agent auto-pilot: when a user doesn't reply, their agent answers from chat context.

Token-frugal by design: Haiku, only the last few turns, tiny max_tokens, and an explicit
'AMP_END' stop so agent↔agent chatter winds down instead of looping forever.
"""
from __future__ import annotations
import os
import anthropic
from . import store, audit

MODEL = os.environ.get("AMP_AUTO_MODEL", "claude-haiku-4-5-20251001")
_key = os.environ.get("ANTHROPIC_API_KEY")
_client = anthropic.Anthropic(api_key=_key) if _key else None
END = "AMP_END"


def set_client(c):
    global _client
    _client = c


def generate(thread_id: str, as_agent: str, as_name: str | None = None) -> str | None:
    """Return the next message text for `as_agent`, or None to stay silent / end."""
    if not _client:
        return None
    msgs = store.thread(thread_id)[-6:]
    if not msgs:
        return None
    name = as_name or as_agent
    lines = []
    for m in msgs:
        who = name if m["from"] == as_agent else m["from"].split(".")[0].capitalize()
        lines.append(f"{who}: {m['body'].get('text','')[:200]}")
    system = (f"You ARE {name} in a group chat. Output ONLY {name}'s next message — a single short, "
              f"natural line (max 16 words, first person, no name prefix). Never explain yourself, "
              f"never ask who you're replying to, never add meta commentary. If there's nothing worth "
              f"adding or the chat has wrapped up, output exactly {END} and nothing else.")
    try:
        m = _client.messages.create(model=MODEL, max_tokens=48, system=system,
                                    messages=[{"role": "user", "content": "Chat so far:\n" + "\n".join(lines) + f"\n\n{name}:"}])
        audit.llm(MODEL, m.usage, "autoreply", actor=as_agent)
        txt = "".join(b.text for b in m.content if b.type == "text").strip()
        if not txt or END in txt:
            return None
        return txt[:200]
    except Exception:
        return None
