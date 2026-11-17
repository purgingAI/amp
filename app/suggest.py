"""Smart reply suggestions — the 'AMP understands the chat and recommends replies' feature.

Deliberately token-cheap: uses the fast Haiku model, only the last few turns, tiny max_tokens,
and caches per (thread, last-message) so repeat views cost nothing.
"""
from __future__ import annotations
import os, json
import anthropic
from . import store, audit

SUGGEST_MODEL = os.environ.get("AMP_SUGGEST_MODEL", "claude-haiku-4-5-20251001")
_key = os.environ.get("ANTHROPIC_API_KEY")
_client = anthropic.Anthropic(api_key=_key) if _key else None
_cache: dict[str, list[str]] = {}


def set_client(c):
    global _client
    _client = c


def suggest_replies(thread_id: str, for_agent: str, k: int = 3) -> list[str]:
    if not _client:
        return []
    msgs = store.thread(thread_id)[-8:]
    if not msgs:
        return []
    sig = thread_id + ":" + (msgs[-1]["id"] if msgs else "") + ":" + for_agent
    if sig in _cache:
        return _cache[sig]
    transcript = "\n".join(f"{'me' if m['from']==for_agent else m['from']}: {m['body'].get('text','')[:300]}"
                           for m in msgs)
    prompt = (f"You are drafting reply options for '{for_agent}' in this chat. Read it and propose "
              f"{k} short, natural replies I could send next (each < 12 words, distinct in intent — "
              f"e.g. agree, ask a question, propose an action). Return ONLY a JSON array of strings.\n\n"
              f"{transcript}")
    try:
        m = _client.messages.create(model=SUGGEST_MODEL, max_tokens=160,
                                    messages=[{"role": "user", "content": prompt}])
        audit.llm(SUGGEST_MODEL, m.usage, "suggest", actor=for_agent)
        txt = "".join(b.text for b in m.content if b.type == "text").strip()
        start, end = txt.find("["), txt.rfind("]")
        arr = json.loads(txt[start:end + 1]) if start >= 0 else []
        arr = [str(x)[:80] for x in arr][:k]
        _cache[sig] = arr
        return arr
    except Exception:
        return []
