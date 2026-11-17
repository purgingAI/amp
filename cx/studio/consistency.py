"""pass^k for a router: a golden case counts only if EVERY way of saying it reaches the right agent.

Agent benchmarks (τ-bench, Anthropic's "Demystifying evals for AI agents", Jan 2026) report pass^k — the chance that
all k trials succeed — because one lucky pass hides an agent that fails customers half the time. Our router is
deterministic (measured 2026-10-28: 40 golden sentences × 3 runs, 0 differences; the reasoning model was never
consulted), so repeating the same sentence k times would only repeat pass@1. What varies for a real customer is the
SURFACE of the message, so a trial here is the original plus k−1 variations of it:

  typo      one letter dropped or two swapped inside one word
  casual    lower-case, no punctuation, a filler word in the sentence's own language ("bhai", "pls", "لو سمحت")
  noise     a stretched letter and an impatient ending ("refunddd??", "jaldi batao")

Variations are seeded by the golden id, so a court is reproducible — the same candidate gets the same verdict
tomorrow. Bump VERSION when the variation rules change: verdicts from different versions are not comparable.
Money-moving agents (kind "action") are gated on pass^k; every other agent is reported, and gated on pass@1 as before.
"""
from __future__ import annotations

import random
import re

VERSION = 1
K = 4                                                       # the original + 3 variations

FILL = {
    "en": (["pls ", "hey ", "umm "], [" pls", " asap", "??"]),
    "hinglish": (["bhai ", "yaar ", "sir "], [" bhai", " jaldi batao", "??"]),
    "hi": (["भाई ", "कृपया "], [" जल्दी बताओ", "??"]),
    "ar": (["لو سمحت ", "مرحبا "], [" بسرعة", "؟؟"]),
}


def _words(text: str) -> list[str]:
    return text.split()


def _typo(text: str, rnd: random.Random) -> str:
    w = _words(text)
    longs = [i for i, x in enumerate(w) if len(re.sub(r"[^\w\u0900-\u097F\u0600-\u06FF]", "", x)) > 3]   # \w alone drops Devanagari vowel signs
    if not longs:
        return text.lower()
    i = rnd.choice(longs)
    s = w[i]
    j = rnd.randrange(1, len(s) - 2)
    w[i] = s[:j] + s[j + 1] + s[j] + s[j + 2:] if rnd.random() < .5 else s[:j] + s[j + 1:]
    return " ".join(w)


def _casual(text: str, lang: str, rnd: random.Random) -> str:
    pre, _ = FILL.get(lang, FILL["en"])
    t = re.sub(r"[?.!,;:؟]+", "", text).strip()
    return rnd.choice(pre) + (t.lower() if lang in ("en", "hinglish") else t)


def _noise(text: str, lang: str, rnd: random.Random) -> str:
    _, post = FILL.get(lang, FILL["en"])
    w = _words(re.sub(r"[?.!؟]+$", "", text.strip()))
    if w:
        i = rnd.randrange(len(w))
        if w[i][-1:].isalpha():
            w[i] = w[i] + w[i][-1] * 2
    return " ".join(w) + rnd.choice(post)


def variants(item: dict) -> list[dict]:
    """The k−1 variations of one golden case (deterministic for its id)."""
    rnd = random.Random(f"{VERSION}:{item.get('id') or item['text']}")
    lang = item.get("lang") or "en"
    out = [{"how": "typo", "text": _typo(item["text"], rnd)},
           {"how": "casual", "text": _casual(item["text"], lang, rnd)},
           {"how": "noise", "text": _noise(item["text"], lang, rnd)}]
    seen, keep = {item["text"]}, []
    for v in out[:K - 1]:
        if v["text"] not in seen:
            seen.add(v["text"])
            keep.append(v)
    return keep


def gate(items: list[dict], trials: dict, g: dict, node) -> dict:
    """items: golden cases; trials: side → list (per case) of bools, original first.
    Gate: on money-moving cases the candidate is consistent at least as often as live, and no case live keeps
    consistent starts to wobble."""
    def money(it):
        return (node(g, it["agent"]) or {}).get("kind") == "action"
    cons = {s: [all(t) for t in trials[s]] for s in trials}
    at1 = {s: [t[0] for t in trials[s]] for s in trials}
    idx_m = [i for i, it in enumerate(items) if money(it)]
    wobble = [items[i]["text"] for i in idx_m if cons["live"][i] and not cons["cand"][i]]
    m = {s: sum(cons[s][i] for i in idx_m) for s in cons}
    ok = m["cand"] >= m["live"] and not wobble
    return {
        "k": K, "version": VERSION,
        "pass1": {s: sum(v) for s, v in at1.items()}, "passk": {s: sum(v) for s, v in cons.items()},
        "money": {"n": len(idx_m), **m}, "wobble": wobble,
        "flaky": {s: [items[i]["text"] for i in range(len(items)) if trials[s][i][0] and not cons[s][i]] for s in trials},
        "pass": ok,
        "detail": (f"pass^{K}: {sum(cons['cand'])}/{len(items)} (live {sum(cons['live'])}) · money-moving {m['cand']}/{len(idx_m)} (live {m['live']})"
                   + (f" · {len(wobble)} money-moving case(s) start to wobble" if wobble else "")),
    }
