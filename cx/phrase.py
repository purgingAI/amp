"""Putting the engine's decision into words, with a model that is never allowed to decide.

Until now every sentence this assistant says is a template: `_tr(a, "spend_more", ...)`
with a hand-written phrasing per language. That is why it is safe, and it is also why it
reads the same every time and why every new sentence costs four translations.

This adds a small generative model that **renders** a decision the engine has already
made. It is the first autoregressive component in this system, which is what makes the
whole prefill/decode/KV-cache vocabulary apply here for real rather than by analogy.

## The rule that shapes the entire file

**The model phrases. It never decides, and it never computes.**

The engine resolves the intent, reads the ledger, applies the guards and produces a set of
FACTS. The model's only job is to turn those facts into a sentence. Then — and this is the
part that makes it safe to ship on a payments surface — **every number in the generated
sentence is checked against the facts, and if one does not appear there, the generation is
thrown away and the template is used.**

A language model that invents "AED 4,549" when the real figure is AED 4,634 is not a
slightly worse writer, it is a liar about somebody's money. Verification is not a quality
gate here, it is the thing that permits the component to exist.

## What this gives us, in the proper terminology

* **Prefill** — the system prompt plus the facts, pushed through in one pass. Compute-bound.
* **Decode** — the sentence, one token at a time, each step attending to everything before
  it. Memory-bandwidth-bound. This is the phase the classifier never had.
* **KV cache** — llama.cpp holds K and V for every token generated so far, so step `t`
  costs `O(t)` instead of recomputing `O(t²)`.
* **Prefix caching** — our system prompt is byte-identical on every call. Its KV is computed
  once and reused, so only the facts and the question are ever prefilled again.
* **Quantization** — Q4_K_M: weights in ~4.5 bits instead of 32. 491MB instead of ~2GB,
  and on CPU it is a genuine speed-up (unlike int8 on the encoder, which we measured and
  rejected — see docs/inference-and-training.md).
* **Latency** — now properly two numbers: TTFT (time to first token) and TPOT (time per
  output token).
"""
from __future__ import annotations

import os
import re
import threading
import time

MODEL = os.environ.get(
    "CX_PHRASE_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"))

#: Off unless asked for. The templates are the default and remain the fallback, so a
#: machine without the model, or with it switched off, behaves exactly as before.
ENABLED = os.environ.get("CX_PHRASE") == "1"

#: Byte-identical on every call, which is the whole point: llama.cpp keeps its KV cache and
#: re-prefills only what comes after it.
SYSTEM = (
    "You are an editor for a payments app. Rewrite the DRAFT sentence so it sounds like a "
    "person talking, not a form. FACTS is reference data — never copy it out.\n"
    "Rules:\n"
    "- Use ONLY numbers and names that appear in FACTS or the DRAFT. Invent nothing.\n"
    "- One or two short sentences. No greeting, no sign-off, no advice, no emoji.\n"
    "- Same language as the DRAFT.\n"
    "- Output only the rewritten sentence.\n"
    "\n"
    "Example\n"
    "FACTS: total=AED 210.00; period=the last 7 days; payments=4\n"
    "DRAFT: You spent AED 210.00 at merchants in the last 7 days.\n"
    "Rewrite: That's AED 210.00 across 4 payments this week."
)

_LLM = None
_LOCK = threading.Lock()
_STATS = {"calls": 0, "kept": 0, "rejected": 0, "ttft_ms": [], "tpot_ms": [], "tokens": 0}

#: Any number in the output must be traceable to the facts. Matches 1,234.56 / 85 / 4.5
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def available() -> bool:
    return ENABLED and os.path.exists(MODEL)


def _llm():
    """One model for the process, loaded on first use.

    `n_ctx` is small on purpose — the prompt is a system block plus a handful of facts, and
    the KV cache is sized from the context window, so a 32k context would reserve memory we
    will never use.
    """
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        _LLM = Llama(model_path=MODEL, n_ctx=1024, n_threads=max(2, (os.cpu_count() or 4) // 2),
                     n_batch=256, verbose=False, logits_all=False)
    return _LLM


def _numbers(text: str) -> set[str]:
    """Numbers as the customer would read them, normalised so 4,549.00 and 4549 match."""
    out = set()
    for m in _NUM.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        out.add(f"{v:.2f}".rstrip("0").rstrip("."))
    return out


def grounded(said: str, facts: str) -> bool:
    """Is every number in the sentence present in the facts?

    Deliberately strict and deliberately dumb: no tolerance, no rounding, no "close
    enough". A figure the engine did not produce has no business in an answer about
    someone's money.

    Necessary, and nowhere near sufficient — see `contradicts`.
    """
    return _numbers(said) <= _numbers(facts)


#: Pairs where swapping one word for the other reverses the meaning while keeping every
#: number intact. Each is a real failure observed from the 0.5B model on this exact task.
#:
#:   "Your 100 AED TO Alex"        -> "That's AED 100.00 FROM Alex"
#:   "in the last 30 days"          -> "this week ... than the previous week"
#:
#: Both passed the number check perfectly. Neither is true. Fluency is not accuracy, and a
#: model small enough to run on this box will cheerfully swap a preposition.
_OPPOSITES = (
    ({"to", "ko", "\u0915\u094b", "\u0625\u0644\u0649"}, {"from", "se", "\u0938\u0947", "\u0645\u0646"}),
    ({"sent", "paid", "spent"}, {"received", "got", "came in", "credited"}),
    ({"delivered", "success", "successful", "complete"},
     {"failed", "pending", "declined", "stuck", "on hold"}),
    ({"refunded"}, {"not refunded", "no refund"}),
    ({"more"}, {"less", "fewer"}),
    ({"week", "7 days"}, {"month", "30 days", "quarter", "90 days", "year"}),
)

#: Digits are kept. The first version stripped them, so a multi-word term containing a
#: number — "30 days", "7 days" — could never match, and "the last 30 days" rewritten as
#: "this week" sailed through the contradiction check untouched. A period is a fact.
_WORD = re.compile(r"[a-z0-9\u0900-\u097f\u0600-\u06ff]+")


def _has(text: str, terms: set) -> bool:
    low = " " + " ".join(_WORD.findall((text or "").lower())) + " "
    return any(f" {t} " in low for t in terms)


#: Verbs that attribute the ACT of paying to whoever is their subject.
_SPEND_VERB = re.compile(r"\b(spent|paid|sent|transferred|owes?)\b", re.I)


def role_swapped(said: str, draft: str, facts: str) -> str | None:
    """Did the rewrite change WHO did the thing?

    Two failures the number check and the opposite-word check both wave through, because
    every digit is right and no antonym appears:

      draft:   "Alex Morgan — AED 3,276.00 across 57 payments"
      rewrite: "Alex Morgan SPENT AED 3,276.00 on 57 transactions"   <- the customer did

      draft:   "I don't have a figure from this conversation to check."
      rewrite: "YOU don't have a figure from this conversation to check."

    The first makes the payee the spender; the second makes the customer the one who is
    confused. Both are fluent, both are grounded, both are false — and on a payments
    surface "who did this" is the fact, not decoration.
    """
    d, s = (draft or ""), (said or "")
    dl, sl = d.lower(), s.lower()

    # Speaker. "I" and "you" are not interchangeable in an assistant's sentence.
    if re.search(r"\bi (?:don'?t|can'?t|couldn'?t|have|haven'?t|see|read|checked)\b", dl) \
            and re.search(r"\byou (?:don'?t|can'?t|couldn'?t|have|haven'?t)\b", sl):
        return "the draft was about what I can do, the rewrite made it about you"

    # Subject of a spending verb. If the draft never says a NAMED person spent/paid, the
    # rewrite must not start saying so.
    names = {w for w in re.findall(r"[A-Z][a-z]{2,}", facts + " " + d)}
    for n in names:
        pat = re.compile(re.escape(n) + r"\s+\w{0,6}\s*" + _SPEND_VERB.pattern, re.I)
        if pat.search(s) and not pat.search(d):
            return f"the rewrite makes {n} the one who paid"
    return None


def contradicts(said: str, draft: str) -> str | None:
    """Did the rewrite flip a fact the draft had the other way round?

    The number check cannot see this: "to Alex" and "from Alex" contain the same digits.
    On a payments surface the direction of the money IS the fact, so a rewrite that
    reverses it is worse than no rewrite at all — it is fluent, confident and backwards.
    """
    for a, b in _OPPOSITES:
        if _has(draft, a) and _has(said, b) and not _has(draft, b):
            return f"draft said {sorted(a)[0]!r}, rewrite said {sorted(b)[0]!r}"
        if _has(draft, b) and _has(said, a) and not _has(draft, a):
            return f"draft said {sorted(b)[0]!r}, rewrite said {sorted(a)[0]!r}"
    return None


def rewrite(draft: str, facts: str, lang: str = "en", max_tokens: int = 90) -> dict:
    """Return a phrased version of `draft`, or the draft itself if anything is off.

    Never raises and never blocks the answer: a phrasing model that is slow, missing or
    hallucinating costs us the nicer sentence and nothing else.
    """
    out = {"text": draft, "used": False, "why": "", "ttft_ms": None, "tpot_ms": None}
    if not available() or not draft:
        out["why"] = "disabled" if not ENABLED else "model missing"
        return out

    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\nFACTS:\n{facts}\n\nDRAFT:\n{draft}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    try:
        with _LOCK:
            llm = _llm()
            _STATS["calls"] += 1
            t0 = time.perf_counter()
            first_at = None
            pieces = []
            # Streamed so TTFT is measurable rather than inferred — the prefill cost is
            # everything before the first token, the decode cost is everything after.
            for chunk in llm(prompt, max_tokens=max_tokens, temperature=0.3, top_p=0.9,
                             stop=["<|im_end|>", "\n\n"], stream=True):
                piece = chunk["choices"][0]["text"]
                if first_at is None and piece.strip():
                    first_at = time.perf_counter()
                pieces.append(piece)
            end = time.perf_counter()
    except Exception as e:                                    # noqa: BLE001
        out["why"] = f"generation failed: {type(e).__name__}"
        return out

    text = "".join(pieces).strip().strip('"')
    n_out = max(1, len(pieces))
    ttft = (first_at - t0) * 1000 if first_at else None
    tpot = ((end - first_at) * 1000 / n_out) if first_at and n_out > 1 else None
    out["ttft_ms"] = round(ttft, 1) if ttft else None
    out["tpot_ms"] = round(tpot, 2) if tpot else None
    if ttft:
        _STATS["ttft_ms"].append(ttft)
    if tpot:
        _STATS["tpot_ms"].append(tpot)
    _STATS["tokens"] += n_out

    if not text:
        out["why"] = "empty"
        _STATS["rejected"] += 1
        return out
    if not grounded(text, facts + " " + draft):
        # The important branch. Keep the template, and record it.
        out["why"] = "ungrounded — a number appeared that the engine never produced"
        _STATS["rejected"] += 1
        return out
    if len(text) > len(draft) * 2.5 + 60:
        out["why"] = "too long"
        _STATS["rejected"] += 1
        return out
    # A small model asked to rewrite will sometimes read the reference data back instead.
    # Every number is "grounded", so the number check passes and the output is still
    # useless — "total=AED 4,634.00; period=the last 30 days; payments=37" is not a
    # sentence. Grounding is about truth; this is about whether it answered at all.
    if text.count("=") >= 2 or text.count(";") >= 2 or text.lower().startswith("facts"):
        out["why"] = "echoed the facts instead of rewriting"
        _STATS["rejected"] += 1
        return out
    flip = contradicts(text, draft)
    if flip:
        out["why"] = f"reversed a fact — {flip}"
        _STATS["rejected"] += 1
        return out
    role = role_swapped(text, draft, facts)
    if role:
        out["why"] = f"changed who did it — {role}"
        _STATS["rejected"] += 1
        return out

    _STATS["kept"] += 1
    out.update(text=text, used=True, why="ok")
    return out


def stats() -> dict:
    def p(xs, q):
        return round(sorted(xs)[min(len(xs) - 1, int(len(xs) * q))], 1) if xs else 0.0
    return {
        "enabled": ENABLED, "model": os.path.basename(MODEL), "present": os.path.exists(MODEL),
        "calls": _STATS["calls"], "kept": _STATS["kept"], "rejected": _STATS["rejected"],
        "tokens": _STATS["tokens"],
        "ttft_p50_ms": p(_STATS["ttft_ms"], 0.5), "ttft_p95_ms": p(_STATS["ttft_ms"], 0.95),
        "tpot_p50_ms": p(_STATS["tpot_ms"], 0.5),
    }
