"""What the assistant keeps from a turn so the next model is better than this one.

## The honest position

Before this file, the answer to "is it actually learning?" was **no**. The `learn` web on the
crawl screen showed what a turn REMEMBERED — the session frame, the pinned language, the task
ledger — which is memory, not learning. Nothing was written anywhere durable and no model was
ever retrained from it. That was deliberate (the web says "candidate for the next retrain",
not "retrained"), but a loop that is never closed is not a loop.

This closes it, and it closes it in the only way that is safe on a money surface:

* **Nothing retrains live.** A classifier that moves money must not change its mind between
  two customers because of something a third one typed. Live self-training on a payments
  surface is how you get an assistant that quietly drifts into sending money to the wrong
  person, with no release, no diff and no way back.
* **What happens instead** is that every turn is written to an append-only store with the
  thing the current corpus cannot express: **the topic that was open when it was said**. The
  next training run reads it. The improvement lands in a model you can version, test against
  the suite, and roll back.

## Why the topic is the whole point

`openwallet-cx-intent-ml-v7` sees one sentence. "was it more than the previous one?" is identical
after a spend breakdown and after a payment, so no amount of single-sentence data can teach
it the difference — that is why the two context bugs were fixed in code, not by training.
Every row here carries `topic`, so the pairs `(topic, utterance) -> intent` are a corpus for
a model that takes the thread as a feature. That model does not exist yet. This is what it
will be trained on.

## What is worth keeping

Not everything. A store that keeps every turn is mostly the same four sentences and teaches
nothing, so a turn is kept when it carries signal:

* **confirmed** — the customer went through with it, so the reading was right
* **corrected** — the next turn contradicted this one, which is the most valuable row in the
  file and the hardest to get any other way
* **rule-reached** — an intent the model cannot predict yet (`spend-rule`, `direction-rule`,
  `sticky`), which is exactly what the next label space needs
* **guarded** — a guard fired, meaning the engine was unsure and said so

Ordinary, confident, unremarkable turns are counted and dropped.
"""
from __future__ import annotations

import json
import os
import threading
import time

#: One file, append-only, newline-delimited. Plain enough to read with `tail` and to feed to
#: a training script without a parser.
STORE = os.environ.get(
    "CX_LEARN_STORE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "learned", "turns.jsonl"))

#: Models reached by rule rather than by the classifier. These are the rows the next retrain
#: most needs, because they are the cases the current label space cannot reach at all.
_RULE_MODELS = {"spend-rule", "direction-rule", "dispute-rule", "sticky", "scope-rule",
                "chosen"}

_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {"seen": 0, "kept": 0, "confirmed": 0, "corrected": 0,
                           "rule": 0, "guarded": 0}
#: The previous turn per session, so a contradiction can be spotted and labelled.
_PREV: dict[str, dict] = {}


def _reason(a, prev: dict | None) -> str | None:
    """Why this turn is worth keeping, or None to drop it."""
    if getattr(a, "confirmed", False) or (a.outcome or "") == "CONFIRMED":
        return "confirmed"
    if getattr(a, "model", "") in _RULE_MODELS:
        return "rule-reached"
    if any(st.get("tag") == "guard" for st in (getattr(a, "lane_trace", None) or [])):
        return "guarded"
    return None


def _corrects(prev: dict | None, a) -> bool:
    """Did this turn contradict the one before it?

    The signal is narrow on purpose: the previous turn asked a question or proposed
    something, and this one came back with a DIFFERENT intent that was not an answer to it.
    That is the customer saying "no, I meant something else", and it is the row that tells
    the next model where this one was wrong.
    """
    if not prev:
        return False
    if prev.get("outcome") not in ("NEED_INFO", "PROPOSED"):
        return False
    return bool(a.intent and a.intent != prev.get("intent")
                and (a.outcome or "") not in ("CANCELLED",))


def record(a, said: str, session: str, topic: str | None = None) -> str | None:
    """Keep this turn if it carries signal. Returns the reason kept, or None.

    Never raises: a failure to learn must not fail an answer. The customer's payment does not
    depend on this file being writable.
    """
    try:
        with _LOCK:
            _COUNTS["seen"] += 1
            prev = _PREV.get(session)
            corrected = _corrects(prev, a)
            reason = "corrected" if corrected else _reason(a, prev)
            _PREV[session] = {"intent": a.intent, "outcome": a.outcome, "said": said}
            if not reason:
                return None
            row = {
                "at": round(time.time()),
                "session": session,
                "said": said,
                #: The feature the current corpus cannot express.
                "topic": topic,
                "intent": a.intent,
                "model": getattr(a, "model", ""),
                "confidence": round(float(getattr(a, "confidence", 0) or 0), 3),
                "lang": getattr(a, "lang", "en"),
                "outcome": a.outcome,
                "slots": {k: v for k, v in (a.slots or {}).items() if v not in (None, "", [])},
                "why_kept": reason,
                #: What the previous turn was, so a correction can be read in context.
                "after": (prev or {}).get("intent"),
            }
            _COUNTS["kept"] += 1
            _COUNTS[{"confirmed": "confirmed", "corrected": "corrected",
                     "rule-reached": "rule", "guarded": "guarded"}[reason]] += 1
            os.makedirs(os.path.dirname(STORE), exist_ok=True)
            with open(STORE, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            return reason
    except Exception:                                     # noqa: BLE001
        return None


def stats() -> dict:
    """Counts for this process, plus the size of the store on disk."""
    out = dict(_COUNTS)
    try:
        with open(STORE) as fh:
            out["rows_on_disk"] = sum(1 for _ in fh)
    except Exception:                                     # noqa: BLE001
        out["rows_on_disk"] = 0
    out["store"] = STORE
    return out


def rows(limit: int = 0) -> list[dict]:
    """Everything kept so far, for a training run to read."""
    try:
        with open(STORE) as fh:
            out = [json.loads(l) for l in fh if l.strip()]
    except Exception:                                     # noqa: BLE001
        return []
    return out[-limit:] if limit else out
