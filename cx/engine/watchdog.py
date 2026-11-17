"""Watch every answer and flag the ones that look wrong, using the customer's own words.

You asked me to watch the conversation and fix what looks wrong, training on what the
customer actually types. Here is the honest split of that, because one half is possible in a
millisecond and the other half is not:

## What happens immediately

Some answers can be judged wrong by the answer alone, with no model involved:

* a money-shaped sentence that got refused,
* a non-English question answered in English,
* a confident answer whose runner-up was nearly as confident,
* a refusal of something containing the customer's own payee or biller names.

Those are caught here, and where there is a safe correction it is applied AT ONCE through
`OVERRIDES` — a table this module writes and the engine consults before the model. No
retraining, effective on the next message. That is the only kind of "train it then and there"
that is truthful: a rule, learned from one real utterance, applied immediately.

## What does not happen immediately

**The model's weights do not change.** Fine-tuning MiniLM on this corpus takes about
25 minutes on this machine, needs the whole corpus to avoid forgetting everything else, and
must be measured against a held-out set before it is trusted — a model that silently
retrained on every message would drift with no way to tell whether it had got better or
worse. So every flagged utterance is written to `data/learned.jsonl` with what the engine
did and what it should have done, and that file is corpus input for the next deliberate
retrain.

Claiming otherwise would be the most tempting lie in this whole system: it would look like
learning and would actually be a rule table with a model-shaped label on it.

## The rule about money

A flag NEVER changes a decision that moves money. It can turn a refusal into a question, and
it can change which language we answer in. It cannot decide a payment the model declined —
"this looks like it should have been allowed" is not a basis for sending somebody's money.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
DATA = os.path.join(CX, "data")
LEARNED = os.path.join(DATA, "learned.jsonl")

_LOCK = threading.Lock()

#: Utterance (lowercased, collapsed) -> intent, learned from a flagged refusal and applied
#: before the model on the next message. Loaded from disk at import so it survives a restart.
OVERRIDES: dict[str, str] = {}

#: What an override is allowed to produce. Never a money-moving intent: a rule learned from
#: one sentence must not be able to authorise a payment the model declined.
_SAFE_TARGETS = {"greeting_help", "bill_due", "split_status", "txn_history",
                 "limits_query", "fees_query", "beneficiary_query", "offers_nearby"}


def _load() -> None:
    try:
        with open(LEARNED) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("fix") and r.get("text"):
                    OVERRIDES[r["text"]] = r["fix"]
    except FileNotFoundError:
        pass
    except Exception:
        pass


def override(said: str) -> str | None:
    """An intent learned from an earlier identical utterance, or None."""
    return OVERRIDES.get(" ".join((said or "").lower().split()))


# ── the checks ───────────────────────────────────────────────────────────────────────────

_MONEYISH = re.compile(
    r"\b(send|transfer|pay|paise|paisa|rupaye|bhej\w*|money|cash|amount|dirham\w*)\b"
    r"|भेज|पैसे|पैसा|रुपये|أرسل|أموال|مال", re.IGNORECASE)

#: A biller word is NOT enough. "fix my internet" names a utility and is a tech-support
#: request we should decline — it was flagged as a mis-refused bill question and would have
#: been LEARNED as one, which is the exact failure mode of correcting yourself
#: automatically. So a bill flag needs the word AND a billing context: owing, paying, being
#: due, or the customer calling it "my bill".
_BILLISH = re.compile(r"\bbill\w*|utilit\w*|बिल|فاتورة", re.IGNORECASE)
_BILL_CONTEXT = re.compile(
    r"\b(due|owe|owing|outstanding|overdue|pay|paid|balance|amount|how much|kitna|"
    r"baaki|bakaya|bhar\w*)\b"
    r"|बाकी|कितना|भर|देना|مستحق|متأخر|كم|ادفع", re.IGNORECASE)


def inspect(said: str, answer) -> list[dict]:
    """Everything that looks wrong about this answer. Empty list is the normal case."""
    flags: list[dict] = []
    low = " ".join((said or "").lower().split())
    intent = getattr(answer, "intent", "")
    say = getattr(answer, "say", "") or ""
    lang = getattr(answer, "lang", "en")

    # 1. a money- or bill-shaped sentence that got refused
    if intent == "out_of_scope":
        if _MONEYISH.search(low):
            flags.append({"kind": "refused_money_request",
                          "why": "mentions money or sending but was declined",
                          "fix": None})
        elif _BILLISH.search(low) and _BILL_CONTEXT.search(low):
            flags.append({"kind": "refused_bill_request",
                          "why": "asks about owing or paying a bill but was declined",
                          "fix": "bill_due"})

    # 2. answered in the wrong language. Checked by SCRIPT, which cannot be argued with: a
    #    Devanagari question answered in Latin is wrong whatever the words are.
    if lang == "hi" and say and not re.search(r"[ऀ-ॿ]", say):
        flags.append({"kind": "answered_in_english",
                      "why": "asked in Hindi, answered without a single Devanagari character",
                      "fix": None})
    elif lang == "ar" and say and not re.search(r"[؀-ۿ]", say):
        flags.append({"kind": "answered_in_english",
                      "why": "asked in Arabic, answered without a single Arabic character",
                      "fix": None})

    # 3. a near-tie presented as a confident answer. The customer sees certainty the model
    #    did not have.
    top = getattr(answer, "top", None) or []
    # Only when the MODEL decided. "yes" is answered from the standing proposal, a
    # continuation from the open frame, a pick from the customer's own tap — in all of those
    # the classifier's ranking was not used, so reporting that it was unsure says nothing
    # about the answer we gave.
    if getattr(answer, "model", "") in ("answered", "learned", "chosen", "continuation",
                                        "sticky", "held-open", "scope-rule", "rules", ""):
        top = []
    try:
        # The field is `p`, not `score`. Reading the wrong key made every probability 0.0,
        # so "0.00 barely beat 0.00" was true of every answer and the flag fired on all of
        # them — noise that would have buried the real ones.
        pairs = [(t.get("intent"), float(t.get("p", t.get("score", 0)))) for t in top] \
            if isinstance(top, list) and top and isinstance(top[0], dict) else []
        if len(pairs) >= 2 and pairs[0][1] - pairs[1][1] < 0.12:
            flags.append({"kind": "near_tie",
                          "why": f"{pairs[0][0]} {pairs[0][1]:.2f} barely beat "
                                 f"{pairs[1][0]} {pairs[1][1]:.2f}", "fix": None})
    except Exception:
        pass

    # 4. an empty answer — EXCEPT when it is deliberate. A confirmed proposal returns no
    #    sentence on purpose: the surface executes it and the result speaks for itself. This
    #    flagged "yes" as a broken answer every time somebody confirmed a payment.
    if (intent != "out_of_scope" and len(say.strip()) < 3
            and not getattr(answer, "confirmed", False)
            and getattr(answer, "outcome", None) not in ("CONFIRMED", "CANCELLED")):
        flags.append({"kind": "empty_answer", "why": "no sentence was produced",
                      "fix": None})
    return flags


def record(said: str, answer, flags: list[dict]) -> None:
    """Append the utterance, what we did, and what it should have been.

    This file is corpus input for the next retrain. It holds what the customer typed and the
    label we believe is right — which is the only training data worth having, and the only
    kind nobody has to invent.
    """
    if not flags:
        return
    low = " ".join((said or "").lower().split())
    fix = next((f["fix"] for f in flags if f.get("fix") in _SAFE_TARGETS), None)
    row = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "text": low,
        "lang": getattr(answer, "lang", "en"),
        "got": getattr(answer, "intent", ""),
        "outcome": getattr(answer, "outcome", None),
        "say": (getattr(answer, "say", "") or "")[:200],
        "flags": [f["kind"] for f in flags],
        "why": [f["why"] for f in flags],
        "fix": fix,
    }
    try:
        os.makedirs(DATA, exist_ok=True)
        with _LOCK:
            with open(LEARNED, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if fix:
                OVERRIDES[low] = fix
    except Exception:
        pass            # never let bookkeeping break an answer


def stats() -> dict:
    """What has been flagged, for the operator screen. Not shown to a customer."""
    rows = []
    try:
        with open(LEARNED) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    except Exception:
        pass
    by: dict[str, int] = {}
    for r in rows:
        for k in r.get("flags", []):
            by[k] = by.get(k, 0) + 1
    return {"total": len(rows), "by_kind": by,
            "overrides": len(OVERRIDES),
            "recent": rows[-12:][::-1]}


_load()
