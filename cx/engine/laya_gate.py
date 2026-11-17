"""Laya as a genuine second opinion — an independent classifier, cross-checked against ours.

## What changed, and why the first attempt was useless

Laya was first asked three yes/no (`noul`) questions. It answered **True to all three for every
input** — a constant function, reading nothing. So it "agreed" only by accident, whenever our
expectation happened to be all-True, and disagreed on every forward-looking intent. That is
why it never seemed to agree with anything.

That was the wrong question TYPE, not a bad model. Asked a `choice` question it is accurate
and well calibrated:

    i want a refund                 -> refund            1.00
    where is my money               -> payment_status    1.00
    send 50 to alex                -> sending_money     1.00
    pay MClothes                    -> paying_a_shop     0.59   (genuinely ambiguous)
    how much is the fee to pakistan -> fees_and_charges  1.00
    any offers near me              -> offers_nearby     1.00
    what is the weather             -> something_else    0.99

7/7, in ~500 ms on CPU, with the confidence dropping exactly where the sentence is ambiguous.

## What it is for

A second, independently-trained model classifying the same sentence into coarse families.
When it agrees with `openwallet-cx-intent-v1`, two unrelated models read the sentence the same way
and we proceed. When it disagrees, that is a real signal the sentence is ambiguous — and the
engine ASKS instead of assuming.

It still decides nothing. It cannot change the refund matrix, the mandate, or an amount. The
most it can do is lower our confidence, which makes the assistant ask a question.
"""
from __future__ import annotations

import threading
import time

#: Coarse families, deliberately broader than our 15 intents. Laya is a cross-check, not a
#: replacement — asking it to reproduce our exact taxonomy would test agreement on labels
#: rather than agreement on meaning.
TOPICS = ("refund", "payment_status", "sending_money", "paying_a_shop",
          "fees_and_charges", "offers_nearby", "how_to_use_the_app",
          "greeting_or_asking_for_help", "something_else")

QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "What is `message` about?",
        "criteria": {t: None for t in TOPICS},
    }
}

#: Our intent -> the family Laya should land on if it reads the sentence the same way.
FAMILY = {
    "refund_request": "refund",
    "refund_status": "refund",
    "txn_status": "payment_status",
    "txn_failed": "payment_status",
    "txn_on_hold": "payment_status",
    "debited_not_credited": "payment_status",
    "send_money": "sending_money",
    "pay_merchant": "paying_a_shop",
    "fees_query": "fees_and_charges",
    "time_query": "fees_and_charges",
    "limits_query": "fees_and_charges",
    "offers_nearby": "offers_nearby",
    "journey_help": "how_to_use_the_app",
    "beneficiary_query": "how_to_use_the_app",
    "greeting_help": "greeting_or_asking_for_help",
    "out_of_scope": "something_else",
}

#: "Debited but not credited" is a complaint about a payment AND a request for money back;
#: either reading is defensible, so neither counts as a disagreement.
ALSO_FINE = {
    # An open "I need help" is reasonably read as any of these.
    "greeting_help": {"how_to_use_the_app", "something_else", "payment_status"},
    "debited_not_credited": {"refund"},
    "refund_status": {"payment_status"},
    "txn_failed": {"refund"},
    "beneficiary_query": {"sending_money"},
    "limits_query": {"how_to_use_the_app"},
    "time_query": {"payment_status"},
}

#: Below this, Laya is unsure enough that its disagreement says more about the model than the
#: sentence, so we do not act on it.
MIN_CONFIDENCE = 0.45

_agent = None
_state = "cold"
_error: str | None = None
_lock = threading.Lock()


def status() -> str:
    return _state


def error() -> str | None:
    return _error


def warm(block: bool = False) -> None:
    global _state
    with _lock:
        if _state in ("loading", "ready"):
            return
        _state = "loading"

    def _load():
        global _agent, _state, _error
        try:
            import laya
            _agent = laya.load("convaiinnovations/laya")
            _state = "ready"
        except Exception as e:
            # Record WHY. Swallowing this silently turned a loading problem into a mystery:
            # the health endpoint just said "unavailable" with no reason anywhere.
            _error = f"{type(e).__name__}: {e}"
            _state = "unavailable"
            print(f"  laya failed to load — {_error}", flush=True)

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    if block:
        t.join()


def _unwrap(value):
    """Laya returns the choice as a dict carrying probabilities; take the label and its
    confidence. Reading it as a bare string is what made a perfect 7/7 look like 0/7."""
    if isinstance(value, dict):
        return value.get("choice"), float(value.get("answer_confidence")
                                          or value.get("confidence") or 0.0)
    return value, 0.0


def second_opinion(message: str, our_intent: str, timeout: float = 6.0) -> dict | None:
    if _state != "ready" or _agent is None:
        return None
    try:
        import laya
        t0 = time.time()
        out = laya.decide(_agent, {"message": message}, questions=QUESTIONS,
                          return_details=True)
        ms = round((time.time() - t0) * 1000)
        choice, conf = _unwrap((out.values or {}).get("topic"))
    except Exception:
        return None

    expected = FAMILY.get(our_intent)
    acceptable = {expected} | ALSO_FINE.get(our_intent, set())
    confident = conf >= MIN_CONFIDENCE
    agrees = (choice in acceptable) if (choice and expected) else True

    return {
        "available": True,
        "topic": choice,
        "confidence": round(conf, 3),
        "expected": expected,
        "agrees": agrees,
        # Only a CONFIDENT disagreement is worth acting on.
        "actionable": (not agrees) and confident,
        "disagrees_on": [] if agrees else ["topic"],
        "ms": ms,
        "role": "advisory second opinion — never decides, never overrules the policy table",
    }


def unavailable(reason: str = "model not loaded") -> dict:
    return {"available": False, "agrees": None, "disagrees_on": [],
            "reason": _error or reason,
            "role": "advisory second opinion — never decides"}
