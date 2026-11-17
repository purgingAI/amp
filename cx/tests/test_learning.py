#!/usr/bin/env python3
"""The learning loop: what gets kept, and what is allowed to become a training label.

The question this answers is "is it actually learning?", and before `learn.py` the honest
answer was no — the learn step recorded what the turn remembered and nothing was written
anywhere. Now every turn goes to an append-only store and `model/promote.py` turns the
trustworthy ones into training rows.

The load-bearing test is the last one. Knowing the right answer for a turn is not the same
as that answer being readable from the sentence, and the row that proves it is:

    "yes"  ->  send_money

true of that moment, false of the word. Promote it and the classifier learns that "yes"
means move money.

Run: ~/amp/.venv/bin/python cx/tests/test_learning.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# a store of our own, so a test never writes to the real one
_TMP = tempfile.mkdtemp(prefix="cx-learn-")
os.environ["CX_LEARN_STORE"] = os.path.join(_TMP, "turns.jsonl")

FAILED = []


def check(name, ok, got=""):
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got}")


def test_a_turn_is_only_kept_when_it_carries_signal():
    import learn
    from engine.engine import ask
    print("\nwhat is kept")
    learn._COUNTS.update({k: 0 for k in learn._COUNTS})
    S = "L-keep"
    a1 = ask("send 50 to alex", want_second_opinion=False, session=S)
    a2 = ask("yes", want_second_opinion=False, session=S)
    check("a confirmed turn is kept", a2.kept == "confirmed", a2.kept)
    check("an ordinary proposal is not", a1.kept in (None, "guarded"), a1.kept)

    a3 = ask("analyse my spending last month", want_second_opinion=False, session="L-rule")
    check("a rule-reached turn is kept", a3.kept == "rule-reached", a3.kept)

    rows = learn.rows()
    check("rows carry the open topic", all("topic" in r for r in rows),
          "a row without a topic is the whole reason this file exists")
    check("rows carry why they were kept", all(r.get("why_kept") for r in rows))


def test_a_contradiction_is_the_row_worth_having():
    import learn
    from engine.engine import ask
    print("\na correction")
    # Deliberately NOT a bill flow. The first version opened with "bill pay karna hai",
    # which needs the bill provider to be reachable — so the day the provider was down this
    # test did not report "bills are unavailable", it reported that the learning store had
    # stopped recording corrections. A test should fail for its own subject or not at all.
    S = "L-corr"
    first = ask("i want a refund", want_second_opinion=False, session=S)
    if first.outcome not in ("NEED_INFO", "PROPOSED"):
        check("SKIPPED: no open question to contradict", True,
              f"refund returned {first.outcome}")
        return
    a = ask("actually send 20 to alex", want_second_opinion=False, session=S)
    check("the contradicting turn is kept as a correction", a.kept == "corrected", a.kept)
    rows = [r for r in learn.rows() if r["said"].startswith("actually send")]
    check("the correction reached the store", bool(rows), "nothing written")
    if rows:
        check("and it records what it contradicted",
              rows[-1].get("after") == "refund_request", rows[-1].get("after"))


def test_learning_never_fails_an_answer():
    """A store that cannot be written must not take an answer down with it."""
    import learn
    from engine.engine import ask
    print("\nthe store is not in the critical path")
    old = learn.STORE
    learn.STORE = "/proc/definitely/not/writable/turns.jsonl"
    try:
        a = ask("show me my payments", want_second_opinion=False, session="L-fail")
        check("the answer is still served", bool(a.say), a.say[:40])
        check("and the turn is simply not kept", a.kept is None, a.kept)
    finally:
        learn.STORE = old


def test_a_label_from_the_offer_never_becomes_a_training_row():
    """The one that matters. `promote.py` splits on HOW the label was reached."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "model"))
    import promote
    print("\nwhat may become a training label")

    rows = [
        # the label came from the standing offer, not the word
        {"said": "yes", "intent": "send_money", "model": "answered",
         "why_kept": "confirmed", "topic": "send_money"},
        # the label came from the open topic, not the sentence
        {"said": "and healthcare?", "intent": "spend_analysis", "model": "sticky",
         "why_kept": "rule-reached", "topic": "spend_analysis"},
        # the sentence itself says what it is
        {"said": "analyse my spending last month", "intent": "spend_analysis",
         "model": "spend-rule", "why_kept": "rule-reached", "topic": None},
        # we were unsure; an unsure turn is not a label
        {"said": "something ambiguous", "intent": "txn_history",
         "model": "openwallet-cx-intent-ml-v7", "why_kept": "guarded", "topic": None},
        # long digits are not needed to learn a sentence shape
        {"said": "show transactions for wallet 971501234567", "intent": "txn_history",
         "model": "openwallet-cx-intent-ml-v7", "why_kept": "confirmed", "topic": None},
    ]
    train, ctx, _ = promote.gather(rows)
    texts = [r["text"] for r in train]
    check("'yes' is NOT a training row", "yes" not in texts, texts)
    check("'and healthcare?' is NOT a training row", "and healthcare?" not in texts, texts)
    check("the self-describing sentence IS", "analyse my spending last month" in texts, texts)
    check("the unsure turn is not promoted at all",
          "something ambiguous" not in texts
          and not any(r["text"] == "something ambiguous" for r in ctx), texts)
    check("both context rows are held with their topic",
          {r["text"] for r in ctx} == {"yes", "and healthcare?"}
          and all(r.get("topic") for r in ctx), ctx)
    check("account numbers are scrubbed",
          any("971501234567" not in r["text"] and "wallet" in r["text"] for r in train),
          texts)


if __name__ == "__main__":
    test_a_turn_is_only_kept_when_it_carries_signal()
    test_a_contradiction_is_the_row_worth_having()
    test_learning_never_fails_an_answer()
    test_a_label_from_the_offer_never_becomes_a_training_row()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
