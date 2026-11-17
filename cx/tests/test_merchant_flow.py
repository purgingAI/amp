#!/usr/bin/env python3
"""A merchant payment must not turn into a payment to somebody else.

Reported from a real session, and it moved the payee:

    "dukan pe pay karna hai"  ->  which merchant?
    "MClothes"                ->  MClothes par kitna?
    "100 kardo"               ->  "Ready to send 100 AED to alex"     <-- WRONG

Two independent defects, each pinned below.

**1. "kardo" was not known as a filler word.** `_is_pure_slot` strips the number and
trailing filler; if anything meaningful survives, the utterance is treated as a NEW request
rather than an answer to the question just asked. The list held English, Devanagari and
Arabic words but no romanised Hinglish, so "kardo" survived, "100 kardo" stopped looking
like an answer, and a confident send_money (0.97) displaced the open merchant payment. The
merchant was dropped and the money pointed at the configured peer.

The strip must stay sharp in the other direction too: "refund 250" keeps "refund" and is
still a new request. Intent-bearing verbs must never be filler.

**2. `"du" in low` matched "dukan".** Biller words were matched as substrings, so the UAE
telecom "du" was found inside "dukan" (shop) and the very first utterance arrived carrying
a bill nobody had mentioned. "dubai" would have done it too.

Run: ~/amp/.venv/bin/python cx/tests/test_merchant_flow.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got!r} (wanted {want!r})")
    else:
        print(f"  ok    {name}")


def test_bare_answers_are_answers():
    """A number plus filler is an answer; a number plus a VERB is a new request."""
    from engine.engine import _is_pure_slot
    from engine import slots as sl
    print("\nbare answer vs new request")
    for said, want in [
        ("100 kardo", True), ("100 kar do", True), ("100 dirhams", True),
        ("bas 100", True), ("100 rupaye kardo", True), ("ठीक 100", True),
        ("100 krdo", True), ("haan 100", True),
        # intent-bearing words must survive the strip
        ("refund 250", False), ("100 bhejdo", False), ("send 100 to asad", False),
        ("split 100", False), ("pay 100 to mclothes", False),
    ]:
        check(f"pure({said!r})", _is_pure_slot(said, sl.extract(said, "").as_dict()), want)


def test_biller_words_are_whole_words():
    from engine import slots as sl
    print("\nbiller words match whole words only")
    for said, want in [
        ("dukan pe pay karna hai", None),      # "du" is inside "dukan"
        ("dukan", None),
        ("dubai me kya hai", None),
        ("internet bill bharna hai", "Home internet"),
        ("bijli ka bill", "Home electricity"),
        ("du ka bill", "Home internet"),       # the real biller still resolves
    ]:
        check(f"bill({said!r})", sl.extract(said, "").bill, want)


def test_the_whole_flow():
    """End to end: the amount must land on the MERCHANT, not on a person."""
    from engine import engine as E
    print("\nthe reported conversation")
    S = "test-merchant-flow"
    E.ask("dukan pe pay karna hai", session=S)
    a2 = E.ask("MClothes", session=S)
    check("asked how much at the merchant", a2.intent, "pay_merchant")
    a3 = E.ask("100 kardo", session=S)
    check("still a merchant payment", a3.intent, "pay_merchant")
    p = a3.proposal or {}
    check("proposal is for the merchant", p.get("kind"), "pay_merchant")
    check("the merchant survived", p.get("merchant"), "MClothes")
    check("the amount landed", p.get("amount"), 100.0)
    check("no payee was invented", p.get("payee"), None)
    # and it must be routed to the merchant's own wallet, not the configured peer
    check("has the merchant's wallet", bool(p.get("receiverMobile")), True)


if __name__ == "__main__":
    test_bare_answers_are_answers()
    test_biller_words_are_whole_words()
    test_the_whole_flow()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
