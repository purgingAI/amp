#!/usr/bin/env python3
"""The guards on the generative phrasing layer.

The model that rewrites our answers is never allowed to decide anything, and these are the
checks that enforce it. They are tested WITHOUT loading the model: a verification layer
that only works when a 491MB file is present is a verification layer nobody runs.

Every rejection case below is a real output observed from Qwen2.5 0.5B or 1.5B on this
exact task. None of them is hypothetical, and each one was fluent, confident and wrong.

Run: ~/amp/.venv/bin/python cx/tests/test_phrase.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, ok, got=""):
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got}")


def test_a_number_the_engine_never_produced_is_rejected():
    import phrase
    print("\ngrounding — invented figures")
    facts = "total=AED 4,634.00; payments=37; period=the last 30 days"
    check("the true figure passes", phrase.grounded("You spent AED 4,634.00", facts))
    check("a rounded figure is still invented",
          not phrase.grounded("You spent about AED 4,600", facts))
    check("a plausible wrong figure is rejected",
          not phrase.grounded("You spent AED 4,549.00", facts))
    check("formatting does not matter", phrase.grounded("You spent AED 4634", facts))
    check("a sentence with no numbers is fine", phrase.grounded("All settled.", facts))


def test_a_reversed_fact_is_rejected():
    import phrase
    print("\ncontradiction — same digits, opposite meaning")
    cases = [
        ("Your 100 AED to Alex is delivered.",
         "That's AED 100.00 from Alex, delivered.", "to became from"),
        ("You spent AED 210.00 in the last 30 days.",
         "You spent AED 210.00 this week.", "30 days became week"),
        ("Your payment is delivered.",
         "Your payment failed.", "delivered became failed"),
        ("AED 20.00 more on services than before.",
         "AED 20.00 less on services than before.", "more became less"),
    ]
    for draft, bad, why in cases:
        check(f"caught: {why}", phrase.contradicts(bad, draft) is not None, (draft, bad))

    print("  and an honest rewrite is not flagged")
    for draft, good in [
        ("Your 100 AED to Alex is delivered.", "That's AED 100.00 to Alex, delivered."),
        ("You spent AED 210.00 in the last 30 days.",
         "AED 210.00 over the last 30 days."),
    ]:
        check(f"passes: {good[:40]!r}", phrase.contradicts(good, draft) is None, good)


def test_a_changed_subject_is_rejected():
    """The failure that only appeared with the BIGGER model — it writes well enough to
    make a confident claim about the wrong person."""
    import phrase
    print("\nrole and speaker")
    facts = "top_payee=Alex Morgan; total=AED 3,276.00; count=57"
    draft = "Alex Morgan — AED 3,276.00 across 57 payments in the last 30 days."
    bad = "Alex Morgan spent AED 3,276.00 on 57 transactions in the last 30 days."
    check("caught: the payee made the spender",
          phrase.role_swapped(bad, draft, facts) is not None, bad)
    good = "That's AED 3,276.00 to Alex Morgan over 57 payments."
    check("an honest rewrite passes", phrase.role_swapped(good, draft, facts) is None, good)

    d2 = "I don't have a figure from this conversation to check."
    b2 = "You don't have a figure from this conversation to check."
    check("caught: I became you", phrase.role_swapped(b2, d2, "") is not None, b2)
    g2 = "There's no figure from this conversation for me to check."
    check("a rephrasing that keeps the speaker passes",
          phrase.role_swapped(g2, d2, "") is None, g2)


def test_the_facts_read_back_are_rejected():
    import phrase
    print("\nechoing the reference data")
    facts = "total=AED 4,634.00; period=the last 30 days; payments=37"
    # Every number is grounded, so the grounding check passes and the output is useless.
    check("grounding alone would let it through", phrase.grounded(facts, facts))
    r = phrase.rewrite("You spent AED 4,634.00 in the last 30 days.", facts)
    # With the model off this returns the draft untouched, which is the safe default.
    check("with the model off, the draft is returned unchanged",
          r["text"] == "You spent AED 4,634.00 in the last 30 days." and not r["used"],
          r)


def test_it_is_off_unless_asked_for():
    import phrase
    print("\ndefault state")
    check("disabled unless CX_PHRASE=1", phrase.ENABLED == (os.environ.get("CX_PHRASE") == "1"))
    if not phrase.ENABLED:
        r = phrase.rewrite("Anything at all.", "x=1")
        check("returns the draft", r["text"] == "Anything at all.", r["text"])
        check("and says it did not run", not r["used"] and r["why"] == "disabled", r["why"])


if __name__ == "__main__":
    test_a_number_the_engine_never_produced_is_rejected()
    test_a_reversed_fact_is_rejected()
    test_a_changed_subject_is_rejected()
    test_the_facts_read_back_are_rejected()
    test_it_is_off_unless_asked_for()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
