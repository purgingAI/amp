"""Multi-turn behaviour: a follow-up must be understood, and must not be over-understood.

Written after the live report: "pay Asad and Sons" → "How much?" → "100 dirhams" →
"I think you want to send money, but I'm not certain enough". The guard was right; the
problem was that the reply was classified from scratch instead of answering the open question.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.engine import ask


def one(text, session):
    return ask(text, want_second_opinion=False, session=session)


def test_amount_completes_an_open_merchant_payment():
    assert one("pay Asad and Sons FZ LLC", "c1").outcome == "NEED_INFO"
    a = one("100 dirhams", "c1")
    assert a.model == "continuation"
    assert a.outcome == "PROPOSED"
    assert a.proposal["merchant"] == "Asad and Sons FZ LLC"
    assert a.proposal["amount"] == 100


def test_bare_number_also_completes_it():
    assert one("pay MClothes", "c2").outcome == "NEED_INFO"
    a = one("250", "c2")
    assert a.outcome == "PROPOSED" and a.proposal["amount"] == 250


def test_a_clear_new_request_is_not_swallowed():
    # "refund 250" carries an amount, but it is plainly a refund request and must not be
    # absorbed into the open payment just because a slot matches.
    one("pay Asad and Sons FZ LLC", "c3")
    a = one("refund 250", "c3")
    assert a.intent == "refund_request", a.intent
    assert a.model != "continuation"


def test_the_frame_is_cleared_once_answered():
    one("pay MClothes", "c4")
    one("250", "c4")
    a = one("50", "c4")          # a stray number must not re-open a finished payment
    assert a.outcome != "PROPOSED", a.outcome


def test_sessions_do_not_leak_into_each_other():
    one("pay MClothes", "c5")
    a = one("100 dirhams", "c6")   # different customer/tab entirely
    assert a.outcome != "PROPOSED"


def test_payee_completes_an_open_send():
    assert one("send money", "c7").outcome in ("NEED_INFO", "NEED_CONFIRMATION")
    a = one("to alex", "c7")
    assert a.slots.get("payee") == "Alex Morgan"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ok   {name}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if failed else 'all green'} ({failed} failing)")
    raise SystemExit(1 if failed else 0)
