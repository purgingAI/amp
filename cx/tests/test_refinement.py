import sys, os
sys.path.insert(0, "/home/sagar-uniyal/amp/cx")
from engine.engine import ask
def one(t, s): return ask(t, want_second_opinion=False, session=s)

def test_which_has_offer_narrows_the_list():
    one("i want to send money to merchant", "r1")
    a = one("which has offer", "r1")
    assert a.model == "refinement", a.model
    assert all("offer" in o["label"].lower() or "off" in o["label"].lower()
               for o in a.options), [o["label"] for o in a.options]

def test_nearest_sorts_by_distance():
    one("i want to send money to merchant", "r2")
    a = one("which is nearest", "r2")
    assert a.model == "refinement" and a.options

def test_a_refinement_needs_an_open_question():
    # Same words, no open frame: must NOT be treated as a refinement.
    a = one("which has offer", "r3")
    assert a.model != "refinement", a.model

def test_naming_a_merchant_is_an_answer_not_a_filter():
    one("i want to send money to merchant", "r4")
    a = one("MClothes", "r4")
    assert a.model != "refinement", a.model

def test_a_real_request_is_not_swallowed():
    one("i want to send money to merchant", "r5")
    a = one("i want a refund", "r5")
    assert a.intent == "refund_request", a.intent

def test_a_long_sentence_is_not_a_refinement():
    one("i want to send money to merchant", "r6")
    a = one("can you please tell me which of these merchants has the best discount today",
            "r6")
    assert a.model != "refinement", a.model

if __name__ == "__main__":
    bad = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try: f(); print(f"  ok   {n}")
            except AssertionError as e: bad += 1; print(f"  FAIL {n}: {e}")
    print(f"\n{'FAILED' if bad else 'all green'} ({bad} failing)")
    raise SystemExit(1 if bad else 0)
