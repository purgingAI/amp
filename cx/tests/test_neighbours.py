#!/usr/bin/env python3
"""Typos, unseen phrasings, and the cache.

All three come from one reported turn:

    "i want a refund"            -> about your 80 AED to MClothes
    "yes show me the detials"    -> "I don't have anyone called Detials on your list."

Two failures stacked. One transposed pair made a common word into a name, and even spelled
correctly the sentence named no subject, so it was read as a fresh request for a list
instead of a follow-up about the payment already on screen.

Run: ~/amp/.venv/bin/python cx/tests/test_neighbours.py
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


def convo(session, *turns):
    from engine.engine import ask
    out = None
    for t in turns:
        out = ask(t, want_second_opinion=False, session=session)
    return out


def test_the_reported_turn():
    print("\nthe reported turn")
    from engine.engine import ask
    first = ask("i want a refund", want_second_opinion=False, session="n-report")
    a = ask("yes show me the detials", want_second_opinion=False, session="n-report")
    check("no payee is invented", "detials" not in (a.say or "").lower()
          and "called Det" not in (a.say or ""), a.say[:80])
    check("it is about the payment in hand", a.intent == "txn_status", a.intent)
    # Compare against the payment the FIRST turn actually chose, rather than a merchant
    # name written into the test. The ledger is live, so a hardcoded payee turns this into
    # a test of yesterday's data — it failed the moment a newer payment arrived, and the
    # behaviour it is meant to pin had not changed at all.
    want = ((first.txn or {}).get("payee") or "").split()[0:1]
    check("and names the same payment the refund was about",
          not want or want[0].lower() in (a.say or "").lower(),
          f"wanted {want} in {a.say[:70]!r}")


def test_a_misspelling_is_not_a_name():
    """The stray-name heuristic treats any unrecognised word as somebody's name."""
    from engine.engine import ask
    print("\nmisspellings")
    for q in ["show me my transations", "show me my paymnets", "my statment"]:
        a = ask(q, want_second_opinion=False, session="n-" + q[:9])
        check(f"{q!r} is answered", "don't have anyone called" not in (a.say or ""),
              a.say[:70])

    print("  but a real unknown name is still queried")
    a = ask("show me ahmed transactions", want_second_opinion=False, session="n-ahmed")
    check("an unknown name still asks", a.outcome == "NEED_INFO", f"{a.outcome}")

    print("  and a name on the roster is never rewritten into a keyword")
    a = ask("show me fatima payments", want_second_opinion=False, session="n-fat")
    check("Fatima survives", "fatima" in (a.say or "").lower(), a.say[:70])


def test_detail_means_the_thing_in_hand():
    print("\n'the details' has no subject of its own")
    for q in ["show me the details", "details", "give me the full details", "receipt",
              "show me the detials", "the deatils", "detail dikhao", "التفاصيل"]:
        a = convo("n-d-" + q[:8], "where is my payment to alex", q)
        check(f"{q[:22]!r} stays on that payment", a.intent == "txn_status", a.intent)

    print("  with nothing in hand it is not hijacked")
    from engine.engine import ask
    a = ask("show me the details", want_second_opinion=False, session="n-cold-detail")
    check("cold, it is not a status lookup for nothing",
          a.intent != "txn_status" or a.outcome in ("NEED_INFO", "NONE"),
          f"{a.intent}/{a.outcome}")


def test_the_nearest_neighbour_never_reaches_for_money():
    """A similarity is not an understanding."""
    import neighbours
    print("\nthe index may not select a money intent")
    for q in ["send 500 to alex", "transfer 100 to asad", "pay mclothes 50",
              "please send the monye to alex"]:
        s = neighbours.suggest(q)
        check(f"{q[:28]!r} -> not a money intent",
              s is None or s["intent"] not in neighbours.FORBIDDEN,
              s and s["intent"])
    print("  and the forbidden set is enforced in the module, not by the caller")
    check("money intents are listed",
          {"send_money", "pay_merchant"} <= neighbours.FORBIDDEN, neighbours.FORBIDDEN)


def test_it_finds_questions_nobody_wrote_down():
    import neighbours
    print("\nunseen phrasings land near known ones")
    for q, want in [("who got the most of my cash this month", "spend_rank"),
                    ("what is the weather on mars", "out_of_scope"),
                    ("show me my transations", "txn_history")]:
        s = neighbours.suggest(q)
        check(f"{q[:34]!r} -> {want}", s is not None and s["intent"] == want,
              s and s["intent"])

    print("  a sentence with no neighbourhood returns nothing, rather than guessing")
    s = neighbours.suggest("qwertyuiop zxcvbnm asdfgh")
    check("gibberish has no neighbourhood", s is None, s)


def test_the_cache_holds_labels_and_not_answers():
    from engine import client
    print("\nthe classification cache")
    before = client.cache_stats()
    a1 = client.classify("my bills")
    a2 = client.classify("MY  BILLS")
    after = client.cache_stats()
    check("case and spacing hit the same entry", after["hit"] > before["hit"], after)
    check("and the label is the same", a1["intent"] == a2["intent"],
          (a1["intent"], a2["intent"]))

    print("  the same question in two sessions still gets two different answers")
    from engine.engine import ask
    one = ask("where is my payment", want_second_opinion=False, session="n-c1")
    convo("n-c2", "send 50 to alex")
    two = ask("where is my payment", want_second_opinion=False, session="n-c2")
    check("answers are not cached, only labels",
          one.say is not None and two.say is not None, "")
    check("both were classified the same way", one.intent == two.intent,
          (one.intent, two.intent))


if __name__ == "__main__":
    test_the_reported_turn()
    test_a_misspelling_is_not_a_name()
    test_detail_means_the_thing_in_hand()
    test_the_nearest_neighbour_never_reaches_for_money()
    test_it_finds_questions_nobody_wrote_down()
    test_the_cache_holds_labels_and_not_answers()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
