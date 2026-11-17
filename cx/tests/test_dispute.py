#!/usr/bin/env python3
"""Being told a figure is wrong.

From a real session, and the assistant had nothing:

    "my last week spends"    -> AED 4,549.00
    "and last month?"        -> AED 4,549.00
    "how can it be the same" -> a quote about UAE transfer times

The customer was asking a fair question. As it happens the number was right — every
merchant payment in the last 30 days fell in the last 6, so both windows hold the same 36
payments — but the assistant could neither say that nor find out. It had no way to be
challenged, so it could not be right for a reason and it could not admit being wrong.

Run: ~/amp/.venv/bin/python cx/tests/test_dispute.py
"""
import os
import re
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


def test_the_reported_conversation():
    print("\nthe reported conversation")
    a = convo("d-report", "my last week spends", "and last month?", "how can it be the same?")
    check("it is read as a dispute", a.intent == "disputed", a.intent)
    low = (a.say or "").lower()
    # Assert the SHAPE, not a count. The first version looked for "same 36 payments" and
    # broke the moment a new payment landed in the ledger — a test of yesterday's data
    # reported as a failure of behaviour that had not changed.
    check("it does not merely repeat the figure",
          bool(re.search(r"same \d+ payments", low)), a.say[:120])
    check("and it says why the windows match", "doesn't change" in low or "don't change" in low,
          a.say[:160])


def test_it_admits_being_wrong():
    """The branch that matters. A recheck that can only ever confirm is not a recheck."""
    from engine.engine import ask
    from engine import conversation
    print("\nwhen the figure really was wrong")
    S = "d-wrong"
    ask("my last month spends", want_second_opinion=False, session=S)
    claim = conversation.last_claim(S)
    real = claim["total"]
    claim["total"] = real / 2.0                       # as a miscalculation would have left it
    a = ask("that cannot be right", want_second_opinion=False, session=S)
    check("outcome is a correction", a.outcome == "CORRECTED", a.outcome)
    check("it says it was wrong", "i was wrong" in (a.say or "").lower(), a.say[:90])
    check("and gives the real figure", f"{real:,.2f}" in (a.say or ""), a.say[:90])
    check("the withdrawn figure is not left standing",
          conversation.last_claim(S)["total"] == real, conversation.last_claim(S)["total"])

    # challenged a second time, the corrected figure is what gets defended
    b = ask("are you sure?", want_second_opinion=False, session=S)
    check("a second challenge stands on the corrected number",
          b.outcome == "ANSWERED" and f"{real:,.2f}" in (b.say or ""), b.say[:90])


def test_a_recheck_recomputes():
    """It must re-read the ledger, not re-read its own cache.

    This is the whole integrity of the feature. Agreeing with yourself out of the cache that
    produced the disputed answer confirms a wrong number as readily as a right one — and
    this codebase has already shipped exactly that bug once, when a warm cache returned
    fewer rows than asked for and a breakdown reported AED 2,159 instead of AED 4,469.
    """
    import history
    import recheck
    import spend
    print("\na recheck goes back to the ledger")
    seen = {"fresh": 0, "cached": 0}
    real = history.recent

    def spy(limit=8, fresh=False):
        seen["fresh" if fresh else "cached"] += 1
        return real(limit=limit, fresh=fresh)

    history.recent = spy
    try:
        claim = recheck.claim_spend(spend.analyse("month"), "spend")
        seen["fresh"] = seen["cached"] = 0
        recheck.recheck(claim, [])
    finally:
        history.recent = real
    check("it asked for fresh rows", seen["fresh"] >= 1, seen)
    check("and did not settle for the cache", seen["cached"] == 0, seen)


def test_every_reason_comes_from_the_data():
    """No reason may be produced that the recheck did not find."""
    import recheck
    import spend
    print("\nreasons are evidence, not reassurance")
    now = spend.analyse("month", fresh=True)
    claim = recheck.claim_spend(now, "spend")
    v = recheck.recheck(claim, [])
    check("a verdict is one of three", v["verdict"] in ("stands", "corrected", "cannot_check"),
          v["verdict"])
    kinds = {w["k"] for w in v["why"]}
    check("every reason is a known kind",
          kinds <= {"same_payments", "quiet_stretch", "window", "count", "added", "gone",
                    "not_live"}, kinds)
    for w in v["why"]:
        if w["k"] == "quiet_stretch":
            check("the quiet stretch is real",
                  w["gap_days"] > w["oldest_days"], w)
        if w["k"] == "window":
            check("the window count matches the recomputation",
                  w["n"] == now["payments"], (w, now["payments"]))


def test_nothing_to_check_says_so():
    print("\nnothing recorded to check")
    import recheck
    v = recheck.recheck(None, [])
    check("a missing claim cannot be checked", v["verdict"] == "cannot_check", v["verdict"])
    # and the rule does not fire at all with no claim standing, so an unrelated sentence
    # is never turned into a recheck of nothing
    from engine.engine import _is_dispute
    check("the rule needs a standing claim",
          _is_dispute("that cannot be right", "d-empty-session") is False)


def test_it_answers_in_their_language():
    print("\nthe challenge is answered in the language it was made in")
    for lang, spend_q, dispute in [
        ("hinglish", "mere last month ke kharche", "ye kaise same ho sakta hai"),
        ("hi", "पिछले महीने का खर्चा", "ये कैसे हो सकता है same"),
        ("ar", "مصروفاتي الشهر الماضي", "كيف يكون نفس الرقم"),
    ]:
        a = convo("d-" + lang, "my last week spends", spend_q, dispute)
        check(f"{lang}: read as a dispute", a.intent == "disputed", a.intent)
        check(f"{lang}: answered in {lang}", a.lang == lang, a.lang)
        # the period names are part of the sentence and must be translated with it
        check(f"{lang}: no English period label leaks",
              "the last 7 days" not in (a.say or ""), a.say[:120])


def test_a_question_about_the_figure_is_not_a_dispute():
    """"is it merchant related" asks what the number covered. It is not a challenge.

    Reported: "how much did I pay to merchants last month" → "36 payments in the last 30
    days, AED 4,549.00" (right rows, but never saying merchants is what it counted), then
    "is it merchant related" → "I don't have a figure from this conversation to check" —
    untrue twice, since there was a figure and we knew exactly what went into it.
    """
    print("\nasking what the last figure covered")
    a = convo("ab-1", "how much did i pay to merchants last month")
    check("the tally says what it counted", "at merchants" in (a.say or ""), a.say[:80])

    b = convo("ab-1", "is it merchant related")
    check("the follow-up is answered, not deflected",
          "don't have a figure" not in (b.say or ""), b.say[:80])
    check("and it confirms the scope", "merchants" in (b.say or "").lower(), b.say[:90])

    # the opposite case: a tally over everything must not claim to be merchants only
    c = convo("ab-2", "how many payments did i make this month", "is that everything")
    check("an unscoped tally says so", "everything" in (c.say or "").lower(), c.say[:90])

    print("  with nothing answered yet it says so rather than inventing a scope")
    from engine.engine import ask
    d = ask("is it merchant related", want_second_opinion=False, session="ab-cold")
    # Either honest answer is fine. v9 can predict `disputed` directly now, so cold the
    # sentence reaches the dispute handler and gets "I don't have a figure from this
    # conversation to check" — which is true and is the right thing to say. What must NOT
    # happen is that it describes the scope of a figure nobody gave.
    check("cold, it does not describe a figure that does not exist",
          "payments at merchants" not in (d.say or "")
          and "that was everything" not in (d.say or "").lower(),
          f"{d.intent}: {d.say[:70]}")


def test_a_ranking_can_be_disputed_too():
    """A dispute must re-run the computation that produced the figure, not a cousin of it."""
    import recheck
    from engine import conversation
    from engine.engine import ask
    print("\na ranking is rechecked as a ranking")
    S = "ab-rank"
    ask("who did i pay the most this month", want_second_opinion=False, session=S)
    claim = conversation.last_claim(S)
    check("the ranking was recorded", claim and claim.get("kind") == "rank",
          claim and claim.get("kind"))
    v = recheck.recheck(claim, [])
    check("and it can be rechecked", v["verdict"] in ("stands", "corrected"), v["verdict"])

    claim["total"] = float(claim["total"] or 0) / 2.0
    v2 = recheck.recheck(claim, [])
    check("a wrong ranking total is corrected", v2["verdict"] == "corrected", v2["verdict"])


def test_an_incomplete_instruction_asks():
    """A bare "pay" is a request we handle, missing its object."""
    from engine.engine import ask
    print("\nan incomplete payment instruction")
    for q in ("pay", "pay something"):
        a = ask(q, want_second_opinion=False, session="ab-" + q[:6])
        check(f"{q!r} asks rather than refuses",
              a.outcome == "NEED_INFO" and a.intent != "out_of_scope",
              f"{a.intent}/{a.outcome}")
    print("  and a genuinely out-of-scope request is still declined")
    for q in ("what is the weather", "how many users do you have"):
        a = ask(q, want_second_opinion=False, session="ab-" + q[:7])
        check(f"{q[:24]!r} declined", a.intent == "out_of_scope", a.intent)


def test_a_biller_failure_is_readable():
    """The customer gets a reason, not an exception."""
    from engine import guard
    print("\na biller that cannot be reached")
    bad = ("I couldn't check Home electricity — URLError: <urlopen error "
           "[Errno 111] Connection refused>")
    try:
        guard.scrub({"say": bad})
        check("the old raw-exception message is caught by the sweep", False, "not caught")
    except Exception:
        check("the old raw-exception message is caught by the sweep", True)
    guard.scrub({"say": "I couldn't check Home electricity — the biller could not be reached."})
    check("the human version passes the sweep", True)


if __name__ == "__main__":
    test_the_reported_conversation()
    test_it_admits_being_wrong()
    test_a_recheck_recomputes()
    test_every_reason_comes_from_the_data()
    test_nothing_to_check_says_so()
    test_it_answers_in_their_language()
    test_a_question_about_the_figure_is_not_a_dispute()
    test_a_ranking_can_be_disputed_too()
    test_an_incomplete_instruction_asks()
    test_a_biller_failure_is_readable()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
