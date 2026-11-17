#!/usr/bin/env python3
"""Which of the things we remember wins, when more than one could answer the sentence.

Every case here is a real wrong answer, and each one came from a different memory being
consulted in the wrong order. The pattern they share: a sentence with no subject of its own
("yes", "split it", "and last week?", "is that more than before?") found a subject, and it
was the wrong one.

Run: ~/amp/.venv/bin/python cx/tests/test_context_precedence.py
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


def test_a_named_confirmation_only_confirms_its_own_action():
    """"split it" was in the same list as "yes", so it confirmed whatever was on offer.

    With a send proposal standing that meant a sentence asking to SPLIT a bill confirmed a
    payment to a person. The milder form is the one that was reported, but this is the one
    that matters: a confirmation that names an action may only ever confirm that action.
    """
    print("\na confirmation that names an action")
    a = convo("p-split-send", "send 50 to alex", "split it")
    check("'split it' does not confirm a send", not a.confirmed, f"confirmed={a.confirmed}")
    check("it is read as a split instead", a.intent == "split_payment", a.intent)

    a = convo("p-send-send", "send 50 to alex", "send it")
    check("'send it' does confirm a send", a.confirmed is True, f"confirmed={a.confirmed}")

    a = convo("p-bare-yes", "send 50 to alex", "yes")
    check("a bare yes still confirms anything", a.confirmed is True, f"confirmed={a.confirmed}")

    # Names the payment: a bare "split it with alex" in a fresh conversation now asks WHICH
    # payment rather than reaching for the latest one in history (2026-10-14).
    a = convo("p-split-split", "split my last payment with alex", "split it")
    check("'split it' confirms a split", a.confirmed is True, f"confirmed={a.confirmed}")


def test_an_offer_lapses_when_the_subject_changes():
    """The reported bug, end to end.

        send 50 to alex / split it with alex / bill pay karna hai / Home electricity
        "split it"  ->  "already split the last one"

    The split offer from turn two was never answered and never dropped, because every turn
    after it was NEED_INFO and left it standing. Four turns later it was still confirmable.
    """
    print("\nan unanswered offer does not outlive its subject")
    a = convo("p-reported", "send 50 to alex", "split it with alex",
              "bill pay karna hai", "Home electricity", "split it")
    check("not treated as a confirmation", not a.confirmed, f"confirmed={a.confirmed}")
    check("it is about the bill on screen", "electricity" in a.say.lower(), a.say[:90])
    check("and it asks who to split with", a.outcome == "NEED_INFO", a.outcome)

    # The lapse rule must not reach back and void an offer that is still the subject.
    a = convo("p-fresh", "split my last payment with alex", "yes")
    check("an offer answered immediately still stands", a.confirmed is True,
          f"confirmed={a.confirmed}")


def test_the_live_topic_beats_an_older_open_task():
    """"is that more than before?" mid-breakdown listed payments to Alex.

    The task ledger's anaphora fallback attached "that" to the one still-open payment and
    carried `payee=Alex` forward, which made the spend-stickiness rule stand down — a
    sentence naming a payee is a new task. Mid-thread, "that" means what we were just
    talking about.
    """
    print("\nthe live topic is the better referent")
    a = convo("p-older", "send 50 to alex", "yes", "analyse my spending last month",
              "is that more than before?")
    check("stays with the breakdown", a.intent == "spend_analysis", a.intent)

    # ...but the ledger still works when the live topic is not the subject.
    a = convo("p-ledger", "send 50 to alex", "is that done?")
    check("a status question still finds the payment", a.intent == "txn_status", a.intent)


def test_a_bare_change_of_window_stays_with_the_breakdown():
    """The worst-failing family in the corpus: 5 of 37 scenarios clean.

    "and last week?" classifies as transaction history, which is a fair reading of the
    sentence alone and the wrong one mid-thread — the customer gets five payments where they
    asked for the same categories over a shorter window.
    """
    print("\na bare change of window")
    for q in ["and last week?", "pichle hafte?", "last 7 days?", "this quarter?",
              "और पिछले हफ्ते?"]:
        a = convo("p-win-" + q[:7], "analyse my spending last month", q)
        check(f"{q!r} stays with spending", a.intent == "spend_analysis", a.intent)

    print("  a window named inside a real request is still that request")
    for q, want in [("show me last week's payments", "txn_history"),
                    ("send 100 to alex", "send_money")]:
        a = convo("p-real-" + q[:7], "analyse my spending last month", q)
        check(f"{q!r} -> {want}", a.intent == want, a.intent)


def test_a_follow_up_with_nothing_to_follow_asks():
    """Said cold, these mean nothing — so the honest answer is to ask.

    Left alone the classifier obliged: "is that more than before?" was read as transaction
    history and the handler went looking for a payee in it, answering "I don't have anyone
    called Than on your list".
    """
    print("\na follow-up with no thread")
    for q in ["is that more than before?", "and healthcare?", "and last week?",
              "and before that"]:
        a = convo("p-cold-" + q[:8], q)
        check(f"{q!r} asks", a.intent == "greeting_help" and a.outcome == "NEED_INFO",
              f"{a.intent}/{a.outcome}")

    print("  and it answers in the language they opened in")
    a = convo("p-cold-hi", "kya pichle se zyada tha?")
    check("hinglish answer", a.lang == "hinglish" and "compare" in a.say.lower(),
          f"{a.lang}: {a.say[:60]}")

    print("  a sentence that names its own subject is NOT swallowed")
    for q, want in [("my bills", "bill_due"), ("my spends", "spend_analysis"),
                    ("show me my payments", "txn_history"), ("hi", "greeting_help")]:
        a = convo("p-own-" + q[:8], q)
        check(f"{q!r} -> {want}", a.intent == want, a.intent)


def test_a_breakdown_is_only_ever_about_the_asker():
    """The subject check the history lane had, and the breakdown lane did not.

    "what did my wife spend last month" answered with the customer's own categories.
    Nothing leaked — every row is scoped to the authenticated customer — but a breakdown
    shown in answer to a question about somebody else reads as being about them.
    """
    print("\na breakdown is about the asker")
    for q in ["what did my wife spend last month", "show me ahmed's spending",
              "spending for wallet 9876543210"]:
        a = convo("p-scope-" + q[:9], q)
        check(f"refuses {q[:30]!r}", a.outcome == "REFUSED_SCOPE", f"{a.intent}/{a.outcome}")

    print("  the customer's own question is still answered")
    for q in ["how much did i spend on food", "what did i spend last month", "my spending"]:
        a = convo("p-mine-" + q[:9], q)
        check(f"answers {q[:30]!r}", a.outcome == "ANSWERED", f"{a.intent}/{a.outcome}")


if __name__ == "__main__":
    test_a_named_confirmation_only_confirms_its_own_action()
    test_an_offer_lapses_when_the_subject_changes()
    test_the_live_topic_beats_an_older_open_task()
    test_a_bare_change_of_window_stays_with_the_breakdown()
    test_a_follow_up_with_nothing_to_follow_asks()
    test_a_breakdown_is_only_ever_about_the_asker()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
