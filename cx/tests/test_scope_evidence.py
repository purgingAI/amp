#!/usr/bin/env python3
"""Small talk and fragments are not requests; a data answer needs a reason in the sentence.

Reported 2026-10-14 from voice: "and how are you" -> bill_due (90%, the customer's bills were
shown), "and" -> txn_history (85%, their payments were listed). The model was confident and
wrong; the engine now requires the SENTENCE (not carried memory) to say something about
payments, bills or money before answering from the customer's data.

Run: ~/amp/.venv/bin/python cx/tests/test_scope_evidence.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILED = []


def check(name, ok, got=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f"    got: {got}"))
    if not ok:
        FAILED.append(name)


from engine import engine as E  # noqa: E402


def ask(t, s):
    return E.ask(t, session=s, want_second_opinion=False)


print("\nsmall talk is answered kindly and declined — nothing looked up")
for t in ["How are you, sab badia", "and how are you", "kaise ho bhai", "كيف حالك", "good morning", "what's up"]:
    a = ask(t, "se-" + t)
    check(f"{t!r}", a.model == "small-talk" and not a.rows and a.intent == "out_of_scope", (a.intent, a.model))

print("\nrandom questions are out of scope")
for t in ["what is the capital of france", "tell me a joke", "who won the match yesterday", "what happened yesterday"]:
    a = ask(t, "se-" + t)
    check(f"{t!r}", a.intent == "out_of_scope" and not a.rows, (a.intent, a.outcome))

print("\na fragment is asked about, never answered from data")
a = ask("and", "se-frag")
check("'and' alone", a.intent == "greeting_help" and not a.rows, (a.intent, a.outcome))
S = "se-frag2"
ask("show my last 5 transactions", S); ask("is it done?", S)
a = ask("and", S)
check("'and' after a thread — carried memory is not evidence", not a.rows and a.intent != "txn_history", (a.intent, a.outcome))

print("\nreal questions and real follow-ups still answer")
for t, want in [("show yesterday's payments", "txn_history"), ("what did i spend this week", "spend_analysis"),
                ("who is on my list", "beneficiary_query"), ("ما هي فواتيري", "bill_due")]:
    a = ask(t, "se-" + t)
    check(f"{t!r} -> {want}", a.intent == want, a.intent)
S = "se-follow"
ask("show my last 5 transactions", S)
a = ask("and before that", S); check("history -> 'and before that'", a.intent == "txn_history", a.intent)
S = "se-follow2"
ask("show my last 5 transactions", S)
a = ask("and last month?", S); check("history -> 'and last month?'", a.intent == "txn_history", a.intent)

print()
print(f"{len(FAILED)} failed" if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
