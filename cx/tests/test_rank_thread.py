#!/usr/bin/env python3
"""A follow-up that only changes the window.

Reported from the model-brief demo:

    "who did i pay the most this month" -> Alex, AED 3,476 over 59 payments
    "and last month?"                   -> "Your last 5 payments last month: AED 495"

Two bugs, stacked:

  1. The follow-up has no "who" and no "most", so the rank rule could not see it; the model
     read it as transaction history (92%) and listed payments. Spend breakdowns had a thread
     rule for exactly this ("and last week?" stays a breakdown); rankings never got one.
  2. Underneath, "this month" and "last month" were the SAME window — both the last 30 days —
     so even routed correctly the answer would have repeated itself. A follow-up naming the
     same span with a "previous" word now means the window before it. On its own, "my
     spending last month" still means the last 30 days, which is what test_spend holds it to.

Run: ~/amp/.venv/bin/python cx/tests/test_rank_thread.py
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
        print(f"  FAIL  {name}    got: {got}")


import spend  # noqa: E402

print("\nwhich window a follow-up means")
for prev, said, want in [
    ("who did i pay the most this month", "and last month?", "prev_month"),
    ("who did i pay the most this month", "and this month?", None),
    ("who did i pay the most this month", "and last week?", None),        # a different span
    ("how many payments this week", "and last week?", "prev_week"),
    ("sabse zyada kisko diye is mahine", "pichle mahine?", "prev_month"),
    ("सबसे ज़्यादा किसको दिया इस महीने", "और पिछले महीने?", "prev_month"),
    ("من دفعت له أكثر هذا الشهر", "والشهر الماضي؟", "prev_month"),
    ("my spending this month", "what about last month", "prev_month"),
]:
    got = spend.follow_up_period(prev, said)
    check(f"{prev[:28]!r} -> {said!r} means {want}", got == want, got)
check("on its own, 'last month' is still the last 30 days", spend.period_from("my spending last month") == "month")
label, off, span = spend.window("prev_month")
check("the window before a month is 30–60 days ago", (off, span) == (720, 720), (off, span))

from engine import engine as E  # noqa: E402

print("\nthrough the engine")


def convo(session, *turns):
    return [E.ask(t, session=session) for t in turns]


a1, a2, a3 = convo("rt-1", "who did i pay the most this month", "and last month?", "and this month?")
check("the follow-up stays a ranking", a2.intent == "spend_rank" and a2.model == "rank-thread",
      (a2.intent, a2.model))
check("…about the window before, and says so", "before that" in a2.say, a2.say[:90])
check("…and does not repeat the first answer", a2.say != a1.say, a2.say[:60])
check("'and this month?' goes back to the current window", a3.say == a1.say, a3.say[:60])

b1, b2 = convo("rt-2", "how many payments did i make this week", "and last week?")
check("a tally follows the window too", b2.intent == "spend_rank" and "7–14 days ago" in b2.say, b2.say[:80])

c1, c2 = convo("rt-3", "my spending this month", "and last month?")
check("a spending breakdown follows the window too", c2.intent == "spend_analysis" and "before that" in c2.say,
      (c2.intent, c2.say[:80]))

d1, d2 = convo("rt-4", "who did i pay the most this month", "show me last month's payments")
check("asking for a LIST after a ranking is still a list", d2.intent == "txn_history", d2.intent)
e1, e2 = convo("rt-5", "who did i pay the most this month", "send 50 to alex")
check("a new task after a ranking is a new task", e2.intent == "send_money", e2.intent)
f1, f2 = convo("rt-6", "show my last 5 transactions", "and last month?")
check("after a list, a window change is still a list", f2.intent == "txn_history", f2.intent)

print()
print(f"{len(FAILED)} failed" if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
