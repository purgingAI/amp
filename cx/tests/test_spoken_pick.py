#!/usr/bin/env python3
"""Picking from what we showed, by voice — and not losing the thread on a mishearing.

Reported 2026-10-14: "split my last payment to a merchant" found the payment; "Alex" was
heard as "because" and REFUSED (the split was lost); the next answer was a list of payments,
and "first one" just showed the same list again.

Run: ~/amp/.venv/bin/python cx/tests/test_spoken_pick.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import engine as E  # noqa: E402

FAILED = []


def run(*turns):
    s = "t-" + uuid.uuid4().hex[:10]
    return [E.ask(t, session=s) for t in turns]


def check(name, ok, got=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f"    got: {got}"))
    if not ok:
        FAILED.append(name)


def steps(a):
    return [t["what"] for t in a.lane_trace]


r = run("split my last payment to a merchant", "because", "first one")
check("a mishearing mid-split is re-asked, not refused", r[1].intent == "split_payment"
      and "reask" in steps(r[1]) and r[1].outcome == "NEED_INFO", (r[1].intent, r[1].outcome, r[1].say[:60]))
check("... the re-ask keeps the payment and asks WHO again",
      (r[1].slots or {}).get("txn_ref") == (r[0].slots or {}).get("txn_ref") and "split_with" in
      {o.get("kind") for o in r[1].options}, (r[1].slots, r[1].say[:60]))
check("... and 'first one' picks the first person shown", "picked" in steps(r[2])
      and r[2].intent == "split_payment" and r[2].outcome in ("NEED_CONFIRMATION", "PROPOSED"),
      (r[2].intent, r[2].outcome, r[2].say[:60]))

r = run("split my last payment", "alex")
check("the split keeps its payment when the reply is just a name",
      r[1].outcome in ("NEED_CONFIRMATION", "PROPOSED") and "split.which" not in steps(r[1]),
      (r[1].outcome, r[1].say[:60]))

r = run("show my last 5 transactions", "the second one")
check("'the second one' opens the second payment listed", r[1].intent == "txn_status"
      and "picked" in steps(r[1]), (r[1].intent, r[1].say[:60]))

r = run("send money", "the first one")
check("headings are skipped: 'the first one' is the first person", r[1].intent == "send_money"
      and (r[1].slots or {}).get("payee"), (r[1].intent, r[1].slots))

r = run("send money", "pehla wala")
check("Hinglish ordinal", "picked" in steps(r[1]), steps(r[1])[:3])

r = run("show my last 5 transactions", "what's my balance")
check("a question is not a pick", "picked" not in steps(r[1]), steps(r[1])[:3])

r = run("send my last 5 transactions to my email", "first one")
check("no list shown last turn -> nothing to pick", not r[0].options and "picked" not in steps(r[1]),
      (len(r[0].options or []), steps(r[1])[:3]))

print("all passed" if not FAILED else f"{len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
