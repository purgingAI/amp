#!/usr/bin/env python3
"""Emailing recent transactions: offered by the engine, sent by the server only after PIN.

The engine part is tested here (no email is sent). The server holds the send behind the
PIN unconditionally, and drops client-supplied `_pin_ok` — see openwallet/server.py cx_act.

Run: ~/amp/.venv/bin/python cx/tests/test_email_statement.py
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


def offer(a):
    return (a.proposal or {}).get("kind") == "email_statement"


for text, n in [("send my last 5 transactions to my email", 5), ("sned my last 5 transaction to email", 5),
                ("email me my last ten payments", 10), ("mere last 3 transactions mail kar do", 3),
                ("أرسل آخر 10 معاملات إلى بريدي", 10), ("mail my statement", 5),
                ("मेरे लेनदेन ईमेल करो", 5)]:
    a = run(text)[0]
    check(f"{text!r} -> offer {n}", offer(a) and a.proposal.get("count") == n and a.outcome == "PROPOSED",
          (a.intent, a.outcome, a.proposal))

a = run("send my last 5 transactions to my email")[0]
check("the offer names a MASKED address, never the full one", "*" in a.proposal.get("to", "")
      and a.proposal["to"] in a.say, a.say)

r = run("send my last 5 transactions to my email", "yes")
check("'yes' confirms it (the app then asks for the PIN)", r[1].confirmed and offer(r[1]), (r[1].outcome,))

r = run("send my last 5 transactions to my email", "no")
check("'no' drops it", r[1].outcome == "CANCELLED", (r[1].outcome,))

for text in ["how do I email my statement", "send money to alex", "show my last 5 transactions",
             "send 50 to alex"]:
    a = run(text)[0]
    check(f"{text!r} is NOT an email offer", not offer(a), (a.intent, a.proposal))

print("all passed" if not FAILED else f"{len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
