#!/usr/bin/env python3
"""Answering "someone you know, or a merchant near you?" is an answer.

Reported 2026-10-14:
  * "send money" -> "Who are you paying — someone you know, or a merchant near you?"
    -> "to a shop" was REFUSED ("I can only help with your OpenWallet payments"): the reply named
    no payee and no merchant, and the no-payee guard never looked at the question we asked.
  * "Send money to a shop" -> "Which merchant?" -> "mcloths" listed MClothes's offer instead
    of paying it: the misspelt name matched nothing, and the model read it as offers_nearby.

Run: ~/amp/.venv/bin/python cx/tests/test_payee_kind.py
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


r = run("send money", "to a shop")
check("'to a shop' after 'who?' opens a merchant payment", r[1].intent == "pay_merchant"
      and r[1].outcome == "NEED_INFO", (r[1].intent, r[1].outcome, r[1].say[:60]))

r = run("send money", "a shop", "mcloths", "100")
check("'mcloths' after 'which merchant?' is MClothes", r[2].intent == "pay_merchant"
      and (r[2].slots or {}).get("merchant") == "MClothes", (r[2].intent, r[2].slots))
check("... and '100' then proposes the payment", r[3].outcome == "PROPOSED", (r[3].intent, r[3].outcome))

r = run("Send money to a shop", "mcloths")
check("'Send money to a shop' then 'mcloths'", r[1].intent == "pay_merchant"
      and (r[1].slots or {}).get("merchant") == "MClothes", (r[1].intent, r[1].slots))

r = run("send money", "a friend")
check("'a friend' narrows to people, not the same question again",
      r[1].intent == "send_money" and r[1].options and all(o.get("kind") == "send_to" for o in r[1].options),
      (r[1].intent, r[1].say[:60], [o.get("kind") for o in r[1].options or []]))

r = run("send money", "Alex")
check("a payee name still answers 'who?'", r[1].intent == "send_money" and (r[1].slots or {}).get("payee"),
      (r[1].intent, r[1].slots))

r = run("send money", "what offers does mclothes have")
check("a real question is still a question", r[1].intent == "offers_nearby", (r[1].intent,))

r = run("send money", "pay my dewa bill")
check("a new instruction is classified, not swallowed", r[1].intent == "bill_pay" and r[1].model != "answered",
      (r[1].intent, r[1].model))

print("all passed" if not FAILED else f"{len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
