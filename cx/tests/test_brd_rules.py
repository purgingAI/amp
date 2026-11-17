#!/usr/bin/env python3
"""The engine rules that were overriding v16 on the BRD prompts, and the three shared leaks.

Fixed 2026-10-14 after v16 went live: the rules were written when the model had no
spend_analysis / spend_rank / cancel_action labels, and several were broader than the
sentences they were written for.

Run: ~/amp/.venv/bin/python cx/tests/test_brd_rules.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import engine as E  # noqa: E402

FAILED = []


def check(text, want, *, not_model=None):
    a = E.ask(text, want_second_opinion=False, session="t-" + uuid.uuid4().hex[:10])
    ok = a.intent == want and (not_model is None or a.model != not_model)
    print(("  ok    " if ok else "  FAIL  ") + f"{text!r} -> {want}" + ("" if ok else f"    got {a.intent} [{a.model}]"))
    if not ok:
        FAILED.append(text)


# the 7 BRD prompts
check("No, cancel", "cancel_action")
check("Pay my gym membership", "bill_pay")
check("How much did I spend?", "spend_analysis")
check("How much did I spend on food last month?", "spend_analysis")
check("Has Ahmed paid me back?", "split_status")
check("How are my spending categories decided?", "knowledge_query")
check("هل وصلت حوالتي إلى أحمد؟", "txn_status")
# the 3 leaks
check("translate good morning into urdu", "out_of_scope")
check("what are your office hours", "out_of_scope")
check("what is the average water bill in sharjah", "out_of_scope")
# ...and what must NOT change
check("how much did I spend at MClothes", "spend_rank")
check("who did I pay the most this month", "spend_rank")
check("how many payments did I make last week", "spend_rank")
check("what is my average bill", "spend_rank")
check("alex has paid me", "payment_claim")
check("pay my rent", "bill_pay")
check("pay", "greeting_help")
check("link my bank account", "link_bank")
check("send an email with my last 5 transactions", "txn_history")

print("all passed" if not FAILED else f"{len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
