#!/usr/bin/env python3
"""A new request in the same conversation is a NEW request.

Reported 2026-10-14, one session:
  * "Dil pay karna hai" (a misheard Hinglish "bill pay karna hai") switched the session to
    Hinglish, and every English question after it was answered in Hinglish.
  * After a spending figure, "Send money to a shop" was answered "No — that was everything
    in pichle 30 dino…": the about-the-last-figure rule saw "shop" and treated a payment
    instruction as a question about the figure.
  * With "who are you paying?" open, "what's my balance" was swallowed as the answer
    ("balance" read as pay-from-wallet).

Context must fill GAPS in a follow-up; it must never turn a new instruction or a new
question into one. A real answer to an open question must still be read as the answer.

Run: ~/amp/.venv/bin/python cx/tests/test_new_request.py
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

CONTEXTS = {"a spending breakdown": ["last month spend analysis"],
            "a ranking": ["who did i pay the most this month"],
            "a history list": ["show my last 5 transactions"],
            "a payment status": ["did my last payment go through"],
            "a disputed figure": ["how much did i spend last week", "that can't be right"],
            "a Hinglish turn": ["dil pay karna hai"],
            "an open 'who?'": ["send money"],
            "an open 'which merchant?'": ["pay a merchant"]}
NEW = [("send money to alex", {"send_money"}),
       ("send money to a shop", {"pay_merchant", "send_money"}),
       ("send money to merchant", {"pay_merchant", "send_money"}),
       ("pay my dewa bill", {"bill_pay"}),
       ("send 50 to alex", {"send_money"}),
       ("what's my balance", {"balance_query", "split_status", "journey_help", "out_of_scope", "limits_query"}),
       ("show my last 5 transactions", {"txn_history"})]
CONTEXT_MODELS = {"about-rule", "sticky", "rank-thread", "in-hand", "answered"}

print("\na new request after each kind of context")
n = 0
for ctx, setup in CONTEXTS.items():
    for text, ok in NEW:
        S = f"newreq-{n}"; n += 1
        for t in setup:
            E.ask(t, session=S, want_second_opinion=False)
        a = E.ask(text, session=S, want_second_opinion=False)
        check(f"after {ctx}: {text!r} is new, in English",
              a.intent in ok and a.model not in CONTEXT_MODELS and a.lang == "en",
              (a.intent, a.model, a.lang, (a.say or "")[:60]))

print("\nreal answers to an open question are still answers")
for setup, reply, want in (("send money", "alex", "send_money"),
                           ("send money to alex", "50", "send_money"),
                           ("pay a merchant", "MClothes", "pay_merchant")):
    S = f"answer-{setup}"
    E.ask(setup, session=S, want_second_opinion=False)
    a = E.ask(reply, session=S, want_second_opinion=False)
    check(f"{setup!r} then {reply!r} continues it", a.intent == want, (a.intent, a.outcome))

print("\nlanguage: a bare number keeps the conversation's language; a sentence decides its own")
S = "lang-1"
E.ask("mujhe paise bhejne hai", session=S, want_second_opinion=False)
a = E.ask("100", session=S, want_second_opinion=False)
check("'100' after Hinglish stays Hinglish", a.lang == "hinglish", a.lang)
a = E.ask("show my last 5 transactions", session=S, want_second_opinion=False)
check("an English sentence after Hinglish is answered in English", a.lang == "en", a.lang)

print()
print(f"{len(FAILED)} failed" if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
