#!/usr/bin/env python3
"""The spelling pre-filter: garbled payment words are fixed; everything else is left alone."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine import prefilter  # noqa: E402

FIXED = [
    ("senmoney", "send money"),
    ("sen money", "send money"),
    ("Sen money to Alex.", "send money to Alex."),
    ("senmoneysenmony", "send money"),
    ("sendmony to mona", "send money to mona"),
    ("sen 200 dirhms to alex", "send 200 dirhams to alex"),
    ("send100aed", "send 100 aed"),
    ("pls pey my dewa bil", "pls pay my dewa bill"),
    ("trasnfer 50.5 to sara", "transfer 50.5 to sara"),
    ("chek my balence?", "check my balance?"),
    ("i want to recharg my phone", "i want to recharge my phone"),
    ("widthdraw 100", "withdraw 100"),
    ("(paymybill)", "(pay my bill)"),
    ("Send money. Send", "Send money."),
    ("Send money, send money, send money", "Send money"),
]
UNTOUCHED = [
    "send money to my son", "end of month spend", "sent 1,000 to ali yesterday, did it reach?",
    "remind me on the 25th", "alex ko 50 bhej do", "mera balance kitna hai", "you are a f*cking idiot",
    "email me at a.b@x.com", "Send 50 to Alex. Send 50 to Sara", "Yes. Yes", "no cancel",
    "أرسل 300 درهم", "मेरा बैलेंस", "100", "what is my balance",
]

bad = 0
for text, want in FIXED:
    got = prefilter.correct(text)
    ok = got == want
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {text!r} -> {got!r}" + ("" if ok else f"   wanted {want!r}"))
for text in UNTOUCHED:
    got = prefilter.correct(text)
    ok = got == text
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {text!r} left alone" + ("" if ok else f"   got {got!r}"))
print("all passed" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
