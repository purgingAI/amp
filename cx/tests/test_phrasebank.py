#!/usr/bin/env python3
"""The phrase bank may vary HOW something is said, never WHAT is said.

Renders every combination (opener × core × closer) of every kind in every language and fails
on: a missing kind, wrong list sizes, a core missing a required slot or using a slot the kind
does not have, a slot in an opener/closer, ANY digit of its own (Latin, Arabic-Indic or
Devanagari), wrong script, a request for a PIN/OTP/password, fewer than 2 passive cores, or a
session hearing the same sentence twice inside the repeat window.

Run: ~/amp/.venv/bin/python cx/tests/test_phrasebank.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phrasebank as PB  # noqa: E402

SCHEMA = open(os.path.join(PB.HERE, "SCHEMA.md"), encoding="utf-8").read()
KINDS = dict(re.findall(r"^\| (\w+) \| [^|]+\| ([^|]*)\|$", SCHEMA, re.M))
KINDS = {k: set(re.findall(r"\{(\w+)\}", v)) for k, v in KINDS.items() if k != "kind"}
SAMPLE = {"payee": "Alex Morgan", "amount": "AED 50.00", "merchant": "MClothes", "when": "6:23 PM",
          "name": "Riley", "balance": "AED 1,234.00", "ref": "H-071361", "what": "Pay DEWA bill",
          "dest": "Pakistan", "rate": "74.50 PKR", "fee": "AED 6.80", "topic": "flights"}
DIGIT = re.compile(r"[0-9٠-٩۰-۹०-९]")
SCRIPT = {"hi": re.compile(r"[ऀ-ॿ]"), "ar": re.compile(r"[؀-ۿ]")}
CREDENTIAL_ASK = re.compile(r"\b(?:enter|type|share|send|tell|give)\b[^.?!]{0,25}\b(?:pin|otp|password)\b", re.I)
PASSIVE = {
    "en": re.compile(r"\b(?:will be|is being|has been|have been|was|were|is|are|been|be)\s+\w+(?:ed|en|t)\b", re.I),
    "hinglish": re.compile(r"\b(?:ho jayega|ho jayegi|ho gaya|ho gayi|kiya jayega|kiya gaya|bheja jayega|bheja gaya|"
                           r"di jayegi|diya gaya|kar diya gaya|jaayega|jayega|jayegi|gaya|gayi|hai)\b", re.I),
}
FAILED = []


def check(name, ok, got=""):
    if not ok:
        print(f"  FAIL  {name}    {str(got)[:160]}")
        FAILED.append(name)


for lang in PB.LANGS:
    b = PB.bank(lang)
    check(f"{lang}: bank file loads", bool(b), lang)
    if not b:
        continue
    for kind, req in KINDS.items():
        k = b.get(kind)
        check(f"{lang}/{kind}: present", k is not None)
        if not k:
            continue
        op, co, cl = k.get("openers", []), k.get("cores", []), k.get("closers", [])
        check(f"{lang}/{kind}: 7/6/5", (len(op), len(co), len(cl)) == (7, 6, 5), (len(op), len(co), len(cl)))
        natural = sum(PB._flows(o, c, l) for o in op for c in co for l in cl)
        check(f"{lang}/{kind}: >= 100 natural wordings after the echo filter", natural >= 100, natural)
        for c in co:
            got = re.findall(r"\{(\w+)\}", c)
            check(f"{lang}/{kind}: core has each required slot once", all(got.count(s) == 1 for s in req), c)
            check(f"{lang}/{kind}: core uses only its slots", set(got) <= req, c)
        for p in op + cl:
            check(f"{lang}/{kind}: no slot in opener/closer", "{" not in p, p)
        for p in op + co + cl:
            check(f"{lang}/{kind}: no digits of its own", not DIGIT.search(p.replace("OpenWallet", "")), p)
            check(f"{lang}/{kind}: never asks for a credential", not CREDENTIAL_ASK.search(p) or kind == "security_never_share", p)
            if lang in SCRIPT and p.strip():
                check(f"{lang}/{kind}: right script", bool(SCRIPT[lang].search(re.sub(r"\{\w+\}", "", p))), p)
        if lang in PASSIVE:
            check(f"{lang}/{kind}: >= 2 passive cores", sum(bool(PASSIVE[lang].search(c)) for c in co) >= 2 or not req,
                  co)
        # all 100 render, and every rendered sentence carries the slot values
        for i in range(len(op)):
            for j in range(len(co)):
                for m in range(len(cl)):
                    s = PB._join(op[i], co[j].format(**SAMPLE), cl[m])
                    check(f"{lang}/{kind}: renders with every fact", all(SAMPLE[x] in s for x in req), s)
    # no repeats inside the window
    for kind in KINDS:
        if kind not in b:
            continue
        seen = [PB.say(kind, lang, "t-repeat", **SAMPLE) for _ in range(PB.REPEAT_WINDOW)]
        check(f"{lang}/{kind}: no repeat inside {PB.REPEAT_WINDOW}", len(set(seen)) == len(seen), len(set(seen)))

n = len(set(FAILED))
print("all passed" if not FAILED else f"{n} distinct checks failed ({len(FAILED)} failures)")
sys.exit(1 if FAILED else 0)
