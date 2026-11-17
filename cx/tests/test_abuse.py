#!/usr/bin/env python3
"""Abuse, and a database that has gone away.

Reported: "gandu" came back as greeting_help — a cheerful welcome for an insult. The model
has no class for abuse (it was trained on polite customer questions), so it picked the
nearest thing it knew. Abuse is now decided by rule before the model is asked:

  * nothing but abuse          -> a calm boundary, in the customer's language, REFUSED
  * abuse around a real request -> the abusive words removed, the request answered, with
                                   an acknowledgement in front

Also here: the circuit breaker. With the VPN down, every Mongo read waited out its own
timeout and a single turn took about sixty seconds before falling back.

Run: ~/amp/.venv/bin/python cx/tests/test_abuse.py
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, ok, got=""):
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}    got: {got}")


from engine import guard  # noqa: E402

# ── the rule itself ────────────────────────────────────────────────────────────────────
ONLY = ["gandu", "GANDU", "gandu!!", "gaaandu", "g@ndu", "tu gandu hai", "tu chutiya hai",
        "madarch0d", "you are a f*cking idiot", "fuck you", "तू कमीना है", "يا كلب"]
for t in ONLY:
    r = guard.abuse(t)
    check(f"nothing but abuse: {t!r}", r is not None and r[2], r)

MIXED = {"gandu app where is my refund": "where is my refund",
         "पैसा कहाँ है चूतिया": "पैसा कहाँ है",
         "انت غبي، وين فلوسي": "وين فلوسي",
         "book a flight you moron": "book a flight"}
for t, keep in MIXED.items():
    r = guard.abuse(t)
    check(f"abuse around a request keeps the request: {t!r}",
          r is not None and not r[2] and keep in r[1], r)

CLEAN = ["send money as soon as possible", "class of service", "assessment fee",
         "is my passport needed", "what the hell happened to my payment",
         "fees kitni hai bc", "shitake mushroom", "scunthorpe", "pass", "hello",
         "كم رسوم التحويل", "पेमेंट फेल क्यों हुआ", "mujhe paise bhejne hai"]
for t in CLEAN:
    check(f"ordinary speech is not abuse: {t!r}", guard.abuse(t) is None, guard.abuse(t))

# no false positives anywhere in our own corpus — 60,000+ real customer phrasings
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hits, n = [], 0
for f in glob.glob(os.path.join(here, "corpus", "*.jsonl")):
    # the abuse corpus and the training file built from it are SUPPOSED to be flagged
    if os.path.basename(f).startswith(("abuse_", "train_v12", "train_v13")):
        continue
    for line in open(f, encoding="utf-8"):
        try:
            t = json.loads(line).get("text")
        except Exception:                                    # noqa: BLE001
            continue
        if t:
            n += 1
            if guard.abuse(t):
                hits.append(t)
check(f"no corpus sentence is flagged ({n:,} scanned)", not hits, hits[:5])

# ── the engine: abuse alone never reaches the model ────────────────────────────────────
from engine import engine as E  # noqa: E402

for t, lang_word in (("gandu", "izzat"), ("you are a f*cking idiot", "respectful"),
                     ("तू कमीना है", "सम्मान"), ("يا كلب", "باحترام")):
    a = E.ask(t, session=f"abuse-{abs(hash(t))}")
    check(f"engine declines {t!r} by rule, in the right language",
          a.intent == "out_of_scope" and a.outcome == "REFUSED" and a.model == "abuse-rule"
          and lang_word in a.say, (a.intent, a.outcome, a.model, a.say[:60]))
    check(f"engine never welcomes {t!r}", a.intent != "greeting_help", a.intent)

# ── policy: help once, then stop ───────────────────────────────────────────────────────
from engine import conversation  # noqa: E402

S = "abuse-policy-1"
a = E.ask("gandu app where is my refund", session=S)
check("first abuse with a real question is answered", a.intent == "refund_status", a.intent)
check("…with a boundary line in front", a.say.startswith("Please keep it respectful."), a.say[:60])
a = E.ask("what are the fees to send money", session=S)
check("a polite question after that is answered normally", a.intent == "fees_query", a.intent)
a = E.ask("you idiot", session=S)
check("abuse a second time ends the conversation", a.model == "abuse-stop" and "support" in a.say,
      (a.model, a.say[:60]))
a = E.ask("sorry, what are the fees", session=S)
check("…and nothing more is answered, even politely", a.model == "abuse-stop", a.model)
a = E.ask("yes", session=S)
check("…including a bare yes to anything offered earlier", a.model == "abuse-stop", a.model)
a = E.ask("what are the fees to send money", session="abuse-policy-fresh")
check("a different conversation is unaffected", a.intent == "fees_query", a.intent)

S = "abuse-policy-2"
E.ask("send 50 to alex", session=S)
check("a payment is on offer", bool(conversation.proposal(S)))
E.ask("gandu", session=S)
E.ask("chutiya", session=S)
check("stopping drops the offer, so no button can complete it", not conversation.proposal(S))

S = "abuse-policy-3"
E.ask("يا كلب", session=S)
a = E.ask("وين فلوسي يا حمار", session=S)
check("the stop is said in the customer's language", "خدمة عملاء" in a.say, a.say[:60])

# ── the breaker ────────────────────────────────────────────────────────────────────────
import breaker  # noqa: E402
from pymongo.errors import ServerSelectionTimeoutError  # noqa: E402

breaker._down_until = 0.0
breaker.record(ValueError("a bug, not an outage"))
check("a bug does not open the breaker", not breaker.is_open())
breaker.record(ServerSelectionTimeoutError("no servers"))
check("a connection failure opens it", breaker.is_open(), breaker.reason())
import history  # noqa: E402
t0 = time.time()
rows = history.recent(fresh=True)
check("while open, a read fails fast instead of waiting out its timeout",
      rows is None and time.time() - t0 < 0.5, f"{time.time() - t0:.2f}s")
breaker._down_until = 0.0
check("it closes again after its cool-down", not breaker.is_open())

print()
print(f"{len(FAILED)} failed" if FAILED else "all passed")
sys.exit(1 if FAILED else 0)
