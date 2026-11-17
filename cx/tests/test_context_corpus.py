#!/usr/bin/env python3
"""Replay every generated conversation and report where the thread is lost.

Each scenario is a fresh session, so nothing leaks between them. For every turn the engine
is asked the sentence and the answer is checked against three things:

* `expect`      — the intent must be one of these, when the scenario can name them
* `forbid`      — the intent must NOT be one of these. This is the load-bearing half: there
                  is often more than one acceptable reply, but answering about a payment
                  when the customer asked about a breakdown is never one of them
* `say_contains`— the SUBJECT must survive into the wording. Pinning the intent alone would
                  have passed the reported bug: it answered `split_payment`, correctly, about
                  entirely the wrong thing

Run:   ~/amp/.venv/bin/python cx/tests/test_context_corpus.py
       ... --family bill-in-hand      one family only
       ... --verbose                  print every failure
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.context_cases import CASES, summary   # noqa: E402


def run(cases, verbose=False):
    from engine import engine as E
    from engine import conversation

    failures, turns = [], 0
    t0 = time.time()
    for i, case in enumerate(cases):
        session = f"ctx-{case['id']}-{i}"
        conversation.clear(session)
        for t_i, turn in enumerate(case["turns"]):
            turns += 1
            try:
                a = E.ask(turn["say"], session=session)
            except Exception as e:                      # noqa: BLE001
                failures.append((case, t_i, turn, f"raised {type(e).__name__}: {e}", ""))
                break
            got, say = a.intent, (a.say or "")
            if "expect" in turn and got not in turn["expect"]:
                failures.append((case, t_i, turn,
                                 f"intent {got}, wanted one of {turn['expect']}", say))
            elif got in (turn.get("forbid") or ()):
                failures.append((case, t_i, turn, f"drifted to {got}", say))
            elif turn.get("say_contains") and turn["say_contains"].lower() not in say.lower():
                failures.append((case, t_i, turn,
                                 f"answered about something else — {turn['say_contains']!r} "
                                 f"is not in the reply", say))
    return failures, turns, time.time() - t0


def main():
    args = sys.argv[1:]
    cases = CASES
    if "--family" in args:
        fam = args[args.index("--family") + 1]
        cases = [c for c in CASES if c["family"] == fam]
    verbose = "--verbose" in args
    if "--limit" in args:
        cases = cases[:int(args[args.index("--limit") + 1])]

    s = summary()
    print(f"{len(cases)} scenarios of {s['cases']} ({s['turns']} turns in the full set)")
    failures, turns, secs = run(cases, verbose)

    by_family = {}
    for case, *_ in failures:
        by_family[case["family"]] = by_family.get(case["family"], 0) + 1
    total_by_family = {}
    for c in cases:
        total_by_family[c["family"]] = total_by_family.get(c["family"], 0) + 1

    print(f"\n{turns} turns in {secs:.0f}s · {len(failures)} failing\n")
    for fam in sorted(total_by_family, key=lambda f: -by_family.get(f, 0)):
        bad, tot = by_family.get(fam, 0), total_by_family[fam]
        mark = "ok  " if not bad else "FAIL"
        print(f"  {mark} {fam:26} {tot - bad:4d}/{tot:<4d} scenarios clean")

    if failures and verbose:
        print("\n--- failures ---")
        for case, t_i, turn, why, say in failures[:60]:
            print(f"\n  [{case['family']}] {case['id']}")
            print(f"    why: {case['why']}")
            for k, prev in enumerate(case["turns"][:t_i]):
                print(f"      {k+1}. {prev['say']}")
            print(f"      {t_i+1}. {turn['say']}   <-- {why}")
            if say:
                print(f"         said: {say[:100]}")
    elif failures:
        print(f"\n  (run with --verbose to see them; showing 3)")
        for case, t_i, turn, why, say in failures[:3]:
            print(f"    [{case['family']}] turn {t_i+1} {turn['say']!r} -> {why}")

    print()
    if failures:
        print(f"{len(failures)} failing turns out of {turns}")
        sys.exit(1)
    print("all good")


if __name__ == "__main__":
    main()
