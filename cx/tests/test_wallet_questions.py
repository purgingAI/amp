#!/usr/bin/env python3
"""Run every question in `corpus/wallet_questions.py` and report where intent detection fails.

This is a measurement, not a pass/fail gate on every row. Some sentences genuinely admit
more than one reading and the corpus says so. What it DOES gate is the forbidden column:
a question answered by offering to move money, or an instruction filed as a question, is a
failure whatever else is true.

Run:
    ~/amp/.venv/bin/python cx/tests/test_wallet_questions.py
    ... --family ranking
    ... --verbose
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.wallet_questions import FAMILIES, rows      # noqa: E402


def main():
    from engine.engine import ask
    args = sys.argv[1:]
    only = args[args.index("--family") + 1] if "--family" in args else None
    verbose = "--verbose" in args

    miss, banned, total = [], [], 0
    by_family: dict[str, list[int]] = {}

    for i, r in enumerate(rows()):
        if only and r["family"] != only:
            continue
        total += 1
        sid = f"wq-{i}"
        # A challenge needs something to challenge, so the thread is set up first. Asking
        # "are you sure?" cold is a different question and is tested elsewhere.
        if r["family"] == "challenge":
            ask("what did i spend last month", want_second_opinion=False, session=sid)
        a = ask(r["text"], want_second_opinion=False, session=sid)
        got = a.intent
        ok = got in r["expect"]
        bad = got in r["forbid"]
        by_family.setdefault(r["family"], [0, 0])
        by_family[r["family"]][1] += 1
        if ok:
            by_family[r["family"]][0] += 1
        else:
            miss.append((r, got, a.say))
        if bad:
            banned.append((r, got, a.say))

    print(f"\n  {total} questions\n")
    for name, _why, _rs in FAMILIES:
        if name not in by_family:
            continue
        hit, n = by_family[name]
        mark = "ok  " if hit == n else "    "
        print(f"  {mark} {name:16} {hit:3}/{n:<3} as expected")

    print(f"\n  {total - len(miss)}/{total} as expected"
          f"   ·   {len(banned)} forbidden")

    if banned:
        print("\n  FORBIDDEN — these are wrong however you read them:")
        for r, got, say in banned:
            print(f"    [{r['family']}] {r['text'][:46]!r} -> {got}")
            print(f"        {(say or '')[:90]}")
    if miss and verbose:
        print("\n  not as expected (may still be defensible):")
        for r, got, say in miss:
            print(f"    [{r['family']}] {r['text'][:46]!r} -> {got}  (wanted {r['expect']})")
            print(f"        {(say or '')[:90]}")
    elif miss:
        print(f"\n  ({len(miss)} not as expected — run with --verbose)")

    # The gate is the forbidden column only.
    return 1 if banned else 0


if __name__ == "__main__":
    raise SystemExit(main())
