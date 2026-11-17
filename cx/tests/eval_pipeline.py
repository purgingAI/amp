"""End-to-end accuracy of the thing we actually ship: scope rules + model + handlers.

The model's own 86.1% is not the product's number. In the engine the deterministic scope
rules run FIRST and can override the model, so the only measurement that means anything is
of the whole pipeline — and the headline metric is not accuracy, it is **leaks**: questions
we should have declined and answered instead.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
sys.path.insert(0, CX)

from engine.engine import ask

import sys as _s
_FILE = _s.argv[1] if len(_s.argv) > 1 else "hard_eval.jsonl"
ROWS = [json.loads(l) for l in open(os.path.join(CX, "corpus", _FILE)) if l.strip()]

t0 = time.time()
correct = leaks = over = 0
_preds = []
wrong = []
for i, r in enumerate(ROWS):
    # A UNIQUE session per row: these utterances are independent, and sharing one
    # session would let an open question from the previous row continue into this one.
    a = ask(r["text"], want_second_opinion=False, session=f"eval-{i}")
    _preds.append(a.intent)
    got, want = a.intent, r["intent"]
    if got == want:
        correct += 1
    else:
        wrong.append((want, got, r["text"], a.model))
        if want == "out_of_scope":
            leaks += 1
        if got == "out_of_scope":
            over += 1

n = len(ROWS)
ms = (time.time() - t0) / n * 1000
print(f"FULL PIPELINE on {n} hand-written utterances from {_FILE}   ({ms:.1f} ms/question)")
import collections as _c
_by = _c.defaultdict(lambda: [0, 0])
for _r, _a in zip(ROWS, _preds):
    _by[_r.get("lang", "en")][1] += 1
    _by[_r.get("lang", "en")][0] += int(_a == _r["intent"])
print("  by language:   " + "   ".join(f"{k} {v[0]}/{v[1]} ({v[0]/v[1]:.0%})" for k, v in sorted(_by.items())))
print(f"  accuracy       {correct / n:.1%}")
print(f"  LEAKED         {leaks}   (should have declined, answered instead)")
print(f"  over-refused   {over}   (declined something we could have answered)")
if wrong:
    print(f"\n  {len(wrong)} wrong:")
    for want, got, text, model in wrong:
        flag = "LEAK " if want == "out_of_scope" else ("OVER " if got == "out_of_scope" else "     ")
        print(f"    {flag} want {want:22} got {got:22} [{model}] {text!r}")
