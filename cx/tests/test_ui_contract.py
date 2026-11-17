"""The surface must be able to render everything the engine can propose.

This exists because of a specific failure, not a hypothetical one. `render()` was a chain of
exclusions — "not refund, not split" — ending in a send-shaped fallback:

    : `Send AED ${fmt(p.amount)} to ${p.payee}`

A bill proposal carries `total` and `bills` but no `amount` and no `payee`, so choosing "pay
from my wallet" produced **"Send AED NaN to undefined"** — and the button beneath it would
have fired a transfer. Every proposal kind added after that fallback was written would have
done the same.

So this test reads the engine for every `kind` it can produce and the page for every kind it
can label, and fails when they drift apart. It is a crude check — it greps source — but it
catches the exact class of bug that shipped, and it costs nothing to run.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
WEB = os.path.join(os.path.dirname(CX), "openwallet", "web")
sys.path.insert(0, CX)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


def ui_kinds() -> set[str]:
    src = open(os.path.join(WEB, "assist.html")).read()
    block = src[src.index("function proposalLabel"):]
    block = block[:block.index("function billReceipt")]
    return set(re.findall(r"case '([a-z_]+)'", block))


def test_every_proposal_kind_can_be_rendered():
    """A proposal the page cannot label is a payment the customer is asked to confirm
    without being told what it is."""
    from engine.engine import PROPOSAL_KINDS
    handled = ui_kinds() | {"refund", "split"}     # both have their own render branches
    missing = set(PROPOSAL_KINDS) - handled
    check("every proposal kind has a UI branch", not missing,
          f"unrenderable: {sorted(missing)}")
    stray = handled - set(PROPOSAL_KINDS) - {"refund", "split"}
    check("the page labels nothing the engine cannot propose", not stray, sorted(stray))


def test_the_fallback_does_not_invent_a_payment():
    src = open(os.path.join(WEB, "assist.html")).read()
    block = src[src.index("function proposalLabel"):]
    block = block[:block.index("function billReceipt")]
    default = block[block.index("default:"):]
    check("an unknown kind yields null, not a guessed send",
          "return null" in default and "Send AED" not in default, default[:80])


def test_bill_proposals_really_lack_amount_and_payee():
    """The fields the old fallback read. If a bill ever grows an `amount`, the NaN would
    hide rather than being fixed — so the absence is asserted, not assumed."""
    from engine.engine import ask
    # A ROUTABLE biller: "Home electricity" (ADDC) has a PENDING service-provider slab, so
    # it is now correctly refused before any proposal exists.
    ask("pay my home internet bill", want_second_opinion=False, session="uc1",
        pick={"intent": "bill_pay"})
    a = ask("pay from my wallet", want_second_opinion=False, session="uc1")
    p = a.proposal or {}
    check("a bill proposal is produced", p.get("kind") == "bill_pay", p.get("kind"))
    check("it has no `amount`", "amount" not in p)
    check("it has no `payee`", "payee" not in p)
    check("it does carry a grand total", p.get("grandTotal") is not None, p.get("grandTotal"))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{'all green' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)} "
          f"({len(FAILED)} failing)")
