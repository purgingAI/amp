#!/usr/bin/env python3
"""The PIN gate: one chokepoint, and the held instruction survives the PIN.

Two real bugs are pinned here.

**1. "I do not know how to do that" after a CORRECT PIN.** `release_pin` resumed the held
action with `action.pop("kind")`, which removes the key before handing the dict to
`cx_act`. The payment was authorised by the wallet and then the instruction describing it
was deleted, so the customer typed the right PIN and was told the assistant did not
understand. Worse than a refusal: they had just been charged the attention of entering a
credential for nothing.

**2. A gate some screens honour and others skip.** The voice send screen started payments
with no PIN at all while the chat asked for one every time. The test for that is
structural — exactly one place may start a payment — because any new screen added later
would otherwise quietly reopen the hole.

Run: ~/amp/.venv/bin/python cx/tests/test_pin_gate.py
"""
import importlib.util
import os
import re
import sys

AMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER = os.path.join(AMP, "openwallet", "server.py")
sys.path.insert(0, os.path.join(AMP, "openwallet"))
sys.path.insert(0, AMP)
sys.path.insert(0, os.path.join(AMP, "cx"))

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got!r} (wanted {want!r})")
    else:
        print(f"  ok    {name}")


def load():
    spec = importlib.util.spec_from_file_location("p10srv", SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_one_chokepoint():
    """Only start_payment may launch a payment thread."""
    print("\nstructure — one way in")
    src = open(SERVER).read()
    starts = re.findall(r"threading\.Thread\(\s*target=run_payment", src)
    check("places that start a payment", len(starts), 1)
    # and it must sit inside start_payment, not somewhere that bypasses the gate
    body = src.split("def start_payment(")[1].split("\ndef ")[0]
    check("that one place is inside start_payment",
          "threading.Thread(target=run_payment" in body, True)


def test_pin_release_keeps_the_instruction(m):
    """A correct PIN must resume the ORIGINAL action, kind and all."""
    print("\nrelease — the held instruction survives")
    from engine import pin as cx_pin

    seen = {}

    def fake_cx_act(req):
        seen["req"] = dict(req)
        return {"ok": True, "ran": True}

    m.cx_act = fake_cx_act
    cx_pin.validate = lambda session, pin: (True, "")      # the wallet says yes

    cx_pin.challenge("t1", {"kind": "send", "payee": "Alex Morgan", "amount": 20,
                            "session": "t1", "_pin_ok": True})
    out = m.release_pin("t1", "1598")
    check("the action ran", out.get("ran"), True)
    check("kind survived the release", seen.get("req", {}).get("kind"), "send")
    check("payee survived", seen.get("req", {}).get("payee"), "Alex Morgan")
    check("amount survived", seen.get("req", {}).get("amount"), 20)


def test_payment_release_goes_to_start_payment(m):
    """A held voice-send resumes through start_payment, with pin_ok set by US, not the body."""
    print("\nrelease — a voice send resumes as a payment")
    from engine import pin as cx_pin

    got = {}

    def fake_start(sentence, amount, receiver_mobile=None, discount=None, session=None,
                   pin_ok=False, who=""):
        got.update(sentence=sentence, amount=amount, pin_ok=pin_ok, who=who)
        return {"ok": True, "started": True}

    m.start_payment = fake_start
    cx_pin.validate = lambda session, pin: (True, "")
    cx_pin.challenge("t2", {"kind": "_payment", "sentence": "send 20 dirhams to alex",
                            "amount": 20.0, "receiver_mobile": None, "discount": None,
                            "session": "t2", "who": "Alex Morgan"})
    out = m.release_pin("t2", "1598")
    check("started", out.get("started"), True)
    check("sentence survived", got.get("sentence"), "send 20 dirhams to alex")
    check("pin_ok set on release", got.get("pin_ok"), True)
    check("payee carried to the sheet", got.get("who"), "Alex Morgan")


def test_nothing_held(m):
    print("\nrelease — nothing to release")
    out = m.release_pin("no-such-session", "1598")
    check("refused", out.get("ok"), False)


if __name__ == "__main__":
    mod = load()
    test_one_chokepoint()
    test_pin_release_keeps_the_instruction(mod)
    test_payment_release_goes_to_start_payment(mod)
    test_nothing_held(mod)
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
