#!/usr/bin/env python3
"""Laptop 1 — turn a spoken sentence into a transfer against a REMOTE monitor.

    ./pay_client.py "send 20 dirhams to alex morgan" --hub https://localhost

The device key never leaves this laptop. What goes over the wire is a signature over the
monitor's single-use nonce, bound to one mandate, payee, amount and currency — useless to an
interceptor and unusable twice.

Intent resolution also stays local: the payee is bound by exact match against the mandate roster
before anything is sent, so a mis-heard name is refused here rather than travelling.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import urllib.request
import datetime as dt

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import device_key as dk, intent as I   # noqa: E402


def post(base: str, path: str, payload: dict, timeout: int = 45) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sentence")
    ap.add_argument("--hub", default="http://127.0.0.1:8300",
                    help="monitor api base, e.g. https://localhost/pay")
    ap.add_argument("--agent", default="agt_sagar")
    ap.add_argument("--step-up", default="")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object instead of prose, for callers like Jarvis")
    ap.add_argument("--plan-only", action="store_true",
                    help="resolve and read back, touch no network, authorize nothing")
    a = ap.parse_args()

    def out(obj: dict) -> int:
        """Structured when asked, prose otherwise. Both say the same thing."""
        if a.json:
            print(json.dumps(obj))
            return 0 if obj.get("ok") else 1
        if not obj.get("ok"):
            print("refused   :", obj.get("reason", ""))
            return 1
        if obj.get("state") == "captured":
            print(f"sent      : {obj['amount']:g} {obj['currency']} to {obj['payee_name']} "
                  f"({obj['tier']}) {obj['transfer_id']}")
        elif obj.get("state") == "requires_step_up":
            print(f"step-up   : confirm on this laptop, then re-run with "
                  f"--step-up {obj['step_up_id']}")
        elif obj.get("state") == "planned":
            print("read back :", obj["read_back"])
        return 0

    try:
        p = I.plan(a.sentence, a.agent)
    except I.IntentError as e:
        return out({"ok": False, "state": "refused", "reason": str(e)})
    if p.intent != "transfer":
        return out({"ok": False, "state": "not_payment",
                    "reason": f"that isn't a payment ({p.intent})"})

    plan_fields = {"ok": True, "read_back": p.read_back(), "amount": p.amount,
                   "currency": p.currency, "payee_name": p.payee_name,
                   "payee_wallet": p.payee_wallet, "mandate_id": p.mandate_id,
                   "transcript": p.text}
    if a.plan_only:
        # Nothing has been sent, no session opened, no nonce spent. This is the
        # read-back a human hears BEFORE deciding, so it must cost nothing.
        return out({**plan_fields, "state": "planned"})
    if not a.json:
        print("read back :", p.read_back())

    eph = X25519PrivateKey.generate()
    pub = eph.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw).hex()
    try:
        s = post(a.hub, "/session", {"agent_id": a.agent, "mandate_id": p.mandate_id,
                                     "eph_pub": pub, "nonce": secrets.token_hex(8)})
    except Exception as e:
        return out({**plan_fields, "ok": False, "state": "unreachable",
                    "reason": f"I couldn't reach the payment monitor ({type(e).__name__})"})
    if "token" not in s:
        return out({**plan_fields, "ok": False, "state": "refused",
                    "reason": s.get("reason") or s.get("error") or "session refused"})
    n = post(a.hub, "/nonce", {"token": s["token"]})["nonce"]

    ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    be = dk.load()                       # the key stays here; only the signature travels
    sig = dk.sign_transfer(be, p.mandate_id, n, p.payee_wallet, p.amount, p.currency, ts)

    r = post(a.hub, "/authorize", {
        "token": s["token"], "nonce": n, "payee_wallet": p.payee_wallet,
        "amount": p.amount, "currency": p.currency, "timestamp": ts,
        "signature": sig["signature"], "step_up_id": a.step_up,
        "evidence": {"transcript": p.text, "resolved_payee": p.payee_name,
                     "backend": sig["backend"],
                     "hardware_resident": sig["hardware_resident"]}})

    state = r.get("state")
    if state == "captured":
        return out({**plan_fields, "state": "captured", "tier": r.get("tier", ""),
                    "transfer_id": r.get("transfer_id", ""),
                    "backend": sig["backend"],
                    "hardware_resident": sig["hardware_resident"]})
    if state == "requires_step_up":
        return out({**plan_fields, "state": "requires_step_up",
                    "step_up_id": r.get("step_up_id", ""), "expires": r.get("expires", "")})
    return out({**plan_fields, "ok": False, "state": "refused",
                "reason": r.get("reason") or "the monitor refused it"})


if __name__ == "__main__":
    sys.exit(main())
