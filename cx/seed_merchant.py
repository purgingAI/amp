"""Give the demo merchant a real, both-signed mandate.

A merchant is a payee like any other: to be paid it needs a mandate the payer signed and the
merchant counter-signed, with its own limits. Shortcutting that for the merchant flow would
mean the till is paid by a path the peer flow does not use, and then the demo proves nothing
about the merchant case.

Idempotent — run it as often as you like.

    ~/amp/.venv/bin/python cx/seed_merchant.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

AMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AMP)
sys.path.insert(0, os.path.join(AMP, "cx"))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app import ledger, mandate_store as ms
from app.mandate import Limits, Mandate, Party

import data as cx_data

KEYS = Path(AMP) / "data" / "keys"
SAGAR, W_SAGAR = cx_data.CUSTOMER["id"], cx_data.CUSTOMER["wallet"]
CUR = "AED"


def file_key(name: str) -> Ed25519PrivateKey:
    KEYS.mkdir(parents=True, exist_ok=True)
    p = KEYS / f"{name}.key"
    if not p.exists():
        p.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw())
        os.chmod(p, 0o600)
    return Ed25519PrivateKey.from_private_bytes(p.read_bytes())


def _resolve_mobile(user_id: str) -> str | None:
    """The merchant's plaintext mobile, decrypted from dev.

    Resolved at RUNTIME and kept in memory. A merchant payment resolves its receiver by
    hashing a plaintext mobile, so we need it — but writing customer PII into a config file
    on a laptop to save a database round trip is not a trade worth making.
    """
    import pii
    from devdb import q
    rows = q("SELECT mobile_number FROM user WHERE id = %s", (user_id,), n=1)
    if not rows:
        return None
    m = pii.decrypt(rows[0]["mobile_number"])
    return m if pii.looks_like_mobile(m) else None


def payer_signer():
    """Sign as the payer with the key ALREADY ENROLLED for them.

    Not a fresh key: `agt_sagar` is enrolled with the device key the voice flow signs with,
    and a mandate signed by anything else fails `verify()` at payment time. Minting a second
    identity to make the seeding easier would produce mandates that look fine here and are
    rejected the moment someone tries to pay with them.
    """
    from app import device_key
    return device_key.load()


#: Where payee private keys live. The payi demo keeps its own set, and a mandate signed by
#: anything other than the payee's ENROLLED key fails verification at payment time.
_KEY_DIRS = (Path(AMP) / "data" / "keys", Path(AMP) / "data" / "payi_keys")


def _payee_key(agent_id: str):
    """The payee's private key whose public half matches what is ENROLLED for them.

    Searching rather than deriving a path, because the enrolled key may have come from a
    different part of the demo. Signing with a freshly-generated key instead produced a
    mandate that looked fine and failed at payment with "that mandate does not carry two
    valid signatures" — which is exactly what happened to Alex when the payer key rotated.
    """
    enrolled = ms.public_key(agent_id)
    if enrolled is None:
        return file_key(agent_id)
    want = enrolled.public_bytes(serialization.Encoding.Raw,
                                 serialization.PublicFormat.Raw)
    for d in _KEY_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.key"):
            try:
                k = Ed25519PrivateKey.from_private_bytes(f.read_bytes())
            except Exception:
                continue
            if k.public_key().public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw) == want:
                return k
    return None


def _remint_existing(payer) -> None:
    """Re-sign every active mandate this payer holds, so they verify under the new key.

    A mandate is only saved once it VERIFIES. Writing one that does not is worse than leaving
    the old one: it fails later, in front of a customer, with a message about signatures that
    tells nobody what actually went wrong.
    """
    import json as _json
    with ms.conn() as c:
        rows = c.execute("SELECT doc FROM mandates WHERE payer_agent=? AND state='active'",
                         (SAGAR,)).fetchall()
    for r in rows:
        m = Mandate.from_dict(_json.loads(r["doc"]))
        payee_key = _payee_key(m.payee.agent_id)
        if payee_key is None:
            print(f"    SKIPPED {m.mandate_id} {m.payee.display_name} — no private key matching "
                  f"their enrolled public key; leaving the existing mandate alone")
            continue
        m.payer_sig = payer.sign(m.canonical()).hex()
        m.sign_as("payee", payee_key)
        payer_pub, payee_pub = ms.public_key(SAGAR), ms.public_key(m.payee.agent_id)
        if not (payer_pub and payee_pub and m.verify(payer_pub, payee_pub)):
            print(f"    REFUSED {m.mandate_id} {m.payee.display_name} — re-signed mandate does "
                  f"not verify; not saving it")
            continue
        ms.save(m, "active")
        print(f"    re-signed {m.mandate_id}  {m.payee.display_name}")


def main() -> None:
    ms.init()
    ledger.init()

    payer = payer_signer()
    label = getattr(payer, "backend", type(payer).__name__)
    enrolled = ms.public_key(SAGAR)
    new_raw = payer.public_key().public_bytes(
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["serialization"]).Encoding.Raw,
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["serialization"]).PublicFormat.Raw).hex()
    old_raw = enrolled.public_bytes(
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["serialization"]).Encoding.Raw,
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["serialization"]).PublicFormat.Raw).hex() if enrolled else None

    rotated = old_raw is not None and old_raw != new_raw
    ms.enrol_key(SAGAR, payer.public_key(), f"payer key: {label}", rotate=True)
    print(f"  payer signs with {label}")
    if rotated:
        # Every existing mandate verifies against the OLD payer key, so leaving them in place
        # would mean every payment fails at m.verify() with a confusing error. Re-mint them.
        print("  payer key rotated — re-minting every existing mandate under the new key")
        _remint_existing(payer)

    for m in cx_data.MERCHANTS:
        agent = "agt_" + str(m["mid"]).lower().replace("-", "")[:24]
        wallet = m.get("wallet") or ("w_" + str(m["mid"]).lower()[:16])
        key = file_key(agent)
        ms.enrol_key(agent, key.public_key(), f"{m['name']} till key")
        try:
            ledger.open_account(wallet, m["name"], CUR, 0)
        except Exception:
            pass                                   # already open

        if ms.find_active(SAGAR, agent):
            print(f"  = {m['name']:20} mandate already active")
            continue

        mandate = Mandate(
            payer=Party(SAGAR, W_SAGAR, cx_data.CUSTOMER["name"]),
            payee=Party(agent, wallet, m["name"]),
            currency=CUR,
            # A till mandate is wider than a peer one -- you buy coffee more often than you
            # wire money to family -- but it is still a cap, and still enforced.
            limits=Limits(500, 1500, 8000, 25),
            green_tier_cap=150,
            not_after="2026-12-31T23:59:59Z",
            purposes=["merchant_checkout"])
        mandate.payer_sig = payer.sign(mandate.canonical()).hex()
        mandate.sign_as("payee", key)
        ms.save(mandate, "active")
        print(f"  + {m['name']:20} {mandate.mandate_id}  green ≤ {mandate.green_tier_cap:g}")


if __name__ == "__main__":
    main()
