"""Decrypt the PII that OpenWallet stores encrypted at rest, exactly as the application does.

Needed for one thing only: a merchant payment resolves its receiver by **plaintext mobile**
(`ReceiverValidator` → `getByMapperByMobileNumber` → `hasher.getHashedKey(mobile)` →
`globalWalletKeyMapper.findById(hash)`), and every merchant's `user.mobile_number` is stored
encrypted. There is no other lookup path — a userId, MID or wallet id will not resolve.

Mirrors `CryptoProcessor/.../DataCipherService.java` precisely:

    AES/GCM/NoPadding, 128-bit tag
    key = Base64.decode(pii.db.encryption.key)        (from Vault)
    iv  = 16 ZERO bytes                               (not 12 — the usual GCM nonce size)
    aad = 128 ZERO bytes                              (their TAG_LENGTH reused as a length)

The IV and AAD look wrong because they are: a fixed all-zero IV defeats the point of GCM, and
`new byte[TAG_LENGTH]` is a 128-*byte* array where 128 was meant as *bits*. Both are
reproduced exactly rather than corrected — this code has to read what the application wrote,
and "fixing" either one produces `InvalidTag` and nothing else.

Read-only. It decrypts; it never writes PII anywhere, and callers should keep what comes out
in memory rather than in a file.
"""
from __future__ import annotations

import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_IV = b"\x00" * 16
_AAD = b"\x00" * 128
_TAG_BYTES = 16

_dek: bytes | None = None


def dek() -> bytes:
    """The data encryption key, from Vault. Cached in memory, never written down."""
    global _dek
    if _dek is None:
        from devdb import _vault_config
        raw = _vault_config().get("pii.db.encryption.key")
        if not raw:
            raise RuntimeError("pii.db.encryption.key is not present in the Vault config")
        _dek = base64.b64decode(raw)
    return _dek


def decrypt(cipher_b64: str | None) -> str | None:
    """Plaintext, or None if it does not decrypt.

    The Java side returns the ciphertext unchanged on failure, which makes a broken decrypt
    look like data. Returning None instead means a caller cannot mistake one for the other.
    """
    if not cipher_b64:
        return None
    try:
        raw = base64.b64decode(cipher_b64)
    except Exception:
        return None
    if len(raw) <= _TAG_BYTES:
        return None
    body, tag = raw[:-_TAG_BYTES], raw[-_TAG_BYTES:]
    try:
        d = Cipher(algorithms.AES(dek()), modes.GCM(_IV, tag)).decryptor()
        d.authenticate_additional_data(_AAD)
        return (d.update(body) + d.finalize()).decode("utf-8")
    except Exception:
        return None


def looks_like_mobile(value: str | None) -> bool:
    v = (value or "").strip().lstrip("+")
    return v.isdigit() and 7 <= len(v) <= 15


if __name__ == "__main__":
    from devdb import q
    rows = q("""SELECT id, business_name, mobile_number, identifier
                FROM user
                WHERE user_type='MERCHANT' AND status='ACTIVE' AND wallet_enable=1
                  AND kyc_status='Successful'
                ORDER BY business_name""", n=200)
    ok = 0
    for r in rows[:int(sys.argv[1]) if len(sys.argv) > 1 else 10]:
        m = decrypt(r["mobile_number"])
        good = looks_like_mobile(m)
        ok += good
        shown = (m[:3] + "*" * max(0, len(m) - 5) + m[-2:]) if good else repr(m)[:28]
        print(f"  {'ok ' if good else '?? '} {r['business_name'][:34]:36} {shown}")
    print(f"\n  {ok} decrypted to something that looks like a mobile number")
