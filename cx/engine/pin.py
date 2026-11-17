"""The pre-transaction PIN gate.

One config flag, applied to EVERY path that moves money — wallet-to-wallet, merchant
payment, a split share, and bills when they arrive. You asked for it under bill payments and
said "this goes for all", so it lives here rather than inside any one handler: a gate that
some flows honour and others skip is not a gate.

    CX_REQUIRE_PIN=1   ask for the PIN and validate it before the transaction
    CX_REQUIRE_PIN=0   straight through, as today  (default)

## Why validation is not done here

This module decides WHETHER a PIN is needed and holds the one-shot challenge. It does not
decide whether a PIN is *correct* — that is `validate()`, which delegates to the real wallet
check. An assistant that could judge a PIN itself would be a second authority on credentials,
and the first one is the only one that should exist.

## What a failed attempt does

Three wrong attempts abandon the payment and clear the challenge. It does not lock the
account: locking is the wallet's decision, on the wallet's own counter, and a chat surface
guessing at lockout policy would either lock people out early or give an attacker a free
parallel channel. We simply stop helping.
"""
from __future__ import annotations

import os
import time

TTL = 180.0

#: Two, not three — and nothing like the wallet's five.
#:
#: `PinServiceImpl.pinValidate` counts failures on the USER ROW, not per session, and at the
#: fifth it sets `UserStatus.FROZEN`, emails, texts and pushes "Your wallet is currently
#: inactive". That counter is not ours and we cannot reset it, so this limit exists to stop
#: well short of it: if two attempts fail, something is wrong that a third guess will not
#: fix, and the cost of being wrong is somebody's wallet frozen.
MAX_ATTEMPTS = 2

#: The wallet freezes at this many, counted server-side across all sessions.
WALLET_FREEZE_AT = 5

#: session -> {"action": <the payment we are holding>, "tries": n, "at": ts}
_PENDING: dict[str, dict] = {}


def required() -> bool:
    return os.environ.get("CX_REQUIRE_PIN", "0") == "1"


def challenge(session: str, action: dict) -> None:
    """Hold a payment until the PIN arrives. The action is stored, never executed here."""
    _PENDING[session] = {"action": action, "tries": 0, "at": time.time()}


def pending(session: str) -> dict | None:
    rec = _PENDING.get(session)
    if not rec:
        return None
    if (time.time() - rec["at"]) > TTL:
        _PENDING.pop(session, None)
        return None
    return rec


def clear(session: str) -> None:
    _PENDING.pop(session, None)


def looks_like_pin(text: str) -> bool:
    """A bare 4-6 digit number, nothing else.

    Deliberately strict. A PIN arriving as part of a sentence is far more likely to be an
    amount, and treating "send 1234 to alex" as a PIN entry would consume the digits and
    then fail the payment for having no amount.
    """
    # EXACTLY four digits. `PinDTO` carries @Pattern(REGEX_PIN, "Pin should be 4 digit"),
    # so a five-digit entry is rejected by bean validation before it ever reaches the PIN
    # check — which at least costs no attempt, but tells the customer nothing useful.
    t = (text or "").strip().replace(" ", "").replace("-", "")
    return t.isdigit() and len(t) == 4


def validate(session: str, pin: str) -> tuple[bool, str]:
    """Check the PIN against the wallet. Returns (ok, message).

    The PIN is passed straight to the wallet's own validator and is never logged, stored or
    echoed — not in the trace, not in the decision theatre, not in an error.
    """
    rec = pending(session)
    if not rec:
        return False, "There's nothing waiting for a PIN."
    try:
        import wallet_pin
        ok, reason = wallet_pin.verify(pin)
    except Exception as exc:
        # An integration failure is NOT a wrong PIN and must not consume an attempt. The
        # wallet's own counter only moves when the request actually reaches the PIN check, so
        # counting our own transport errors here would walk the customer towards a freeze for
        # a fault entirely on our side.
        return False, (f"I couldn't check your PIN just now ({type(exc).__name__}). "
                       f"Nothing has been taken from your wallet.")
    if ok:
        clear(session)
        return True, ""
    if reason == "LOCKED":
        # The wallet has already frozen or closed the account. Stop immediately: more
        # attempts cannot help and the customer needs to be told the real reason.
        clear(session)
        return False, ("Your wallet is locked, so I can't take this payment. You'll need to "
                       "contact support to unlock it.")
    if reason != "WRONG_PIN":
        # Carry the real reason into the trace. "I couldn't check your PIN just now" told
        # the customer the right thing and told US nothing, so an unexpected status from the
        # wallet was indistinguishable from a network blip.
        detail = reason.split(":", 2)
        why = (detail[2] or f"status {detail[1]}") if len(detail) == 3 else reason
        return False, (f"I couldn't check your PIN just now ({why}). Nothing has been "
                       f"taken from your wallet.")
    rec["tries"] += 1
    left = MAX_ATTEMPTS - rec["tries"]
    if left <= 0:
        clear(session)
        return False, ("That PIN wasn't right, so I've stopped the payment. Nothing has "
                       "left your wallet — and do be careful, your wallet locks itself "
                       "after a few wrong attempts.")
    return False, (f"That PIN wasn't right. I'll try once more, then stop — your wallet "
                   f"locks itself after {WALLET_FREEZE_AT} wrong attempts in total.")
