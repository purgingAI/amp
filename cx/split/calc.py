"""Share arithmetic, kept away from the handler and away from the model.

The backend recomputes every split and rejects the request if its arithmetic disagrees with
the `displayValue` figures we send. That is the right design — it means the assistant cannot
quietly round somebody's share — but it also means this module has to match the backend
exactly rather than approximately.

Two rules follow from money being in whole fils:

1. **Shares are floored to 2dp and the INITIATOR absorbs the remainder.** 100 three ways is
   33.33 / 33.33 / 33.34, not three times 33.3333. Giving the remainder to a participant
   would mean the person who did not pay the bill is charged the extra fil.

2. **The shares must sum to the bill, exactly.** Asserted, not hoped for: a split that does
   not add up is either a short-changed merchant or a customer paying for a rounding error,
   and both are the kind of bug that is invisible until somebody reconciles.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


def _q(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def equal(total: float, n_participants: int) -> tuple[list[float], float]:
    """Split `total` between `n_participants` people PLUS the initiator.

    Returns (participant shares, initiator share). The initiator takes the remainder.
    """
    heads = n_participants + 1
    if heads < 2:
        raise ValueError("a split needs at least one other person")
    t = _q(total)
    base = (t / heads).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    shares = [base] * n_participants
    mine = t - sum(shares)
    out = [float(s) for s in shares]
    assert _q(sum(out) + float(mine)) == t, f"shares do not sum to {t}"
    return out, float(mine)


def by_amount(total: float, amounts: list[float]) -> float:
    """Participants pay named amounts; the initiator pays what is left.

    Raises if the named amounts already exceed the bill — silently making the initiator's
    share negative would ask them to receive money from a merchant payment.
    """
    t = _q(total)
    named = sum(_q(a) for a in amounts)
    if named > t:
        raise ValueError(f"those shares add up to {named}, more than the {t} bill")
    return float(t - named)


def by_percentage(total: float, percents: list[float]) -> tuple[list[float], float]:
    """Percentages per participant; the initiator keeps the balance."""
    t = _q(total)
    if sum(percents) > 100:
        raise ValueError(f"those percentages add up to {sum(percents):g}%")
    shares = [(t * _q(p) / 100).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
              for p in percents]
    mine = t - sum(shares)
    return [float(s) for s in shares], float(mine)
