"""Fail fast while the wallet database is unreachable.

Every Mongo read in the assistant has its own timeout (6 s here, 15 s there), and a single
turn makes several of them. With the database unreachable — the VPN dropped, say — a turn
waited out every timeout in turn before falling back: six reads, about sixty seconds, for
a question whose answer was going to come from the fallback anyway.

So the first connection failure opens this breaker, and for the next minute every read
skips straight to its fallback. After the minute, the next read tries the database again:
if it answers, normal service resumes; if not, the breaker opens for another minute.

Only CONNECTION failures trip it. A bad query or a missing field is a bug in our code and
must keep failing loudly, not be absorbed as "the database is down".
"""
from __future__ import annotations

import time

COOL_SECONDS = 60
_down_until = 0.0
_reason = ""


def is_open() -> bool:
    """True while reads should not even try."""
    return time.time() < _down_until


def reason() -> str:
    return _reason


def record(exc: BaseException) -> None:
    """Call from an except block around a Mongo read. Opens the breaker on network errors."""
    global _down_until, _reason
    try:
        from pymongo.errors import ConnectionFailure
    except Exception:                                      # noqa: BLE001
        return
    if isinstance(exc, ConnectionFailure):
        _down_until = time.time() + COOL_SECONDS
        _reason = f"{type(exc).__name__} — retrying the database in {COOL_SECONDS}s"


class Open(RuntimeError):
    """Raised instead of attempting a read while the breaker is open."""


def check() -> None:
    if is_open():
        raise Open(f"wallet database unreachable ({_reason})")
