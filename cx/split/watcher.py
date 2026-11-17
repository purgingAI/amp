"""Watch open splits and announce when somebody actually pays their share.

Without this the customer only learns that Alex settled by asking. The useful moment is the
one where it happens, and a chat that says "Alex just paid his 42.50" the moment he does is
the difference between a ledger you query and an assistant that is paying attention.

## Why polling, and why that is not a cop-out

The settlement happens in the Transaction service and is published to Kafka, but consuming
dev MSK from here means broker credentials, a consumer group, and an offset to manage — a
lot of moving parts for a demo surface, and every one of them a way to miss an event
silently. A poll of `/split-payment/{id}` is one authenticated read whose failure mode is
"we find out a few seconds later", which is the right failure mode for a notification.

## What it will not do

It never announces the customer's OWN share, which is marked PAID from the moment the split
is created — announcing that would be the assistant telling you about something you just
did. It announces a transition, not a state, so a participant who was already PAID when we
first saw the split is never reported as having "just paid".
"""
from __future__ import annotations

import threading
import time

from . import api

#: How often to look. Fast enough that the message still feels like news, slow enough that a
#: screen left open all afternoon is not hammering the API.
INTERVAL = 12.0

#: Give up on a split after this long with no change — a split nobody is paying should not
#: be polled forever.
STALE_AFTER = 3600.0

_WATCHED: dict[str, dict] = {}        # splitId -> {"seen": {userId: status}, "since": ts}
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_EMIT = None


def watch(split_id: str, announce: bool = True) -> None:
    """Start following a split. Safe to call repeatedly for the same id.

    `announce=False` follows it silently. Adopted splits are followed that way so a restart
    does not lose track of them, without turning this customer's five historical groups into
    a stream of chat messages about bills they settled weeks ago.
    """
    with _LOCK:
        if split_id not in _WATCHED:
            _WATCHED[split_id] = {"seen": {}, "since": time.time(),
                                  "announce": announce}
        elif announce:
            _WATCHED[split_id]["announce"] = True


def start(emit) -> None:
    """Begin the poll loop, pushing events through `emit(stage, kind, detail, tag)`."""
    global _THREAD, _EMIT
    _EMIT = emit
    if _THREAD and _THREAD.is_alive():
        return
    _THREAD = threading.Thread(target=_loop, daemon=True)
    _THREAD.start()


def adopt_open_splits() -> None:
    """Pick up splits that already exist, SILENTLY.

    This customer has five groups and a dozen historical splits. Announcing a transition in
    any of them filled the chat with "Alex just paid their 90.00 share… their 40.00 share…"
    for bills that have nothing to do with what is on screen. The chat should speak about the
    split the customer is actually in the middle of; everything else is answered when asked.

    They are still followed, so a split created before a restart is not forgotten — it simply
    does not interrupt.
    """
    gs = api.groups()
    if gs is None:
        return
    for g in gs:
        for row in api.group_history(g.get("groupId") or ""):
            sid = row.get("splitId")
            if sid and str(row.get("status") or "").upper() not in ("SETTLED", "REVERSED"):
                watch(sid, announce=False)


def _loop() -> None:
    while True:
        try:
            _tick()
        except Exception:
            pass                       # a watcher that dies silently is worse than a slow one
        time.sleep(INTERVAL)


def _tick() -> None:
    with _LOCK:
        ids = list(_WATCHED)
    for sid in ids:
        d = api.detail(sid)
        if not d:
            continue
        with _LOCK:
            rec = _WATCHED.get(sid)
            if rec is None:
                continue
            seen, first_look = rec["seen"], not rec["seen"]
            changed = []
            for p in d.get("participants") or []:
                uid = p.get("userId")
                if not uid:
                    continue
                was, now = seen.get(uid), str(p.get("status") or "").upper()
                seen[uid] = now
                # A transition, not a state — and never the initiator, whose share is PAID
                # from the moment the split exists.
                if (not first_look and was and was != now and now == "PAID"
                        and not p.get("initiator")):
                    changed.append(p)
            if changed:
                rec["since"] = time.time()
            status = str(d.get("splitStatus") or "").upper()
            done = status in ("SETTLED", "REVERSED") or all(
                str(p.get("status") or "").upper() in ("PAID", "REVERSED", "EXPIRED")
                for p in (d.get("participants") or []))
            if done or (time.time() - rec["since"]) > STALE_AFTER:
                _WATCHED.pop(sid, None)

        if not rec or not rec.get("announce"):
            continue                       # followed, but not this conversation's business
        for p in changed:
            first = (p.get("name") or "someone").split()[0]
            amount = p.get("shareAmount")
            _EMIT and _EMIT("CX · Split", "split.paid",
                            f"{first} paid their {amount:g} share of {sid}", "act")
        if changed and done:
            _EMIT and _EMIT("CX · Split", "split.settled",
                            f"{sid} is fully settled", "act")
