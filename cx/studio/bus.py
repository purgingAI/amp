"""Live events: what changed, pushed to every open Studio (GET /events, server-sent events).

publish(kind, by, **data) from wherever something changes; each open page holds one stream and
reacts (a toast saying who did what, and the view it affects refreshes). In memory only — a page that
was closed simply loads fresh state when it opens. Bounded: at most MAX_SUBS open streams, and a slow
reader's queue drops its oldest events rather than holding memory.
"""
from __future__ import annotations

import queue
import threading
import time

MAX_SUBS = 64
_subs: list[queue.Queue] = []
_lock = threading.Lock()


def subscribe() -> queue.Queue | None:
    with _lock:
        if len(_subs) >= MAX_SUBS:
            return None
        q: queue.Queue = queue.Queue(maxsize=200)
        _subs.append(q)
        return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        if q in _subs:
            _subs.remove(q)


def publish(kind: str, by: str = "", **data) -> None:
    ev = {"kind": kind, "by": (by or "system").split("@")[0], "at": time.time(), **data}
    with _lock:
        subs = list(_subs)
    for q in subs:
        try:
            q.put_nowait(ev)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(ev)
            except (queue.Empty, queue.Full):
                pass
