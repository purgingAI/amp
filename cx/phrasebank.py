"""Many ways to say the same thing — without ever saying a different thing.

Each message KIND (phrases/SCHEMA.md) is written per language as 5 openers × 5 cores × 4
closers, so one reply has 100 wordings. Cores mix active and passive voice ("I'll send AED 50
to Alex" / "AED 50 will be sent to Alex"); openers and closers carry the warmth.

What makes it safe on a payments surface:
  * facts enter ONLY through {slots} filled by the engine — the bank contains no numbers,
    names or amounts of its own, and tests/test_phrasebank.py renders every combination and
    fails the build if any phrase adds a digit, drops a required slot or uses an unknown one;
  * no generative model: every sentence a customer can see has been written and checked.

What makes it feel human:
  * a session never hears the same full sentence twice in its last REPEAT_WINDOW replies of
    that kind, and never the same opener twice running;
  * the language follows the customer (en / hi / hinglish / ar), falling back to English.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import threading

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phrases")
LANGS = ("en", "hi", "hinglish", "ar")
REPEAT_WINDOW = 30

_LOCK = threading.Lock()
_BANK: dict[str, dict] = {}
_SEEN: dict[tuple, collections.deque] = {}
_LAST_OPENER: dict[tuple, int] = {}
_SLOT = re.compile(r"\{(\w+)\}")


def bank(lang: str) -> dict:
    with _LOCK:
        if lang not in _BANK:
            try:
                _BANK[lang] = json.load(open(os.path.join(HERE, f"{lang}.json"), encoding="utf-8"))
            except (OSError, ValueError):
                _BANK[lang] = {}
        return _BANK[lang]


def kinds(lang: str = "en") -> list[str]:
    return sorted(bank(lang))


def _join(*parts: str) -> str:
    out = " ".join(p.strip() for p in parts if p and p.strip())
    return re.sub(r"\s+([,.!?؟،])", r"\1", out)


_STEM = re.compile(r"[a-z\u0900-\u097f\u0600-\u06ff]{5,}", re.I)
#: Words that sound clumsy when two parts of one reply both use them ("Ready. I'm ready to...",
#: "Please confirm... Would you like to confirm?").
_ECHO = ("ready", "confirm", "happy", "sure", "help", "glad", "please", "noted", "taiyaar", "confirm",
         "madad", "bilkul", "تأكيد", "جاهز", "مساعد", "पुष्टि", "तैयार", "मदद")


def _flows(opener: str, core: str, closer: str) -> bool:
    """No echo between the three parts: the same cue word may appear in only one of them."""
    parts = [p.lower() for p in (opener, re.sub(r"\{\w+\}", "", core), closer)]
    for w in _ECHO:
        if sum(w in p for p in parts) > 1:
            return False
    return True


def say(kind: str, lang: str = "en", session: str = "default", opener: bool = True, **slots) -> str | None:
    """A fresh wording of `kind` in `lang`, or None when the bank has no such kind (callers
    then keep their existing sentence). Slot values are inserted as given."""
    lang = lang if lang in LANGS else "en"
    k = bank(lang).get(kind) or (bank("en").get(kind) if lang != "en" else None)
    if not k:
        return None
    op, co, cl = k.get("openers") or [""], k.get("cores") or [], k.get("closers") or [""]
    if not co:
        return None
    key = (session, lang, kind)
    with _LOCK:
        seen = _SEEN.setdefault(key, collections.deque(maxlen=REPEAT_WINDOW))
        last_op = _LAST_OPENER.get(key)
        combos = [(i, j, m) for i in range(len(op)) for j in range(len(co)) for m in range(len(cl))
                  if (i, j, m) not in seen and (len(op) == 1 or i != last_op)
                  and _flows(op[i], co[j], cl[m])]
        if not combos:                              # window exhausted (tiny lists): start over
            seen.clear()
            combos = [(i, j, m) for i in range(len(op)) for j in range(len(co)) for m in range(len(cl))
                      if _flows(op[i], co[j], cl[m])] or [(0, 0, 0)]
        i, j, m = random.choice(combos)
        seen.append((i, j, m))
        _LAST_OPENER[key] = i
    try:
        core = co[j].format(**slots)
    except (KeyError, IndexError, ValueError):
        return None                                 # a slot we were not given: keep the old sentence
    return _join(op[i] if opener else "", core, cl[m])


def required_slots(kind: str, lang: str = "en") -> set[str]:
    k = bank(lang).get(kind) or {}
    sets = [set(_SLOT.findall(c)) for c in k.get("cores") or []]
    return set.intersection(*sets) if sets else set()


def reset(session: str) -> None:
    with _LOCK:
        for key in [k for k in _SEEN if k[0] == session]:
            _SEEN.pop(key, None)
            _LAST_OPENER.pop(key, None)
