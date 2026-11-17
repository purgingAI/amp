"""Refinements: follow-ups that NARROW an open question rather than answering it.

When the assistant has asked "who are you paying?" and offered a list, there are three things
a reply can be, and the engine previously only knew two:

  1. an **answer**        — "MClothes"            -> fill the slot (continuation)
  2. a **new request**    — "actually, refund me" -> start again
  3. a **refinement**     — "which has offers?"   -> same question, smaller list   <-- missing

Without the third, "which has offers" fell into the hold-the-question path and the assistant
re-asked, unchanged, with the same list. That reads as not listening, because it isn't: the
customer narrowed the question and got the original back.

A refinement is deliberately detected by rule rather than by the classifier. It is not a
15-way intent decision — it is a handful of modifiers over a list we are already showing, and
a table of them is auditable and cannot drift. The classifier has nothing to add: "which has
offers" is weak and ambiguous on its own, and only means anything relative to the list
currently on screen.
"""
from __future__ import annotations

import re

#: modifier -> the patterns that signal it. Order matters: the first match wins, so the more
#: specific phrasings sit above the general ones.
PATTERNS: dict[str, tuple[str, ...]] = {
    "with_offer": (
        r"\b(which|who|any|ones?)\b.*\b(offer|offers|discount|deal|deals|promo|"
        r"cashback|deals?|saving|savings)\b",
        r"\b(with|has|have|having|got)\b.*\b(offer|offers|discount|deal|promo|cashback)\b",
        r"\bonly\b.*\b(offer|discount|deal)\b",
        r"\b(offer|offers|discount|deals?)\b\s*\??$",
    ),
    "nearest": (
        r"\b(which|who)\b.*\b(nearest|closest|near(by)?|close)\b",
        r"\b(nearest|closest)\b",
        r"\bnear(est)?\s+(me|here)\b",
    ),
    "cheapest": (
        r"\b(cheapest|best value|lowest fee|least fee|cheaper)\b",
        r"\b(which|who)\b.*\b(cheap|cheapest|best)\b",
    ),
    "more": (
        r"\b(more|others|other ones|anyone else|anything else|show all|all of them)\b",
    ),
}

_COMPILED = {k: tuple(re.compile(p, re.I) for p in v) for k, v in PATTERNS.items()}

#: A refinement must not carry a real answer. "MClothes has an offer right?" names a merchant
#: and should be treated as the answer, not as a filter.
_ANSWERY = ("amount", "payee", "merchant", "txn_ref")

#: A request that names its OWN subject and scope is a new question, not a narrowing of
#: ours. "which has offers" filters the list on screen; "what offers are near me" asks for
#: offers near the customer and would be the same question with no list open at all.
#:
#: This mattered in a real conversation: with "which merchant?" open, "what offers are near
#: me" was read as "narrow those merchants to the ones with offers" and the customer's new
#: request vanished into the old one. Starting a different flow has to stay possible.
_FRESH_REQUEST = re.compile(
    r"^\s*(what|what's|whats|show|list|give|tell|find|kaun[ -]?se|kya|konse|dikha\w*)\b"
    r".*\b(offer|offers|discount|deal|deals|promo|cashback|merchant|merchants|shop|"
    r"shops|dukaan)\b"
    r"|\bmy\s+(offers?|deals?|discounts?)\b"
    r"|\b(near|around)\s+me\b.*\b(offer|offers|deal|deals)\b",
    re.I)


def detect(text: str, slots: dict | None = None) -> str | None:
    """Which refinement this is, or None.

    Returns None when the utterance also supplies a slot — then it is an answer, and answering
    beats filtering.
    """
    if slots and any(slots.get(k) not in (None, "") for k in _ANSWERY):
        return None
    low = " ".join((text or "").lower().split())
    if not low or len(low.split()) > 8:        # a refinement is short; a sentence is not
        return None
    if _FRESH_REQUEST.search(low):
        return None
    for key, pats in _COMPILED.items():
        if any(p.search(low) for p in pats):
            return key
    return None


def apply_to_merchants(kind: str, merchants: list[dict]) -> tuple[list[dict], str | None]:
    """Narrow a merchant list. Returns (rows, why-nothing-matched)."""
    if kind == "with_offer":
        rows = [m for m in merchants if m.get("offer")]
        if not rows:
            return [], "none of the merchants near you has a live offer right now"
        return rows, None
    if kind == "nearest":
        rows = sorted([m for m in merchants if m.get("metres") is not None],
                      key=lambda m: m["metres"])
        return (rows or merchants), None
    if kind == "cheapest":
        # "Cheapest" over merchants means the biggest saving — there is no price list here,
        # only what each offer takes off.
        rows = sorted([m for m in merchants if m.get("offer")],
                      key=lambda m: -float((m["offer"] or {}).get("value") or 0))
        if not rows:
            return [], "none of the merchants near you has a live offer to compare"
        return rows, None
    return merchants, None


def describe(kind: str) -> str:
    return {"with_offer": "with a live offer",
            "nearest": "nearest first",
            "cheapest": "best saving first",
            "more": "all of them"}.get(kind, "")
