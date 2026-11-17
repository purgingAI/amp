"""Varied phrasing, so the assistant does not read like a form letter.

## Why this is not a language model

Every sentence this assistant says contains an amount, a status, a payee or a policy. Handing
those to a generative model to reword is handing it the chance to change one — to say "your
payment succeeded" where the truth is "pending", or to round AED 85.15 to AED 85. On a
customer-facing money surface that is not a style risk, it is a correctness risk, and it would
be invisible until someone complained.

So the split is:

* **The facts are composed, never generated.** Amounts, references, statuses and the refund
  policy sentences are built from data and are byte-identical every time.
* **Everything around them varies** — the opening, the acknowledgement, how a question is
  asked, how a result is handed over. That is where the "same message again" feeling actually
  comes from, and it is safe to vary because none of it asserts anything.

`pick()` is seeded per session and remembers its last choice, so the same phrasing does not
come back twice in a row within one conversation — which is the thing that makes a reply feel
canned more than the words themselves do.

If a generative paraphraser is wanted later, the seam is here: it would feed `pick()`, still
never touching the composed facts.
"""
from __future__ import annotations

import random

#: Phrase pools. Each must be interchangeable — same meaning, same commitments, no new claims.
POOLS: dict[str, tuple[str, ...]] = {
    "ack": ("Right.", "Got it.", "Okay.", "Sure.", "Of course.", "No problem."),

    "greet": (
        "I can help with your payments.",
        "Happy to help with your payments.",
        "Sure — payments are what I can help with.",
        "Of course. I handle anything to do with your payments.",
        "I'm here for anything about your payments.",
    ),
    "greet_offer": (
        "Ask me where a payment got to, about a refund, fees and timings, your limits, "
        "offers near you — or tell me who to pay.",
        "You can ask where money went, request a refund, check fees, timings or limits, "
        "see offers near you, or just tell me who to pay.",
        "Try me on a payment's status, a refund, fees and timings, your limits, nearby "
        "offers, or paying someone.",
        "Whereabouts of a payment, refunds, fees, timings, limits, nearby offers, or sending "
        "money — whichever you need.",
    ),

    "which_txn": (
        "Which payment do you mean?",
        "Which one are we talking about?",
        "Which payment should I look at?",
        "Tell me which payment and I'll pull it up.",
    ),
    "which_payee": (
        "Who are you paying — someone you know, or a merchant near you?",
        "Who's this going to — someone on your list, or a shop nearby?",
        "Who should this go to? Someone you've paid before, or a merchant near you?",
        "Who am I paying — one of your people, or somewhere nearby?",
    ),
    "how_much": (
        "How much?",
        "How much shall I send?",
        "What's the amount?",
        "How much are we talking?",
    ),

    #: Handing over a LIST. The counts and the total that follow are composed, so these
    #: openers can vary freely — none of them asserts anything about the money.
    "here_they_are": (
        "Here are your last {n} payments{scope}.",
        "These are your last {n} payments{scope}.",
        "Your last {n} payments{scope}:",
        # Every variant carries the noun: {n} may arrive as "2 failed", and an adjective
        # with nothing to qualify reads as "Found 2 failed — here they are."
        "Found {n} payments{scope} — here they are.",
        "Got them. Your last {n} payments{scope}:",
    ),
    "list_tail": (
        "Tap any one and I'll open it.",
        "Tap one to look into it.",
        "Pick any of them and I'll dig in.",
        "Anything there you want me to open?",
    ),
    "about": (
        "About",
        "Looking at",
        "On",
        "Regarding",
    ),
    "guessed": (
        "I picked the one that looks like the problem",
        "I've gone with the one that looks wrong",
        "this is the one that looks like the issue",
        "assuming you mean the one that didn't go smoothly",
    ),
    "ready": (
        "Confirm and it goes through the usual checks.",
        "Say the word and I'll put it through.",
        "Confirm when you're ready and it runs the usual checks.",
        "Give me the go-ahead and it goes through.",
    ),
    "no_qr": (
        "Confirm to pay — no QR needed.",
        "Confirm and it's paid, no scanning needed.",
        "Just confirm — you won't need to scan anything.",
        "Confirm to pay. No QR, no scanning.",
    ),
}

#: Last index used per (session, key), so a phrase never repeats back to back in a chat.
_last: dict[tuple[str, str], int] = {}


def pick(key: str, session: str = "default") -> str:
    """One phrase from the pool, never the same one twice running in a session."""
    pool = POOLS.get(key)
    if not pool:
        return ""
    if len(pool) == 1:
        return pool[0]
    prev = _last.get((session, key))
    rng = random.Random()
    choices = [i for i in range(len(pool)) if i != prev]
    i = rng.choice(choices)
    _last[(session, key)] = i
    return pool[i]


def reset(session: str) -> None:
    for k in [k for k in _last if k[0] == session]:
        _last.pop(k, None)
