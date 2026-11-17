"""What the assistant says when the customer goes quiet.

A chat that sits in silence after asking a question reads as broken — the customer cannot
tell whether it is waiting, thinking, or dead. So after a while it speaks again.

Three rules make this helpful rather than irritating, and all three are policy decisions
rather than implementation details:

1. **A nudge never acts.** It is a sentence and some buttons. Nothing here can move money,
   raise a refund or commit to anything; the most it does is re-offer a choice the customer
   already had. An assistant that gets bored and sends the payment anyway is not a feature.

2. **It is finite.** Two nudges, then silence. Software that keeps tapping someone on the
   shoulder is worse than software that waits, and the second nudge explicitly says it will
   stop so the silence afterwards reads as deliberate rather than as another failure.

3. **It says something.** "Are you still there?" is noise — it asks the customer to do the
   work of restarting the conversation. Every nudge here carries the actual next step, taken
   from the open question if there is one, and from what they were just doing if there isn't.
"""
from __future__ import annotations

import random
import time

from . import conversation

#: Openers, so the second nudge is not the first one again. Repeating a sentence verbatim is
#: the single clearest signal that nothing is actually paying attention — which is the exact
#: impression this whole feature exists to remove.
_OPENER = {
    1: ("Still there?", "No rush —", "Whenever you're ready —"),
    2: ("Last nudge from me:", "One more from me:", "Before I leave you to it:"),
}


def _open(n: int) -> str:
    return random.choice(_OPENER.get(n, _OPENER[1]))

#: When to speak. The first is short enough to feel attentive and long enough that a customer
#: reading the previous answer is not interrupted mid-sentence; the second is a last call.
FIRST = 45.0
SECOND = 150.0
MAX = 2

#: session -> (how many nudges sent, when the last one went)
_SENT: dict[str, tuple[int, float]] = {}


def reset(session: str) -> None:
    """The customer spoke. Everything starts again."""
    _SENT.pop(session, None)


def _describe(p: dict) -> str:
    """One phrase for whatever is on offer, so the nudge names it rather than saying "it"."""
    kind = p.get("kind")
    if kind == "split":
        who = ", ".join(x["name"].split()[0] for x in p.get("participants") or [])
        return f"The split of {p.get('currency', 'AED')} {p.get('total', 0):g} with {who}"
    if kind == "pay_merchant":
        return f"The {p.get('currency', 'AED')} {p.get('payable', p.get('amount', 0)):g} " \
               f"to {p.get('merchant', 'them')}"
    if kind == "send":
        return f"The {p.get('currency', 'AED')} {p.get('amount', 0):g} to " \
               f"{p.get('payee', 'them')}"
    return "That"


def _re_ask(frame, n: int) -> tuple[str, list[dict]] | None:
    """We asked a question and got nothing back. Ask it again, more helpfully."""
    asked, intent = frame.asked_for, frame.intent
    payee = frame.slots.get("payee") or frame.slots.get("merchant")
    o = _open(n)
    if asked == "amount" and payee:
        return (f"{o} just tell me how much to send {payee} and I'll get it ready — or say "
                f"cancel and we'll leave it." if n == 1 else
                f"{o} how much should {payee} get? Nothing has left your wallet.",
                [{"kind": "ask_text", "label": "Cancel that", "text": "cancel"}])
    if asked in ("payee", "merchant"):
        return (f"{o} tell me who it's going to — someone you've paid before, or a merchant "
                f"near you.",
                [{"kind": "ask_text", "label": "Show me who I can pay",
                  "text": "who can i send money to"},
                 {"kind": "ask_text", "label": "Offers near me", "text": "any offers near me"}])
    if asked == "txn_ref":
        return (f"{o} if you're not sure which payment you mean, I can list your recent ones "
                f"and you can point at it.",
                [{"kind": "ask_text", "label": "Show my last 5",
                  "text": "show me my last 5 transactions"}])
    if intent:
        return (f"{o} we were in the middle of {intent.replace('_', ' ')} — happy to carry on "
                f"whenever you are.", [])
    return None


def _next_step(topic: str | None) -> tuple[str, list[dict]] | None:
    """No open question, but we know what they were doing. Offer the natural next thing."""
    if topic == "txn_history":
        return ("Anything there you want me to look into? I can open any one of them, or "
                "show you older ones.",
                [{"kind": "ask_text", "label": "Show me older",
                  "text": "show me more transactions"},
                 {"kind": "ask_text", "label": "Only the failed ones",
                  "text": "show me my failed transactions"}])
    if topic in ("txn_status", "txn_failed", "debited_not_credited", "txn_on_hold"):
        return ("Does that answer it? If the money still hasn't turned up I can raise a "
                "refund for you.",
                [{"kind": "ask_text", "label": "Raise a refund", "text": "i want a refund"},
                 {"kind": "ask_text", "label": "Show my other payments",
                  "text": "show me my last 5 transactions"}])
    if topic in ("refund_request", "refund_status"):
        return ("I'll keep an eye on that refund. Anything else while you're here?",
                [{"kind": "ask_text", "label": "My transactions",
                  "text": "show me my last 5 transactions"}])
    if topic == "offers_nearby":
        return ("Want me to work out what you'd actually pay at any of those?",
                [{"kind": "ask_text", "label": "Which is nearest", "text": "which is nearest"},
                 {"kind": "ask_text", "label": "Best offer", "text": "which has the best offer"}])
    if topic in ("split_payment", "split_status"):
        return ("Anything else on your splits?",
                [{"kind": "ask_text", "label": "Who owes me?", "text": "who owes me money"},
                 {"kind": "ask_text", "label": "Split my last payment",
                  "text": "split my last payment"}])
    if topic in ("send_money", "pay_merchant"):
        return ("Still want to send that? Nothing has left your wallet.",
                [{"kind": "ask_text", "label": "Yes, carry on", "text": "yes"},
                 {"kind": "ask_text", "label": "Not now", "text": "cancel"}])
    return None


def due(session: str, idle: float) -> dict | None:
    """What to say after `idle` seconds of silence, or None to stay quiet.

    Returns the same shape as an answer so the page renders it as an ordinary assistant
    message — a nudge is not a different kind of thing, and dressing it up as a system notice
    would make it read like an error.
    """
    count, _ = _SENT.get(session, (0, 0.0))
    if count >= MAX:
        return None
    threshold = FIRST if count == 0 else SECOND
    if idle < threshold:
        return None

    # The most useful thing to say after a merchant payment is "want to split that?" —
    # asked once, at the only moment everyone who shared the bill is still together.
    offer = conversation.split_offer(session)
    if offer and count == 0:
        conversation.mark_split_offered(session)
        who = (offer["merchant"] or "").split()[0]
        _SENT[session] = (count + 1, time.time())
        return {"say": f"Was that {offer['amount']:g} at {who} shared with anyone? I can "
                       f"split it and ask them for their share.",
                "options": [{"kind": "ask_text", "label": "Split it",
                             "text": f"split my payment to {offer['merchant']} with"},
                            {"kind": "dismiss", "label": "No, it was just me"}],
                "nudge": True, "n": count + 1}

    # Something is on offer and we are waiting for a decision — ask for the DECISION. The
    # slot-shaped re-ask ("how much should I send Alex?") is for a half-filled request, and
    # using it here asked about a figure the customer had already been shown.
    held = conversation.proposal(session)
    if held:
        o = _open(count + 1)
        _SENT[session] = (count + 1, time.time())
        what = _describe(held)
        say = (f"{o} {what} is still waiting on your go-ahead. Nothing has left your wallet."
               if count == 0 else
               f"{o} say yes and I'll do it, or no and I'll drop it. Nothing has moved.")
        if count + 1 >= MAX:
            say += " I'll leave it there — just type whenever."
        return {"say": say, "nudge": True, "n": count + 1,
                "options": [{"kind": "ask_text", "label": "Yes, go ahead", "text": "yes"},
                            {"kind": "ask_text", "label": "No, cancel", "text": "no"}]}

    frame = conversation.get(session)
    out = _re_ask(frame, count + 1) if frame else None
    if not out:
        out = _next_step(conversation.topic(session))
    if not out and count == 0:
        # Nothing in flight at all. Offer something real rather than a greeting repeated.
        out = ("Take your time. When you're ready I can show you your recent payments, "
               "what a transfer costs, or what's on offer near you.",
               [{"kind": "ask_text", "label": "My last 5 payments",
                 "text": "show me my last 5 transactions"},
                {"kind": "ask_text", "label": "Offers near me", "text": "any offers near me"}])
    if not out:
        return None

    say, options = out
    if count + 1 >= MAX:
        say += " I'll leave it there — just type whenever."
    _SENT[session] = (count + 1, time.time())
    return {"say": say, "options": options, "nudge": True, "n": count + 1}
