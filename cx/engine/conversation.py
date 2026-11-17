"""Short-term conversation memory, so a follow-up answer is understood as an answer.

Without this the engine is stateless and every utterance is judged alone, which breaks the
most ordinary exchange there is:

    "pay Asad and Sons"        -> "How much at Asad and Sons?"
    "100 dirhams"              -> "I think you want to send money, but I'm not certain
                                   enough to set it up. Tell me who and how much."

The second reply is nonsense: we had just asked the question it answers. On its own "100
dirhams" names nobody, so the classifier scores a weak `send_money` and the money-confidence
guard — correctly — refuses to act on it. The guard is not the bug. The bug is that the
sentence should never have been classified from scratch: it is a **slot filling an open
question**, and the merchant was already known.

So we keep one open frame per session: the intent we asked about and the slots already
gathered. When the next utterance supplies a missing slot, we merge and resume that intent
instead of re-classifying.

Three limits, because conversational state is also how assistants do surprising things:

* **It expires** (`TTL`). An amount said four minutes later is not an answer to a forgotten
  question.
* **It only ever fills slots.** A follow-up can complete a pending request; it can never
  change what the request *is*. "Pay Asad" then "actually, refund me" starts fresh, because
  the new utterance classifies strongly on its own.
* **It never survives an answer.** Once the frame is complete and acted on, it is cleared,
  so a stray "100" afterwards cannot re-trigger a payment.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

#: How long an open question stays open. 180s was far too short: a person typing a reply in a
#: chat easily takes longer, and when the frame expired "106" stopped being the answer to
#: "how much?" and became a fresh, payee-less payment request — which is exactly the reported
#: "I answered and it asked me again".
TTL = 1800.0         # 30 minutes


@dataclass
class Frame:
    intent: str
    slots: dict = field(default_factory=dict)
    asked_for: str | None = None        # which slot we were waiting on
    at: float = field(default_factory=time.time)

    @property
    def fresh(self) -> bool:
        return (time.time() - self.at) < TTL


_FRAMES: dict[str, Frame] = {}

#: What this session has already DONE, as opposed to what it is part-way through. A payment
#: made two messages ago is the single most likely subject of "I want a refund", and before
#: this the engine reached past it for an older, more broken-looking one.
_DONE: dict[str, dict] = {}


def remember_payment(session: str, ref: str, payee: str = "", amount: float = 0.0) -> None:
    _DONE[session] = {"ref": ref, "payee": payee, "amount": amount, "at": time.time()}


def last_payment(session: str) -> dict | None:
    rec = _DONE.get(session)
    if not rec or (time.time() - rec["at"]) > TTL:
        return None
    return rec


#: The last intent this session actually ANSWERED, with a timestamp. Distinct from the open
#: frame: a frame means "we are waiting for something", this means "this is what we were just
#: talking about". A completed answer clears the frame — correctly, so a stray number cannot
#: reopen a payment — but the SUBJECT survives, and follow-ups need the subject.
_TOPIC: dict[str, tuple[str, float]] = {}


#: The language this session is being conducted in, so surfaces without an Answer — the PIN
#: sheet, the server's own messages — answer in the same language as the conversation.
_LANG: dict[str, str] = {}


def note_lang(session: str, lang: str) -> None:
    if lang and lang != "en":
        _LANG[session] = lang
    elif lang == "en":
        _LANG.pop(session, None)


def last_lang(session: str) -> str:
    return _LANG.get(session, "en")


#: What the conversation is ABOUT, as opposed to what we last asked or what intent we last
#: ran. A bill, a payment, a person — whatever the customer and the assistant have jointly
#: settled on.
#:
#: This is the thing that was missing. The frame remembers an open QUESTION and clears once
#: answered; the topic remembers the last INTENT. Neither remembers the subject, so a
#: customer who was quoted "AED 506.00 for Home electricity" and then said "split kardo" was
#: starting from nothing: no bill named in that utterance, no bill in the frame, and the
#: router fell through to splitting an unrelated merchant payment. A person would have
#: remembered which bill they were just discussing.
_SUBJECT: dict[str, dict[str, tuple[object, float]]] = {}

#: How long a subject stays current. Long enough to cover a real conversation with pauses,
#: short enough that tomorrow's "split it" does not reach back to yesterday's bill.
SUBJECT_TTL = 1800.0


def note_subject(session: str, kind: str, value) -> None:
    """Remember what this conversation is about. Called when a handler RESOLVES something —
    a specific bill, a specific payment — not when it merely guesses."""
    if not session or value in (None, "", []):
        return
    _SUBJECT.setdefault(session, {})[kind] = (value, time.time())


def subject(session: str, kind: str):
    """The current subject of this kind, or None if there is none or it has gone stale."""
    rec = (_SUBJECT.get(session) or {}).get(kind)
    if not rec or (time.time() - rec[1]) > SUBJECT_TTL:
        return None
    return rec[0]


def subjects(session: str) -> dict:
    """Everything still current, for filling gaps in a new utterance."""
    now = time.time()
    return {k: v for k, (v, at) in (_SUBJECT.get(session) or {}).items()
            if (now - at) <= SUBJECT_TTL}


def forget_subject(session: str, kind: str) -> None:
    (_SUBJECT.get(session) or {}).pop(kind, None)


#: Abuse in this conversation: (count, time of the latest). Policy: the first abusive
#: message that carries a real question is answered, with a boundary; any abuse after that
#: ends the conversation — nothing more is answered until it lapses (TTL after the last
#: abusive message) or a new session starts.
_ABUSE: dict[str, tuple[int, float]] = {}


def note_abuse(session: str) -> int:
    """Record an abusive message; returns how many this conversation has had, this one included."""
    n, _ = _ABUSE.get(session) or (0, 0.0)
    if _ABUSE.get(session) and time.time() - _ABUSE[session][1] > TTL:
        n = 0
    _ABUSE[session] = (n + 1, time.time())
    return n + 1


def abuse_stopped(session: str) -> bool:
    """True once the conversation has been ended for repeated abuse, until it lapses."""
    rec = _ABUSE.get(session)
    return bool(rec and rec[0] >= 2 and time.time() - rec[1] <= TTL)


def note_topic(session: str, intent: str) -> None:
    _TOPIC[session] = (intent, time.time())


def topic(session: str) -> str | None:
    rec = _TOPIC.get(session)
    if not rec or (time.time() - rec[1]) > TTL:
        return None
    return rec[0]


#: A merchant payment this session just made that has not been offered for splitting yet.
#: Separate from `_DONE` because it is consumed once: the offer is made, and whether they say
#: yes or ignore it, it is not made again for the same payment. An assistant that asks "want
#: to split that?" after every message about the same bill is nagging.
_SPLIT_OFFER: dict[str, dict] = {}


def offer_split(session: str, ref: str, merchant: str, amount: float) -> None:
    _SPLIT_OFFER[session] = {"ref": ref, "merchant": merchant, "amount": amount,
                             "at": time.time(), "offered": False}


def split_offer(session: str) -> dict | None:
    """The pending offer, if there is one and it is still fresh. Does not consume it."""
    rec = _SPLIT_OFFER.get(session)
    if not rec or rec["offered"] or (time.time() - rec["at"]) > 600:
        return None
    return rec


def mark_split_offered(session: str) -> None:
    if session in _SPLIT_OFFER:
        _SPLIT_OFFER[session]["offered"] = True


#: The last thing we OFFERED to do, held so that a bare "yes" can mean it.
#:
#: Without this the proposal existed only inside one HTTP response. The customer saw
#: "Splitting the AED 80 with Alex", said "yes", and the classifier — with nothing to go on
#: but the word "yes" — read it as a greeting and replied "Of course. I handle anything to do
#: with your payments." The offer had evaporated between two messages.
_PROPOSAL: dict[str, tuple[dict, float]] = {}
PROPOSAL_TTL = 900.0


def hold_proposal(session: str, proposal: dict | None) -> None:
    if proposal:
        _PROPOSAL[session] = (proposal, time.time())
    else:
        _PROPOSAL.pop(session, None)


def proposal(session: str) -> dict | None:
    rec = _PROPOSAL.get(session)
    if not rec or (time.time() - rec[1]) > PROPOSAL_TTL:
        _PROPOSAL.pop(session, None)
        return None
    return rec[0]


def drop_proposal(session: str) -> None:
    _PROPOSAL.pop(session, None)


def get(session: str) -> Frame | None:
    f = _FRAMES.get(session)
    if f and not f.fresh:
        _FRAMES.pop(session, None)
        return None
    return f


def hold(session: str, intent: str, slots: dict, asked_for: str | None) -> None:
    """Remember that we asked a question, and what we already knew when we asked it."""
    _FRAMES[session] = Frame(intent=intent,
                             slots={k: v for k, v in slots.items() if v not in (None, "")},
                             asked_for=asked_for)


def clear(session: str) -> None:
    _FRAMES.pop(session, None)


def merge(session: str, new_slots: dict) -> tuple[str, dict] | None:
    """If a question is open and this utterance helps answer it, return (intent, merged slots).

    "Helps answer it" is deliberately strict: the utterance must supply the slot we actually
    asked for, or at least one slot the pending intent still needs. A reply that contributes
    nothing relevant is not treated as a continuation, so unrelated chatter cannot creep into
    an open payment.
    """
    f = get(session)
    if not f:
        return None
    useful = {k: v for k, v in new_slots.items() if v not in (None, "")}
    if not useful:
        return None
    if f.asked_for and f.asked_for not in useful:
        # They answered something, but not what we asked. Only continue if it is still a slot
        # this intent cares about; otherwise let the normal classifier take it.
        if not (set(useful) & set(f.slots.keys() | {"amount", "payee", "merchant", "txn_ref"})):
            return None
    merged = dict(f.slots)
    merged.update(useful)
    return f.intent, merged

# ── the task ledger ──────────────────────────────────────────────────────────────────────
#
# A frame is what the customer is part-way through RIGHT NOW. It is one slot, it expires in
# half an hour, and starting anything else destroys it. That is correct for finishing a
# sentence and useless for the way people actually talk:
#
#     "pay my electricity bill"        -> 506 due
#     "split it with Alex"            -> asked him
#     "what are my offers"             -> ... different subject entirely
#     "send 50 to Asad"                -> ... and another
#     "alex ne paise bhej diye hai?"  -> WHICH Alex thing? The split, from four turns ago.
#
# The frame is long gone by the last line, and a single "subject" slot has been overwritten
# twice. So alongside the frame we keep a LEDGER of the things this session has started:
# what kind, what it was about, where it got to. A reference can then reach back past any
# number of topic changes, which is the whole difference between a form and a conversation.
#
# Tasks are not the frame and must not become it. The ledger says what exists and how it
# ended; it never resumes a payment or fills a slot on its own. Reaching back into an old
# task re-opens the SUBJECT, and the customer still says what they want done.

#: Tasks outlive frames by a long way — the point is to still be there after the
#: conversation has been somewhere else. Six hours, and session-scoped regardless.
TASK_TTL = 6 * 3600.0

#: How much of a reference has to match before we believe it points at a given task.
#: Below this we ask instead of guessing: picking the wrong past task is worse than
#: admitting the reference was ambiguous, because the customer will not re-read the answer
#: closely enough to notice we answered about a different bill.
_MATCH_FLOOR = 0.34


@dataclass
class Task:
    id: str
    kind: str                       # bill_split | split | send | pay_merchant | bill_pay …
    label: str                      # "Home electricity split with Alex"
    key: dict = field(default_factory=dict)     # split_id, bill, payee, amount, ref …
    status: str = "open"            # open | waiting | done | failed | cancelled
    at: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    note: str = ""

    @property
    def fresh(self) -> bool:
        return (time.time() - self.touched) < TASK_TTL

    @property
    def live(self) -> bool:
        return self.status in ("open", "waiting")

    def words(self) -> set[str]:
        """Everything this task could be referred to by."""
        bits = [self.label, self.kind.replace("_", " "), self.note]
        for v in self.key.values():
            if isinstance(v, str):
                bits.append(v)
            elif isinstance(v, (list, tuple)):
                bits += [str(x) for x in v]
        return {t for t in _TOKENS.split(" ".join(bits).lower()) if len(t) > 2}


_TASKS: dict[str, list[Task]] = {}
_TOKENS = re.compile(r"[^\wऀ-ॿ؀-ۿ]+")

#: Words that point BACK at something without naming it. On their own they are not enough
#: to choose a task, but they say that a reference is being made at all.
#: Deliberately excludes "the", "it", "this", "jo", "tha" and "thi". They were in this set
#: and "what is the weather" matched an open bill split — one live task, the word "the",
#: and the fallback below fired. A word that appears in most English sentences is not
#: somebody pointing at an earlier task.
_ANAPHORA = {
    "that", "same", "earlier", "before", "previous", "last", "old",
    "wo", "woh", "vo", "usko", "uska", "uske", "wala", "wali", "waali",
    "pehle", "pehla", "pichla", "pichle", "purana", "purani",
    "ذلك", "السابق", "نفس",
}

#: A bare back-reference is SHORT. "what about that one" is pointing at something; a long
#: sentence that happens to contain "last" is talking about something of its own.
_ANAPHORA_MAX_WORDS = 6


def open_task(session: str, kind: str, label: str, key: dict | None = None,
              status: str = "open") -> Task:
    """Record that this session has started something. Returns the task.

    If a live task of the same kind already covers the same subject, that one is updated
    rather than duplicated — four collections for one electricity bill is exactly the
    shape of bug this prevents at the conversation layer too.
    """
    key = dict(key or {})
    lst = _TASKS.setdefault(session, [])
    for t in lst:
        if t.kind == kind and t.live and t.fresh and _same_subject(t.key, key):
            t.label, t.touched = label or t.label, time.time()
            t.key.update(key)
            t.status = status
            return t
    t = Task(id=f"T{len(lst) + 1}", kind=kind, label=label, key=key, status=status)
    lst.append(t)
    del lst[:-20]                   # a session does not need more history than this
    return t


def _same_subject(a: dict, b: dict) -> bool:
    for k in ("split_id", "ref", "bill", "identifier", "payee", "merchant"):
        if a.get(k) and b.get(k):
            return str(a[k]).lower() == str(b[k]).lower()
    return False


def tasks(session: str, kind: str | None = None, live_only: bool = False) -> list[Task]:
    """This session's tasks, most recently touched first."""
    out = [t for t in _TASKS.get(session, [])
           if t.fresh and (kind is None or t.kind == kind)
           and (not live_only or t.live)]
    return sorted(out, key=lambda t: t.touched, reverse=True)


def update_task(session: str, task_id: str, status: str | None = None,
                note: str | None = None, **key) -> Task | None:
    for t in _TASKS.get(session, []):
        if t.id == task_id:
            if status:
                t.status = status
            if note is not None:
                t.note = note
            t.key.update({k: v for k, v in key.items() if v is not None})
            t.touched = time.time()
            return t
    return None


def find_task(session: str, said: str, kind: str | None = None) -> Task | None:
    """Which earlier task is this utterance talking about? None if it is not clear.

    Scored on the words the utterance shares with the task — the bill's name, the person's
    name, the kind of thing it was — with a nudge for recency and for still being open. A
    bare "wo wala" matches nothing on content, so it falls back to the most recent live
    task, which is what "that one" means when only one thing is outstanding.
    """
    pool = tasks(session, kind)
    if not pool:
        return None
    words = {t for t in _TOKENS.split((said or "").lower()) if len(t) > 2}
    if not words:
        return None

    best, best_score = None, 0.0
    now = time.time()
    for t in pool:
        tw = t.words()
        if not tw:
            continue
        hits = words & tw
        if not hits:
            continue
        # Proportion of the TASK's identity that was named, so naming the bill counts for
        # more than sharing one common word with a long label.
        score = len(hits) / max(3.0, len(tw) ** 0.5 + 2)
        score += 0.12 if t.live else 0.0
        score += max(0.0, 0.15 * (1 - (now - t.touched) / TASK_TTL))
        if score > best_score:
            best, best_score = t, score

    if best and best_score >= _MATCH_FLOOR:
        best.touched = now
        return best

    # No content match. If the utterance is clearly pointing back ("wo wala", "that one")
    # and exactly one thing is outstanding, that is what it means.
    if (words & _ANAPHORA) and len(said.split()) <= _ANAPHORA_MAX_WORDS:
        live = [t for t in pool if t.live]
        if len(live) == 1:
            live[0].touched = now
            return live[0]
    return None


def describe_tasks(session: str) -> str:
    """One line per live task, for when we have to ask which one they mean."""
    live = [t for t in tasks(session, live_only=True)]
    return "; ".join(t.label for t in live[:4])


# ---------------------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------------------
#: What we have ASSERTED in this conversation, newest last. Kept so that a customer saying
#: "that can't be right" has something concrete to be right or wrong about: without it the
#: only possible reply to a challenge is to repeat the sentence they are challenging.
#:
#: Deliberately small. A claim is a figure plus the working behind it, not a transcript —
#: `recheck` re-derives everything else from the ledger rather than trusting what is here,
#: because the point of a recheck is to not trust what is here.
_CLAIMS: dict[str, list[dict]] = {}
CLAIMS_KEPT = 6


def note_claim(session: str, claim: dict) -> None:
    if not claim:
        return
    got = _CLAIMS.setdefault(session, [])
    got.append(claim)
    del got[:-CLAIMS_KEPT]


def claims(session: str) -> list[dict]:
    """Everything still standing, oldest first."""
    import recheck
    got = recheck.live(_CLAIMS.get(session, []))
    _CLAIMS[session] = got
    return got


def last_claim(session: str) -> dict | None:
    got = claims(session)
    return got[-1] if got else None


def note_correction(session: str, claim: dict, now: float) -> None:
    """Replace a claim we have just had to correct, so the corrected figure is what any
    later challenge is measured against — not the number we already withdrew."""
    for c in _CLAIMS.get(session, []):
        if c is claim:
            c["total"] = now
            c["corrected"] = True
            return
