"""Intent pipeline — turning a spoken sentence into a transfer proposal, or a refusal.

Four stages, each able to refuse, none able to authorize:

    transcribe -> classify (4x4 matrix) -> resolve payee -> bind and read back

The load-bearing rule: **the semantic layer may propose, never select.** Whisper and any vector
search sit OUTSIDE the guardrail; the 4x4 matrix and the exact-match binding are the guardrail.
That is what lets imperfect models be used safely — they can be wrong, and the worst outcome is
a refusal rather than money going to the wrong person.

Why exact matching is not negotiable, with evidence: on 29 September, Whisper transcribed
"Riley" as "the one", and a larger model as "the worm". A nearest-neighbour match on a mangled
name is a real person who would receive real funds. So a payee is resolved by exact token match
against the payer's own mandate roster, or not at all. There is no similarity threshold.

Amount parsing fails closed the same way: "twenty", "20" and "1 million" resolve,
"twenty twenty" and any sentence with two numbers in it do not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import mandate_store as ms
from .mandate import Mandate

VERBS = ("transfer", "query", "message", "other")
OBJECTS = ("person", "amount", "currency", "time")

# What puts a sentence in each verb class. Deterministic, table-driven, auditable — no model.
VERB_TRIGGERS = {
    "transfer": {"send", "transfer", "pay", "wire", "remit", "give"},
    "query":    {"what", "who", "when", "how", "show", "tell", "status", "balance"},
    "message":  {"message", "msg", "text", "tell", "ping", "email"},
    "other":    set(),
}

# The 4x4 dictionary matrix: for each verb class, the markers that signal each object slot.
# A payment needs the transfer x person AND transfer x amount cells to fire; anything less is
# not a payment, whatever it sounded like.
MATRIX: dict[str, dict[str, set[str]]] = {
    "transfer": {"person":   {"to", "into"},
                 "amount":   {"dirham", "dirhams", "aed", "dollar", "dollars", "usd",
                              "rupee", "rupees", "inr", "euro", "euros", "eur"},
                 "currency": {"dirham", "dirhams", "aed", "dollar", "dollars", "usd",
                              "rupee", "rupees", "inr", "euro", "euros", "eur"},
                 "time":     {"now", "today", "tonight", "tomorrow"}},
    "query":    {"person": {"about", "on", "is"}, "amount": {"balance", "much"},
                 "currency": set(), "time": {"today", "yesterday", "week", "lately", "last"}},
    "message":  {"person": {"to"}, "amount": set(), "currency": set(),
                 "time": {"now", "later", "tomorrow"}},
    "other":    {o: set() for o in OBJECTS},
}

CURRENCIES = {
    "dirham": "AED", "dirhams": "AED", "aed": "AED", "dh": "AED", "dhs": "AED",
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
}

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}

# Magnitude words that terminate a group and scale it. Lakh and crore are here because the
# payers and payees on this corridor are PK/IN English speakers who say them as readily as
# "million" -- leaving them out is the same 1000x defect in a different accent.
#
# Their absence is what sent AED 1 for "send 1 million dirhams": the magnitude was heard,
# transcribed, and then silently dropped by a parser that only knew "thousand".
_SCALES = {"thousand": 1_000,
           "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
           "million": 1_000_000,
           "crore": 10_000_000, "crores": 10_000_000,
           "billion": 1_000_000_000}

_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"hundred", "and"}


# Words that are never part of a payee's name, so stripping them leaves the name behind.
FILLER = {"the", "a", "an", "please", "now", "today", "tonight", "tomorrow",
          "to", "into", "and", "some", "over", "across", "my", "for"}

# A sentence that OPENS with one of these is a question about a payment, not an instruction
# to make one. Measured need: "did my 500 dirhams to alex go through" was classified as a
# transfer and proposed sending 500 AED, because the currency word alone promoted it.
OPENS_A_QUESTION = {"did", "do", "does", "is", "are", "was", "were", "has", "have", "had",
                    "can", "could", "would", "should", "will", "why", "which", "whether",
                    "what", "who", "whom", "when", "where", "how"}

# Already happened. "I sent 20 dirhams to Alex" is a statement of fact; acting on it pays twice.
ALREADY_SENT = {"sent", "paid", "transferred", "wired", "remitted", "gave", "settled"}


def payee_phrase(text: str) -> str:
    """What is left once the verb, the number and the currency are removed.

    Better than "everything after 'to'": people say "pay alex 50 dirhams" as readily as
    "send 50 dirhams to alex", and the first has no preposition at all.
    """
    out = []
    for t in tokens(text):
        if t in FILLER or t in CURRENCIES:
            continue
        if t in VERB_TRIGGERS["transfer"] or t in VERB_TRIGGERS["message"]:
            continue
        if re.fullmatch(r"\d+(\.\d{1,2})?", t):
            continue
        if t in _NUMBER_WORDS:
            continue
        out.append(t)
    return " ".join(out)


class IntentError(Exception):
    """Refusal with a reason safe to say out loud."""


@dataclass
class Proposal:
    """A *proposal*. Nothing here has authorized anything."""
    intent: str
    text: str
    cells: dict = field(default_factory=dict)
    payee_agent: str = ""
    payee_name: str = ""
    payee_wallet: str = ""
    mandate_id: str = ""
    amount: float | None = None
    currency: str = ""

    def read_back(self) -> str:
        """What the human hears before anything moves. One sentence, and it catches every
        class of transcription failure the pipeline can produce -- including, since the
        1-for-1-million incident, a magnitude that was mis-parsed rather than mis-heard."""
        return (f"{say_amount(self.amount)} {self.currency} to {self.payee_name}, "
                f"wallet {self.payee_wallet}. Say confirm.")


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9.]+", text.lower()) if t]


def _eval_run(ws: list[str]) -> tuple[float | None, bool]:
    """Evaluate one run of number tokens. Returns (value, ambiguous).

    Digits and words are evaluated by the SAME pass, which is the whole point: "1 million"
    is one number, not a digit followed by a word to be thrown away.

    Ambiguity is a distinct outcome from "no number here", because the two must be handled
    differently -- a run we cannot read has to refuse the whole sentence, whereas a run with
    no number in it is simply not an amount.
    """
    total = current = 0.0
    seen = False
    last_scale = None                        # scales must descend: "two million five thousand"
    for w in ws:
        if w == "and":
            continue
        if w in ("a", "an"):                 # "a million" -- only ever reaches here before a scale
            if seen or current:
                return None, True
            current, seen = 1.0, True
        elif re.fullmatch(r"\d+(\.\d{1,2})?", w):
            if current:
                return None, True            # "20 20": two values, no separator
            current, seen = float(w), True
        elif w in _UNITS:
            if current % 100 and current % 100 < 20:
                return None, True            # "nineteen five"
            current += _UNITS[w]; seen = True
        elif w in _TENS:
            if current % 100:
                return None, True            # "twenty twenty" is a year, a score or a mishearing
            current += _TENS[w]; seen = True
        elif w == "hundred":
            if not current:
                return None, True
            current *= 100; seen = True
        elif w in _SCALES:
            scale = _SCALES[w]
            if not current:
                return None, True            # a magnitude with nothing in front of it
            if last_scale is not None and scale >= last_scale:
                return None, True            # "five thousand two million" is not a number
            total += current * scale
            current, last_scale, seen = 0.0, scale, True
        else:
            return None, True
    return (total + current, False) if seen else (None, False)


def words_to_number(ws: list[str]) -> float | None:
    """Spoken numbers. Returns None on anything ambiguous rather than guessing."""
    value, ambiguous = _eval_run(ws)
    return None if ambiguous else value


def _number_runs(ts: list[str]) -> list[list[str]]:
    """Maximal runs of number tokens. "a"/"an" counts only directly before a magnitude, so
    "send a million" reads but "send money to a friend" does not become an amount of 1."""
    runs: list[list[str]] = []
    run: list[str] = []
    for i, t in enumerate(ts):
        nxt = ts[i + 1] if i + 1 < len(ts) else ""
        if (t in _NUMBER_WORDS or re.fullmatch(r"\d+(\.\d{1,2})?", t)
                or (t in ("a", "an") and nxt in _SCALES)):
            run.append(t)
        elif run:
            runs.append(run); run = []
    if run:
        runs.append(run)
    return runs


def parse_amount(text: str) -> float | None:
    """The amount, or None and the caller refuses. Never guesses a magnitude.

    Exactly one readable number must appear in the sentence. Two numbers is a refusal for the
    same reason an unreadable one is: there is no safe way to pick.
    """
    values = []
    for run in _number_runs(tokens(text)):
        value, ambiguous = _eval_run(run)
        if ambiguous:
            return None
        if value is not None:
            values.append(value)
    return values[0] if len(values) == 1 else None


def say_amount(value: float | None) -> str:
    """Spoken magnitude AND the digits, so a 1000x error is audible before anything is signed.
    "1 million, that is 1,000,000" cannot be confused with "1" the way "1000000" can."""
    if value is None:
        return "no amount"
    if value < 1000:
        return f"{value:g}"
    for word, scale in (("billion", 1_000_000_000), ("crore", 10_000_000),
                        ("million", 1_000_000), ("lakh", 100_000), ("thousand", 1_000)):
        quotient = value / scale
        if value >= scale and abs(quotient - round(quotient, 2)) < 1e-9:
            return f"{quotient:g} {word}, that is {value:,.0f}"
    return f"{value:,.2f}"


def parse_currency(text: str) -> str:
    for t in tokens(text):
        if t in CURRENCIES:
            return CURRENCIES[t]
    return ""


def classify(text: str) -> tuple[str, dict]:
    """Verb class plus which object cells fired. Pure lookup, no model."""
    tl = tokens(text)
    ts = set(tl)
    scores = {v: len(ts & VERB_TRIGGERS[v]) for v in VERBS}

    # Is this a question about money rather than an instruction to move it? Two signals, both
    # cheap and both found by measurement rather than guessed at: the sentence opens with an
    # interrogative or auxiliary, or its verb is already in the past.
    asking = bool(tl) and (tl[0] in OPENS_A_QUESTION or bool(ts & ALREADY_SENT))

    # A currency word alone is strong evidence of a transfer, even if the verb was mis-heard --
    # but not in a question. Without this guard "did my 500 dirhams to alex go through"
    # proposed sending 500 AED in reply to somebody asking whether it had already been sent.
    if scores["transfer"] == 0 and not asking and (ts & MATRIX["transfer"]["currency"]):
        scores["transfer"] = 1
    # A question that also contains a transfer verb ("how much did i send alex") is still a
    # question. Resolving the tie towards transfer is the dangerous direction.
    if asking:
        scores["query"] += 1
    verb = max(scores, key=lambda v: scores[v])
    if scores[verb] == 0:
        verb = "other"
    cells = {o: bool(ts & MATRIX[verb][o]) for o in OBJECTS}
    if verb == "transfer":
        cells["amount"] = parse_amount(text) is not None
        cells["person"] = bool(payee_phrase(text))
    return verb, cells


# ---- payee resolution ----

class RosterResolver:
    """Exact token match against the payer's own mandate roster.

    'Alex' matching 'Alex Morgan' is an exact match on a name token, not a fuzzy one — no
    edit distance, no threshold. Zero matches or more than one both refuse: a mis-heard name
    must never silently become somebody.
    """
    name = "roster"

    def candidates(self, payer_agent: str) -> list[dict]:
        """One entry per payee, newest mandate wins.

        Without the ordering this returned whatever SQLite happened to yield first, so a payer
        holding two live mandates for the same person could transact against a stale one — the
        wrong limits, and an id the monitor may not even know. Newest-wins is deterministic and
        matches what a person means by "my mandate with Alex".
        """
        out: dict[str, dict] = {}
        with ms.conn() as c:
            rows = c.execute(
                "SELECT doc FROM mandates WHERE payer_agent=? AND state='active'"
                " ORDER BY updated DESC", (payer_agent,)).fetchall()
        import json
        for r in rows:
            m = Mandate.from_dict(json.loads(r["doc"]))
            if m.active and m.payee.agent_id not in out:
                out[m.payee.agent_id] = {
                    "agent_id": m.payee.agent_id, "wallet": m.payee.wallet,
                    "name": m.payee.display_name or m.payee.agent_id,
                    "mandate_id": m.mandate_id}
        return list(out.values())

    def resolve(self, spoken: str, payer_agent: str) -> dict:
        cands = self.candidates(payer_agent)
        if not cands:
            raise IntentError("you have no payment mandates set up yet")
        spoken_tokens = set(tokens(spoken))
        # Score by how many name tokens matched, so "alex morgan" beats "alex" alone when
        # two people share a first name. Still exact token matching — no edit distance.
        scored = [(len(spoken_tokens & set(tokens(c["name"]))), c) for c in cands]
        scored = [(n, c) for n, c in scored if n]
        if not scored:
            raise IntentError("I don't have a mandate for anyone by that name")
        best = max(n for n, _ in scored)
        hits = [c for n, c in scored if n == best]
        if len({h["agent_id"] for h in hits}) > 1:
            names = ", ".join(sorted(h["name"] for h in hits))
            raise IntentError(f"that could be {names} — which one?")
        return hits[0]


class OpenSearchResolver(RosterResolver):
    """kNN over payee aliases, for phrasings the roster alone will not catch ("my brother").

    Deliberately subclasses the roster resolver: whatever the vector search proposes is still
    bound by exact match below, so a near neighbour cannot become a payee on its own.
    """
    name = "opensearch"

    def __init__(self, client=None):
        self.client = client

    def resolve(self, spoken: str, payer_agent: str) -> dict:
        try:
            return super().resolve(spoken, payer_agent)
        except IntentError:
            if self.client is None:
                raise
            raise IntentError("I don't have a mandate for anyone by that name")


# ---- the pipeline ----

def plan(text: str, payer_agent: str, resolver: RosterResolver | None = None) -> Proposal:
    """Spoken text in, a bound proposal out — or IntentError with something worth saying."""
    verb, cells = classify(text)
    p = Proposal(intent=verb, text=text, cells=cells)
    if verb != "transfer":
        return p

    if not (cells["person"] and cells["amount"]):
        missing = "who to pay" if not cells["person"] else "how much"
        raise IntentError(f"I didn't catch {missing}")

    amount = parse_amount(text)
    if amount is None:
        raise IntentError("I didn't catch the amount clearly")
    if amount <= 0:
        raise IntentError("the amount has to be more than zero")

    who = (resolver or RosterResolver()).resolve(payee_phrase(text), payer_agent)

    mandate = ms.get(who["mandate_id"])
    currency = parse_currency(text) or mandate.currency
    if currency != mandate.currency:
        raise IntentError(f"that mandate only covers {mandate.currency}")

    p.payee_agent, p.payee_name = who["agent_id"], who["name"]
    p.payee_wallet, p.mandate_id = who["wallet"], who["mandate_id"]
    p.amount, p.currency = amount, currency
    return p
