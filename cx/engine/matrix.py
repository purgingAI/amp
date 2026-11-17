"""The refund decision matrix.

A refund is a money-moving decision, so it is made by a table, not by a model. The model
chooses which question we are answering; this file decides what happens about it, and it is
a table precisely so that a compliance reviewer can read the whole policy on one screen and
a test can pin every cell.

The two axes are the only two things that actually matter:

  * **state** — what our system says happened.
  * **position** — where the money physically is. This is the axis people forget, and it is
    the one that decides whether a refund is ours to give. "Failed" with the money still in
    our ledger is a reversal we can do instantly. "Failed" with the money already pushed to
    HBL is a claim against HBL, and promising the customer an instant refund there is a
    promise we cannot keep.

Elapsed time against the corridor SLA then modifies the outcome: inside SLA, "pending" is
normal and the honest answer is "wait"; past SLA the same cell becomes an investigation.

Nothing in here executes anything. Every outcome is a PROPOSAL carrying the reason it was
reached, and a refund still goes through the normal approval path before money moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"
    ON_HOLD = "ON_HOLD"          # stopped by compliance / treasury
    REVERSED = "REVERSED"        # already returned
    UNKNOWN = "UNKNOWN"


class Position(str, Enum):
    NOT_DEBITED = "NOT_DEBITED"      # never left the customer's wallet
    HELD_BY_US = "HELD_BY_US"        # debited, still in our ledger
    WITH_PARTNER = "WITH_PARTNER"    # pushed to the payout partner
    CREDITED = "CREDITED"            # the beneficiary has it
    RETURNED = "RETURNED"            # back in the customer's wallet


class Outcome(str, Enum):
    AUTO_REVERSE = "AUTO_REVERSE"
    REFUND_ELIGIBLE = "REFUND_ELIGIBLE"
    AWAIT_PARTNER = "AWAIT_PARTNER"
    ALREADY_CREDITED = "ALREADY_CREDITED"
    ALREADY_REFUNDED = "ALREADY_REFUNDED"
    WAIT_SLA = "WAIT_SLA"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NOTHING_TO_REFUND = "NOTHING_TO_REFUND"
    NEED_INFO = "NEED_INFO"


#: What the customer is told, per outcome. Written to be said out loud: no internal nouns,
#: no hedging, and every one of them states what happens next and when.
SAY: dict[Outcome, str] = {
    Outcome.AUTO_REVERSE:
        "That transfer failed before the money left us, so it goes straight back to your "
        "wallet. You should see it within a few minutes.",
    Outcome.REFUND_ELIGIBLE:
        "That one qualifies for a refund. I can raise it now — our team approves refunds "
        "before the money moves, and you will get a notification when it lands.",
    Outcome.AWAIT_PARTNER:
        "The money has already gone to our payout partner, so I have to confirm with them "
        "before returning it. I will raise the claim now; this normally takes 1–2 working "
        "days and you will hear from us either way.",
    Outcome.ALREADY_CREDITED:
        "That payment went through — the money reached the beneficiary. There is nothing to "
        "refund, but I can show you the details if something still looks wrong.",
    Outcome.ALREADY_REFUNDED:
        "That one has already been refunded and the money is back in your wallet.",
    Outcome.WAIT_SLA:
        "That transfer is still in progress and within its normal delivery time. Give it a "
        "little longer — I will flag it the moment it goes past the expected window.",
    Outcome.HUMAN_REVIEW:
        "That payment is being reviewed by our team. I am not able to release or refund it "
        "myself, but I have flagged your request and someone will come back to you.",
    Outcome.NOTHING_TO_REFUND:
        "Nothing was taken from your wallet for that one, so there is nothing to refund.",
    Outcome.NEED_INFO:
        "I need to know which payment you mean. Tell me the reference, or pick it from your "
        "recent transactions.",
}

#: True when the outcome results in a refund proposal being raised for approval.
RAISES_REFUND = frozenset({Outcome.AUTO_REVERSE, Outcome.REFUND_ELIGIBLE, Outcome.AWAIT_PARTNER})

#: True when a human must look at it. These never auto-anything.
ESCALATES = frozenset({Outcome.HUMAN_REVIEW, Outcome.AWAIT_PARTNER})


#: ---------------------------------------------------------------------------------------
#: THE MATRIX. (state, position) -> outcome. Every combination is listed, including the ones
#: that should be impossible: "impossible" states are exactly the ones that show up in a
#: production incident, and a KeyError in front of a customer is not an answer.
#: ---------------------------------------------------------------------------------------
MATRIX: dict[tuple[State, Position], Outcome] = {
    # --- the transfer worked ---
    (State.SUCCESS, Position.CREDITED):     Outcome.ALREADY_CREDITED,
    (State.SUCCESS, Position.WITH_PARTNER): Outcome.WAIT_SLA,      # sent, not yet confirmed paid
    (State.SUCCESS, Position.HELD_BY_US):   Outcome.HUMAN_REVIEW,  # contradiction: investigate
    (State.SUCCESS, Position.NOT_DEBITED):  Outcome.HUMAN_REVIEW,  # contradiction: investigate
    (State.SUCCESS, Position.RETURNED):     Outcome.ALREADY_REFUNDED,

    # --- the transfer failed: the money's position decides everything ---
    (State.FAILED, Position.NOT_DEBITED):   Outcome.NOTHING_TO_REFUND,
    (State.FAILED, Position.HELD_BY_US):    Outcome.AUTO_REVERSE,
    (State.FAILED, Position.WITH_PARTNER):  Outcome.AWAIT_PARTNER,
    (State.FAILED, Position.CREDITED):      Outcome.ALREADY_CREDITED,
    (State.FAILED, Position.RETURNED):      Outcome.ALREADY_REFUNDED,

    # --- still moving ---
    (State.PENDING, Position.NOT_DEBITED):  Outcome.NOTHING_TO_REFUND,
    (State.PENDING, Position.HELD_BY_US):   Outcome.WAIT_SLA,
    (State.PENDING, Position.WITH_PARTNER): Outcome.WAIT_SLA,
    (State.PENDING, Position.CREDITED):     Outcome.ALREADY_CREDITED,
    (State.PENDING, Position.RETURNED):     Outcome.ALREADY_REFUNDED,

    # --- compliance stopped it: never ours to decide, whatever the money is doing ---
    (State.ON_HOLD, Position.NOT_DEBITED):  Outcome.HUMAN_REVIEW,
    (State.ON_HOLD, Position.HELD_BY_US):   Outcome.HUMAN_REVIEW,
    (State.ON_HOLD, Position.WITH_PARTNER): Outcome.HUMAN_REVIEW,
    (State.ON_HOLD, Position.CREDITED):     Outcome.HUMAN_REVIEW,
    (State.ON_HOLD, Position.RETURNED):     Outcome.ALREADY_REFUNDED,

    # --- already given back ---
    (State.REVERSED, Position.RETURNED):    Outcome.ALREADY_REFUNDED,
    (State.REVERSED, Position.NOT_DEBITED): Outcome.NOTHING_TO_REFUND,
    (State.REVERSED, Position.HELD_BY_US):  Outcome.AUTO_REVERSE,
    (State.REVERSED, Position.WITH_PARTNER): Outcome.AWAIT_PARTNER,
    (State.REVERSED, Position.CREDITED):    Outcome.HUMAN_REVIEW,
}


@dataclass
class Evidence:
    """What we know about the transaction in question."""
    state: State = State.UNKNOWN
    position: Position = Position.HELD_BY_US
    hours_elapsed: float = 0.0
    sla_hours: float = 0.5
    amount: float = 0.0
    currency: str = "AED"
    corridor: str = "AE-AE"
    txn_ref: str | None = None

    @property
    def sla_breached(self) -> bool:
        return self.hours_elapsed > self.sla_hours

    @property
    def sla_approaching(self) -> bool:
        return not self.sla_breached and self.hours_elapsed >= self.sla_hours * 0.75


@dataclass
class Decision:
    """A proposal, with the whole reasoning chain attached.

    `trace` is not decoration — it is what the watch screen renders and what an auditor
    reads. A decision nobody can reconstruct is not auditable, and refunds have to be.
    """
    outcome: Outcome
    say: str
    evidence: Evidence
    trace: list[dict] = field(default_factory=list)
    raises_refund: bool = False
    escalates: bool = False
    confidence: float = 1.0
    advisory: dict | None = None      # Laya's second opinion, when consulted

    def step(self, lane: str, what: str, detail: str, tag: str = "rule") -> None:
        self.trace.append({"lane": lane, "what": what, "detail": detail, "tag": tag})


def decide(ev: Evidence) -> Decision:
    """Run the matrix. Pure function of the evidence — same input, same answer, always."""
    if ev.txn_ref is None:
        d = Decision(Outcome.NEED_INFO, SAY[Outcome.NEED_INFO], ev)
        d.step("decide", "identify", "no transaction identified in the request")
        return d

    base = MATRIX.get((ev.state, ev.position))
    d = Decision(base or Outcome.HUMAN_REVIEW, "", ev)
    d.step("decide", "matrix",
           f"state={ev.state.value} × position={ev.position.value} → "
           f"{(base or Outcome.HUMAN_REVIEW).value}")
    if base is None:
        # Unreachable by construction today; kept because "unreachable" ages badly and a
        # human review is the only safe default for a combination we have not thought about.
        d.step("decide", "unmapped",
               "that combination is not in the policy table — sending it to a person",
               tag="guard")

    # --- the time modifier -----------------------------------------------------------
    if d.outcome is Outcome.WAIT_SLA and ev.sla_breached:
        d.outcome = Outcome.REFUND_ELIGIBLE if ev.position is Position.HELD_BY_US \
            else Outcome.AWAIT_PARTNER
        d.step("decide", "sla",
               f"{ev.hours_elapsed:.1f}h elapsed vs {ev.sla_hours:.1f}h expected — past the "
               f"window, so 'please wait' is no longer an honest answer → {d.outcome.value}")
    elif d.outcome is Outcome.WAIT_SLA and ev.sla_approaching:
        d.step("decide", "sla",
               f"{ev.hours_elapsed:.1f}h of {ev.sla_hours:.1f}h — close to the limit, "
               f"flagged for proactive follow-up")

    d.say = SAY[d.outcome]
    d.raises_refund = d.outcome in RAISES_REFUND
    d.escalates = d.outcome in ESCALATES
    if d.raises_refund:
        d.step("act", "refund.propose",
               f"raise a refund proposal for {ev.amount:g} {ev.currency} — approval required "
               f"before any money moves", tag="act")
    if d.escalates:
        d.step("act", "escalate", "a person has to look at this one", tag="guard")
    return d


def explain_matrix() -> list[dict]:
    """The whole policy as rows, for the UI to render as a real grid."""
    return [{"state": s.value, "position": p.value, "outcome": o.value,
             "raises_refund": o in RAISES_REFUND, "escalates": o in ESCALATES}
            for (s, p), o in MATRIX.items()]
