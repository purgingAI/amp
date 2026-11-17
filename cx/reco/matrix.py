"""The recommendation decision matrix.

The real decision a recommender makes is not "which offer is biggest" — it is **whether to
interrupt someone at all, and how loudly**. So this is a table, like the refund policy, and
for the same reason: a reviewer can read the whole thing on one screen and a test can pin
every cell.

Two axes, both computed from things we actually hold. Neither needs a shred of customer
behaviour or transaction history, which is the point — this works on day one, for a customer
we have never seen before, and it will give the same answer next week for the same inputs.

    STRENGTH   how much the offer is worth, relative to what they'd spend
               NONE   <  AED 5 or nothing applies
               SMALL  <  5% of the basket
               GOOD   5–12%
               STRONG > 12%

    REACH      can they actually use it with a normal basket for that category
               BLOCKED      not eligible at all (segment, targeting, expired)
               STRETCH      needs a bigger basket than they'd normally spend
               READY        eligible right now
               AT_HAND      eligible and on their doorstep (< 100m)

The cell is the ACTION — and most cells are deliberately quiet. A recommender's failure mode
is not being wrong, it is being boring often enough that people stop reading, so only the
corner where a strong offer meets a ready customer earns a notification.

Urgency (a campaign nearly expired or nearly out of budget) can promote an action by one
step — the same way elapsed SLA modifies the refund matrix. It can never promote something
BLOCKED, because urgency about an offer you cannot use is just noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Strength(str, Enum):
    NONE = "NONE"
    SMALL = "SMALL"
    GOOD = "GOOD"
    STRONG = "STRONG"


class Reach(str, Enum):
    BLOCKED = "BLOCKED"
    STRETCH = "STRETCH"
    READY = "READY"
    AT_HAND = "AT_HAND"


class Action(str, Enum):
    STAY_QUIET = "STAY_QUIET"              # do not surface at all
    LIST_ONLY = "LIST_ONLY"                # appears if they go looking
    SUGGEST = "SUGGEST"                    # shown in the recommendation panel
    RECOMMEND = "RECOMMEND"                # top of the panel, with reasons
    NOTIFY = "NOTIFY"                      # worth a push — draft a notification
    SUGGEST_BIGGER_BASKET = "SUGGEST_BIGGER_BASKET"   # "spend AED 20 more and you save 25"


#: What each action means out loud, and whether it may interrupt.
SAY: dict[Action, str] = {
    Action.STAY_QUIET: "Not worth showing.",
    Action.LIST_ONLY: "Shown only if the customer browses offers.",
    Action.SUGGEST: "Shown in the offers panel.",
    Action.RECOMMEND: "Shown first, with the reasoning.",
    Action.NOTIFY: "Worth interrupting for — draft a notification for approval.",
    Action.SUGGEST_BIGGER_BASKET: "Worth showing with the catch stated: a slightly larger "
                                  "basket unlocks it.",
}

INTERRUPTS = frozenset({Action.NOTIFY})

#: ------------------------------------------------------------------------------------
#: THE MATRIX. (strength, reach) -> action. Every combination listed; no defaults.
#: ------------------------------------------------------------------------------------
MATRIX: dict[tuple[Strength, Reach], Action] = {
    # Nothing applies. Being blocked is not a reason to talk about it.
    (Strength.NONE, Reach.BLOCKED):   Action.STAY_QUIET,
    (Strength.NONE, Reach.STRETCH):   Action.STAY_QUIET,
    (Strength.NONE, Reach.READY):     Action.STAY_QUIET,
    (Strength.NONE, Reach.AT_HAND):   Action.STAY_QUIET,

    # Small money. Findable, never pushed — this is the row that protects the channel.
    (Strength.SMALL, Reach.BLOCKED):  Action.STAY_QUIET,
    (Strength.SMALL, Reach.STRETCH):  Action.STAY_QUIET,
    (Strength.SMALL, Reach.READY):    Action.LIST_ONLY,
    (Strength.SMALL, Reach.AT_HAND):  Action.SUGGEST,

    # Worth knowing about.
    (Strength.GOOD, Reach.BLOCKED):   Action.STAY_QUIET,
    (Strength.GOOD, Reach.STRETCH):   Action.SUGGEST_BIGGER_BASKET,
    (Strength.GOOD, Reach.READY):     Action.SUGGEST,
    (Strength.GOOD, Reach.AT_HAND):   Action.RECOMMEND,

    # Genuinely good money.
    (Strength.STRONG, Reach.BLOCKED): Action.STAY_QUIET,
    (Strength.STRONG, Reach.STRETCH): Action.SUGGEST_BIGGER_BASKET,
    (Strength.STRONG, Reach.READY):   Action.RECOMMEND,
    (Strength.STRONG, Reach.AT_HAND): Action.NOTIFY,     # the only cell that may interrupt
}

#: Promotion order for the urgency modifier.
_LADDER = [Action.STAY_QUIET, Action.LIST_ONLY, Action.SUGGEST, Action.RECOMMEND, Action.NOTIFY]

MIN_WORTH_SAYING = 5.0          # AED. Below this, nothing is "strong" however large the %.

#: "On their doorstep". Real store locations are spread across a city, so this is the
#: distance at which an offer is worth interrupting someone for.
AT_HAND_M = 400.0


@dataclass
class Signals:
    """Everything the matrix needs. All computable without any behavioural data."""
    saving: float = 0.0
    basket: float = 0.0
    eligible: bool = True
    min_txn: float | None = None
    metres: float | None = None
    days_left: float | None = None
    budget_left: float | None = None
    blocked_by: str | None = None


@dataclass
class Verdict:
    strength: Strength
    reach: Reach
    action: Action
    say: str
    interrupts: bool
    urgency: str | None = None
    trace: list[dict] = field(default_factory=list)

    def step(self, what: str, detail: str) -> None:
        self.trace.append({"what": what, "detail": detail})


def strength_of(s: Signals) -> Strength:
    if not s.eligible or s.saving < MIN_WORTH_SAYING:
        return Strength.NONE
    pct = (s.saving / s.basket) if s.basket else 0.0
    if pct > 0.12:
        return Strength.STRONG
    if pct >= 0.05:
        return Strength.GOOD
    return Strength.SMALL


def reach_of(s: Signals) -> Reach:
    if not s.eligible:
        return Reach.BLOCKED
    if s.min_txn is not None and s.basket < s.min_txn:
        return Reach.STRETCH
    # An UNKNOWN distance is not "at hand". `(s.metres or 0) < 100` treated None as zero and
    # promoted every merchant with no store location straight into the notify corner.
    if s.metres is None:
        return Reach.READY
    return Reach.AT_HAND if s.metres < AT_HAND_M else Reach.READY


def urgency_of(s: Signals) -> str | None:
    """Real scarcity only. We never manufacture it."""
    if s.days_left is not None and s.days_left <= 3:
        return f"ends in {int(s.days_left)} day(s)"
    if s.budget_left is not None and 0 < s.budget_left < 500:
        return "campaign budget nearly used up"
    return None


def decide(s: Signals) -> Verdict:
    strength, reach = strength_of(s), reach_of(s)
    action = MATRIX[(strength, reach)]
    v = Verdict(strength, reach, action, SAY[action], action in INTERRUPTS)
    v.step("strength", f"AED {s.saving:g} on a AED {s.basket:g} basket "
                       f"({(s.saving / s.basket * 100) if s.basket else 0:.0f}%) → {strength.value}")
    where = ("no store location on file" if s.metres is None
             else f"{s.metres / 1000:.1f}km away" if s.metres >= 1000
             else f"{s.metres:.0f}m away")
    v.step("reach", (f"not eligible — {s.blocked_by}" if not s.eligible else
                     f"needs AED {s.min_txn:g}, basket is AED {s.basket:g}"
                     if reach is Reach.STRETCH else where) + f" → {reach.value}")
    v.step("matrix", f"{strength.value} × {reach.value} → {action.value}")

    urgency = urgency_of(s)
    if urgency and action is not Action.STAY_QUIET and action in _LADDER:
        i = _LADDER.index(action)
        if i + 1 < len(_LADDER):
            promoted = _LADDER[i + 1]
            v.step("urgency", f"{urgency} — promoting {action.value} → {promoted.value}")
            v.action, v.say = promoted, SAY[promoted]
            v.interrupts = promoted in INTERRUPTS
        v.urgency = urgency
    elif urgency:
        # Deliberately inert: urgency about something unusable is noise.
        v.step("urgency", f"{urgency} — ignored, the offer is not usable anyway")
        v.urgency = urgency
    return v


def explain_matrix() -> list[dict]:
    return [{"strength": st.value, "reach": re.value, "action": ac.value,
             "interrupts": ac in INTERRUPTS}
            for (st, re), ac in MATRIX.items()]
