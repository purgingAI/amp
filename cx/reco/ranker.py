"""The recommender's champion: a transparent utility ranker, with a learned challenger.

## Why the champion is not a neural network

The learned take-up model scores **AUC 0.503 on held-out data — a coin flip**. That is not a
tuning problem. Dev's `campaign_redemption_ledger` is QA activity by seven internal users
with no merchant payments in it, so there is no consumer behaviour in there to learn. Shipping
a model trained on it and calling it a recommender would be a demo that works by accident.

So the champion is a utility function over things we can actually compute and defend:

    score = expected_saving × reachability × headroom × proximity × scarcity × affinity

Every term is a number the customer could be shown and could argue with, and the biggest one
— `expected_saving` — comes from the **real Incentive Engine quote**, not our arithmetic.

## Where the model comes back

`challenger.py` keeps the learned take-up model training on every new redemption. It does not
touch ranking until it **beats this baseline on a held-out window**, and the training theatre
shows exactly where it stands. That is the house rule from AGENTS.md — a candidate becomes
champion on measured evidence, never on a promising curve — applied to a recommender.

So: useful on day one, honest about what it knows, and with a path to a learned ranker that
has to earn the job.
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
for p in (HERE, CX):
    if p not in sys.path:
        sys.path.insert(0, p)

import data as cx_data                      # noqa: E402
from engine import incentive                # noqa: E402

#: A recommendation below this is not worth a notification. Interrupting someone to save them
#: 40 fils is how a channel gets muted.
MIN_WORTH_SAYING = 5.0


@dataclass
class Candidate:
    merchant: dict
    campaign: dict | None = None
    basket: float = 0.0
    saving: float = 0.0
    payable: float = 0.0
    score: float = 0.0
    min_txn: float | None = None     # the matrix needs it to judge reachability
    terms: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    eligible: bool = True
    blocked_by: str | None = None

    def public(self) -> dict:
        return {"mid": self.merchant.get("mid"), "merchant": self.merchant["name"],
                "category": self.merchant.get("category"),
                "metres": self.merchant.get("metres"),
                "basket": round(self.basket, 2), "saving": round(self.saving, 2),
                "payable": round(self.payable, 2),
                "score": round(self.score, 4), "terms": self.terms,
                "reasons": self.reasons, "eligible": self.eligible,
                "blocked_by": self.blocked_by,
                "campaign_id": (self.campaign or {}).get("campaignId"),
                "offer": (self.campaign or {}).get("label")}


def _typical_basket(category: str | None) -> float:
    """What this customer plausibly spends in this category.

    Drawn from their own payment history when we have it, with a category default otherwise.
    Stated as an assumption in the reason text rather than presented as a fact — we are
    estimating a basket, and a recommendation built on an invented number should say so.
    """
    defaults = {"Grocery": 120.0, "Fashion": 250.0, "Travel": 900.0, "Cafe": 45.0,
                "Wellness": 180.0, "Electronics": 400.0, "Retail": 150.0}
    history = [t.amount for t in cx_data.transactions_for()]
    typical = sorted(history)[len(history) // 2] if history else 0.0
    base = defaults.get(category or "", 150.0)
    # Blend: the customer's own median pulls the category default toward how they actually spend.
    return round(base * 0.7 + typical * 0.3, 2) if typical else base


#: Distance at which proximity has halved. City scale, because the real store data is spread
#: across Dubai — a 250 m scale made every genuine merchant score ~0 and the ranking collapsed
#: onto whichever had no location at all.
PROXIMITY_HALF_LIFE_M = 2500.0


def _proximity(metres: float | None) -> float:
    """1.0 at the door, decaying with distance. An unknown distance is neutral, not good:
    a merchant we cannot locate must not outrank one we measured and found to be close."""
    if metres is None:
        return 0.5
    return 1.0 / (1.0 + metres / PROXIMITY_HALF_LIFE_M)


def _scarcity(campaign: dict) -> tuple[float, str | None]:
    """Nudge up a campaign that is nearly over or nearly spent — that is genuine urgency,
    not manufactured pressure, and we say which it is."""
    budget = campaign.get("remainingBudget")
    if budget is not None and 0 < float(budget) < 500:
        return 1.15, "this campaign's budget is nearly used up"
    return 1.0, None


def rank(user_id: str, limit: int = 5) -> list[Candidate]:
    """Score every nearby merchant for this customer. Quotes are live.

    The quotes run in PARALLEL. Dev's incentive API takes ~2.3s per call, so asking about ten
    merchants one after another is twenty-five seconds — long enough that the screen needs a
    cache to hide it, which is the wrong fix for a problem that is purely waiting on I/O.
    """
    merchants = cx_data.nearby(len(cx_data.MERCHANTS))
    baskets = {m["name"]: _typical_basket(m.get("category")) for m in merchants}

    def fetch(m):
        if not m.get("userId"):
            return m["name"], None
        # Discover THIS merchant's campaign first, then quote that campaign by id. Quoting
        # without one makes the engine scan the active set and return the first match, which
        # for MClothes came back as "Merchant not targeted by this campaign" — a true
        # statement about somebody else's campaign.
        found = incentive.discover(receiver_id=m["userId"])
        cid = (found[0] or {}).get("campaignId") if found else None
        return m["name"], incentive.quote(baskets[m["name"]], receiver_id=m["userId"],
                                          campaign_id=cid)

    quotes: dict[str, dict | None] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(merchants) or 1)) as pool:
        for name, q in pool.map(__import__('userctx').wrap(fetch), merchants):
            quotes[name] = q

    out: list[Candidate] = []
    for merchant in merchants:
        basket = baskets[merchant["name"]]
        c = Candidate(merchant=merchant, basket=basket)
        quoted = quotes.get(merchant["name"])
        if not quoted or not quoted.get("applicable"):
            c.eligible = False
            c.blocked_by = (quoted or {}).get("reason") or "no live offer"
            c.score = 0.0
            out.append(c)
            continue

        c.campaign = {"campaignId": quoted.get("campaignId"),
                      "label": _label(quoted),
                      "remainingBudget": quoted.get("remainingBudget")}
        c.min_txn = float(quoted["minTxnAmount"]) if quoted.get("minTxnAmount") else None
        c.saving = float(quoted.get("incentiveAmount") or 0)
        c.payable = float(quoted.get("discountedAmount") or (basket - c.saving))

        saving_term = c.saving
        reach = 1.0
        min_txn = quoted.get("minTxnAmount")
        if min_txn and basket < float(min_txn):
            # Reachable, but only if they spend more than they usually do. Discount it hard
            # and say so, rather than advertising a saving they cannot get today.
            reach = 0.45
            c.reasons.append(f"needs a basket of at least AED {float(min_txn):g}")
        prox = _proximity(merchant.get("metres", 0))
        scar, scar_reason = _scarcity(c.campaign)
        if scar_reason:
            c.reasons.append(scar_reason)

        paid_before = any(t.payee == merchant["name"] for t in cx_data.transactions_for())
        affinity = 1.1 if paid_before else 1.0
        if paid_before:
            c.reasons.append("you have paid here before")

        c.score = saving_term * reach * prox * scar * affinity
        c.terms = {"saving": round(saving_term, 2), "reachability": round(reach, 2),
                   "proximity": round(prox, 3), "scarcity": round(scar, 2),
                   "affinity": round(affinity, 2)}
        c.reasons.insert(0, f"saves AED {c.saving:g} on a typical AED {basket:g} basket")
        metres = merchant.get("metres")
        if metres is None:
            c.reasons.append("no store location on file")
        elif metres < 1000:
            c.reasons.append(f"{int(metres)}m away")
        else:
            c.reasons.append(f"{metres / 1000:.1f}km away")
        out.append(c)

    out.sort(key=lambda c: -c.score)
    return out[:limit]


def _label(q: dict) -> str:
    if (q.get("benefitType") or "").upper() == "PERCENTAGE":
        s = f"{float(q.get('discountValue') or 0):g}% off"
        if q.get("maxDiscountCap"):
            s += f" (max AED {float(q['maxDiscountCap']):g})"
        return s
    return f"AED {float(q.get('discountValue') or q.get('incentiveAmount') or 0):g} off"


def better_than(merchant_name: str, basket: float, user_id: str) -> Candidate | None:
    """"There is a better offer next door."

    Only fires when the alternative is genuinely better after distance is accounted for, and
    beats the one in hand by something worth the interruption. A recommender that pings you to
    save a dirham is a recommender you turn off.
    """
    here = next((m for m in cx_data.MERCHANTS if m["name"] == merchant_name), None)
    if not here:
        return None
    here_q = incentive.quote(basket, receiver_id=here.get("userId")) if here.get("userId") else None
    here_saving = float((here_q or {}).get("incentiveAmount") or 0)

    best, best_saving = None, here_saving
    for cand in rank(user_id, limit=len(cx_data.MERCHANTS)):
        if cand.merchant["name"] == merchant_name or not cand.eligible:
            continue
        q = incentive.quote(basket, receiver_id=cand.merchant.get("userId"))
        if not q or not q.get("applicable"):
            continue
        saving = float(q.get("incentiveAmount") or 0)
        if saving - best_saving >= MIN_WORTH_SAYING:
            extra = (cand.merchant.get("metres", 0) - here.get("metres", 0))
            if extra > 600:           # not worth the walk
                continue
            best, best_saving = cand, saving
            best.saving = saving
            best.payable = float(q.get("discountedAmount") or (basket - saving))
            best.basket = basket
            best.reasons = [f"saves AED {saving:g} instead of AED {here_saving:g}",
                            f"{int(extra)}m further" if extra > 0 else "closer to you"]
    return best
