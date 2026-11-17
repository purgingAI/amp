"""Build the recommender's training set from REAL dev data.

The label is the only honest one available: **did the customer actually take the offer?**
`campaign_redemption_ledger` records a reserve and its outcome, so

    movement_type = RESERVED, status = CONSUMED  -> 1  (they went through with it)
    movement_type = RESERVED, status = RELEASED  -> 0  (they reserved, then did not)

That is a real decision by a real person, not a proxy we invented.

**Be clear about the size.** 192 labelled events across 7 users and 6 merchants, and none of
them are merchant payments — dev simply has not seen that flow yet. So this trains a model
of *"will an offer of this shape, to a customer with this history, actually be taken"*, which
generalises to merchants because the features are the offer's economics, not merchant
identity. What it is NOT is collaborative filtering; "customers like you also bought" needs
thousands of users and we have seven. Saying otherwise would be inventing a capability.

Leakage is the easy mistake here, so two guards:
  * the split is **chronological** — train on the past, test on the future, never random.
  * the user's prior conversion rate is an **expanding mean over earlier rows only**, so a
    row never sees its own outcome or any later one.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
sys.path.insert(0, CX)

from devdb import q  # noqa: E402

OUT = os.path.join(HERE, "data")

#: Every feature the model sees, in a fixed order. Named, because an unnamed feature vector
#: is unauditable and the whole point of this build is that a human can follow the decision.
FEATURES = (
    "benefit_abs",          # how much money off, in AED
    "benefit_pct",          # as a fraction of the basket -- 10 off 20 is not 10 off 2000
    "amount",               # basket size
    "amount_log",           # spend is long-tailed; the log is what actually correlates
    "budget_headroom",      # campaign budget left, scaled -- a nearly-spent campaign is a
                            # bad thing to advertise
    "user_prior_rate",      # this customer's own past take-up (expanding, no leakage)
    "user_prior_n",         # how much history that rate is based on
    "is_percentage",        # benefit shape
    "is_cashback",
    "is_fee_discount",
    "is_cross_border",
    "is_request_to_pay",
)


@dataclass
class Row:
    user_id: str
    campaign_id: int
    merchant_id: str | None
    created_at: str
    label: int
    features: dict
    # kept for the UI so a human can read the example rather than a vector
    human: str


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build() -> list[Row]:
    raw = q("""
        SELECT user_id, campaign_id, merchant_id, product_code, incentive_type, benefit_type,
               transaction_amount, incentive_amount, remaining_budget, status, created_at
        FROM campaign_redemption_ledger
        WHERE movement_type = 'RESERVED' AND status IN ('CONSUMED','RELEASED')
        ORDER BY created_at
    """, n=5000)

    seen: dict[str, list[int]] = {}
    rows: list[Row] = []
    for r in raw:
        amount = _f(r["transaction_amount"])
        benefit = _f(r["incentive_amount"])
        budget = _f(r["remaining_budget"])
        prior = seen.get(r["user_id"], [])
        prior_rate = sum(prior) / len(prior) if prior else 0.28   # the base rate, not 0
        label = 1 if r["status"] == "CONSUMED" else 0

        feats = {
            "benefit_abs": benefit,
            "benefit_pct": benefit / amount if amount else 0.0,
            "amount": amount,
            "amount_log": __import__("math").log1p(max(amount, 0.0)),
            "budget_headroom": min(budget / 5000.0, 2.0),
            "user_prior_rate": prior_rate,
            "user_prior_n": float(len(prior)),
            "is_percentage": 1.0 if (r["benefit_type"] or "").upper() == "PERCENTAGE" else 0.0,
            "is_cashback": 1.0 if (r["incentive_type"] or "").upper() == "CASHBACK" else 0.0,
            "is_fee_discount": 1.0 if (r["incentive_type"] or "").upper() == "FEE_DISCOUNT" else 0.0,
            "is_cross_border": 1.0 if r["product_code"] == "CROSS_BORDER_TRANSFER" else 0.0,
            "is_request_to_pay": 1.0 if r["product_code"] == "REQUEST_TO_PAY" else 0.0,
        }
        human = (f"AED {benefit:g} off a AED {amount:g} "
                 f"{(r['product_code'] or 'payment').replace('_', ' ').lower()} "
                 f"({(r['benefit_type'] or '?').lower()} {(r['incentive_type'] or '?').lower()}), "
                 f"customer has taken {prior_rate:.0%} of {len(prior)} past offers")

        rows.append(Row(user_id=r["user_id"], campaign_id=r["campaign_id"],
                        merchant_id=r["merchant_id"],
                        created_at=str(r["created_at"]), label=label,
                        features=feats, human=human))
        seen.setdefault(r["user_id"], []).append(label)
    return rows


def split(rows: list[Row], test_frac: float = 0.25):
    """Chronological. A random split would let the model see the future, and on 192 rows that
    flatters it enormously."""
    cut = int(len(rows) * (1 - test_frac))
    return rows[:cut], rows[cut:]


def vectors(rows: list[Row]):
    X = [[r.features[f] for f in FEATURES] for r in rows]
    y = [r.label for r in rows]
    return X, y


def main() -> None:
    rows = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reco.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    pos = sum(r.label for r in rows)
    tr, te = split(rows)
    print(f"  {len(rows)} labelled reserve->outcome events from dev")
    print(f"    taken (CONSUMED)   {pos}  ({pos / len(rows):.0%})")
    print(f"    not taken          {len(rows) - pos}")
    print(f"    users {len({r.user_id for r in rows})}  campaigns "
          f"{len({r.campaign_id for r in rows})}  merchants "
          f"{len({r.merchant_id for r in rows if r.merchant_id})}")
    print(f"    chronological split: train {len(tr)}  test {len(te)}")
    print(f"    test-period base rate {sum(r.label for r in te) / max(len(te), 1):.0%}")
    print(f"  -> {os.path.join(OUT, 'reco.jsonl')}")


if __name__ == "__main__":
    main()
