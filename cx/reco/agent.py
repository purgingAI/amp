"""The recommendation agent: decides what is worth saying, drafts it, and stops.

Three hard rules, enforced in code rather than intended:

1. **It can only ever target the demo user.** `ALLOWED_USERS` is a whitelist, and before
   drafting anything the agent re-reads the target segment's membership from the database
   and refuses if it contains anybody else. Not "the segment should only contain Sagar" —
   it checks, every time, because segments are dynamic and a rule change could silently
   widen the audience of an already-drafted campaign.

2. **It drafts, it never sends.** Campaigns are written with `status='DRAFT'`. It calls no
   send endpoint and cannot approve its own work. Approval is a human action through the
   normal maker-checker path, exactly as AGENTS.md requires of anything that proposes
   changes: it can propose, it cannot deploy.

3. **It stays quiet unless it has something worth saying.** A recommendation below
   `MIN_WORTH_SAYING` is dropped. The failure mode of a recommender is not being wrong, it
   is being boring often enough that people stop reading.

Everything it recommends is grounded in the live Incentive Engine quote and the customer's
own history — nothing is generated, invented or inferred from other customers.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
for p in (HERE, CX):
    if p not in sys.path:
        sys.path.insert(0, p)

import ranker                               # noqa: E402
import matrix as rmatrix                    # noqa: E402
from devdb import conn, q                   # noqa: E402

#: The ONLY customer this agent may target. Widening this is a deliberate edit, reviewed.
ALLOWED_USERS = {"676fa52f-d308-4e74-b375-a0df004668fb"}

#: The segment it drafts against. Verified to contain only allowed users before every draft.
TARGET_SEGMENT = 104

TAG = "CXRECO"


class Refused(Exception):
    """The agent declined to act. Always carries a reason that can be shown."""


@dataclass
class Recommendation:
    kind: str                     # nearby_best | better_elsewhere
    headline: str
    body: str
    merchant: str
    saving: float
    score: float
    campaign_id: int | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    #: The matrix decides whether this may be shown, recommended or pushed. The ranker only
    #: ORDERS candidates; what we are allowed to do with them is a policy question.
    action: str = "SUGGEST"
    strength: str = ""
    reach: str = ""
    interrupts: bool = False
    urgency: str | None = None
    matrix_trace: list = field(default_factory=list)

    def public(self) -> dict:
        return {"kind": self.kind, "headline": self.headline, "body": self.body,
                "merchant": self.merchant, "saving": round(self.saving, 2),
                "score": round(self.score, 3), "campaign_id": self.campaign_id,
                "reasons": self.reasons, "evidence": self.evidence,
                "action": self.action, "strength": self.strength, "reach": self.reach,
                "interrupts": self.interrupts, "urgency": self.urgency,
                "matrix_trace": self.matrix_trace}


def _verdict_for(c) -> "rmatrix.Verdict":
    """Run one ranked candidate through the policy matrix."""
    return rmatrix.decide(rmatrix.Signals(
        saving=c.saving, basket=c.basket, eligible=c.eligible,
        min_txn=c.min_txn, metres=c.merchant.get("metres", 0),
        budget_left=(c.campaign or {}).get("remainingBudget"),
        days_left=(c.campaign or {}).get("daysLeft"),
        blocked_by=c.blocked_by))


# ---------------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------------

def assert_allowed(user_id: str) -> None:
    if user_id not in ALLOWED_USERS:
        raise Refused(
            f"This agent is restricted to the demo customer and will not generate "
            f"recommendations for {user_id}.")


def assert_segment_safe(segment_id: int) -> list[str]:
    """Read the segment's live membership and refuse if anyone else is in it."""
    rows = q("SELECT user_id FROM segment_customers WHERE segment_id = %s",
             (segment_id,), n=500)
    members = [r["user_id"] for r in rows]
    if not members:
        raise Refused(f"Segment {segment_id} has no members — refusing to draft against "
                      f"an empty or dynamically-evaluated audience.")
    outside = [m for m in members if m not in ALLOWED_USERS]
    if outside:
        raise Refused(
            f"Segment {segment_id} contains {len(outside)} customer(s) outside the allowed "
            f"set. Refusing to draft a campaign that could reach real customers.")
    return members


# ---------------------------------------------------------------------------------------
# recommending
# ---------------------------------------------------------------------------------------

def recommend(user_id: str, limit: int = 3) -> list[Recommendation]:
    assert_allowed(user_id)
    ranked = [c for c in ranker.rank(user_id, limit=10) if c.eligible]
    out: list[Recommendation] = []

    for c in ranked:
        v = _verdict_for(c)
        if v.action is rmatrix.Action.STAY_QUIET:
            continue                       # the matrix says this is not worth showing
        body = (f"{c.campaign['label']} on a typical AED {c.basket:g} basket — "
                f"you'd pay AED {c.payable:g}. {c.merchant.get('metres')}m away.")
        if v.action is rmatrix.Action.SUGGEST_BIGGER_BASKET and c.min_txn:
            short = c.min_txn - c.basket
            body = (f"{c.campaign['label']}, but it needs a basket of AED {c.min_txn:g} — "
                    f"about AED {short:g} more than you usually spend here.")
        out.append(Recommendation(
            kind="nearby_best",
            headline=f"Save AED {c.saving:g} at {c.merchant['name']}",
            body=body,
            merchant=c.merchant["name"], saving=c.saving, score=c.score,
            campaign_id=(c.campaign or {}).get("campaignId"),
            reasons=c.reasons,
            evidence={"terms": c.terms, "basket": c.basket, "payable": c.payable,
                      "quoted_by": "incentive engine (live)"},
            action=v.action.value, strength=v.strength.value, reach=v.reach.value,
            interrupts=v.interrupts, urgency=v.urgency, matrix_trace=v.trace))
        if len(out) >= limit:
            break
    return out


def better_elsewhere(user_id: str, merchant: str, basket: float) -> Recommendation | None:
    """"You're about to pay here — there's more off next door." """
    assert_allowed(user_id)
    alt = ranker.better_than(merchant, basket, user_id)
    if not alt:
        return None
    return Recommendation(
        kind="better_elsewhere",
        headline=f"{alt.merchant['name']} saves you more",
        body=(f"On AED {basket:g}, {alt.merchant['name']} takes off AED {alt.saving:g} "
              f"({alt.campaign['label']}) — you'd pay AED {alt.payable:g}."),
        merchant=alt.merchant["name"], saving=alt.saving, score=alt.score,
        campaign_id=(alt.campaign or {}).get("campaignId"),
        reasons=alt.reasons,
        evidence={"compared_with": merchant, "basket": basket,
                  "quoted_by": "incentive engine (live)"})


# ---------------------------------------------------------------------------------------
# drafting — never sending
# ---------------------------------------------------------------------------------------

def draft_notification(user_id: str, rec: Recommendation,
                       segment_id: int = TARGET_SEGMENT) -> dict:
    """Write a NOTIFICATION campaign in DRAFT. Returns what it wrote.

    Nothing is sent. The row sits in DRAFT until a human moves it, which is the whole point:
    an agent that can both write and approve its own customer messaging has no control at all.
    """
    assert_allowed(user_id)
    if not rec.interrupts:
        raise Refused(
            f"The policy matrix decided this is {rec.action} ({rec.strength} offer, "
            f"{rec.reach}) — only an offer strong enough AND usable right now earns a "
            f"notification. Refusing to draft one.")
    members = assert_segment_safe(segment_id)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    notif_id = f"NEN-{TAG}{random.randint(100000, 999999)}"
    with conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO campaign
              (campaign_type, name, status, description, display_text, entity_type,
               notification_id, schedule_type, is_app, is_email, is_sms,
               created_at, updated_at, created_by, created_by_name, tnc)
            VALUES ('NOTIFICATION', %s, 'DRAFT', %s, %s, 'CUSTOMER',
                    %s, 'IMMEDIATE', b'1', b'0', b'0',
                    %s, %s, %s, %s, %s)
        """, (rec.headline[:255], rec.body[:255], rec.headline[:255],
              notif_id, now, now, "cx-reco-agent", "CX Recommendation Agent",
              f"Drafted by the recommendation agent ({TAG}). Dev only. Awaiting approval."))
        cid = cur.lastrowid
        cur.execute("INSERT INTO campaign_segment (campaign_id, segment_id) VALUES (%s,%s)",
                    (cid, segment_id))
        c.commit()

    return {"campaign_id": cid, "notification_id": notif_id, "status": "DRAFT",
            "segment_id": segment_id, "audience": len(members),
            "headline": rec.headline, "body": rec.body,
            "awaiting": "human approval — the agent cannot approve or send its own drafts"}


def list_drafts() -> list[dict]:
    rows = q("""
        SELECT c.id, c.name, c.description, c.status, c.notification_id, c.created_at,
               (SELECT COUNT(*) FROM campaign_segment s WHERE s.campaign_id = c.id) segments
        FROM campaign c
        WHERE c.campaign_type = 'NOTIFICATION' AND c.created_by = 'cx-reco-agent'
        ORDER BY c.id DESC
    """, n=50)
    return [{"campaign_id": r["id"], "headline": r["name"], "body": r["description"],
             "status": r["status"], "notification_id": r["notification_id"],
             "created_at": str(r["created_at"]), "segments": r["segments"]} for r in rows]


def undo_drafts() -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id FROM campaign WHERE created_by='cx-reco-agent'")
        ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            return 0
        marks = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM campaign_segment WHERE campaign_id IN ({marks})", ids)
        cur.execute(f"DELETE FROM campaign WHERE id IN ({marks})", ids)
        c.commit()
        return len(ids)


if __name__ == "__main__":
    import json
    user = next(iter(ALLOWED_USERS))
    if "--undo" in sys.argv:
        print(f"removed {undo_drafts()} drafted campaigns")
        sys.exit(0)
    if "--list" in sys.argv:
        for d in list_drafts():
            print(f"  [{d['status']:7}] #{d['campaign_id']} {d['headline']}")
        sys.exit(0)

    print("recommendations:")
    recs = recommend(user)
    for r in recs:
        print(f"  {r.score:7.2f}  {r.headline}")
        print(f"           {r.body}")
    print("\nrefusal check (another customer):")
    try:
        recommend("37a9f2b2-692c-4750-b71c-17dae0c36b14")
    except Refused as e:
        print(f"  refused: {e}")

    if "--draft" in sys.argv and recs:
        print("\ndrafting:")
        print("  " + json.dumps(draft_notification(user, recs[0]), indent=2).replace("\n", "\n  "))
