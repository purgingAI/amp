"""Seed the demo's merchants and live incentive campaigns into dev.

Writes directly to `APP_DB_dev`, which deserves an explanation. The correct path is
`POST /incentive/campaign` then `/campaign/action` through maker-checker, but dev's portal
users are checker-only (and finance L3 is a separate role again), so a campaign created that
way cannot be moved to ACTIVE here. These are demo rows in dev, created idempotently and
removable with `--undo`, and every value matches what the API would have written.

The combination is not guesswork — it is what `IncentiveEvaluationServiceImpl` actually
requires, read from the source:

  * `campaign.status = ACTIVE`  — `findByProductIdAndStatusAndCampaignType(id, ACTIVE,
    INCENTIVE)`. APPROVED is NOT enough; dev's one existing campaign is APPROVED and is
    invisible to the evaluator for exactly that reason.
  * `product_id = 90` (`MERCHANT_PAYMENT`) — `WalletToWalletHandler` maps both `SEND_MONEY`
    and `MERCHANT_PAYMENT`. `SEND_MONEY_MERCHANT` (62) has NO handler and silently falls back
    to wallet-to-bank.
  * `incentive_type = FEE_DISCOUNT` + `applies_on = TRANSFER_AMOUNT` → "Instant discount on
    transfer amount", `discountedAmount = amount - benefit`. That is the merchant discount.
  * **a segment containing the payer** — `userInAnySegment` returns FALSE for an empty set,
    so a campaign with no segment matches nobody at all.
  * `merchant_target_type = MID` with rows in `campaign_merchant_mid`, whose `mid` column
    holds the merchant's **user UUID** — that is also what `receiverId` must be at quote time.

    python seed_dev_campaigns.py            # create (idempotent)
    python seed_dev_campaigns.py --undo     # remove exactly what this script created
    python seed_dev_campaigns.py --show     # what is live right now
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from devdb import conn  # noqa: E402

PRODUCT_ID = 90                 # MERCHANT_PAYMENT
PRODUCT_NAME = "Merchant Payment"
SEGMENT_ID = 104                # "ALL Indian Nationals" — contains our demo payer
PAYER = "676fa52f-d308-4e74-b375-a0df004668fb"
TAG = "CXDEMO"                  # every row this script creates carries it, so --undo is exact

#: Ten real ACTIVE dev merchants with wallets and successful KYC. `metres` is demo-only —
#: there is no location data in dev, and inventing a distance is honest as long as nothing
#: downstream claims it was measured.
MERCHANTS = [
    {"user_id": "7a81dade-73b1-4d05-a8f8-370c2626cc04", "name": "Asad and Sons FZ LLC",
     "wallet": "24853406", "category": "Grocery", "metres": 40},
    {"user_id": "e480184b-358d-40c4-8e7b-f9bec5a37d8a", "name": "MClothes",
     "wallet": "45413083", "category": "Fashion", "metres": 85},
    {"user_id": "d093c8e6-78e3-404d-9d7b-7bcb57a79f86", "name": "Cystal Healing Global FZC",
     "wallet": "73360694", "category": "Wellness", "metres": 120},
    {"user_id": "928c70df-c6cf-4b39-9e52-8c26ce407b55", "name": "Pluto Travels (L.L.C)",
     "wallet": "48228747", "category": "Travel", "metres": 160},
    {"user_id": "7704e517-3522-4783-8d10-dad2f53b6e4d", "name": "Leather",
     "wallet": "93773672", "category": "Fashion", "metres": 210},
    {"user_id": "802940c3-5a93-4efb-8e56-4f8f9dca80dd", "name": "Aorespace Technology",
     "wallet": "55122456", "category": "Electronics", "metres": 260},
    {"user_id": "8e6cd221-d19d-4042-bbea-016c572b5166", "name": "Sharjah National Travel",
     "wallet": "83356658", "category": "Travel", "metres": 310},
    {"user_id": "b5281064-d83d-45b0-abf4-a1507b662457", "name": "Isam TN",
     "wallet": "79693552", "category": "Cafe", "metres": 350},
    {"user_id": "168a0726-0775-4206-88b4-a43b8ad3f824", "name": "OPS-Onboard POS2",
     "wallet": "87132695", "category": "Retail", "metres": 420},
    {"user_id": "d1d93421-5d28-44b8-86ed-5be424a1521f", "name": "OpenWallet Trading NiGq",
     "wallet": "61653534", "category": "Retail", "metres": 480},
]

#: Three campaigns on three of them, deliberately different shapes so the recommender has
#: something real to compare: a percentage with a cap, a flat amount, and a bigger-ticket
#: percentage with a high minimum.
#: Campaigns go ONLY on merchants with an ACTIVE wallet. Three of the ten (Asad and Sons,
#: Sharjah National Travel, Pluto Travels) have no wallet_master row, so a payment to them
#: fails with "No wallet found" — advertising an offer that cannot be paid is worse than
#: showing no offer at all.
CAMPAIGNS = [
    {"merchant": "d093c8e6-78e3-404d-9d7b-7bcb57a79f86",
     "name": "Cystal Healing — 15% off",
     "code": f"{TAG}-WELLNESS15", "display": "15% off, up to AED 25",
     "discount_type": "PERCENTAGE", "value": 15, "cap": 25,
     "min_txn": 50, "max_txn": 2000, "budget": 5000, "per_user_cap": 75,
     "per_user_limit": 5},
    {"merchant": "e480184b-358d-40c4-8e7b-f9bec5a37d8a",
     "name": "MClothes — AED 20 off",
     "code": f"{TAG}-STYLE20", "display": "AED 20 off over AED 100",
     "discount_type": "FLAT", "value": 20, "cap": 20,
     "min_txn": 100, "max_txn": 3000, "budget": 4000, "per_user_cap": 60,
     "per_user_limit": 3},
    {"merchant": "802940c3-5a93-4efb-8e56-4f8f9dca80dd",
     "name": "Aorespace — 10% off electronics",
     "code": f"{TAG}-TECH10", "display": "10% off, up to AED 50",
     "discount_type": "PERCENTAGE", "value": 10, "cap": 50,
     "min_txn": 200, "max_txn": 10000, "budget": 8000, "per_user_cap": 150,
     "per_user_limit": 2},
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def show() -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.name, c.incentive_code, c.status, c.product_id, c.end_at,
                   d.id did, d.incentive_type, d.applies_on, d.discount_type, d.discount_value,
                   d.max_discount_cap, d.min_txn_amount, d.total_budget, d.budget_used,
                   d.merchant_target_type
            FROM campaign c JOIN campaign_incentive_details d ON d.campaign_id = c.id
            WHERE c.product_id = %s AND c.campaign_type = 'INCENTIVE'
            ORDER BY c.id DESC
        """, (PRODUCT_ID,))
        rows = cur.fetchall()
        print(f"{len(rows)} incentive campaigns on product {PRODUCT_ID}")
        for r in rows:
            cur.execute("SELECT mid FROM campaign_merchant_mid WHERE incentive_details_id=%s",
                        (r["did"],))
            mids = [m["mid"] for m in cur.fetchall()]
            cur.execute("SELECT segment_id FROM campaign_segment WHERE campaign_id=%s", (r["id"],))
            segs = [s["segment_id"] for s in cur.fetchall()]
            print(f"  [{r['status']:8}] #{r['id']} {r['name']}")
            print(f"       {r['incentive_type']}/{r['applies_on']} {r['discount_type']} "
                  f"{r['discount_value']} cap {r['max_discount_cap']} min {r['min_txn_amount']}"
                  f"  budget {r['budget_used']}/{r['total_budget']}  ends {r['end_at']}")
            print(f"       segments={segs}  target={r['merchant_target_type']} mids={mids}")


def undo() -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id FROM campaign WHERE incentive_code LIKE %s", (TAG + "-%",))
        ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            print("nothing to remove")
            return
        marks = ",".join(["%s"] * len(ids))
        cur.execute(f"SELECT id FROM campaign_incentive_details WHERE campaign_id IN ({marks})", ids)
        dids = [r["id"] for r in cur.fetchall()]
        if dids:
            dmarks = ",".join(["%s"] * len(dids))
            cur.execute(f"DELETE FROM campaign_merchant_mid WHERE incentive_details_id IN ({dmarks})", dids)
            cur.execute(f"DELETE FROM campaign_incentive_details WHERE id IN ({dmarks})", dids)
        cur.execute(f"DELETE FROM campaign_segment WHERE campaign_id IN ({marks})", ids)
        cur.execute(f"DELETE FROM campaign WHERE id IN ({marks})", ids)
        c.commit()
        print(f"removed campaigns {ids}")


def seed() -> None:
    now = _now()
    end = now + timedelta(days=90)
    created = []
    with conn() as c, c.cursor() as cur:
        for spec in CAMPAIGNS:
            cur.execute("SELECT id FROM campaign WHERE incentive_code=%s", (spec["code"],))
            existing = cur.fetchone()
            if existing:
                print(f"  = {spec['code']:20} already exists (#{existing['id']})")
                created.append(existing["id"])
                continue

            cur.execute("""
                INSERT INTO campaign
                  (campaign_type, name, status, description, display_text, incentive_code,
                   incentive_type, product_id, product_name, entity_type, corridor,
                   start_at, end_at, created_at, updated_at, created_by, created_by_name,
                   is_app, is_email, is_sms, tnc)
                VALUES ('INCENTIVE', %s, 'ACTIVE', %s, %s, %s,
                        'FEE_DISCOUNT', %s, %s, 'CUSTOMER', NULL,
                        %s, %s, %s, %s, %s, %s, b'1', b'0', b'0', %s)
            """, (spec["name"], f"CX demo campaign ({TAG})", spec["display"], spec["code"],
                  PRODUCT_ID, PRODUCT_NAME, now, end, now, now,
                  "cx-demo", "CX Recommender", "Demo campaign. Dev only."))
            cid = cur.lastrowid

            cur.execute("""
                INSERT INTO campaign_incentive_details
                  (campaign_id, incentive_type, applies_on, apply_mode, discount_type,
                   discount_value, max_discount_cap, min_txn_amount, max_txn_amount,
                   total_budget, budget_used, per_user_cap, per_user_redemption_limit,
                   promo_code, promo_enabled, promo_code_type, merchant_target_type,
                   is_specific_amount, created_at, updated_at)
                VALUES (%s, 'FEE_DISCOUNT', 'TRANSFER_AMOUNT', 'AUTO_APPLY', %s,
                        %s, %s, %s, %s,
                        %s, 0, %s, %s,
                        %s, b'1', 'STATIC', 'MID',
                        b'0', %s, %s)
            """, (cid, spec["discount_type"], spec["value"], spec["cap"],
                  spec["min_txn"], spec["max_txn"], spec["budget"],
                  spec["per_user_cap"], spec["per_user_limit"], spec["code"], now, now))
            did = cur.lastrowid

            cur.execute("INSERT INTO campaign_merchant_mid (incentive_details_id, mid) VALUES (%s,%s)",
                        (did, spec["merchant"]))
            cur.execute("INSERT INTO campaign_segment (campaign_id, segment_id) VALUES (%s,%s)",
                        (cid, SEGMENT_ID))
            created.append(cid)
            merchant = next(m["name"] for m in MERCHANTS if m["user_id"] == spec["merchant"])
            print(f"  + #{cid:<4} {spec['code']:20} {merchant:28} {spec['display']}")
        c.commit()
    print(f"\n  {len(created)} campaigns live on product {PRODUCT_ID}, segment {SEGMENT_ID}")


def write_secrets() -> None:
    """Put the merchants into openwallet/.secrets.json so the demo picks up the real ones."""
    path = os.path.join(os.path.dirname(HERE), "openwallet", ".secrets.json")
    cfg = json.load(open(path))
    cfg["merchants"] = [
        {"mid": m["user_id"], "name": m["name"], "category": m["category"],
         "metres": m["metres"], "userId": m["user_id"], "wallet": m["wallet"]}
        for m in MERCHANTS
    ]
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  wrote {len(MERCHANTS)} merchants into openwallet/.secrets.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show()
    elif args.undo:
        undo()
    else:
        seed()
        write_secrets()
        print()
        show()
