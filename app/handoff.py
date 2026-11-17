"""Carrying a capsule between two machines on your hand.

The flow the gestures drive:

    laptop A   pinch and pull   -> GRAB     capsule leaves the sending agent, in flight
    laptop B   open palm        -> RELEASE  capsule lands in whichever agent is signed in there

Your hand is the carrier. The same palm does both gestures, and that is what links the two
machines: nothing is typed, no code is read off a screen, no pairing step. GRAB files the
capsule under the palm that took it; RELEASE looks up whoever is presenting and hands over
what they are carrying.

Why the palm is not the whole gate. Hand geometry has a measured false-accept rate around
7% (tools/palm_bench.py) — roughly one presentation in fourteen, another person's hand
scores as yours. On its own that is not an identity. So a handoff also requires:

  * an authenticated agent session on BOTH machines — the receiver is never "whoever waved";
  * a short time to live, so an in-flight capsule is not sitting there to be claimed later;
  * single use, so one grab cannot be released onto several machines;
  * a margin test, because a hand that fits two enrolled people almost equally has
    identified neither, however good its absolute score.

The capsule itself is signed with the same HMAC envelope the rest of AMP uses, so the
receiving side can tell it came from this hub rather than from whoever can reach the port.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import capsule as capsule_mod
from . import palm as palm_mod
from . import protocol

DATA = Path(__file__).resolve().parents[1] / "data"
DB = DATA / "amp.db"

# Long enough to walk to the next desk, short enough that a forgotten capsule expires
# rather than waiting to be picked up by the next hand that looks similar.
TTL_SECONDS = 180

# A capsule is only released if the winning hand beats the runner-up by this much. Without
# it, two colleagues with similar hands can each claim the other's capsule whenever both are
# enrolled and the absolute score happens to pass.
MIN_MARGIN = 0.35

_SCHEMA = """
CREATE TABLE IF NOT EXISTS palm_enrolments (
    subject   TEXT PRIMARY KEY,
    agent_id  TEXT,
    payload   TEXT NOT NULL,
    updated   REAL
);
CREATE TABLE IF NOT EXISTS handoffs (
    id         TEXT PRIMARY KEY,
    subject    TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent   TEXT,
    topic      TEXT,
    capsule    TEXT NOT NULL,
    envelope   TEXT NOT NULL,
    state      TEXT NOT NULL,
    created    REAL NOT NULL,
    expires    REAL NOT NULL,
    consumed   REAL
);
"""


def _c():
    DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=6000")
    c.executescript(_SCHEMA)
    # An earlier build stored only the signature and rebuilt the envelope from columns to
    # verify it. That could not work: `ts` is an ISO string in the signed form and a float
    # in the column, so the canonical string never matched and every release was refused as
    # tampered. Carry the envelope verbatim instead.
    cols = {r[1] for r in c.execute("PRAGMA table_info(handoffs)")}
    if "sig" in cols:
        # Keyed on the legacy `sig` column, not on `envelope` being absent: an earlier
        # attempt added `envelope` with ALTER, which left a hybrid table carrying BOTH, and
        # a guard testing for the new column then never fired again while every insert still
        # had to satisfy `sig NOT NULL`. Rebuilt rather than altered because rows here are
        # in-flight capsules with a 180-second life — there is nothing durable to preserve.
        c.execute("DROP TABLE handoffs")
        c.executescript(_SCHEMA)
        c.commit()
    return c


# ---------------------------------------------------------------- enrolment

def enrol_palm(subject: str, agent_id: str, samples: list, meta: dict | None = None) -> dict:
    """Record what one person's hand looks like, from several frames."""
    e = palm_mod.enrol(subject, samples, meta=meta)
    with _c() as c:
        c.execute("INSERT INTO palm_enrolments (subject,agent_id,payload,updated) VALUES (?,?,?,?) "
                  "ON CONFLICT(subject) DO UPDATE SET agent_id=excluded.agent_id, "
                  "payload=excluded.payload, updated=excluded.updated",
                  (subject, agent_id, json.dumps(e.to_dict()), time.time()))
        c.commit()
    return {"subject": subject, "samples": e.samples, "features": len(e.mean)}


def _enrolments() -> list:
    with _c() as c:
        return [palm_mod.Enrolment.from_dict(json.loads(r["payload"]))
                for r in c.execute("SELECT payload FROM palm_enrolments").fetchall()]


def identify(landmarks: list, threshold: float = palm_mod.DEFAULT_THRESHOLD) -> dict:
    """Who is this hand? Returns the match with its numbers, and applies the margin rule."""
    res = palm_mod.match(_enrolments(), landmarks, threshold=threshold)
    if res["accepted"] and res["margin"] is not None and res["margin"] < MIN_MARGIN:
        res = dict(res)
        res["accepted"] = False
        res["subject"] = None
        res["reason"] = (f"margin {res['margin']:.2f} below {MIN_MARGIN}: this hand fits more "
                         f"than one enrolled person about equally well")
    return res


# ---------------------------------------------------------------- the two gestures

def grab(from_agent: str, topic: str, landmarks: list, secret: str) -> dict:
    """Pinch and pull: lift a topic out of the sending agent and hold it on the palm."""
    who = identify(landmarks)
    if not who["accepted"]:
        return {"ok": False, "stage": "grab", "error": "hand not recognised", "palm": who}

    cap = capsule_mod.build(topic, owner=from_agent)
    if cap["total"] == 0:
        return {"ok": False, "stage": "grab", "palm": who,
                "error": f"nothing known about '{topic}' — there is nothing to carry"}

    env = protocol.make(frm=from_agent, to="*", intent="knowledge.handoff",
                        body=capsule_mod.manifest(cap), secret=secret)
    now = time.time()
    with _c() as c:
        # One hand carries one capsule. An earlier grab that was never released is dropped
        # rather than queued: the gesture means "this one", and a queue would release
        # something the person had forgotten they were holding.
        c.execute("UPDATE handoffs SET state='superseded' WHERE subject=? AND state='in_flight'",
                  (who["subject"],))
        c.execute("INSERT INTO handoffs (id,subject,from_agent,to_agent,topic,capsule,envelope,"
                  "state,created,expires) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (env["id"], who["subject"], from_agent, None, topic, json.dumps(cap),
                   json.dumps(env), "in_flight", now, now + TTL_SECONDS))
        c.commit()
    return {"ok": True, "stage": "grab", "handoff": env["id"], "palm": who,
            "manifest": capsule_mod.manifest(cap), "expires_in": TTL_SECONDS, "envelope": env}


def release(to_agent: str, landmarks: list, secret: str) -> dict:
    """Open palm on the other machine: hand the capsule to the agent signed in there."""
    who = identify(landmarks)
    if not who["accepted"]:
        return {"ok": False, "stage": "release", "error": "hand not recognised", "palm": who}

    now = time.time()
    with _c() as c:
        c.execute("UPDATE handoffs SET state='expired' WHERE state='in_flight' AND expires < ?", (now,))
        c.commit()
        row = c.execute("SELECT * FROM handoffs WHERE subject=? AND state='in_flight' "
                        "ORDER BY created DESC LIMIT 1", (who["subject"],)).fetchone()
        if not row:
            return {"ok": False, "stage": "release", "palm": who,
                    "error": "this hand is not carrying anything (nothing grabbed, or it expired)"}

        cap = json.loads(row["capsule"])
        env = json.loads(row["envelope"])
        # Three things must hold: the envelope is ours, the manifest is the one that was
        # signed, and the payload still hashes to the id in that manifest. The last is what
        # catches an edit to the content itself, which leaves the manifest untouched.
        if (not protocol.verify(env, secret)
                or capsule_mod.manifest(cap) != env.get("body")
                or not capsule_mod.verify_integrity(cap)):
            c.execute("UPDATE handoffs SET state='tampered' WHERE id=?", (row["id"],))
            c.commit()
            return {"ok": False, "stage": "release", "palm": who,
                    "error": "capsule signature does not verify — refusing to apply it"}

        if row["from_agent"] == to_agent:
            return {"ok": False, "stage": "release", "palm": who,
                    "error": "sender and receiver are the same agent"}

        result = capsule_mod.apply(cap, receiver=to_agent)
        c.execute("UPDATE handoffs SET state='delivered', to_agent=?, consumed=? WHERE id=?",
                  (to_agent, now, row["id"]))
        c.commit()

    return {"ok": True, "stage": "release", "handoff": row["id"], "palm": who,
            "from_agent": row["from_agent"], "to_agent": to_agent,
            "manifest": capsule_mod.manifest(cap), "applied": result}


def in_flight(subject: str | None = None) -> list:
    """What is currently on someone's palm — drives the orb's 'holding' state."""
    now = time.time()
    with _c() as c:
        c.execute("UPDATE handoffs SET state='expired' WHERE state='in_flight' AND expires < ?", (now,))
        c.commit()
        sql = "SELECT id,subject,from_agent,topic,created,expires FROM handoffs WHERE state='in_flight'"
        args: tuple = ()
        if subject:
            sql += " AND subject=?"
            args = (subject,)
        return [dict(r) for r in c.execute(sql + " ORDER BY created DESC", args).fetchall()]


def enrolments_summary() -> list:
    """Who is enrolled, without exposing the feature vectors themselves.

    The means and spreads are the biometric template; they leave this module only as
    distances. A caller that wants to know who is enrolled does not need the template.
    """
    with _c() as c:
        rows = c.execute("SELECT subject, agent_id, updated FROM palm_enrolments "
                         "ORDER BY subject").fetchall()
    out = []
    for r in rows:
        e = next((x for x in _enrolments() if x.subject == r["subject"]), None)
        out.append({"subject": r["subject"], "agent_id": r["agent_id"],
                    "updated": r["updated"], "samples": e.samples if e else None})
    return out


def forget_palm(subject: str) -> dict:
    """Drop one person's enrolment. Re-enrolling is the normal fix for a bad capture."""
    with _c() as c:
        cur = c.execute("DELETE FROM palm_enrolments WHERE subject=?", (subject,))
        c.commit()
    return {"subject": subject, "removed": cur.rowcount}


def score_all(landmarks: list) -> dict:
    """Distance from this hand to EVERY enrolment.

    identify() answers "who is this", which is the product question. This answers "how far
    is it from each", which is the measurement question: a false-accept rate can only be
    computed from impostor scores, and those are exactly the distances identify() discards.
    """
    es = _enrolments()
    if not es:
        return {"scores": []}
    vec = palm_mod.signature(landmarks)
    return {"scores": sorted(
        ({"subject": e.subject, "distance": palm_mod.distance(e, vec)} for e in es),
        key=lambda d: d["distance"])}
