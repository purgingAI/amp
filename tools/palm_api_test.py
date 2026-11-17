#!/usr/bin/env python3
"""End-to-end test of the enrolment/handoff HTTP surface.

The maths is already covered by palm_bench.py and handoff_test.py. What this covers is the
part the browser tool depends on and those two cannot see: that the routes exist, that they
refuse an unauthenticated caller, that landmarks survive JSON as [x,y,z] triples, and that
score/identify/grab/release agree with each other over HTTP.

Run: ./.venv/bin/python tools/palm_api_test.py
"""
from __future__ import annotations
import os, sys, random, tempfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Point every store at a throwaway directory BEFORE importing the app, so a test run cannot
# touch the real enrolments or the live knowledge graph.
_TMP = tempfile.mkdtemp(prefix="palm-api-")
os.environ["AMP_DATA"] = _TMP
os.environ["ACE_HANDOFF_SECRET"] = "test-secret-not-a-real-key"

from fastapi.testclient import TestClient          # noqa: E402
from app import hub, handoff, palm                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- synthetic hands
def hand(seed: int, pose: int = 0) -> list:
    """A plausible 21-landmark right hand, in metres, with a per-person shape and a pose.

    Same generator as palm_bench so the numbers here are comparable with it.
    """
    rnd = random.Random(seed)
    scale = rnd.uniform(0.85, 1.15)
    fing = [rnd.uniform(0.9, 1.1) for _ in range(5)]
    spread = rnd.uniform(0.85, 1.15)
    lm = [(0.0, 0.0, 0.0)]
    # thumb
    base = (-0.022 * spread, 0.030, 0.004)
    for i in range(4):
        lm.append((base[0] - 0.014 * i * fing[0], base[1] + 0.017 * i * fing[0], base[2] + 0.002 * i))
    # four fingers
    for f, x0 in enumerate([-0.008, 0.008, 0.023, 0.037]):
        mcp = (x0 * spread, 0.072 * scale, 0.0)
        L = (0.042 + 0.006 * f * -1) * fing[f + 1] * scale
        for i in range(4):
            lm.append((mcp[0], mcp[1] + L * i / 3.0, mcp[2] - 0.002 * i))
    lm = [(x * scale, y * scale, z * scale) for x, y, z in lm]

    # pose: rigid rotation + translation, which canonicalise() must remove entirely
    pr = random.Random(seed * 977 + pose)
    import math
    a, b = pr.uniform(-0.5, 0.5), pr.uniform(-0.4, 0.4)
    out = []
    for x, y, z in lm:
        y2, z2 = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
        x2, z3 = x * math.cos(b) + z2 * math.sin(b), -x * math.sin(b) + z2 * math.cos(b)
        out.append([x2 + 0.3, y2 + 0.1, z3 - 0.5])
    return out


def noisy(seed: int, pose: int, sigma: float = 0.0015) -> list:
    rnd = random.Random(seed * 131 + pose * 7 + 1)
    return [[c + rnd.gauss(0, sigma) for c in p] for p in hand(seed, pose)]


AUTH = None


def main():
    global AUTH
    c = TestClient(hub.app)

    print("\nhttp surface")
    r = c.post("/ace/palm/enrol", json={"subject": "x", "samples": []})
    check("unauthenticated enrol is refused", r.status_code == 401, f"got {r.status_code}")
    r = c.post("/ace/palm/score", json={"landmarks": hand(1)})
    check("unauthenticated score is refused", r.status_code == 401, f"got {r.status_code}")
    r = c.get("/palm/me")
    check("unauthenticated /palm/me is refused", r.status_code == 401, f"got {r.status_code}")
    r = c.get("/palm")
    check("/palm serves the enrolment page", r.status_code == 200 and b"Palm Enrolment" in r.content,
          f"got {r.status_code}")

    # a real ACE session, through the real login path
    d = c.post("/palm/login", json={"email": "tester@example.com", "name": "Tester"}).json()
    if not d.get("token"):
        check("login returned a session", False, str(d)[:120])
        return
    AUTH = {"X-Ace-Agent": d["agent_id"], "X-Ace-Token": d["token"]}
    check("login returned a session", True)

    print("\nenrolment")
    r = c.post("/ace/palm/enrol", json={"subject": "alice", "samples": [noisy(11, p) for p in range(3)]},
               headers=AUTH)
    check(f"fewer than {palm.MIN_SAMPLES} samples is rejected", r.status_code == 400, f"got {r.status_code}")

    for name, seed in (("alice", 11), ("bob", 22), ("carol", 33)):
        r = c.post("/ace/palm/enrol",
                   json={"subject": name, "samples": [noisy(seed, p) for p in range(8)]}, headers=AUTH)
        check(f"enrolled {name}", r.status_code == 200 and r.json().get("features") == 30,
              str(r.json())[:100])

    r = c.get("/ace/palm/enrolments", headers=AUTH).json()
    check("roster lists three hands", len(r["enrolments"]) == 3, str(len(r["enrolments"])))
    check("roster reports sample counts", all(e["samples"] == 8 for e in r["enrolments"]))
    check("roster does NOT leak the template",
          "payload" not in str(r) and "mean" not in str(r))
    check("roster publishes the operating point",
          r["threshold"] == palm.DEFAULT_THRESHOLD and r["min_margin"] == handoff.MIN_MARGIN)

    print("\nidentify / score")
    r = c.post("/ace/palm/identify", json={"landmarks": noisy(11, 99)}, headers=AUTH).json()
    check("alice's hand identifies as alice", r["accepted"] and r["subject"] == "alice",
          f"{r.get('subject')} d={r.get('distance')}")
    # Whether an unenrolled hand lands FAR from everyone is a question about the score
    # distributions, which palm_bench.py measures across many people. All this asserts is that
    # the route applies both rules — threshold and margin — and says which one fired.
    r = c.post("/ace/palm/identify", json={"landmarks": noisy(77, 1)}, headers=AUTH).json()
    check("an unenrolled hand is not accepted", not r["accepted"], r.get("reason", "")[:90])
    r = c.post("/ace/palm/identify", json={"landmarks": [[0, 0, 0]] * 5}, headers=AUTH)
    check("a malformed hand is a 400, not a 500", r.status_code == 400, f"got {r.status_code}")

    s = c.post("/ace/palm/score", json={"landmarks": noisy(22, 4)}, headers=AUTH).json()["scores"]
    check("score returns every enrolment", len(s) == 3, str(len(s)))
    check("score is sorted nearest first", s == sorted(s, key=lambda x: x["distance"]))
    check("score's nearest agrees with identify", s[0]["subject"] == "bob", str(s[:2]))

    print("\nthe two gestures over HTTP")
    g = c.post("/ace/handoff/grab", json={"topic": "incentive engine", "landmarks": noisy(11, 5)},
               headers=AUTH)
    check("grab succeeds for an enrolled hand", g.status_code == 200 and g.json().get("ok"),
          str(g.json())[:140])
    gid = g.json().get("handoff")   # grab names the capsule "handoff"; the in-flight row calls it "id"

    fl = c.get("/ace/handoff/in-flight", headers=AUTH).json()["in_flight"]
    check("the capsule is in flight", any(x.get("id") == gid for x in fl), str(fl)[:120])

    bad = c.post("/ace/handoff/release", json={"landmarks": noisy(77, 2)}, headers=AUTH)
    check("a stranger's palm cannot release it", bad.status_code == 409, f"got {bad.status_code}")

    # receiving agent = a second ACE session, as on the second laptop
    d2 = c.post("/palm/login", json={"email": "receiver@example.com", "name": "Receiver"}).json()
    AUTH2 = {"X-Ace-Agent": d2["agent_id"], "X-Ace-Token": d2["token"]}
    rel = c.post("/ace/handoff/release", json={"landmarks": noisy(11, 6)}, headers=AUTH2)
    check("the same palm releases it to the other agent",
          rel.status_code == 200 and rel.json().get("ok"), str(rel.json())[:160])
    again = c.post("/ace/handoff/release", json={"landmarks": noisy(11, 7)}, headers=AUTH2)
    check("it cannot be released twice", again.status_code == 409, f"got {again.status_code}")

    print("\nforget")
    r = c.post("/ace/palm/forget", json={"subject": "carol"}, headers=AUTH).json()
    check("forget removes the enrolment", r.get("removed") == 1, str(r))
    n = len(c.get("/ace/palm/enrolments", headers=AUTH).json()["enrolments"])
    check("roster is down to two", n == 2, str(n))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
