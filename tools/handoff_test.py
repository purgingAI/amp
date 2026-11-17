"""End-to-end: grab on one agent, release on another, with the failure modes."""
import random, sqlite3, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app import handoff, capsule as capsule_mod
import palm
sys.path.insert(0, str(ROOT / "tools"))
from palm_bench import person, pose, jitter

SECRET = "test-secret"
rnd = random.Random(11)
ok = fail = 0

def check(label, cond, detail=""):
    global ok, fail
    if cond: ok += 1;  print(f"  PASS  {label}")
    else:    fail += 1; print(f"  FAIL  {label}  {detail}")

# a clean slate in this run's tables only
with handoff._c() as c:
    c.execute("DELETE FROM handoffs"); c.execute("DELETE FROM palm_enrolments"); c.commit()

# seed something worth carrying
capsule_mod._ensure_learnings(capsule_mod.AMP_DB)
con = sqlite3.connect(capsule_mod.AMP_DB)
con.execute("DELETE FROM learnings WHERE question LIKE 'HANDOFF-TEST%'")
for i in range(4):
    con.execute("INSERT INTO learnings (question,answer,rating,keywords,agent,ts) VALUES (?,?,?,?,?,?)",
                (f"HANDOFF-TEST widgets q{i}", f"widgets answer {i}", "good", "widgets", "alice", time.time()))
con.commit(); con.close()

alice_hand, bob_hand = person(501), person(777)
def frames(h, n): return [jitter(pose(h, rnd), rnd, 0.012) for _ in range(n)]

print("\n1. enrolment")
r = handoff.enrol_palm("alice", "agent.alice", frames(alice_hand, 8))
check("alice's hand enrolled", r["samples"] == 8, r)
handoff.enrol_palm("bob", "agent.bob", frames(bob_hand, 8))

print("\n2. grab — alice pinches and pulls 'widgets'")
g = handoff.grab("agent.alice", "widgets", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("grab accepted", g["ok"], g.get("error"))
check("identified alice", g.get("palm", {}).get("subject") == "alice", g.get("palm"))
check("capsule has the learnings", g.get("manifest", {}).get("counts", {}).get("learnings", 0) >= 4, g.get("manifest"))
check("shows as in flight", len(handoff.in_flight("alice")) == 1)

print("\n3. release — the same hand, on bob's machine")
r = handoff.release("agent.bob", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("release accepted", r["ok"], r.get("error"))
check("learnings landed", r.get("applied", {}).get("learnings_added", 0) >= 4, r.get("applied"))
check("no longer in flight", len(handoff.in_flight("alice")) == 0)

print("\n4. single use — the same hand again")
again = handoff.release("agent.bob", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("second release refused", not again["ok"], again)

print("\n5. idempotent — same knowledge transferred twice does not duplicate")
handoff.grab("agent.alice", "widgets", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
r2 = handoff.release("agent.bob", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("nothing added the second time", r2["applied"]["learnings_added"] == 0, r2["applied"])
check("and it says it skipped them", r2["applied"]["learnings_skipped"] >= 4, r2["applied"])

print("\n6. a different hand cannot collect alice's capsule")
handoff.grab("agent.alice", "widgets", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
b = handoff.release("agent.bob", jitter(pose(bob_hand, rnd), rnd, 0.012), SECRET)
check("bob's hand gets nothing of alice's", not b["ok"], b)

print("\n7. sender and receiver must differ")
same = handoff.release("agent.alice", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("self-handoff refused", not same["ok"], same)

print("\n8. a tampered capsule is refused")
handoff.grab("agent.alice", "widgets", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
with handoff._c() as c:
    row = c.execute("SELECT id,capsule FROM handoffs WHERE state='in_flight'").fetchone()
    bad = row["capsule"].replace("widgets answer 0", "widgets answer TAMPERED")
    c.execute("UPDATE handoffs SET capsule=? WHERE id=?", (bad, row["id"])); c.commit()
t = handoff.release("agent.bob", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("tampering detected", not t["ok"] and "signature" in (t.get("error") or ""), t)

print("\n9. expiry")
handoff.grab("agent.alice", "widgets", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
with handoff._c() as c:
    c.execute("UPDATE handoffs SET expires=? WHERE state='in_flight'", (time.time() - 1,)); c.commit()
e = handoff.release("agent.bob", jitter(pose(alice_hand, rnd), rnd, 0.012), SECRET)
check("expired capsule not delivered", not e["ok"], e)

print("\n10. an unknown hand is refused")
u = handoff.grab("agent.alice", "widgets", jitter(pose(person(999), rnd), rnd, 0.012), SECRET)
check("stranger cannot grab", not u["ok"], u.get("error"))

print(f"\n{'=' * 60}\n  {ok} passed, {fail} failed\n{'=' * 60}")
sys.exit(1 if fail else 0)
