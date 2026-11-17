"""Measure what the hand-geometry matcher can and cannot do.

Two questions, answered with numbers rather than assurances:

  1. Is the signature pose-invariant? Move, rotate and rescale the SAME hand; the signature
     must not move. If it does, the "biometric" is partly measuring camera distance.
  2. Can it tell people apart? Generate synthetic people whose hands differ by a realistic
     margin, add per-frame landmark noise, and compare genuine against impostor distances.

Synthetic hands can only falsify, never confirm: passing here means the maths is sound, not
that real hands separate. Run tools/palm_enrol_live.py on actual colleagues before trusting
any threshold in production.
"""
import math, random, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import palm


def base_hand():
    """A neutral right hand, roughly to human proportion, in arbitrary units."""
    return {
        palm.WRIST: (0.00, 0.00, 0.0),
        palm.THUMB_CMC: (0.20, -0.28, 0.02), palm.THUMB_MCP: (0.42, -0.46, 0.03),
        palm.THUMB_IP: (0.60, -0.60, 0.03),  palm.THUMB_TIP: (0.74, -0.72, 0.03),
        palm.INDEX_MCP: (0.92, -0.26, 0.0),  palm.INDEX_PIP: (1.30, -0.30, 0.0),
        palm.INDEX_DIP: (1.54, -0.32, 0.0),  palm.INDEX_TIP: (1.74, -0.33, 0.0),
        palm.MIDDLE_MCP: (1.00, 0.00, 0.0),  palm.MIDDLE_PIP: (1.42, 0.02, 0.0),
        palm.MIDDLE_DIP: (1.68, 0.03, 0.0),  palm.MIDDLE_TIP: (1.90, 0.04, 0.0),
        palm.RING_MCP: (0.95, 0.24, 0.0),    palm.RING_PIP: (1.33, 0.30, 0.0),
        palm.RING_DIP: (1.56, 0.34, 0.0),    palm.RING_TIP: (1.74, 0.37, 0.0),
        palm.PINKY_MCP: (0.86, 0.46, 0.0),   palm.PINKY_PIP: (1.16, 0.56, 0.0),
        palm.PINKY_DIP: (1.34, 0.62, 0.0),   palm.PINKY_TIP: (1.48, 0.66, 0.0),
    }


def person(seed):
    """One person's skeleton: finger lengths and knuckle spread shifted a few percent.

    Real adult hands vary far more than this between individuals; keeping the synthetic
    spread tight makes the test harder than reality rather than easier.
    """
    rnd = random.Random(seed)
    finger_scale = {f: rnd.uniform(0.90, 1.10) for f in ("thumb", "index", "middle", "ring", "pinky")}
    spread = rnd.uniform(0.92, 1.08)
    h = base_hand()
    groups = {
        "thumb": [palm.THUMB_CMC, palm.THUMB_MCP, palm.THUMB_IP, palm.THUMB_TIP],
        "index": [palm.INDEX_MCP, palm.INDEX_PIP, palm.INDEX_DIP, palm.INDEX_TIP],
        "middle": [palm.MIDDLE_MCP, palm.MIDDLE_PIP, palm.MIDDLE_DIP, palm.MIDDLE_TIP],
        "ring": [palm.RING_MCP, palm.RING_PIP, palm.RING_DIP, palm.RING_TIP],
        "pinky": [palm.PINKY_MCP, palm.PINKY_PIP, palm.PINKY_DIP, palm.PINKY_TIP],
    }
    out = dict(h)
    for finger, idxs in groups.items():
        s = finger_scale[finger]
        for i in idxs:
            x, y, z = h[i]
            out[i] = (x * s, y * spread, z)
    return [out[i] for i in range(palm.N_LANDMARKS)]


def pose(hand, rnd):
    """Same hand, seen from somewhere else: rotated, moved and rescaled."""
    ax, ay, az = (rnd.uniform(-0.5, 0.5) for _ in range(3))
    s = rnd.uniform(0.4, 2.5)
    tx, ty, tz = (rnd.uniform(-3, 3) for _ in range(3))
    def rot(p):
        x, y, z = p
        y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
        x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
        x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
        return (x * s + tx, y * s + ty, z * s + tz)
    return [rot(p) for p in hand]


def jitter(hand, rnd, sigma):
    """Per-frame landmark noise, standing in for MediaPipe's own wobble."""
    return [(x + rnd.gauss(0, sigma), y + rnd.gauss(0, sigma), z + rnd.gauss(0, sigma))
            for x, y, z in hand]


def main():
    rnd = random.Random(7)
    print("=" * 74)
    print("1. POSE INVARIANCE — the same hand, moved/rotated/rescaled")
    print("=" * 74)
    h = person(1)
    ref = palm.signature(h)
    worst = 0.0
    for _ in range(300):
        v = palm.signature(pose(h, rnd))
        worst = max(worst, max(abs(a - b) for a, b in zip(ref, v)))
    print(f"   worst per-feature drift across 300 random poses: {worst:.2e}")
    print(f"   -> {'INVARIANT' if worst < 1e-6 else 'NOT INVARIANT — signature tracks pose'}")

    print()
    print("=" * 74)
    print("2. DISCRIMINATION — genuine vs impostor, with landmark noise")
    print("=" * 74)
    SIGMA = 0.012          # ~1.2% of the reference bone, a fair webcam figure
    people = {f"p{i}": person(100 + i) for i in range(12)}

    enrolments = []
    for name, hand in people.items():
        samples = [jitter(pose(hand, rnd), rnd, SIGMA) for _ in range(8)]
        enrolments.append(palm.enrol(name, samples))

    genuine, impostor = [], []
    for name, hand in people.items():
        mine = next(e for e in enrolments if e.subject == name)
        for _ in range(40):
            probe = palm.signature(jitter(pose(hand, rnd), rnd, SIGMA))
            genuine.append(palm.distance(mine, probe))
            for other in enrolments:
                if other.subject != name:
                    impostor.append(palm.distance(other, probe))

    def describe(label, xs):
        xs = sorted(xs)
        print(f"   {label:<9} n={len(xs):<5} mean={statistics.mean(xs):.2f}  "
              f"p5={xs[int(.05*len(xs))]:.2f}  median={xs[len(xs)//2]:.2f}  "
              f"p95={xs[int(.95*len(xs))]:.2f}")
    describe("genuine", genuine)
    describe("impostor", impostor)

    print()
    print(f"   {'threshold':>10} {'FRR':>8} {'FAR':>8}   (FRR: own hand rejected, FAR: someone else accepted)")
    best = None
    for t in [x / 4 for x in range(2, 25)]:
        frr = sum(1 for d in genuine if d > t) / len(genuine)
        far = sum(1 for d in impostor if d <= t) / len(impostor)
        if best is None or abs(frr - far) < abs(best[1] - best[2]):
            best = (t, frr, far)
        if t in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            print(f"   {t:>10.2f} {frr:>7.1%} {far:>7.1%}")
    print(f"\n   equal-error point ~ threshold {best[0]:.2f}  (FRR {best[1]:.1%}, FAR {best[2]:.1%})")
    print(f"   module default DEFAULT_THRESHOLD = {palm.DEFAULT_THRESHOLD}")

    print()
    print("=" * 74)
    print("3. IDENTIFICATION — pick the right person out of 12")
    print("=" * 74)
    hits = tot = 0
    margins = []
    for name, hand in people.items():
        for _ in range(40):
            r = palm.match(enrolments, jitter(pose(hand, rnd), rnd, SIGMA))
            tot += 1
            hits += (r["subject"] == name)
            if r["margin"] not in (None, float("inf")):
                margins.append(r["margin"])
    print(f"   correct identifications: {hits}/{tot} ({hits/tot:.1%})")
    print(f"   median margin to runner-up: {statistics.median(margins):.2f}")
    print()
    print("   Synthetic hands only. These numbers say the maths separates people who really")
    print("   differ; they say nothing about YOUR colleagues under YOUR lighting. Measure that")
    print("   with tools/palm_enrol_live.py before relying on the threshold.")


if __name__ == "__main__":
    main()
