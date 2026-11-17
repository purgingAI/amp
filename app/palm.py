"""Hand-geometry biometrics from MediaPipe Hands landmarks.

What this is, precisely, so nobody builds on a wrong assumption:

  * It identifies a person from the SHAPE of their hand — finger-length ratios, knuckle
    spacing, palm proportions — computed from the 21 landmarks MediaPipe returns.
  * It is NOT palm-print recognition. Ridge and crease patterns need contact imaging at
    hundreds of DPI; a laptop webcam at arm's length resolves nothing of the sort. Anyone
    promising palm prints from a webcam is describing hand geometry and calling it something
    grander.

Hand geometry is a real modality — it ran physical access control for decades — but a weak
one. Expect it to separate a dozen colleagues in decent light, not to withstand someone
deliberately trying to match another person's hand. So `match()` returns a score and a
margin, never a bare boolean, and the caller decides what a score is worth. In the transfer
flow it is one factor beside a signed session, not the whole gate.

Everything here is pose-invariant by construction: landmarks are translated to the wrist,
scaled by a reference bone, and rotated into a canonical frame before any feature is read.
Without that the "biometric" mostly measures how far away the hand was held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

N_LANDMARKS = 21

# The knuckle row plus the wrist: the rigid part of the hand. Features are built from these
# and from bone lengths, never from fingertip positions alone, because tips move with flexion
# and would make the signature depend on how the hand happened to be posed.
_PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# (name, chain) — summed segment lengths, so a bent finger measures the same as a straight one.
_BONES = [
    ("thumb",  [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]),
    ("index",  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]),
    ("middle", [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]),
    ("ring",   [RING_MCP, RING_PIP, RING_DIP, RING_TIP]),
    ("pinky",  [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]),
]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def _dist(a, b):
    return _norm(_sub(a, b))


def canonicalise(landmarks: list) -> list:
    """Put a hand into a fixed frame: wrist at the origin, unit scale, palm in a known plane.

    Three degrees of freedom are removed in order — position, size, orientation — because a
    signature that survives none of them is just a record of where the camera was.

    Scale uses the wrist-to-middle-knuckle bone. It is the most stable length on the hand:
    it does not change with finger flexion, and it is visible in every frame where the hand
    is detected at all.
    """
    if len(landmarks) != N_LANDMARKS:
        raise ValueError(f"expected {N_LANDMARKS} landmarks, got {len(landmarks)}")
    pts = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in landmarks]

    origin = pts[WRIST]
    pts = [_sub(p, origin) for p in pts]

    scale = _norm(pts[MIDDLE_MCP])
    if scale < 1e-9:
        raise ValueError("degenerate hand: wrist and middle knuckle coincide")
    pts = [(p[0] / scale, p[1] / scale, p[2] / scale) for p in pts]

    # x along the wrist->middle-knuckle axis; z normal to the palm; y completes the frame.
    x_axis = _unit(pts[MIDDLE_MCP])
    across = _sub(pts[PINKY_MCP], pts[INDEX_MCP])
    z_axis = _unit(_cross(x_axis, across))
    if _norm(z_axis) < 1e-9:
        raise ValueError("degenerate hand: knuckles collinear with the palm axis")
    y_axis = _cross(z_axis, x_axis)
    return [(_dot(p, x_axis), _dot(p, y_axis), _dot(p, z_axis)) for p in pts]


def signature(landmarks: list) -> list:
    """A scale-free feature vector for one hand.

    Ratios throughout: every length is already divided by the reference bone in
    canonicalise, so the vector describes proportion rather than size. Two hands of
    different absolute size but identical proportion are deliberately the same signature —
    distance from the camera must not be a biometric.
    """
    p = canonicalise(landmarks)
    feats: list[float] = []

    # finger lengths, summed along each chain so flexion does not shorten them
    lengths = {}
    for name, chain in _BONES:
        total = sum(_dist(p[chain[i]], p[chain[i + 1]]) for i in range(len(chain) - 1))
        lengths[name] = total
        feats.append(total)

    # length ratios between fingers — the part people differ on most
    order = [n for n, _ in _BONES]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = lengths[order[i]], lengths[order[j]]
            feats.append(a / b if b > 1e-9 else 0.0)

    # knuckle spacing: the rigid geometry of the palm
    for i in range(len(_PALM)):
        for j in range(i + 1, len(_PALM)):
            feats.append(_dist(p[_PALM[i]], p[_PALM[j]]))

    # palm breadth against palm length, and how far the thumb sits down the hand
    feats.append(_dist(p[INDEX_MCP], p[PINKY_MCP]))
    feats.append(_dist(p[WRIST], p[THUMB_CMC]))

    # splay: the angle each knuckle makes at the wrist, which is skeletal rather than postural
    for idx in (INDEX_MCP, RING_MCP, PINKY_MCP):
        v1, v2 = _unit(p[MIDDLE_MCP]), _unit(p[idx])
        feats.append(math.acos(max(-1.0, min(1.0, _dot(v1, v2)))))

    return feats


FEATURE_COUNT = len(signature([
    (math.cos(i * 0.3) * (1 + i * 0.05), math.sin(i * 0.3) * (1 + i * 0.05), i * 0.01)
    for i in range(N_LANDMARKS)
]))


@dataclass
class Enrolment:
    """What a person's hand looks like, learned from several samples.

    Per-feature spread is kept, not just the mean: finger-length ratios are far more
    repeatable than splay angles, and a match that weights them equally is dominated by the
    noisiest feature. Storing the spread lets match() weight each one by how reliably that
    person reproduces it.
    """
    subject: str
    mean: list
    spread: list
    samples: int
    created_at: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"subject": self.subject, "mean": self.mean, "spread": self.spread,
                "samples": self.samples, "created_at": self.created_at, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "Enrolment":
        return Enrolment(subject=d["subject"], mean=list(d["mean"]), spread=list(d["spread"]),
                         samples=int(d.get("samples", 0)), created_at=float(d.get("created_at", 0.0)),
                         meta=dict(d.get("meta") or {}))


# A floor on per-feature spread. Without it, a feature that happened to be identical across
# a handful of enrolment samples gets a near-zero denominator in match() and a tiny real
# difference explodes into a huge z-score — one accidentally-consistent feature would then
# reject the very person it was enrolled from.
_MIN_SPREAD = 0.01
# Relative floor: no feature may claim to be more repeatable than 4% of its own magnitude. The
# absolute floor alone is generous for features near 0.2 and absurdly tight for features near 3.3,
# and the consequence was measured on a real hand — its own enrolment refused it at 2.28-3.20
# because the denominators came from eight unusually quiet frames. See the Go port, which is now
# the deployed implementation; these two must stay bit-identical or the cross-check test is a lie.
_MIN_REL_SPREAD = 0.04
# Drop the worst 3 of 30 per-feature z-scores: measured, the worst three contributed 30% of the
# total distance. An impostor differs across many features; a genuine hand in an awkward frame
# usually differs badly in only a few.
_TRIM_WORST = 3

MIN_SAMPLES = 5


def enrol(subject: str, samples: list, meta: dict | None = None) -> Enrolment:
    """Build an enrolment from several frames of the same hand."""
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} samples, got {len(samples)}")
    import time as _t
    vecs = [signature(s) if len(s) == N_LANDMARKS else list(s) for s in samples]
    n, dim = len(vecs), len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError("samples produced feature vectors of differing length")
    mean = [sum(v[i] for v in vecs) / n for i in range(dim)]
    spread = []
    for i in range(dim):
        var = sum((v[i] - mean[i]) ** 2 for v in vecs) / max(1, n - 1)
        spread.append(max(math.sqrt(var), _MIN_SPREAD, _MIN_REL_SPREAD * abs(mean[i])))
    return Enrolment(subject=subject, mean=mean, spread=spread, samples=n,
                     created_at=_t.time(), meta=meta or {})


def distance(enrolment: Enrolment, landmarks_or_vec: list) -> float:
    """Mean absolute z-score of a presented hand against an enrolment.

    Roughly: how many standard deviations, averaged over features, this hand sits from the
    enrolled one. Scale-free, so a single threshold is meaningful across people — unlike a
    raw euclidean distance, which would drift with hand size.
    """
    vec = (signature(landmarks_or_vec)
           if len(landmarks_or_vec) == N_LANDMARKS and not isinstance(landmarks_or_vec[0], float)
           else list(landmarks_or_vec))
    if len(vec) != len(enrolment.mean):
        raise ValueError("feature length mismatch against enrolment")
    zs = sorted(abs(v - m) / s for v, m, s in zip(vec, enrolment.mean, enrolment.spread))
    keep = zs[:-_TRIM_WORST] if len(zs) > _TRIM_WORST * 2 else zs
    return sum(keep) / len(keep)


# The equal-error point measured by tools/palm_bench.py on synthetic hands with realistic
# landmark noise: FRR 8.8%, FAR 6.9%. The first draft of this file said 2.0, which the same
# benchmark showed accepts roughly one impostor presentation in five — the number was picked
# by eye and was wrong, which is the argument for having the benchmark at all.
#
# Even at the equal-error point this is a weak gate: ~7% of the time another person's hand
# scores as yours. Hand geometry is a FACTOR, not an identity. The transfer flow pairs it
# with a signed session for that reason.
#
# Re-measure on real hands before trusting it. A threshold carried over from synthetic data
# is a guess wearing a number's clothing.
DEFAULT_THRESHOLD = 1.5


def match(enrolments: list, landmarks: list, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Identify a hand against everyone enrolled.

    Returns the best subject, its distance, and the margin to the runner-up. The margin is
    the part worth gating on: a hand that fits two people almost equally well has identified
    nobody, however good its absolute score. Callers get the numbers and decide; this never
    returns a bare yes.
    """
    if not enrolments:
        return {"subject": None, "distance": None, "margin": None, "accepted": False,
                "reason": "no enrolments"}
    vec = signature(landmarks)
    scored = sorted(((distance(e, vec), e.subject) for e in enrolments), key=lambda t: t[0])
    best_d, best_s = scored[0]
    margin = (scored[1][0] - best_d) if len(scored) > 1 else float("inf")
    accepted = best_d <= threshold
    reason = "ok" if accepted else f"distance {best_d:.2f} over threshold {threshold:.2f}"
    return {"subject": best_s if accepted else None, "distance": best_d, "margin": margin,
            "accepted": accepted, "reason": reason,
            "runners_up": [{"subject": s, "distance": d} for d, s in scored[1:4]]}
