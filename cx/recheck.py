"""Answering "that can't be right".

A customer who pushes back on a number is doing the assistant a favour, and the assistant
had no way to take it. Asked for last week's spending, then last month's, it gave the same
figure both times; asked how that could be, it had nothing — the sentence classified as
something unrelated and the thread died. Two different failures hide in that one moment:

* the figure might be **wrong**, in which case the only acceptable answer is to say so and
  give the right one, and
* the figure might be **right and surprising**, in which case repeating it is useless and
  the answer is the reason it is right.

Both need the same first step: actually work it out again.

## Re-checking has to mean re-computing

A recheck that reuses the cached rows behind the disputed answer is not a check. It is the
same arithmetic over the same input, and it will confirm a wrong number exactly as
confidently as a right one. So every path here re-reads the ledger with `fresh=True`,
bypassing the cache, and compares the new result with what was claimed.

That has already caught one real bug in this codebase: `history.recent` was returning fewer
rows than asked for from a warm cache, so a breakdown said AED 2,159 where the truth was
AED 4,469. A customer disputing that figure would have been told it was correct.

## What a verdict may say

* `corrected` — the recomputation disagrees. The claim was wrong; the new number is given
  and the correction is recorded.
* `stands` — the recomputation agrees, with **evidence** for why the surprising thing is
  true. Evidence is read off the data, never composed: if there is no evidence, the verdict
  is `stands` with an empty reason and the answer says only that it was checked again.
* `cannot_check` — there is nothing recorded to re-check, or the recomputation failed. It
  says that, rather than guessing which way it fell.

Never a fourth option where the assistant defends a number it has not recomputed.
"""
from __future__ import annotations

import time

import spend as spendmod

#: Two totals this close are the same number, not a disagreement. Rounding in the ledger and
#: rounding here are both to the fil, so anything under a fil apart is the same figure.
TOLERANCE = 0.01

#: How long a claim stays disputable. Long enough to cover a real conversation, short enough
#: that "are you sure?" never reaches back to something from a different session of work.
CLAIM_TTL = 45 * 60


def claim_spend(result: dict, said: str) -> dict:
    """Record what we just asserted, and enough of its working to test it again later."""
    return {
        "kind": "spend",
        "at": time.time(),
        "said": said,
        "period": result.get("period"),
        "period_label": result.get("period_label"),
        "total": result.get("total"),
        "payments": result.get("payments"),
        "oldest_hours": result.get("oldest_hours"),
        "next_outside_hours": result.get("next_outside_hours"),
        "refs": list(result.get("refs") or []),
        "live": result.get("live"),
    }


def live(claims: list[dict]) -> list[dict]:
    now = time.time()
    return [c for c in claims if now - c.get("at", 0) < CLAIM_TTL]


def describe(claim: dict) -> dict | None:
    """What a claim actually covered, so a question ABOUT it can be answered.

    "is it merchant related?" after a tally is not a dispute and not a new question — it
    asks what the figure we just gave was counted over. Before this it reached the dispute
    handler and got "I don\'t have a figure from this conversation to check", which was
    both unhelpful and untrue: there was a figure, and we knew exactly what was in it.
    """
    if not claim:
        return None
    scope = claim.get("scope")
    return {
        "kind": claim.get("kind"),
        "scope": scope,
        "covers": {"merchant": "merchants", "person": "people"}.get(scope, "everything"),
        "only": scope in ("merchant", "person"),
        "payee": claim.get("payee"),
        "period": claim.get("period"),
        "period_label": claim.get("period_label"),
        "payments": claim.get("payments"),
        "total": claim.get("total"),
        "name": claim.get("name"),
    }


def recheck(claim: dict, others: list[dict] | None = None) -> dict:
    """Work the claim out again from the ledger and say whether it holds."""
    if not claim:
        return {"verdict": "cannot_check", "why": [], "claim": claim}
    if claim.get("kind") == "rank":
        return _recheck_rank(claim)
    if claim.get("kind") != "spend":
        return {"verdict": "cannot_check", "why": [], "claim": claim}
    try:
        now = spendmod.analyse(claim.get("period") or "month", fresh=True)
    except Exception:                                         # noqa: BLE001
        return {"verdict": "cannot_check", "why": [], "claim": claim}

    was, is_now = float(claim.get("total") or 0.0), float(now.get("total") or 0.0)
    out = {"claim": claim, "now": now, "was": was, "is_now": is_now,
           "period_label": now.get("period_label")}
    if abs(was - is_now) > TOLERANCE:
        out["verdict"] = "corrected"
        out["why"] = _why_changed(claim, now)
        return out
    out["verdict"] = "stands"
    out["why"] = _why_it_holds(claim, now, others or [])
    return out


def _why_changed(claim: dict, now: dict) -> list[dict]:
    """What moved between the two runs. Facts only — if we cannot name the cause, we say the
    figure changed and leave the cause out rather than inventing one."""
    why = []
    before, after = int(claim.get("payments") or 0), int(now.get("payments") or 0)
    if after != before:
        why.append({"k": "count", "before": before, "after": after})
    old_refs, new_refs = set(claim.get("refs") or []), set(now.get("refs") or [])
    if old_refs and new_refs:
        added, gone = new_refs - old_refs, old_refs - new_refs
        if added:
            why.append({"k": "added", "n": len(added)})
        if gone:
            why.append({"k": "gone", "n": len(gone)})
    return why


def _why_it_holds(claim: dict, now: dict, others: list[dict]) -> list[dict]:
    """Evidence that a surprising figure is nevertheless the right one.

    Each entry is a fact read off the recomputation, for the engine to put into words in the
    customer's own language. Nothing here is phrased, and nothing is added when the data does
    not support it — an explanation the data cannot back is worse than no explanation,
    because it is indistinguishable from one it can.
    """
    why: list[dict] = []
    oldest = now.get("oldest_hours")
    outside = now.get("next_outside_hours")
    payments = now.get("payments") or 0

    # The case that started this: two different windows, the same number. It is not a bug if
    # both windows hold the same payments, and that is checkable.
    twin = next((c for c in others
                 if c.get("kind") == "spend"
                 and c.get("period") != claim.get("period")
                 and abs(float(c.get("total") or 0) - float(now.get("total") or 0)) <= TOLERANCE),
                None)
    if twin and set(twin.get("refs") or []) == set(now.get("refs") or []) and payments:
        # The period KEYS, not the English labels. The engine says this sentence in the
        # customer's own language, and a reason that reads "the last 7 days aur the last 30
        # days" is a translated sentence with two untranslated holes in it.
        why.append({"k": "same_payments", "n": payments,
                    "a": twin.get("period"), "b": now.get("period")})

    # ...because nothing happened in the rest of the longer window.
    if oldest is not None and outside is not None and outside > (now.get("span_hours") or 0):
        why.append({"k": "quiet_stretch",
                    "oldest_days": round(oldest/24.0, 1),
                    "gap_days": round(outside/24.0, 1)})
    elif oldest is not None and payments:
        why.append({"k": "window", "oldest_days": round(oldest/24.0, 1), "n": payments})

    if not now.get("live"):
        why.append({"k": "not_live"})
    return why


def claim_rank(result: dict, said: str) -> dict:
    """A ranking or a tally, recorded the same way a breakdown is, so it can be disputed.

    Stored under its own kind: re-checking "who did I pay the most" means re-running the
    ranking, not the category breakdown, and a `recheck` that quietly substituted one for
    the other would answer a challenge about the wrong number.
    """
    return {
        "kind": "rank",
        "at": time.time(),
        "said": said,
        "rank_kind": result.get("kind"),
        "period": result.get("period"),
        "period_label": result.get("period_label"),
        "scope": result.get("scope"),
        "payee": result.get("payee"),
        "total": (result.get("answer") or {}).get("total") if result.get("answer")
                 else result.get("total"),
        "name": (result.get("answer") or {}).get("name") if result.get("answer") else None,
        "payments": result.get("payments"),
        "live": result.get("live"),
    }


def _recheck_rank(claim: dict) -> dict:
    """A ranking or a tally, worked out again from the ledger.

    Separate from the spend path because re-running the wrong computation is not a check.
    A customer disputing "who did I pay the most" is owed the ranking again, not a category
    breakdown that happens to share a window.
    """
    import aggregate
    kind = claim.get("rank_kind") or "top"
    period = claim.get("period") or "month"
    scope = claim.get("scope") or "any"
    try:
        if kind == "tally":
            now = aggregate.tally(period, scope, payee=claim.get("payee"), fresh=True)
            is_now = float(now.get("total") or 0.0)
        elif kind in ("biggest", "smallest"):
            now = aggregate.extreme(kind, period, scope, fresh=True)
            is_now = float((now.get("answer") or {}).get("total") or 0.0)
        else:
            now = aggregate.rank(kind, period, scope, fresh=True)
            is_now = float((now.get("answer") or {}).get("total") or 0.0)
    except Exception:                                         # noqa: BLE001
        return {"verdict": "cannot_check", "why": [], "claim": claim}

    was = float(claim.get("total") or 0.0)
    out = {"claim": claim, "now": now, "was": was, "is_now": is_now,
           "period_label": now.get("period_label")}
    if abs(was - is_now) > TOLERANCE:
        out["verdict"] = "corrected"
        out["why"] = [{"k": "count", "before": claim.get("payments") or 0,
                       "after": now.get("payments") or 0}] \
            if (claim.get("payments") or 0) != (now.get("payments") or 0) else []
        return out
    out["verdict"] = "stands"
    why = []
    if now.get("payments"):
        why.append({"k": "window", "oldest_days": 0,
                    "n": now["payments"]})
    if not now.get("live"):
        why.append({"k": "not_live"})
    out["why"] = why
    return out
