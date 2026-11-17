"""Questions whose answer is a name or a single number, not a list.

    "who did I pay the most this month"
    "what's my biggest payment"
    "how many payments did I make"
    "how much did I send Alex"
    "which shop do I go to most"

Every one of these was wrong before this file existed, and wrong in a way that looked like
an answer. "who did I pay the most" listed five payments and announced "I don't have anyone
called Who on your list"; "which merchant do I go to most" offered a discount at a shop
1km away. The classifier was reaching for the nearest intent it had — and it did not have
this one, because a ranking question is not a history request with different words. A list
is not an answer to "who".

## The shape of an answer here

One fact, with the arithmetic behind it. `top_payee` returns the name AND the amount AND
how many payments it took AND what the runner-up was, because "Alex" on its own is
unverifiable and the second place is what makes the first place mean something.

## Ties are reported, not broken

Two people on exactly the same total is a real outcome, and picking whichever happened to
sort first would be inventing a fact. `tied` carries the others.

## People and merchants are counted separately

"Who did I pay the most" is about people; "which shop" is about merchants. Mixing them
would let a supermarket win a question about friends. `scope` decides, and it is derived
from the question rather than defaulted.
"""
from __future__ import annotations

import spend as spendmod

#: The windows, borrowed wholesale from the spend report so that "this month" means the
#: same thing in a ranking as it does in a breakdown. Two definitions of "month" in one
#: assistant is a bug waiting for someone to notice.
PERIODS = spendmod._PERIODS


def _rows(period: str, scope: str, fresh: bool = False):
    """Payments in the window, split the way the question asks for."""
    label, off, span = spendmod.window(period)
    rows, live = spendmod._rows(600, fresh=fresh)
    out = []
    for r in rows:
        if str(r.get("state", "SUCCESS")).upper() not in ("SUCCESS", "CREDITED"):
            continue
        amt = float(r.get("amount") or 0.0)
        h = float(r.get("hours_ago") or 0.0)
        if amt <= 0 or h < off or h >= off + span:
            continue
        is_merchant = bool(r.get("is_merchant"))
        if scope == "merchant" and not is_merchant:
            continue
        if scope == "person" and is_merchant:
            continue
        out.append(r)
    return out, label, live


def _totals(rows) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for r in rows:
        name = (r.get("payee") or "").strip() or "someone unnamed"
        e = by.setdefault(name, {"name": name, "total": 0.0, "count": 0,
                                 "merchant": bool(r.get("is_merchant"))})
        e["total"] += float(r.get("amount") or 0.0)
        e["count"] += 1
    for e in by.values():
        e["total"] = round(e["total"], 2)
    return by


def rank(kind: str = "top", period: str = "month", scope: str = "any",
         fresh: bool = False) -> dict:
    """Who or what got the most (or least) of the customer's money in the window."""
    rows, label, live = _rows(period, scope, fresh)
    out = {"kind": kind, "period": period, "period_label": label, "scope": scope,
           "live": live, "currency": "AED", "payments": len(rows)}
    if not rows:
        out["answer"] = None
        return out
    by = sorted(_totals(rows).values(), key=lambda e: e["total"],
                reverse=(kind != "bottom"))
    first = by[0]
    # A tie is a fact about the data, not a problem to be resolved by sort order.
    tied = [e for e in by[1:] if abs(e["total"] - first["total"]) < 0.005]
    out.update({"answer": first, "tied": tied,
                "runner_up": by[len(tied) + 1] if len(by) > len(tied) + 1 else None,
                "distinct": len(by),
                "total": round(sum(e["total"] for e in by), 2)})
    return out


def extreme(kind: str = "biggest", period: str = "month", scope: str = "any",
            fresh: bool = False) -> dict:
    """The single largest or smallest payment in the window."""
    rows, label, live = _rows(period, scope, fresh)
    out = {"kind": kind, "period": period, "period_label": label, "scope": scope,
           "live": live, "currency": "AED", "payments": len(rows)}
    if not rows:
        out["answer"] = None
        return out
    pick = (max if kind != "smallest" else min)(rows, key=lambda r: float(r.get("amount") or 0))
    out["answer"] = {"name": (pick.get("payee") or "").strip(),
                     "total": round(float(pick.get("amount") or 0), 2),
                     "count": 1, "merchant": bool(pick.get("is_merchant")),
                     "hours_ago": round(float(pick.get("hours_ago") or 0), 2),
                     "ref": pick.get("ref")}
    return out


def _same_person(wanted: str, row_name: str | None) -> bool:
    """Is this ledger row a payment to the person the question named?

    Substring matching is not enough and it failed on the first real name it met. The
    roster calls him "Alex Morgan"; the ledger calls him "Alex Umashankar Morgan
    Umashankar Gaurishankar". Neither string contains the other, so "how much did I send
    Alex this month" answered "I can't see any payments" about 57 of them.

    What actually identifies a person across the two is the name parts they share.
    """
    a = {t for t in _tokens(wanted) if len(t) > 2}
    b = {t for t in _tokens(row_name or "") if len(t) > 2}
    if not a or not b:
        return False
    return bool(a & b)


def _tokens(s: str) -> list[str]:
    # The engine's own tokeniser, which handles Devanagari and Arabic. A bare \W+ split
    # shreds Hindi names into single consonants — that bug cost nine call sites once.
    from engine import slots as slotlib
    return [t.lower() for t in slotlib.token_list((s or "").lower())]


def tally(period: str = "month", scope: str = "any", payee: str | None = None,
          fresh: bool = False) -> dict:
    """How many, how much in total, and the average — optionally for one payee only."""
    rows, label, live = _rows(period, scope, fresh)
    if payee:
        rows = [r for r in rows if _same_person(payee, r.get("payee"))]
    total = round(sum(float(r.get("amount") or 0) for r in rows), 2)
    return {"kind": "tally", "period": period, "period_label": label, "scope": scope,
            "live": live, "currency": "AED", "payee": payee,
            "payments": len(rows), "total": total,
            "average": round(total / len(rows), 2) if rows else 0.0,
            "distinct": len(_totals(rows))}
