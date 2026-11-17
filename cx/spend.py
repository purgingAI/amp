"""What the customer actually spends their money on, by category and by period.

## Why categories live here and not in the merchant record (yet)

A merchant's category belongs on the merchant, set once when the merchant is onboarded.
It is not there today, so this file carries a static table instead. That is a deliberate
staging post, not a design: when `category` lands on merchant creation, `categorise()`
becomes a single field read and `MERCHANT_CATEGORY` becomes the one-time migration script.
Everything else in this module — the windows, the aggregation, the comparison — is written
against `categorise()` and does not care where the answer comes from.

## Uncategorised is a real answer

Dev ledgers contain merchants called "aaa" and "KVVIZM WRGGZ". Guessing a category for those
would put invented numbers in front of a customer looking at their own money, so anything the
table and the keywords cannot place lands in `other` and is reported as uncategorised. A
spend breakdown that admits it could not place 12% of the spend is worth more than one that
silently files it under "services".

## The comparison is the point

"You spent AED 1,240 on food" means very little on its own. "AED 1,240 on food — AED 380 more
than the month before, and the increase is almost entirely one merchant" is something a
customer can act on. So every category carries its previous-period figure, and the headline is
chosen by the largest movement rather than the largest total.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

#: The closed set, mirroring the five the product asked for plus an honest escape hatch.
#: `other` is never shown as a category name — it is reported as "uncategorised".
CATEGORIES: tuple[str, ...] = (
    "food", "electronics", "healthcare", "entertainment", "services", "other",
)

LABELS = {
    "food": "Food & groceries",
    "electronics": "Electronics",
    "healthcare": "Healthcare",
    "entertainment": "Entertainment",
    "services": "Services & bills",
    "other": "Uncategorised",
}

#: Exact trading names we know. Checked before the keywords so a merchant whose name happens
#: to contain a keyword ("OpenWallet Trading") is never dragged into the wrong category.
MERCHANT_CATEGORY: dict[str, str] = {
    # real dev merchants
    "mclothes": "services",              # a clothing retailer — see the note on `retail` below
    "cystal healing global fzc": "healthcare",
    "etisalat landline": "services",
    "du": "services",
    # the seeded demo corpus
    "carrefour": "food", "lulu hypermarket": "food", "talabat": "food",
    "starbucks": "food", "costa coffee": "food",
    "sharaf dg": "electronics", "jumbo electronics": "electronics", "apple store": "electronics",
    "aster clinic": "healthcare", "life pharmacy": "healthcare", "medcare hospital": "healthcare",
    "vox cinemas": "entertainment", "reel cinemas": "entertainment", "spotify": "entertainment",
    "netflix": "entertainment",
    "dewa": "services", "salik": "services", "emirates nbd": "services",
}

#: Word-boundary keyword rules, tried in order. These are whole-word matches on purpose:
#: substring matching is how the UAE telecom "du" got found inside "dukan" once already.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("food",          ("grocery", "groceries", "supermarket", "hypermarket", "restaurant",
                       "cafe", "coffee", "kitchen", "foods", "bakery", "catering", "mart")),
    ("electronics",   ("electronics", "electronic", "computers", "mobiles", "digital",
                       "gadget", "appliances")),
    ("healthcare",    ("clinic", "pharmacy", "hospital", "medical", "healing", "health",
                       "dental", "optical", "wellness")),
    ("entertainment", ("cinema", "cinemas", "movies", "gaming", "games", "entertainment",
                       "music", "tickets", "park")),
    ("services",      ("telecom", "landline", "mobile", "internet", "electricity", "water",
                       "utility", "utilities", "insurance", "salon", "laundry", "clothing",
                       "clothes", "tailor", "rent", "parking", "toll", "bank")),
)


def categorise(name: str | None) -> tuple[str, str]:
    """`(category, basis)` for a merchant name. `basis` says WHY, so the UI can show it.

    Returned rather than logged because "why is my cinema ticket under services" is a
    question a customer will ask, and the answer has to be inspectable.
    """
    low = (name or "").strip().lower()
    if not low:
        return "other", "no merchant name"
    exact = MERCHANT_CATEGORY.get(low)
    if exact:
        return exact, "known merchant"
    for cat, words in _KEYWORDS:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                return cat, f"matched “{w}”"
    return "other", "no category on this merchant yet"


# ---------------------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------------------

#: (label, hours_back_start, hours_back_end). The window is [end, start) hours ago, so
#: "last 30 days" is 0..720 and the period before it is 720..1440 — same length, immediately
#: before, which is the only comparison that means anything.
_PERIODS: dict[str, tuple[str, float]] = {
    "week":      ("the last 7 days",  24 * 7),
    "fortnight": ("the last 14 days", 24 * 14),
    "month":     ("the last 30 days", 24 * 30),
    "quarter":   ("the last 90 days", 24 * 90),
}

#: The window BEFORE one of the above: same length, immediately earlier. Never read from a
#: sentence on its own — "my spending last month" means the last 30 days to most people, and
#: the tests hold it to that. It is chosen only for a FOLLOW-UP that contrasts with the
#: window just answered: "who did I pay the most this month" -> "and last month?" means the
#: month before, and answering the same 30 days again (as it did) is answering nothing.
PREV = "prev_"


def window(period: str) -> tuple[str, float, float]:
    """(label, hours ago the window starts, its length in hours)."""
    if period.startswith(PREV):
        base = period[len(PREV):]
        _, span = _PERIODS.get(base, _PERIODS["month"])
        return period_label(period), span, span
    label, span = _PERIODS.get(period, _PERIODS["month"])
    return label, 0.0, span


#: Words that say "the one before", which is what turns a repeated span into the previous one.
_PREVIOUS_WORDS = re.compile(
    r"\b(?:last|previous|prior|before|pichle|pichhle|pichla|pichli|pehle|pahle)\b", re.I)
_PREVIOUS_WORDS_INTL = ("पिछले", "पिछला", "पिछली", "पहले", "الماضي", "الماضية", "السابق",
                        "السابقة", "قبل")


def follow_up_period(previous_question: str, follow_up: str) -> str | None:
    """The window a follow-up means, given the question it follows.

    "this month" -> "and last month?" names the SAME span with a "previous" word, so it means
    the window before. Anything else — a different span ("and last week?"), or no previous
    word — means exactly what it says, and None is returned so the caller reads it as usual.
    """
    new = period_named(follow_up)
    if not new or new.startswith(PREV):
        return None
    old = period_from(previous_question)
    low = (follow_up or "").lower()
    says_previous = bool(_PREVIOUS_WORDS.search(low)) or any(w in low for w in _PREVIOUS_WORDS_INTL)
    if new == old and says_previous:
        return PREV + new
    return None


_PERIOD_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("week",      ("this week", "last week", "past week", "7 days", "seven days",
                   "is hafte", "pichle hafte")),
    ("fortnight", ("fortnight", "14 days", "two weeks", "2 weeks")),
    ("quarter",   ("quarter", "90 days", "three months", "3 months", "teen mahine")),
    ("month",     ("this month", "last month", "past month", "30 days", "thirty days",
                   "mahine", "mahina", "is mahine", "pichle mahine")),
)

#: Period words in scripts where `in` is the right test — see the note in the engine on why
#: `\b` cannot be used here.
_PERIOD_WORDS_INTL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("week",    ("हफ़्ते", "हफ्ते", "सप्ताह", "أسبوع")),
    ("quarter", ("तीन महीने", "तिमाही", "ثلاثة أشهر")),
    ("month",   ("महीने", "महीना", "شهر")),
)


#: Words that name a category in the question itself. Whole-word, same reason as everywhere
#: else: "services" must not be found inside another word, and "du" must never match "dukan".
_CATEGORY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("food",          ("food", "grocery", "groceries", "eating", "restaurants", "khana",
                       "khane", "grocery ka")),
    ("electronics",   ("electronics", "gadgets", "devices", "phone", "laptop")),
    ("healthcare",    ("healthcare", "health", "medical", "doctor", "pharmacy", "clinic",
                       "dawai", "ilaaj")),
    ("entertainment", ("entertainment", "movies", "cinema", "fun", "subscriptions")),
    ("services",      ("services", "bills", "utilities", "utility", "bijli", "recharge")),
)


#: Category names in other scripts, matched by containment for the same reason.
_CATEGORY_WORDS_INTL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("food",          ("खाने", "खाना", "किराना", "الطعام", "بقالة")),
    ("electronics",   ("इलेक्ट्रॉनिक", "إلكترونيات")),
    ("healthcare",    ("सेहत", "दवाई", "इलाज", "डॉक्टर", "صحة", "دواء")),
    ("entertainment", ("मनोरंजन", "फ़िल्म", "फिल्म", "ترفيه")),
    ("services",      ("बिजली", "बिल", "सेवा", "فواتير", "خدمات")),
)


def focus_from(said: str | None) -> str | None:
    """The category the customer actually named, if any. `None` means "the whole picture"."""
    low = (said or "").lower()
    for cat, words in _CATEGORY_WORDS_INTL:
        for w in words:
            if w in low:
                return cat
    for cat, words in _CATEGORY_WORDS:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                return cat
    return None


def period_named(said: str | None) -> str | None:
    """The window the customer ACTUALLY named, or None if they named none.

    `period_from` cannot answer this, because it falls back to a month — the right default
    for "my spending", and useless for deciding whether a sentence is about a window at all.
    Telling the two apart is what lets "and last week?" be recognised as carrying on a
    spend conversation rather than starting a new one.
    """
    got = period_from(said)
    return got if _NAMES_A_PERIOD(said) else None


def _NAMES_A_PERIOD(said: str | None) -> bool:
    low = (said or "").lower()
    if any(w in low for _, ws in _PERIOD_WORDS_INTL for w in ws):
        return True
    if any(w in low for _, ws in _PERIOD_WORDS for w in ws):
        return True
    return _spanned(low) is not None


def period_from(said: str | None) -> str:
    """Which window the customer asked about. Defaults to a month — the period people mean
    when they say "my spending" without qualifying it."""
    low = (said or "").lower()
    for key, words in _PERIOD_WORDS_INTL:
        for w in words:
            if w in low:
                return key
    for key, words in _PERIOD_WORDS:
        for w in words:
            if w in low:
                return key
    got = _spanned(low)
    return got or "month"


#: Numbers people say rather than type. "what about three months" named a window perfectly
#: clearly and was read as a bill-scheduling request, because the period parser only knew
#: digits.
_WORD_NUM = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
             "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "chhe": 6,
             "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6,
             "واحد": 1, "شهرين": 2, "اثنين": 2, "ثلاثة": 3, "أربعة": 4, "ستة": 6}

#: How many days a unit is worth, so "three months" and "90 days" land on the same window.
_UNIT_DAYS = {"day": 1, "days": 1, "din": 1, "दिन": 1, "يوم": 1, "أيام": 1,
              "week": 7, "weeks": 7, "hafta": 7, "hafte": 7, "हफ़्ते": 7, "हफ्ते": 7,
              "सप्ताह": 7, "أسبوع": 7, "أسابيع": 7,
              "month": 30, "months": 30, "mahina": 30, "mahine": 30, "महीने": 30,
              "महीना": 30, "شهر": 30, "أشهر": 30, "شهور": 30,
              "year": 365, "years": 365, "saal": 365, "साल": 365, "سنة": 365}

_SPAN = re.compile(
    r"\b(\d{1,3}|" + "|".join(sorted(_WORD_NUM, key=len, reverse=True)) + r")\s*"
    r"(" + "|".join(sorted(_UNIT_DAYS, key=len, reverse=True)) + r")\b", re.I)


def _spanned(low: str) -> str | None:
    """A window written as a number and a unit: "7 days", "three months", "do hafte"."""
    m = _SPAN.search(low)
    if not m:
        return None
    n = m.group(1)
    n = int(n) if n.isdigit() else _WORD_NUM.get(n.lower())
    if not n:
        return None
    days = n * _UNIT_DAYS.get(m.group(2).lower(), 1)
    if days <= 10:
        return "week"
    if days <= 20:
        return "fortnight"
    if days <= 45:
        return "month"
    return "quarter"


# ---------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------

#: Period and category names in the customer's language. Without these the translated
#: sentence comes out half-English — "the last 30 days me aapne AED 5,229 kharch kiye" — which
#: is the same mistake as having English-only code behind a multilingual model, just one layer
#: further out. The data labels have to travel with the sentence.
PERIOD_LABELS: dict[str, dict[str, str]] = {
    "en":       {"week": "the last 7 days", "fortnight": "the last 14 days",
                 "month": "the last 30 days", "quarter": "the last 90 days",
                 "prev_week": "the 7 days before that (7–14 days ago)",
                 "prev_fortnight": "the 14 days before that (14–28 days ago)",
                 "prev_month": "the 30 days before that (30–60 days ago)",
                 "prev_quarter": "the 90 days before that (90–180 days ago)"},
    "hi":       {"week": "पिछले 7 दिनों", "fortnight": "पिछले 14 दिनों",
                 "month": "पिछले 30 दिनों", "quarter": "पिछले 90 दिनों",
                 "prev_week": "उससे पहले के 7 दिनों (7–14 दिन पहले)",
                 "prev_fortnight": "उससे पहले के 14 दिनों (14–28 दिन पहले)",
                 "prev_month": "उससे पहले के 30 दिनों (30–60 दिन पहले)",
                 "prev_quarter": "उससे पहले के 90 दिनों (90–180 दिन पहले)"},
    "hinglish": {"week": "pichle 7 dino", "fortnight": "pichle 14 dino",
                 "month": "pichle 30 dino", "quarter": "pichle 90 dino",
                 "prev_week": "usse pehle ke 7 dino (7–14 din pehle)",
                 "prev_fortnight": "usse pehle ke 14 dino (14–28 din pehle)",
                 "prev_month": "usse pehle ke 30 dino (30–60 din pehle)",
                 "prev_quarter": "usse pehle ke 90 dino (90–180 din pehle)"},
    "ar":       {"week": "آخر 7 أيام", "fortnight": "آخر 14 يوماً",
                 "month": "آخر 30 يوماً", "quarter": "آخر 90 يوماً",
                 "prev_week": "الأيام السبعة التي سبقتها (قبل 7–14 يوماً)",
                 "prev_fortnight": "الأربعة عشر يوماً التي سبقتها (قبل 14–28 يوماً)",
                 "prev_month": "الثلاثين يوماً التي سبقتها (قبل 30–60 يوماً)",
                 "prev_quarter": "التسعين يوماً التي سبقتها (قبل 90–180 يوماً)"},
}

CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "en":       dict(LABELS),
    "hi":       {"food": "खाने-पीने", "electronics": "इलेक्ट्रॉनिक्स", "healthcare": "सेहत",
                 "entertainment": "मनोरंजन", "services": "सेवाओं और बिलों",
                 "other": "बिना श्रेणी"},
    "hinglish": {"food": "khane-peene", "electronics": "electronics", "healthcare": "sehat",
                 "entertainment": "entertainment", "services": "services aur bills",
                 "other": "bina category"},
    "ar":       {"food": "الطعام والبقالة", "electronics": "الإلكترونيات",
                 "healthcare": "الرعاية الصحية", "entertainment": "الترفيه",
                 "services": "الخدمات والفواتير", "other": "غير مصنّف"},
}


def period_label(period: str, lang: str = "en") -> str:
    table = PERIOD_LABELS.get(lang) or PERIOD_LABELS["en"]
    return table.get(period) or PERIOD_LABELS["en"].get(period, period)


def category_label(category: str, lang: str = "en") -> str:
    table = CATEGORY_LABELS.get(lang) or CATEGORY_LABELS["en"]
    return table.get(category) or LABELS.get(category, category)


@dataclass
class CategorySpend:
    category: str
    label: str
    total: float = 0.0
    count: int = 0
    previous: float = 0.0
    merchants: dict = field(default_factory=dict)   # name -> total, this period

    @property
    def delta(self) -> float:
        return round(self.total - self.previous, 2)

    @property
    def delta_pct(self) -> float | None:
        if not self.previous:
            return None
        return round((self.total - self.previous) / self.previous * 100.0, 1)

    def top_merchant(self) -> tuple[str, float] | None:
        if not self.merchants:
            return None
        name, amt = max(self.merchants.items(), key=lambda kv: kv[1])
        return name, round(amt, 2)

    def public(self) -> dict:
        top = self.top_merchant()
        return {"category": self.category, "label": self.label,
                "total": round(self.total, 2), "count": self.count,
                "previous": round(self.previous, 2), "delta": self.delta,
                "delta_pct": self.delta_pct,
                "top_merchant": top[0] if top else None,
                "top_merchant_amount": top[1] if top else None}


def _demo_rows() -> list[dict]:
    """A realistic three-month merchant corpus, for showing the breakdown when the dev ledger
    cannot (its merchants are named "aaa"). Behind CX_SPEND_DEMO=1 and never mixed with live
    rows — the same rule `transactions_for` already follows, and for the same reason."""
    plan = [
        # (merchant, amount, hours_ago)  — roughly two months, with a deliberate food jump
        ("Carrefour", 412, 60), ("Carrefour", 268, 180), ("Carrefour", 331, 300),
        ("Lulu Hypermarket", 290, 120), ("Talabat", 88, 40), ("Talabat", 102, 150),
        ("Talabat", 76, 220), ("Starbucks", 32, 30), ("Starbucks", 28, 200),
        ("Sharaf DG", 1899, 210), ("Jumbo Electronics", 349, 500),
        ("Aster Clinic", 260, 90), ("Life Pharmacy", 118, 260),
        ("VOX Cinemas", 95, 110), ("Spotify", 24, 70), ("Netflix", 42, 400),
        ("du", 310, 80), ("DEWA", 455, 140), ("Salik", 50, 340),
        # the period before, so the comparison has something to compare against
        ("Carrefour", 240, 800), ("Lulu Hypermarket", 198, 900), ("Talabat", 64, 850),
        ("Sharaf DG", 1299, 1000), ("Aster Clinic", 300, 820), ("Life Pharmacy", 90, 980),
        ("VOX Cinemas", 150, 870), ("Netflix", 42, 1100), ("du", 310, 810),
        ("DEWA", 420, 860), ("Salik", 70, 1040), ("Starbucks", 46, 760),
    ]
    return [{"payee": n, "amount": float(a), "currency": "AED", "hours_ago": float(h),
             "is_merchant": True, "state": "SUCCESS", "ref": f"DEMO{i:04d}",
             "product": "Merchant Payment"}
            for i, (n, a, h) in enumerate(plan)]


def _rows(limit: int = 600, fresh: bool = False) -> tuple[list[dict], bool]:
    """Merchant payments to analyse, and whether they are live. Demo rows REPLACE live ones
    rather than topping them up: a breakdown that blends real spending with invented spending
    is not something a customer could ever reconcile against their own statement."""
    if os.environ.get("CX_SPEND_DEMO") == "1":
        return _demo_rows(), False
    try:
        import history
        rows = history.recent(limit=limit, fresh=fresh)
    except Exception:
        rows = None
    if not rows:
        return _demo_rows(), False
    return rows, True


def analyse(period: str = "month", limit: int = 600, focus: str | None = None,
            fresh: bool = False) -> dict:
    """The whole report for one window, with the window before it for comparison.

    `focus` names the category the customer asked about, so "how much did I spend on food"
    leads with food instead of with whatever moved most. The rest of the breakdown is still
    returned — the question is about one category, but the comparison is what makes the
    number mean anything.
    """
    label, off, span = window(period)
    rows, live = _rows(limit, fresh=fresh)

    cats: dict[str, CategorySpend] = {
        c: CategorySpend(category=c, label=LABELS[c]) for c in CATEGORIES
    }
    basis: dict[str, str] = {}
    total = prev_total = 0.0
    n_rows = 0
    oldest = newest = next_out = None
    refs: set[str] = set()

    for r in rows:
        # Only merchant spending. A transfer to a person is not a category of spend — filing
        # "sent 500 to Alex" under a category would make the breakdown add up to something
        # the customer never spent in a shop.
        if not r.get("is_merchant"):
            continue
        if str(r.get("state", "SUCCESS")).upper() not in ("SUCCESS", "CREDITED"):
            continue
        h = float(r.get("hours_ago") or 0.0) - off      # hours before the window's newest edge
        amt = float(r.get("amount") or 0.0)
        if amt <= 0 or h < 0:
            continue
        name = r.get("payee") or ""
        cat, why = categorise(name)
        basis.setdefault(name, why)
        if h < span:
            c = cats[cat]
            c.total += amt
            c.count += 1
            c.merchants[name] = c.merchants.get(name, 0.0) + amt
            total += amt
            n_rows += 1
            oldest = h if oldest is None else max(oldest, h)
            newest = h if newest is None else min(newest, h)
            if r.get("ref"):
                refs.add(str(r["ref"]))
        else:
            next_out = h if next_out is None else min(next_out, h)
        if span <= h < span * 2:
            cats[cat].previous += amt
            prev_total += amt

    present = [c for c in cats.values() if c.total or c.previous]
    present.sort(key=lambda c: c.total, reverse=True)

    # The headline is the biggest MOVE, not the biggest total — the biggest total is usually
    # the same category every month and tells the customer nothing they did not know.
    movers = [c for c in present if c.previous and c.category != "other"]
    mover = max(movers, key=lambda c: abs(c.delta), default=None)
    # A named category always wins: they asked about food, so answer about food.
    if focus:
        named = next((c for c in present if c.category == focus), None)
        if named is not None:
            mover = named

    uncategorised = cats["other"].total
    return {
        #: The SHAPE of the window, not just its total. Without these a challenge to the
        #: figure can only be answered by repeating the figure. "Last week and last month are
        #: the same number" is a fair thing to doubt, and the answer to it is that both
        #: windows hold the same 36 payments and the oldest is 5 days old — which needs the
        #: oldest and the gap, not the total.
        "span_hours": span,
        # hours ago from NOW, whatever window was asked for
        "oldest_hours": round(oldest + off, 2) if oldest is not None else None,
        "newest_hours": round(newest + off, 2) if newest is not None else None,
        "next_outside_hours": round(next_out + off, 2) if next_out is not None else None,
        "refs": sorted(refs),
        "period": period,
        "period_label": label,
        "live": live,
        "currency": "AED",
        "total": round(total, 2),
        "previous_total": round(prev_total, 2),
        "delta": round(total - prev_total, 2),
        "payments": n_rows,
        "categories": [c.public() for c in present],
        "headline": _headline(mover, total, prev_total, label),
        #: The headline in parts, so the engine can say it in the customer's language rather
        #: than translating a finished English sentence.
        "headline_parts": _headline_parts(mover, total, period),
        "focus": focus,
        "mover": mover.category if mover else None,
        "mover_label": mover.label if mover else None,
        "uncategorised": round(uncategorised, 2),
        "uncategorised_pct": round(uncategorised / total * 100.0, 1) if total else 0.0,
        "basis": basis,
    }


def _headline(mover: CategorySpend | None, total: float, prev: float, label: str) -> str:
    """One sentence a customer can act on. Deliberately plain: this is the line that gets
    read out, and a sentence with three clauses and a percentage is not read out well."""
    if not total:
        return f"You have no merchant spending in {label}."
    if mover is None or not mover.delta:
        return f"You spent AED {total:,.2f} at merchants in {label}."
    direction = "more" if mover.delta > 0 else "less"
    line = (f"You spent AED {total:,.2f} in {label} — "
            f"AED {abs(mover.delta):,.2f} {direction} on {mover.label.lower()} "
            f"than the period before.")
    top = mover.top_merchant()
    if top and mover.delta > 0 and top[1] >= abs(mover.delta) * 0.6:
        line += f" Most of that is {top[0]}."
    return line


def _headline_parts(mover: CategorySpend | None, total: float, period: str) -> dict:
    """The pieces of the headline, for translation. `key` picks the sentence shape; `vals`
    fills it. Kept beside `_headline` so the English and the translated versions cannot drift
    apart without this function changing."""
    if not total:
        return {"key": "spend_none", "vals": {}, "period": period, "category": None}
    if mover is None or not mover.delta:
        return {"key": "spend_flat",
                "vals": {"total": f"{total:,.2f}", "cur": "AED"},
                "period": period, "category": None}
    top = mover.top_merchant()
    blame = bool(top and mover.delta > 0 and top[1] >= abs(mover.delta) * 0.6)
    # Separate keys per direction rather than a {direction} placeholder: "more" and "less"
    # are the part of this sentence that has to be translated, and a placeholder would carry
    # the English word straight through into the Hindi answer.
    up = mover.delta > 0
    key = ("spend_more" if up else "spend_less") + ("_blame" if blame else "")
    return {
        "key": key,
        "vals": {"total": f"{total:,.2f}", "cur": "AED",
                 "delta": f"{abs(mover.delta):,.2f}",
                 "merchant": top[0] if top else ""},
        "period": period, "category": mover.category,
    }
