"""Pull the facts out of the sentence. Deterministic, and it refuses rather than guesses.

Shares the amount parser with the voice flow (`app.intent.parse_amount`) on purpose: that
parser is the one that learned, expensively, that "1 million" is not 1. Having a second
amount parser here is how that bug comes back.
"""
from __future__ import annotations

import os
import re
import sys

AMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if AMP not in sys.path:
    sys.path.insert(0, AMP)

CX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CX not in sys.path:
    sys.path.insert(0, CX)

from app.intent import parse_amount, parse_currency          # noqa: E402
from taxonomy import COUNTRY_TO_CORRIDOR, Slots              # noqa: E402
import data                                                  # noqa: E402
import journeys                                              # noqa: E402

#: OpenWallet references are long digit strings. Anything shorter is an amount, not a reference —
#: treating "250" as a transaction id is how a customer gets told about the wrong payment.
_REF = re.compile(r"\b(\d{13,20}|[A-Z]{2,5}\d{4,12})\b")


#: "last 5", "last ten", "show me 3". The written-out forms are needed in every language the
#: assistant takes, because "paanch" and "خمسة" arrive as often as "5".
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "panch": 5, "chhe": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "bees": 20,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "बीस": 20,
    "واحد": 1, "اثنان": 2, "ثلاثة": 3, "أربعة": 4, "خمسة": 5, "ستة": 6, "سبعة": 7,
    "ثمانية": 8, "تسعة": 9, "عشرة": 10, "عشرين": 20,
}

#: A bare number next to one of these is a QUANTITY, not money. Without this, "show me my
#: last 5 transactions" parsed 5 as an amount and the engine saw a payment request with the
#: amount already filled — the single most dangerous misreading in the whole surface, because
#: every guardrail downstream assumes the amount came from someone asking to send money.
_COUNT_CUE = re.compile(
    r"(?:last|latest|recent|past|previous|next|top|first|show|list|only|just)\s+"
    r"(\d{1,3}|[a-z\u0900-\u097f\u0600-\u06ff]+)"
    r"|(\d{1,3}|[a-z\u0900-\u097f\u0600-\u06ff]+)\s+"
    r"(?:transactions?|payments?|transfers?|txns?|lenden|लेनदेन|ट्रांजैक्शन|معاملات|معاملة|دفعات)",
    re.IGNORECASE)

#: Relative time windows, as the customer says them. Mapped to a day count because that is
#: what the ledger filter needs; "this month" is approximated rather than calendar-exact and
#: the answer says which dates it actually covered, so the approximation is never silent.
_WINDOWS: list[tuple[str, int]] = [
    ("in the last 30 days", 30), ("in the last 7 days", 7), ("last 30 days", 30),
    ("last 7 days", 7), ("last month", 60), ("this month", 30), ("last week", 14),
    ("this week", 7), ("yesterday", 2), ("today", 1), ("since monday", 7),
    ("pichle mahine", 60), ("is mahine", 30), ("pichle hafte", 14), ("is hafte", 7),
    ("aaj", 1), ("kal", 2),
    ("पिछले महीने", 60), ("इस महीने", 30), ("पिछले हफ्ते", 14), ("इस हफ्ते", 7),
    ("आज", 1), ("कल", 2),
    ("الشهر الماضي", 60), ("هذا الشهر", 30), ("الأسبوع الماضي", 14), ("هذا الأسبوع", 7),
    ("اليوم", 1), ("أمس", 2),
]


def _parse_count(low: str) -> tuple[int, str] | None:
    """Returns (count, the exact token that said it), or None.

    Every match is considered, not just the first. "show me list of last 5 transactions" has
    a cue at "show me" whose next word is "me" — taking the first match and giving up made
    the headline example of this whole feature come back empty. The token is returned so the
    caller can remove precisely it from the sentence the amount parser sees; removing a
    reconstructed digit missed the written-out forms ("last ten") and left 10 to be read as
    ten dirhams.
    """
    for m in _COUNT_CUE.finditer(low):
        tok = (m.group(1) or m.group(2) or "").strip().lower()
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= 100:
                return n, tok
        elif tok in _COUNT_WORDS:
            return _COUNT_WORDS[tok], tok
    return None


#: How a customer names a payment's state, mapped to the State the ledger uses.
_STATUS_WORDS: list[tuple[tuple[str, ...], str]] = [
    (("failed", "failure", "unsuccessful", "bounced", "declined", "rejected",
      "fail hui", "फेल", "फ़ेल", "فشلت", "فاشلة"), "FAILED"),
    (("pending", "in progress", "still going", "not yet", "atka", "अटका", "लंबित",
      "معلقة", "قيد"), "PENDING"),
    (("on hold", "held", "being reviewed", "under review", "रोक", "معلق", "مراجعة"),
     "ON_HOLD"),
    (("refunded", "reversed", "returned", "वापस", "مستردة"), "REVERSED"),
    (("successful", "success", "delivered", "completed", "went through", "सफल",
      "ناجحة", "مكتملة"), "SUCCESS"),
]


#: Biller words in every language, mapped to what the saved bills are actually called.
_BILL_WORDS = {
    "electricity": ("electricity", "power", "bijli", "बिजली", "الكهرباء", "addc", "dewa",
                    "adcc", "aadc"),
    "water": ("water", "paani", "pani", "पानी", "الماء", "sewage", "sewa", "fewa"),
    "internet": ("internet", "broadband", "wifi", "इंटरनेट", "الإنترنت", "etisalat", "du"),
    "landline": ("landline", "phone", "लैंडलाइन", "الهاتف"),
    "gas": ("gas", "गैस", "الغاز", "lootah"),
}


def _parse_bill(low: str) -> str | None:
    """A biller the customer named, resolved against their OWN saved bills.

    Matched against what their bills are really called — nickname, biller, kind — so "bijli"
    finds "Home electricity" without the handler having to know Hindi. Returns the nickname
    so the rest of the conversation has something stable to carry.
    """
    try:
        from bills import catalogue
        saved = catalogue.saved_bills()
    except Exception:
        return None
    if not saved:
        return None
    toks = tokens(low)
    # Their own words first: a nickname they chose beats any generic biller word.
    #
    # Scored, not first-match. Nicknames share words — "Home electricity" and "Home
    # internet" both start with "Home" — and taking the first bill with ANY token in common
    # meant answering "Home internet" at a which-bill question selected Home electricity,
    # because "home" matched it first. A token carries weight in inverse proportion to how
    # many of their bills contain it, so the shared prefix decides nothing and "internet"
    # decides everything.
    from collections import Counter
    nicks = {b["nickname"]: tokens(str(b.get("nickname") or ""), 3) for b in saved
             if b.get("nickname")}
    spread = Counter(t for nt in nicks.values() for t in nt)
    scored = sorted(
        ((sum(1.0 / spread[t] for t in (nt & toks)), nick) for nick, nt in nicks.items()),
        reverse=True)
    if scored and scored[0][0] > 0:
        # A tie means they named something every one of those bills answers to — "home" on
        # its own. That is not a selection, it is the same question again, so say nothing
        # and let the caller ask rather than picking one and sounding certain.
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1]
    for kind, words in _BILL_WORDS.items():
        # WHOLE WORDS, not substrings. "du" is the UAE telecom, and `"du" in low` made
        # "dukan" (shop) an internet bill — so "dukan pe pay karna hai" arrived at the
        # merchant flow already carrying a bill nobody mentioned. "dubai", "during" and
        # "produce" would all have done the same. `tokens()` is the shared script-aware
        # splitter, so this still works for बिजली and الكهرباء.
        if not (toks & set(words)):
            continue
        for b in saved:
            hay = " ".join(str(b.get(k) or "").lower()
                           for k in ("nickname", "biller", "kind"))
            hay_toks = tokens(hay)
            if kind in hay_toks or (hay_toks & set(words)):
                return b["nickname"]
    return None


def _parse_status(low: str) -> str | None:
    for words, state in _STATUS_WORDS:
        for w in words:
            pattern = rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w)
            if re.search(pattern, low):
                return state
    return None


def _parse_window(low: str) -> tuple[str, int] | None:
    for phrase, days in _WINDOWS:          # longest/most specific first, by construction
        if phrase in low:
            return phrase, days
    return None


#: How the customer names the way a bill divides. Mapped to the backend's SplitType enum.
#: EQUAL/CUSTOM apply before the merchant is paid, SHARE/AMOUNT/PERCENTAGE after — the
#: handler picks the right pair for the flow, so these resolve to the POST names and are
#: translated when the flow turns out to be pre-transaction.
_SPLIT_EQUAL = ("equally", "equal", "evenly", "even", "same", "fifty fifty", "50 50",
                "50-50", "half each", "half-half", "halves", "barabar", "aadha aadha",
                "बराबर", "आधा आधा", "بالتساوي", "النصف لكل")
_SPLIT_PERCENT = ("percent", "percentage", "%", "pct", "परसेंट", "प्रतिशत",
                  "النسبة المئوية", "بالنسبة")
_SPLIT_AMOUNT = ("custom", "different amounts", "specific amount", "exact amount",
                 "alag alag", "अलग अलग", "मبالغ مختلفة", "مبالغ مختلفة")

#: "split this three ways", "{count} ways", "between 4 of us". Distinct from a participant
#: list: it says HOW MANY without saying who, which is a different question to ask next.
_WAYS = re.compile(
    r"(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|do|teen|char|paanch|"
    r"दो|तीन|चार|पांच|اثنين|ثلاثة|أربعة|خمسة)\s*"
    r"(?:ways?|way|hisso|hisse|हिस्सों|हिस्से|people|of us|logo|लोगों|أشخاص|على)",
    re.IGNORECASE)

_WAYS_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
               "eight": 8, "nine": 9, "ten": 10,
               "do": 2, "teen": 3, "char": 4, "paanch": 5,
               "दो": 2, "तीन": 3, "चार": 4, "पांच": 5,
               "اثنين": 2, "ثلاثة": 3, "أربعة": 4, "خمسة": 5}


#: Word characters, INCLUDING the combining marks Python's `\w` leaves out.
#:
#: `re.split(r"\W+", "विकास के साथ बांटो")` returns
#: `['व','क','स','क','स','थ','ब','ट']` — every Devanagari word shredded into bare
#: consonants, because its vowel signs are Unicode category Mc/Mn and `\w` matches neither.
#: Nine places in this engine tokenised that way, so payee matching, merchant matching, the
#: history vocabulary and the name comparator were all quietly broken for Hindi while
#: appearing to work for English, Hinglish and (by luck of its script) Arabic.
_WORD = r"\w\u0900-\u097f\u0a80-\u0aff\u0600-\u06ff\u0640\u064b-\u065f"
_SPLIT = re.compile(f"[^{_WORD}]+")


def tokens(text: str, min_len: int = 1) -> set[str]:
    r"""The words in `text`, lowercased. Script-agnostic — use this, never ``\W+``."""
    return {t for t in _SPLIT.split((text or "").lower()) if len(t) >= min_len}


def token_list(text: str) -> list[str]:
    return [t for t in _SPLIT.split((text or "").lower()) if t]


#: How a customer names WHEN. Resolved to an absolute instant here so nothing downstream
#: re-parses English — a scheduled payment that fires on the wrong day is worse than one
#: that was never scheduled.
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
             "saturday": 5, "sunday": 6,
             "somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "shukravar": 4,
             "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "शुक्रवार": 4,
             "शनिवार": 5, "रविवार": 6,
             "الاثنين": 0, "الثلاثاء": 1, "الأربعاء": 2, "الخميس": 3, "الجمعة": 4,
             "السبت": 5, "الأحد": 6}

_WHEN_WORDS = (
    ("day after tomorrow", 2), ("tomorrow", 1), ("tonight", 0), ("today", 0),
    ("next week", 7), ("next month", 30), ("end of the month", None),
    ("parso", 2), ("kal", 1), ("aaj", 0), ("agle hafte", 7), ("agle mahine", 30),
    ("परसों", 2), ("कल", 1), ("आज", 0), ("अगले हफ्ते", 7), ("अगले महीने", 30),
    ("بعد غد", 2), ("غدا", 1), ("اليوم", 0), ("الأسبوع القادم", 7), ("الشهر القادم", 30),
)

_ORDINAL = re.compile(
    # "on the 1st", "for the 5th", "by the 10th", "the 1st" — people say all of them, and
    # matching only "on the" lost "schedule it FOR the 1st", which is the commoner phrasing
    # when they are asked to choose a date.
    r"\b(?:on|for|by|from)?\s*the\s+(\d{1,2})(?:st|nd|rd|th)\b"
    r"|\b(\d{1,2})(?:st|nd|rd|th)\s+of\b"
    r"|\bपहली तारीख\b|\bpehli tareekh\b|\bفي الأول\b", re.IGNORECASE)
_CLOCK = re.compile(r"\b(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)

#: Wallet or open banking, as the customer says it.
_PAY_MODE = (
    (("open banking", "openbanking", "open-banking", "bank account", "from my bank",
      "my bank", "bank transfer", "ob", "اوبن بانكنج", "من البنك", "बैंक से",
      "bank se", "bank wala"), "OPEN_BANKING"),
    (("wallet", "openwallet wallet", "my wallet", "from my wallet", "balance",
      "wallet se", "वॉलेट", "वॉलेट से", "المحفظة", "من المحفظة"), "WALLET"),
)


def _parse_pay_mode(low: str) -> str | None:
    for words, mode in _PAY_MODE:
        for w in words:
            pattern = rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w)
            if re.search(pattern, low):
                return mode
    return None


_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
              "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60,
              "do": 2, "teen": 3, "char": 4, "paanch": 5, "das": 10}
_UNIT = r"(min|mins|minute|minutes|minut|hr|hrs|hour|hours|ghante|ghanta|minat)"
#: A time FROM NOW: "in 2 minutes", "in next 2 minutes", "in the next two minutes", "after an
#: hour", "within 5 mins", "5 minute baad", "بعد 10 دقائق". Shared by every flow, and its
#: numbers are never an amount ("send 300 to Alex in 1 minute" has ONE amount: 300).
REL_TIME = re.compile(
    r"\b(?:in|after|within)\s+(?:the\s+)?(?:next\s+)?(\d{1,3}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
    + r")\s*" + _UNIT + r"\b"
    r"|\b(\d{1,3}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")\s*" + _UNIT
    + r"\s+(?:ke\s+)?(?:baad|bad|me|mein|main|mai|mei|tak)\b"
    r"|(\d{1,3})\s*(मिनट|घंटे|घंटा)\s*(?:बाद|में)"
    r"|بعد\s+(\d{1,3})\s*(دقيقة|دقائق|ساعة|ساعات)", re.I)


def rel_time(low: str):
    """(phrase, minutes from now) for a relative time, or None."""
    m = REL_TIME.search(low or "")
    if not m:
        return None
    g = m.groups()
    num = g[0] or g[2] or g[4] or g[6]
    unit = (g[1] or g[3] or g[5] or g[7] or "").lower()
    n = int(num) if num.isdigit() else _NUM_WORDS.get(num.lower())
    if not n:
        return None
    hours = unit.startswith(("h", "ghant", "ساع", "घंट"))
    return m.group(0), n * 60 if hours else n


def _parse_when(low: str) -> tuple[str, str] | None:
    """Returns (what they said, ISO instant) or None.

    Defaults to 09:00 UTC when no time is given, and never schedules in the past: "on the
    5th" said on the 20th means NEXT month's 5th, which is what a person means.
    """
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    rel = rel_time(low)
    if rel:
        return rel[0], (now + _dt.timedelta(minutes=rel[1])).replace(microsecond=0).isoformat(timespec="seconds")
    hour, minute = 9, 0
    m = _CLOCK.search(low)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        minute = int(m.group(2) or 0)

    for word, days in _WHEN_WORDS:
        if word in low:
            if days is None:                      # "end of the month"
                nxt = (now.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
                when = nxt - _dt.timedelta(days=1)
            else:
                when = now + _dt.timedelta(days=days)
            when = when.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if when <= now:
                when += _dt.timedelta(days=1)
            return word, when.isoformat(timespec="seconds")

    for name, idx in _WEEKDAYS.items():
        pattern = rf"\b{re.escape(name)}\b" if name.isascii() else re.escape(name)
        if re.search(pattern, low):
            ahead = (idx - now.weekday()) % 7 or 7
            when = (now + _dt.timedelta(days=ahead)).replace(
                hour=hour, minute=minute, second=0, microsecond=0)
            return name, when.isoformat(timespec="seconds")

    m = _ORDINAL.search(low)
    if m:
        day = int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else 1
        if 1 <= day <= 28:
            when = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if when <= now:                        # that day has passed; they mean next month
                when = (when.replace(day=28) + _dt.timedelta(days=5)).replace(day=day)
            return m.group(0), when.isoformat(timespec="seconds")
    return None


def _parse_split_type(low: str) -> str | None:
    for words, name in ((_SPLIT_PERCENT, "PERCENTAGE"), (_SPLIT_AMOUNT, "AMOUNT"),
                        (_SPLIT_EQUAL, "SHARE")):
        for w in words:
            if w in low:
                return name
    return None


def _parse_ways(low: str) -> tuple[int, str] | None:
    """Returns (n, the token that said it) so the caller can take it out of the sentence.

    "split this three ways" parsed 3 as an AMOUNT — the same collision that made "last 5
    transactions" look like a request to send five dirhams. A quantity and a sum of money
    must never both own the same number.
    """
    m = _WAYS.search(low)
    if not m:
        return None
    tok = m.group(1).lower()
    n = int(tok) if tok.isdigit() else _WAYS_WORDS.get(tok)
    return (n, tok) if n and 2 <= n <= 25 else None


#: Roster names in the scripts customers actually type. The roster itself is Latin — it
#: comes from KYC — so "विकास के साथ बांटो" resolved to nobody even though the model read the
#: sentence correctly. Third time this class of bug has appeared: the model speaks four
#: languages and the lookup table speaks one. Kept as data next to the matcher so adding a
#: payee's native spelling is an edit, not a code change.
_NAME_ALIASES = {
    "alex": ("विकास", "فيكاس", "ويكاس"),
    "morgan": ("दाधीच", "داديش"),
    "imran": ("इमरान", "عمران"),
    "sheikh": ("शेख", "شيخ"),
    "rahul": ("राहुल", "راهول"),
    "verma": ("वर्मा", "فيرما"),
    "fatima": ("फातिमा", "فاطمة"),
    "noor": ("नूर", "نور"),
    "ahmed": ("अहमद", "أحمد"),
    "khan": ("खान", "خان"),
    "jordan": ("सुनील", "سونيل"),
    "sagar": ("सागर", "ساجر"),
}


def _alias_tokens(name: str) -> set[str]:
    """A roster name's Latin tokens plus every native-script spelling we know for them."""
    out = tokens(name, 3)
    for t in list(out):
        out.update(_NAME_ALIASES.get(t, ()))
    return out


def _parse_participants(low: str, said: str) -> list[dict] | None:
    """EVERY person named, not just the first.

    `extract` resolves a single payee and stops, which is right for a transfer — one sender,
    one recipient. A split names several, and dropping all but the first would silently
    exclude somebody from a bill they agreed to share. Matching is the same exact-roster rule
    used for payees: a participant is a real userId on this customer's own list or they are
    not a participant.
    """
    import data
    said = tokens(low)
    found: list[dict] = []
    seen: set[str] = set()
    for b in data.sendable():
        name_tokens = _alias_tokens(b["name"])
        if name_tokens & said and b["name"] not in seen:
            seen.add(b["name"])
            found.append({"name": b["name"], "userId": b.get("userId")})
    if not found:
        return None
    # Resolve a real OpenWallet userId for each. The roster comes from the mandate config and
    # carries no userId, so without this every participant arrived unidentified and no split
    # could be built. A name that resolves to nothing stays unresolved on purpose — the
    # handler refuses rather than inventing an id.
    try:
        from split import api as split_api
        for p in found:
            if not p.get("userId"):
                p["userId"] = split_api.resolve_user_id(p["name"])
    except Exception:
        pass
    return found


def extract(text: str, intent: str) -> Slots:
    s = Slots()
    low = " ".join((text or "").lower().split())

    m = _REF.search(text or "")
    if m:
        s.txn_ref = m.group(1)

    # Parse the amount from the sentence with the REFERENCE REMOVED. Otherwise "refund
    # 1202461006221144 please" is read as a request for one and a half trillion dirhams --
    # a reference is the longest number in the sentence, so it wins every time.
    for_amount = low.replace(s.txn_ref.lower(), " ") if s.txn_ref else low

    # Quantity before money. "show me my last 5 transactions" contains the number 5, and read
    # as an amount it turns a request to LOOK at history into a half-formed request to SEND
    # five dirhams. Once a count cue has claimed the number it is removed from the sentence
    # the amount parser sees, so the two can never both own it.
    # WHICH saved bill they mean. Declared on Slots but never parsed, so the conversation
    # could not carry it: answering "which bill?" with "bijli ka bill" left the slot empty
    # and the next turn started over.
    s.bill = _parse_bill(low)
    s.status = _parse_status(low)
    s.pay_mode = _parse_pay_mode(low)
    got_when = _parse_when(low)
    if got_when:
        s.when, s.when_at = got_when
    s.split_type = _parse_split_type(low)
    ways = _parse_ways(low)
    s.participants = _parse_participants(low, text or "")
    got = _parse_count(low)
    win = _parse_window(low)
    if win:
        s.time_window, s.window_days = win
    if got:
        s.count, tok = got
        for_amount = for_amount.replace(tok, " ", 1)
    if ways:
        s.ways, wtok = ways
        for_amount = for_amount.replace(wtok, " ", 1)
    # a time from now is never the amount: "send 300 to alex in 1 minute"
    for_amount = REL_TIME.sub(" ", for_amount)
    s.amount = parse_amount(for_amount)
    s.currency = parse_currency(low) or None

    # Longest first, so "sri lanka" wins over a stray "lk", and \b only for ASCII — Python's
    # word boundary does not fire between a space and Devanagari or Arabic script.
    for word in sorted(COUNTRY_TO_CORRIDOR, key=len, reverse=True):
        pattern = rf"\b{re.escape(word)}\b" if word.isascii() else re.escape(word)
        if re.search(pattern, low):
            s.corridor = COUNTRY_TO_CORRIDOR[word]
            break

    # Payee: exact match against the people this customer can actually pay. No fuzzy
    # matching — the voice flow learned that one the hard way ("Riley" heard as "the worm").
    for b in data.sendable():
        if _alias_tokens(b["name"]) & tokens(low):
            s.payee = b["name"]
            break

    # Merchants by NAME TOKENS, not substring. Real trading names carry suffixes nobody says
    # out loud — "Asad and Sons FZ LLC" is asked for as "asad and sons" — so an exact
    # substring test matches almost nothing. Score by distinctive tokens, and refuse when two
    # merchants tie rather than picking one: paying the wrong shop is the same class of error
    # as paying the wrong person.
    _NOISE = {"and", "the", "llc", "fz", "fzc", "ltd", "l", "c", "co", "inc", "trading",
              "global", "general", "group", "company", "est", "national"}
    said_tokens = tokens(low)
    # Spacing is not spelling. Trading names run words together ("MClothes", "HomePay",
    # "SpeedPay") and customers type them apart — "M clothes" shares no TOKEN with
    # "mclothes", so the merchant was not matched at all and a split fell back to an
    # unrelated payment. Comparing with the spaces removed catches it both ways round.
    said_squashed = re.sub(r"[^a-z0-9\u0900-\u097f\u0600-\u06ff]", "", low)
    scored = []
    for merchant in data.MERCHANTS:
        name_tokens = tokens(merchant["name"])
        strong = {t for t in (name_tokens & said_tokens) if t not in _NOISE and len(t) > 2}
        if strong:
            scored.append((len(strong), merchant))
        elif len(said_squashed) > 4:
            # No token matched. Try the name with its spaces removed, both directions.
            squashed = re.sub(r"[^a-z0-9]", "", merchant["name"].lower())
            head = re.sub(r"[^a-z0-9]", "",
                          " ".join(w for w in merchant["name"].lower().split()
                                   if w not in _NOISE))
            if len(head) > 4 and (head in said_squashed or
                                  (len(said_squashed) > 5 and said_squashed in squashed)):
                scored.append((1, merchant))
        elif merchant.get("category", "").lower() in low:
            scored.append((1, merchant))
    if not scored and not s.payee:
        # Close spelling. Speech and thumbs give "mcloths" for MClothes, and an exact token
        # test misses it, so the customer answering "which merchant?" was not heard. Only
        # one merchant may be close, and only words of five letters or more count, so a short
        # common word never turns into a shop. The confirmation card still names the merchant.
        from difflib import SequenceMatcher
        words = [w for w in said_tokens if len(w) >= 5] + ([said_squashed] if len(said_squashed) >= 5 else [])
        close = []
        for merchant in data.MERCHANTS:
            head = re.sub(r"[^a-z0-9]", "", " ".join(
                w for w in merchant["name"].lower().split() if w not in _NOISE))
            if len(head) >= 5 and any(SequenceMatcher(None, w, head).ratio() >= 0.85 for w in words):
                close.append(merchant)
        if len(close) == 1:
            scored.append((1, close[0]))
    if scored:
        top = max(n for n, _ in scored)
        winners = [m for n, m in scored if n == top]
        if len(winners) == 1:
            s.merchant = winners[0]["name"]

    journey = journeys.find(low)
    if journey:
        s.journey = journey["key"]

    # A second, older window parser used to run here and OVERWRITE the one above, turning
    # "last week" into the token "last_week" and discarding the day count with it. Nothing
    # ever read those tokens, so they only ever did harm — the phrase ended up in a sentence
    # shown to a customer ("Found 5 last_week"). It now only fills what the richer parser
    # left empty, and in the words a person would actually read.
    if not s.time_window:
        if re.search(r"\byesterday\b", low):
            s.time_window, s.window_days = "yesterday", 2
        elif re.search(r"\btoday\b|\bthis morning\b", low):
            s.time_window, s.window_days = "today", 1
        elif re.search(r"\blast week\b", low):
            s.time_window, s.window_days = "last week", 14

    return s


#: Phrases that name a payment by recency rather than by reference.
_RECENCY = re.compile(r"\b(last|latest|recent|most recent|just (made|now|sent|paid)|"
                      r"this one|that one|previous)\b")


def resolve_txn(slots: Slots, intent: str, said: str = "", session: str = ""):
    """Which payment are we talking about? Returns (txn, how_we_chose).

    When the customer names one, use it. When they do not, we may INFER only for complaint
    intents, and we say so out loud in the answer — an inferred transaction that is wrong is
    recoverable if the customer can see which one we picked, and dangerous if they cannot.
    """
    if slots.txn_ref:
        txn = data.find_txn(slots.txn_ref)
        return (txn, "named by the customer") if txn else (None, "reference not recognised")

    # NAMED A PERSON OR SHOP? Then it is their payment we are being asked about, and no
    # other payment is an acceptable answer.
    #
    # This function used to ignore `payee` and `merchant` entirely and fall through to
    # "most recent payment". With the task ledger now carrying a name back from several
    # topics ago, "asad wala kya hua" arrived here with payee=Asad and was answered with a
    # confident status report about an 80 AED payment to MClothes — a different payment,
    # a different amount, presented as the answer. Not finding it has to be an answer of
    # its own.
    who = (slots.payee or slots.merchant or "").strip()
    if who:
        match = data.find_txn_by_name(who)
        if match:
            return match, f"the payment to {match.payee}"
        return None, f"no payment to {who} found"

    # A payment made EARLIER IN THIS CONVERSATION is almost certainly what they mean. Before
    # this, saying "I want a refund" right after paying reached past it for an older, more
    # broken-looking payment — which reads as the assistant not having been present.
    if session:
        from engine import conversation
        recent_pay = conversation.last_payment(session)
        if recent_pay:
            txn = data.find_txn(recent_pay["ref"])
            if txn:
                return txn, "the payment you made a moment ago in this conversation"

    # An explicit recency cue beats our severity ranking. The customer told us which payment
    # they mean; picking an older, more broken-looking one instead is ignoring them.
    if _RECENCY.search((said or "").lower()):
        txn = data.latest()
        if txn:
            return txn, "the customer asked about their most recent payment"

    complaint = intent in ("txn_failed", "debited_not_credited", "txn_on_hold",
                           "refund_request", "refund_status")
    if complaint:
        txn = data.latest_problem(intent)
        if txn:
            return txn, (f"inferred: most likely payment for a '{intent}' complaint "
                         f"(worst state first, then most recent)")
    if intent == "txn_status":
        txn = data.latest()
        if txn:
            return txn, "inferred: most recent payment"
    return None, "no transaction identified"
