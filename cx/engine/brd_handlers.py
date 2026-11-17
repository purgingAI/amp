"""Handlers for the intents the customer-app BRD (v.34) adds.

Each one follows the BRD's prompt-and-response table for its feature, and each one keeps
the BRD's core principle (§6): the model understands; amounts, payees, balances and status
come from deterministic services. Nothing here moves money. The two that lead to money
moving (recharge, retry) end in a PROPOSAL that goes through the same confirm-card + PIN
path as every other payment.

    mobile_recharge   §2.3   R1-R7
    reminder_set/list §2.6   M1-M7
    retry_payment     §2.1 T11-T12, §2.2 P8-P10, §2.7 S2-S4
    cancel_action     §2.1 T10
    balance_query     §2.8 #4
    human_handoff     §4.1 (handoff on request, context passed)
    wrong_transfer    §2.8 #14, #65
    report_fraud      §2.8 #66
    knowledge_query   §2.8 (the RAG lane — documents not loaded yet)
    add_money / withdraw_money / link_bank   objective 2 (navigation)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
STORE = os.path.join(CX, "data")
os.makedirs(STORE, exist_ok=True)

_DIGITS_AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _t(a, key: str, english: str, **vals) -> str:
    from engine import lang as langmod
    return langmod.t(key, a.lang, **vals) or english


# ---------------------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------------------
def wallet_balance() -> tuple[float | None, str]:
    """The wallet's own figure: wallet_master, else the latest ledger closing balance."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(CX), "openwallet"))
        from db import db_conn                                     # noqa: E402
        import openwallet_api                                           # noqa: E402
        cur = db_conn().cursor()
        import userctx
        cur.execute("select balance from wallet_master where user_id=%s", (userctx.user_id(),))
        r = cur.fetchone()
        if r is not None:
            return float(r["balance"]), "wallet"
    except Exception:                                              # noqa: BLE001
        pass
    try:
        import history
        db, uid = history._mongo()
        row = db["walletLedger"].find_one({"userId": uid, "transactionStatus": "SUCCESS"},
                                          sort=[("createdDate", -1)])
        if row and row.get("closingBal") is not None:
            return float(str(row["closingBal"])), "ledger"
    except Exception:                                              # noqa: BLE001
        pass
    return None, "unavailable"


def h_balance(a, s, said):
    bal, src = wallet_balance()
    if bal is None:
        a.say = _t(a, "balance_unavailable", "I can't read your balance right now. Please try "
                                             "again in a moment — I won't guess a figure.")
        a.outcome = "UNAVAILABLE"
        return a
    a.step("resolve", "balance", f"read from the {src} — not estimated", tag="db")
    a.say = _ph(a, "balance", _t(a, "balance_is", f"Your wallet balance is **AED {bal:,.2f}**.", amount=f"{bal:,.2f}"),
                balance=f"**AED {bal:,.2f}**")
    a.outcome = "ANSWERED"
    a.options = [{"kind": "ask_text", "label": "Show my last 5 transactions",
                  "text": "show my last 5 transactions"}]
    return a


# ---------------------------------------------------------------------------------------
# mobile recharge
# ---------------------------------------------------------------------------------------
#: UAE mobile prefixes. Operator is auto-detected from the number (BRD R4 allows this).
DU = ("052", "055", "058")
ETISALAT = ("050", "054", "056")
QUICK = (25, 50, 100)


def _phone(said: str) -> str | None:
    t = (said or "").translate(_DIGITS_AR)
    m = re.search(r"(?:\+?971[\s-]?|0)?(5\d)[\s-]?(\d{3})[\s-]?(\d{4})\b", t)
    if m:
        return "0" + m.group(1) + m.group(2) + m.group(3)
    return None


def _bad_number(said: str) -> str | None:
    """A digit run that is clearly meant as a phone number but is not one (BRD R7)."""
    t = (said or "").translate(_DIGITS_AR)
    for m in re.finditer(r"(?<!\d)\d{5,}(?!\d)", t):
        if not _phone(m.group(0)):
            return m.group(0)
    return None


def _amount(said: str) -> float | None:
    """The AED amount, never a phone number. The general slot extractor read
    "0552223344" as AED 552,223,344 — the BRD's zero-tolerance "wrong amount" case."""
    from engine import slots as slotlib
    t = (said or "").translate(_DIGITS_AR)
    t = re.sub(r"(?:\+?971[\s-]?|0)?5\d[\s-]?\d{3}[\s-]?\d{4}", " ", t)   # drop phone numbers
    t = re.sub(r"(?<!\d)\d{5,}(?!\d)", " ", t)                         # and any other long digit run
    amt = slotlib.extract(t, "mobile_recharge").amount
    return float(amt) if amt and 0 < float(amt) <= 5000 else None


_FOREIGN = re.compile(r"\b(?:rupees?|rs\.?|inr|usd|dollars?|\$|pkr|bdt|taka|euros?|pounds?|lakhs?|crores?)\b|₹|روبية|دولار|रुपये|रुपए", re.I)


def answers_open(intent: str, asked: str | None, said: str) -> bool:
    """Is this short reply the answer to the recharge / reminder question we just asked?"""
    t = " ".join((said or "").lower().split())
    if len(t.split()) > 6:
        return False
    if intent == "mobile_recharge":
        return bool(_phone(t) or _bad_number(t) or re.fullmatch(
            r"(aed\s*)?\d+(\.\d+)?\s*(aed|dirhams?|dhs?|درهم)?|(du|etisalat|e&|virgin)( number)?|"
            r"my (own )?number|same number", t))
    if intent == "reminder_set":
        d, tm, rec = _when(t)
        return bool(d or tm or rec or re.fullmatch(r"(on )?(the )?\d{1,2}(st|nd|rd|th)?", t))
    return False


def _operator(said: str, number: str | None) -> str | None:
    low = (said or "").lower()
    if re.search(r"\b(e&|etisalat|اتصالات)\b", low):
        return "Etisalat (e&)"
    if re.search(r"\bdu\b", low):
        return "du"
    if re.search(r"\bvirgin\b", low):
        return "Virgin Mobile"
    if number:
        return "du" if number[:3] in DU else "Etisalat (e&)" if number[:3] in ETISALAT else None
    return None


def _own_mobile() -> str | None:
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(CX), "openwallet"))
        import openwallet_api
        c = openwallet_api.cfg()
        for p in (c.get("people") or {}).values():
            if (p.get("name") or "").upper().startswith("SAGAR") and p.get("mobile"):
                m = str(p["mobile"])
                return m if m.startswith("0") else "0" + m[-9:]
    except Exception:                                              # noqa: BLE001
        pass
    return None


def _mask(n: str) -> str:
    return f"{n[:3]} ••• {n[-4:]}" if n and len(n) >= 7 else n


def h_recharge(a, s, said):
    from engine import conversation
    bad = _bad_number(said)
    number = _phone(said)
    if bad and not number:
        a.say = _t(a, "recharge_bad_number", "That number doesn't look valid — a UAE mobile "
                   "starts with 05 and has 10 digits. Please check and try again.")
        a.outcome = "NEED_INFO"
        a.step("decide", "recharge.invalid", f"{bad} is not a UAE mobile number", tag="guard")
        return a
    low = (said or "").lower()
    someone_else = re.search(r"\b(son|daughter|wife|husband|mother|father|brother|sister|"
                             r"beta|beti|bhai|bhabhi|ابني|ابنتي|زوجتي)\b", low)
    sess = a.session or ""
    # In-progress details live in the subject store: the engine's _remember() rewrites the
    # held frame's slots after every turn, and a phone number is not one of its slot fields.
    held = conversation.get(sess)
    prev = (conversation.subject(sess, "recharge_state") or {}) if held and held.intent == "mobile_recharge" else {}
    number = number or prev.get("phone")
    if not number and re.search(r"\b(my (own )?number|same number|mera number)\b", low):
        number = _own_mobile()
    amount = _amount(said) or prev.get("amount")
    # BRD 2.1 T4 "currency unclear": a recharge is in AED. "150 rupees" was proposed as AED 150
    # — the currency word was simply ignored. Say which currency we will charge, and ask.
    if _amount(said) and _FOREIGN.search(said or ""):
        amt = _amount(said)
        conversation.note_subject(sess, "recharge_state", dict(prev, amount=None))
        conversation.hold(sess, "mobile_recharge", {}, "amount")
        a.say = _t(a, "recharge_currency", f"Recharges are charged in AED. Did you mean AED {amt:g}?", amt=f"{amt:g}")
        a.options = [{"kind": "ask_text", "label": f"Yes, AED {amt:g}", "text": f"recharge with {amt:g} AED"},
                     {"kind": "ask_text", "label": "No, a different amount", "text": "a different amount"}]
        a.outcome = "NEED_CONFIRMATION"
        a.step("decide", "currency", f"amount said in another currency — confirming AED {amt:g} first", tag="guard")
        return a
    if not number:
        own = _own_mobile()
        a.slots = dict(a.slots or {}, amount=amount)
        conversation.note_subject(sess, "recharge_state", {"amount": amount})
        conversation.hold(sess, "mobile_recharge", {"amount": amount}, "phone")
        if someone_else or not own:
            a.say = _t(a, "recharge_which_number", "What is the number?")
        else:
            a.say = _t(a, "recharge_saved_or_new",
                       f"Which number? Your saved number ({_mask(own)}) or a new one?", mask=_mask(own))
            a.options = [{"kind": "ask_text", "label": f"My number {_mask(own)}",
                          "text": f"recharge {own}" + (f" with {amount:g}" if amount else "")}]
        a.outcome = "NEED_INFO"
        return a
    op = _operator(said, number) or prev.get("operator")
    if not amount:
        conversation.note_subject(sess, "recharge_state", {"phone": number, "operator": op})
        conversation.hold(sess, "mobile_recharge", {}, "amount")
        a.say = _t(a, "recharge_how_much", "How much?")
        a.options = [{"kind": "ask_text", "label": f"{q} AED", "text": f"recharge {number} with {q} AED"}
                     for q in QUICK]
        a.outcome = "NEED_INFO"
        return a
    if not op:
        conversation.note_subject(sess, "recharge_state", {"phone": number, "amount": amount})
        conversation.hold(sess, "mobile_recharge", {"amount": amount}, "operator")
        a.say = _t(a, "recharge_operator", "Is this a du or Etisalat (e&) number?")
        a.options = [{"kind": "ask_text", "label": o, "text": f"recharge {number} {o} with {amount:g}"}
                     for o in ("du", "Etisalat")]
        a.outcome = "NEED_INFO"
        return a
    a.step("resolve", "recharge.operator", f"{op} — from the number prefix {number[:3]}"
           if not re.search(r"\b(du|etisalat|e&|virgin)\b", low) else f"{op} — named", tag="rule")
    conversation.forget_subject(sess, "recharge_state")
    a.proposal = {"kind": "recharge", "phone": number, "operator": op, "amount": float(amount),
                  "currency": "AED", "payee": f"{op} · {_mask(number)}"}
    a.say = _t(a, "recharge_confirm", f"Recharge {op} {_mask(number)} with AED {amount:,.2f}?",
               op=op, mask=_mask(number), amount=f"{amount:,.2f}")
    a.outcome = "PROPOSED"
    return a


# ---------------------------------------------------------------------------------------
# reminders — they NOTIFY; they never pay (BRD §2.8 #59)
# ---------------------------------------------------------------------------------------
REMINDERS = os.path.join(STORE, "reminders.json")
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
BILLERS = ("dewa", "sewa", "fewa", "addc", "etisalat", "du", "empower", "salik", "nol", "rent",
           "internet", "water", "electricity", "broadband", "landline", "gas", "phone")


def _owner() -> str:
    import userctx
    return userctx.user_id()


def _me() -> dict:
    """The customer a reminder belongs to and is delivered to."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(CX), "openwallet"))
    import userctx
    import openwallet_api
    w = openwallet_api.me()
    return {"user_id": w["user_id"], "mobile": w.get("mobile"), "name": userctx.name() or w.get("name") or ""}


def _load_all() -> list[dict]:
    try:
        return json.load(open(REMINDERS))
    except Exception:                                              # noqa: BLE001
        return []


def _load() -> list[dict]:
    """THIS customer's reminders. Rows written before logins existed belong to the configured
    demo customer."""
    import userctx
    me = _owner()
    mine_by_default = userctx.is_configured_sender()
    return [r for r in _load_all() if r.get("owner", me if mine_by_default else None) == me]


def _save(rows: list[dict]) -> None:
    import userctx
    me = _owner()
    mine_by_default = userctx.is_configured_sender()
    others = [r for r in _load_all() if r.get("owner", me if mine_by_default else None) != me]
    json.dump(others + [dict(r, owner=me) for r in rows], open(REMINDERS, "w"), indent=1)


def _when(said: str, now: dt.datetime | None = None) -> tuple[dt.date | None, str | None, str | None]:
    """(date, time, recurrence) named in the sentence, each None if not named."""
    now = now or dt.datetime.now()
    low = (said or "").lower().translate(_DIGITS_AR)
    recur = "monthly" if re.search(r"every month|monthly|har mahine|हर महीने|كل شهر", low) else \
        "weekly" if re.search(r"every week|weekly|har hafte|كل أسبوع", low) else None
    tm = None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        tm = f"{h:02d}:{int(m.group(2) or 0):02d}"
    d = None
    # "in 5 minutes", "in 2 hours", "5 minute baad", "بعد 10 دقائق" — exact, for today
    # ("in next 2 minutes", "in the next two minutes" too — one parser, shared with payments)
    from engine import slots as _sl
    rel = _sl.rel_time(low)
    if rel:
        t = now + dt.timedelta(minutes=rel[1])
        return t.date(), t.strftime("%H:%M"), recur
    if re.search(r"\btomorrow|kal\b|कल|غدا|غداً", low):
        d = (now + dt.timedelta(days=1)).date()
    elif re.search(r"\btoday|aaj\b|आज|اليوم", low):
        d = now.date()
    m = re.search(r"in (\d+) days?", low)
    if m:
        d = (now + dt.timedelta(days=int(m.group(1)))).date()
    for i, wd in enumerate(_WEEKDAYS):
        if re.search(rf"\b{wd}\b", low):
            ahead = (i - now.weekday()) % 7 or 7
            d = (now + dt.timedelta(days=ahead)).date()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b|\bon the (\d{1,2})\b|(\d{1,2}) (?:ko|तारीख)|يوم (\d{1,2})", low)
    if m:
        day = int(next(g for g in m.groups() if g))
        if 1 <= day <= 31:
            y, mo = now.year, now.month
            if day < now.day or (day == now.day and now.hour >= 9):
                mo += 1
                if mo > 12:
                    y, mo = y + 1, 1
            try:
                d = dt.date(y, mo, day)
            except ValueError:
                d = None
    if re.search(r"end of the month", low):
        nxt = (now.replace(day=28) + dt.timedelta(days=4))
        d = (nxt - dt.timedelta(days=nxt.day)).date()
    return d, tm, recur


#: Numbers that belong to WHEN, not HOW MUCH: "in 30 minutes", "at 5 pm", "on the 25th".
_TIME_NUMS = re.compile(r"\b(?:in|after|within)\s+(?:the\s+)?(?:next\s+)?\d{1,3}\s*(?:min\w*|hr\w*|hours?|days?)\b"
                        r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}(?:st|nd|rd|th)\b"
                        r"|\bon the \d{1,2}\b|\b\d{1,3}\s*(?:minute|min|ghante|ghanta)\s+(?:baad|me|mein)\b"
                        r"|بعد\s+\d{1,3}\s*\S+|يوم\s+\d{1,2}", re.I)


def _reminder_amount(said: str, s) -> float | None:
    """The amount to remind about — never a number that is part of the time.
    (Named apart from `_amount`, the recharge parser: reusing that name replaced it and
    every recharge crashed.)"""
    # Read the amount from the sentence WITHOUT its time phrase: "send 200 tomorrow at 6 pm"
    # has two numbers, and the amount parser rightly refuses to pick one; "in 30 minutes" has
    # one, and it is not an amount.
    from engine import slots as slotlib
    rest = _TIME_NUMS.sub(" ", said or "")
    return slotlib.extract(rest, "send_money").amount or None


def _what(said: str, s) -> str | None:
    low = (said or "").lower()
    for b in BILLERS:
        if re.search(rf"\b{re.escape(b)}\b", low):
            label = {"nol": "Nol card", "rent": "rent", "phone": "phone bill"}.get(b, b.upper() if len(b) <= 5 else b)
            return f"Pay {label}" + ("" if b in ("rent", "phone") else " bill")
    if s.payee:
        amt = _reminder_amount(said, s)
        return f"Send money to {s.payee}" + (f" (AED {amt:g})" if amt else "")
    m = re.search(r"(?:send|pay|transfer)\s+(?:money\s+)?(?:to\s+)?([A-Za-z]{3,})", said or "")
    if m and m.group(1).lower() not in ("money", "the", "my", "bill", "rent"):
        return f"Send money to {m.group(1).title()}"
    if re.search(r"\b(rent|kiraya|किराय|إيجار|الإيجار)", low):
        return "Pay rent"
    return None


def h_reminder_set(a, s, said):
    from engine import conversation
    sess = a.session or ""
    held = conversation.get(sess)
    prev = (conversation.subject(sess, "reminder_state") or {}) if held and held.intent == "reminder_set" else {}
    what = _what(said, s) or prev.get("what")
    d, tm, recur = _when(said)
    d = d or (dt.date.fromisoformat(prev["date"]) if prev.get("date") else None)
    recur = recur or prev.get("recur")
    if not what:
        conversation.note_subject(sess, "reminder_state", {"date": d and d.isoformat(), "recur": recur})
        conversation.hold(sess, "reminder_set", {}, "what")
        a.say = _t(a, "reminder_what", "What should I remind you about — a bill, or sending money to someone?")
        a.outcome = "NEED_INFO"
        return a
    if not d and recur != "monthly":
        conversation.note_subject(sess, "reminder_state", {"what": what, "recur": recur})
        conversation.hold(sess, "reminder_set", {}, "when")
        a.say = _ph(a, "ask_when", _t(a, "reminder_when", "When should I remind you?"))
        a.options = [{"kind": "ask_text", "label": l, "text": f"remind me {l.lower()}"}
                     for l in ("Tomorrow", "On the 1st", "On the 25th")]
        a.outcome = "NEED_INFO"
        return a
    if recur == "monthly" and not d:
        conversation.note_subject(sess, "reminder_state", {"what": what, "recur": recur})
        conversation.hold(sess, "reminder_set", {}, "when")
        a.say = _t(a, "reminder_day", "On which day of the month?")
        a.outcome = "NEED_INFO"
        return a
    # BRD M3: time defaults to 9:00 AM (a product decision; stated in the reply, not hidden)
    time_ = tm or "09:00"
    import reminders_store as store
    me = _me()
    try:
        live = store.mine(me["user_id"])
    except store.StoreDown:
        a.say = _t(a, "reminder_down", "I can't save reminders right now — please try again in a moment.")
        a.outcome = "NONE"
        a.step("act", "reminder.store", "the reminder store is unreachable — nothing was saved", tag="guard")
        return a
    rows = [{"what": r["what"], "date": dt.datetime.fromtimestamp(r["due"]).date().isoformat()} for r in live]
    dup = next((r for r in rows if r["what"].lower() == what.lower() and r["date"] == d.isoformat()), None)
    if dup:
        a.say = _t(a, "reminder_dup", f"You already have a reminder to {what.lower()} on "
                   f"{d.day} {_MONTHS[d.month - 1]}. Keep both, or update the existing one?")
        a.options = [{"kind": "ask_text", "label": "Keep the existing one", "text": "show my reminders"}]
        a.outcome = "NEED_CONFIRMATION"
        return a
    hh, mm = map(int, time_.split(":"))
    due = dt.datetime.combine(d, dt.time(hh, mm))
    if due.timestamp() < time.time() - 60:
        a.say = _t(a, "reminder_past", "That time has already passed — when should I remind you?")
        conversation.note_subject(sess, "reminder_state", {"what": what, "recur": recur})
        conversation.hold(sess, "reminder_set", {}, "when")
        a.outcome = "NEED_INFO"
        return a
    try:
        rec = store.add(me, what, due, recur, _reminder_amount(said, s))
    except store.StoreDown:
        a.say = _t(a, "reminder_down", "I can't save reminders right now — please try again in a moment.")
        a.outcome = "NONE"
        return a
    rid = rec["id"]
    conversation.forget_subject(sess, "reminder_state")
    conversation.clear(sess)
    hh, mm = map(int, time_.split(":"))
    when_txt = f"{d.day} {_MONTHS[d.month - 1]}, {(hh % 12) or 12}:{mm:02d} {'AM' if hh < 12 else 'PM'}"
    a.say = _ph(a, "reminder_saved", _t(a, "reminder_set", f"Reminder set: {what}, {when_txt}"
               + (", every month" if recur == "monthly" else "") + ". "
               "It will only notify you — you confirm every payment yourself.",
               what=what, when=when_txt),
               what=what, when=when_txt + (", every month" if recur == "monthly" else ""))
    if not tm:
        a.step("decide", "reminder.time", "no time named — 9:00 AM by default (BRD §2.6 M3)", tag="rule")
    a.outcome = "SAVED"
    a.step("act", "reminder.saved", f"{rid}: {what} on {d.isoformat()} {time_} — kept until it falls "
           f"due, then sent to the customer's app as a notification", tag="db")
    return a


_CLEAR_ALL = re.compile(
    r"\b(?:clear|reset|wipe|delete|remove|cancel|erase|hata\w*|mita\w*)\b[^.?!]{0,20}\b(?:all\s+)?(?:my\s+|the\s+)?"
    r"(?:reminders|saare\s+reminders?|sab\s+reminders?)\b|"
    r"\b(?:all|saare|sab|sabhi)\s+(?:my\s+)?reminders?\b[^.?!]{0,20}\b(?:clear|delete|remove|hata\w*|mita\w*|cancel)\b|"
    r"احذف\s+(?:كل\s+)?(?:جميع\s+)?التذكيرات|امسح\s+(?:كل\s+)?التذكيرات|सारे\s+रिमाइंडर", re.I)


def h_reminder_list(a, s, said):
    import reminders_store as store
    me = _me()
    try:
        live = store.mine(me["user_id"])
    except store.StoreDown:
        a.say = _t(a, "reminder_down", "I can't reach your reminders right now — please try again in a moment.")
        a.outcome = "NONE"
        return a
    rows = [{"id": r["id"], "what": r["what"],
             "date": dt.datetime.fromtimestamp(r["due"]).date().isoformat(),
             "time": dt.datetime.fromtimestamp(r["due"]).strftime("%H:%M"), "recur": r.get("recur")}
            for r in live]
    low = (said or "").lower()
    # "Clear my reminders", "reset my reminders", "delete all reminders", "saare reminders hata do".
    # All of them, so it asks once with the count; the confirmed form says "yes" / "confirm".
    if _CLEAR_ALL.search(low) and rows:
        if re.search(r"\b(yes|confirm|haan|ha|ji|نعم|अवश्य|हाँ)\b", low):
            n = 0
            for r in rows:
                n += 1 if store.delete(me["user_id"], r["id"]) else 0
            a.say = _t(a, "reminders_cleared", f"Done — deleted all {n} of your reminders.", n=n)
            a.outcome = "DONE"
            a.step("act", "reminders.cleared", f"{n} reminders deleted on the customer's confirmation", tag="act")
            return a
        a.say = _t(a, "reminders_clear_confirm",
                   f"Delete all {len(rows)} of your reminders? This can't be undone.", n=len(rows))
        a.options = [{"kind": "ask_text", "label": f"Yes, delete all {len(rows)}",
                      "text": "yes, clear all my reminders"},
                     {"kind": "ask_text", "label": "No, keep them", "text": "show my reminders"}]
        a.outcome = "NEED_CONFIRMATION"
        return a
    if re.search(r"\b(delete|remove|cancel|turn off|hata|हटा|احذف)\b", low):
        target = next((r for r in rows if any(w in r["what"].lower() for w in low.split() if len(w) > 2)), None)
        if target:
            store.delete(me["user_id"], target["id"])
            a.say = _t(a, "reminder_deleted", f"Deleted: {target['what']} ({target['date']}).")
            a.outcome = "DONE"
            return a
    if not rows:
        a.say = _t(a, "reminder_none", "You have no reminders set. Want me to set one?")
        a.options = [{"kind": "ask_text", "label": "Remind me to pay DEWA on the 18th",
                      "text": "remind me to pay DEWA on the 18th"}]
        a.outcome = "NONE"
        return a
    rows.sort(key=lambda r: (r["date"], r["time"]))
    a.say = _t(a, "reminder_list", f"You have {len(rows)} reminder{'s' if len(rows) > 1 else ''}:")
    a.rows = [{"payee": r["what"], "status": ("every month · " if r.get("recur") == "monthly" else "")
               + r["time"], "when": r["date"]} for r in rows[:10]]
    a.options = [{"kind": "ask_text", "label": f"Delete: {r['what']}", "text": f"delete my {r['what']} reminder"}
                 for r in rows[:3]]
    a.outcome = "LISTED"
    return a


# ---------------------------------------------------------------------------------------
# retry — safe only when the last attempt definitely FAILED
# ---------------------------------------------------------------------------------------
def h_retry(a, s, said):
    import data
    from engine.matrix import State
    txns = data.transactions_for(limit=5)
    last = txns[0] if txns else None
    if not last:
        a.say = _t(a, "retry_nothing", "I can't see a recent payment to retry.")
        a.outcome = "NONE"
        return a
    who, amt = last.payee, last.amount
    if last.state in (State.PENDING, State.UNKNOWN, State.ON_HOLD):
        a.step("decide", "retry.blocked", f"{last.ref} is {last.state.value} — a retry now could pay twice", tag="guard")
        a.say = _t(a, "retry_unknown", f"Your payment of AED {amt:,.2f} to {who} is still being "
                   "confirmed. Retrying now could send the money twice, so I won't — I'll update you "
                   "as soon as it settles.", amount=f"{amt:,.2f}", who=who)
        a.outcome = "RETRY_BLOCKED"
        return a
    if last.state == State.SUCCESS:
        a.say = _t(a, "retry_succeeded", f"Your last payment of AED {amt:,.2f} to {who} went through. "
                   f"Reference: {last.ref}. No need to retry.", amount=f"{amt:,.2f}", who=who, ref=last.ref)
        a.outcome = "ANSWERED"
        return a
    # failed: a NEW confirmation with a new key, never a silent re-send
    a.step("decide", "retry.allowed", f"{last.ref} FAILED and nothing was taken — offering a new attempt", tag="rule")
    kind = "pay_merchant" if last.is_merchant else "send"
    a.proposal = {"kind": kind, ("merchant" if last.is_merchant else "payee"): who,
                  "amount": float(amt), "currency": last.currency or "AED", "retry_of": last.ref}
    a.say = _t(a, "retry_failed", f"Your previous payment of AED {amt:,.2f} to {who} failed and no "
               "money was deducted. Try again?", amount=f"{amt:,.2f}", who=who)
    a.outcome = "PROPOSED"
    return a


# ---------------------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------------------
def h_cancel(a, s, said):
    from engine import conversation
    had = conversation.proposal(a.session or "") or conversation.get(a.session or "")
    # A payment held for later ("pay alex 100 in 2 minutes") is cancelled by name, or when
    # nothing else is in progress.
    if re.search(r"schedul|later|that payment|wo payment", said or "", re.I) or not had:
        try:
            import scheduled_pay, userctx
            uid = userctx.user_id()
            held = sorted(scheduled_pay.mine(uid), key=lambda j: j["due"])
            if held and scheduled_pay.cancel(uid, held[-1]["id"]):
                j = held[-1]
                a.say = _t(a, "cancel_scheduled",
                           f"Cancelled — your scheduled payment of AED {j['amount']:g} to {j['who']} "
                           f"won't go. Nothing was sent.")
                a.outcome = "CANCELLED"
                a.step("act", "unschedule", f"{j['id']} cancelled before it ran", tag="act")
                return a
        except Exception:                                          # noqa: BLE001
            pass
    conversation.drop_proposal(a.session or "")
    conversation.clear(a.session or "")
    a.say = (_ph(a, "payment_cancelled", _t(a, "cancel_done", "Payment cancelled. No money was sent.")) if had
             else _ph(a, "nothing_to_cancel", _t(a, "cancel_nothing", "There's nothing in progress — nothing has moved.")))
    a.outcome = "CANCELLED"
    return a


# ---------------------------------------------------------------------------------------
# handoffs: a real record, so the context really does pass (BRD §4.1)
# ---------------------------------------------------------------------------------------
HANDOFFS = os.path.join(STORE, "handoffs.jsonl")


def _handoff(a, reason: str, extra: dict | None = None) -> str:
    from engine import conversation
    ref = f"H-{int(time.time()) % 1_000_000:06d}"
    rec = {"ref": ref, "reason": reason, "session": a.session, "at": time.time(),
           "said": a.said, "topic": conversation.topic(a.session or ""), **(extra or {})}
    with open(HANDOFFS, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    a.step("act", "handoff", f"{ref} recorded for {reason}, with this conversation attached", tag="db")
    return ref


def _ph(a, kind: str, fallback: str, **slots) -> str:
    """A fresh wording from the phrase bank in the customer's language, or the old sentence."""
    try:
        import phrasebank
        return phrasebank.say(kind, a.lang or "en", a.session or "default", **slots) or fallback
    except Exception:                                              # noqa: BLE001
        return fallback


def h_handoff(a, s, said):
    ref = _handoff(a, "customer asked for a person")
    a.say = _ph(a, "handoff", _t(a, "handoff", f"I'm connecting you with OpenWallet customer support. I've passed on this "
               f"conversation so you won't have to repeat yourself. Your reference is {ref}.", ref=ref), ref=ref)
    a.outcome = "HANDOFF"
    return a


def h_wrong_transfer(a, s, said):
    import data
    t = data.find_txn_by_name(s.payee) if s.payee else None
    t = t or next((x for x in data.transactions_for(limit=5) if not x.is_merchant), None)
    ref = _handoff(a, "wrong recipient — recall", {"txn": t.ref if t else None})
    if t:
        a.say = _t(a, "wrong_transfer", f"I'll raise a recall request for AED {t.amount:,.2f} sent to "
                   f"{t.payee} (reference {t.ref}) and connect you with support ({ref}). A recall needs "
                   "the recipient's agreement, so it isn't guaranteed — the sooner it's raised, the better.",
                   amount=f"{t.amount:,.2f}", who=t.payee, txn=t.ref, ref=ref)
        a.txn = t.public()
    else:
        a.say = _t(a, "wrong_transfer_which", f"I'll connect you with support to recall it ({ref}). "
                   "Which payment was it?")
    a.outcome = "HANDOFF"
    return a


def h_report_fraud(a, s, said):
    ref = _handoff(a, "unrecognised transaction — fraud team")
    a.say = _t(a, "report_fraud", f"I'm connecting you with our fraud team right away ({ref}). Don't "
               "share your PIN or OTP with anyone — OpenWallet will never ask for them.", ref=ref)
    a.outcome = "HANDOFF"
    return a


# ---------------------------------------------------------------------------------------
# knowledge — the RAG lane. Documents are not loaded yet.
# ---------------------------------------------------------------------------------------
#: Only answers the BRD itself states word for word. Everything else waits for the approved
#: documents (§2.8: "RAG answers only general product knowledge from approved documents").
BRD_ANSWERS = [
    (r"reminder.*(pay|charge).*(automatic|itself|by itself)|will a reminder pay",
     "No. Reminders only notify you. You confirm every payment."),
    (r"charged twice.*retry|retry.*charged twice|pay twice",
     "No. Retry is blocked until your first payment is confirmed, and a failed payment is never charged."),
]


def h_knowledge(a, s, said):
    import journeys
    low = (said or "").lower()
    for pat, ans in BRD_ANSWERS:
        if re.search(pat, low):
            a.say = ans
            a.outcome = "ANSWERED"
            a.step("resolve", "knowledge.brd", "answer stated in the BRD itself", tag="db")
            return a
    j = journeys.find(said)
    if j:
        a.say = j["title"] + ": " + " ".join(f"{i}. {st}" for i, st in enumerate(j["steps"], 1))
        a.outcome = "EXPLAINED"
        a.step("resolve", "knowledge.journey", f"curated help content — {j['key']}", tag="db")
        return a
    a.step("resolve", "knowledge.rag", "retrieval lane — the approved documents are not loaded yet, "
           "so the standard no-answer message is given rather than a generated one", tag="guard")
    a.say = _t(a, "knowledge_none", "I don't have an approved answer for that yet. I can connect "
               "you with support, or help with a payment, a bill or your transactions.")
    a.options = [{"kind": "ask_text", "label": "Talk to support", "text": "connect me to customer support"}]
    a.outcome = "NO_ANSWER"
    return a


# ---------------------------------------------------------------------------------------
# navigation to the app's own screens
# ---------------------------------------------------------------------------------------
SCREENS = {"add_money": ("Add Money", "Opening **Add Money** — top up your wallet from your card or bank."),
           "withdraw_money": ("Withdraw Money", "Opening **Withdraw Money** — move wallet money to your bank."),
           "link_bank": ("Connect Bank", "Opening **Connect Bank** — link your bank account to your wallet.")}


def _navigate(key):
    def h(a, s, said):
        title, text = SCREENS[key]
        a.say = _t(a, f"nav_{key}", text)
        a.options = [{"kind": "navigate", "screen": key, "label": f"Open {title}"}]
        a.outcome = "NAVIGATE"
        a.step("act", "navigate", f"opens the app's {title} screen — the assistant does not move money here", tag="rule")
        return a
    return h


def handlers() -> dict:
    from taxonomy import Intent
    return {
        Intent.BALANCE_QUERY: h_balance,
        Intent.MOBILE_RECHARGE: h_recharge,
        Intent.REMINDER_SET: h_reminder_set,
        Intent.REMINDER_LIST: h_reminder_list,
        Intent.RETRY_PAYMENT: h_retry,
        Intent.CANCEL_ACTION: h_cancel,
        Intent.HUMAN_HANDOFF: h_handoff,
        Intent.WRONG_TRANSFER: h_wrong_transfer,
        Intent.REPORT_FRAUD: h_report_fraud,
        Intent.KNOWLEDGE_QUERY: h_knowledge,
        Intent.ADD_MONEY: _navigate("add_money"),
        Intent.WITHDRAW_MONEY: _navigate("withdraw_money"),
        Intent.LINK_BANK: _navigate("link_bank"),
    }
