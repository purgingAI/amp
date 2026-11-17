"""Split payments: the arithmetic, the direction, the guardrails, the real API contract.

The direction tests are the important ones. A split is NOT a transfer — "split 50 with
Alex" records that Alex owes the customer 25 for a bill the customer paid a merchant,
whereas "send 50 to Alex" takes 50 off the customer. One word apart, opposite money flows,
and getting it backwards is the worst thing this surface can do short of paying the wrong
person.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine import ask
from engine.slots import extract, tokens
from split import calc

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


# ── arithmetic ───────────────────────────────────────────────────────────────────────────
def test_equal_shares_sum_exactly():
    for total, n in [(100, 2), (85, 1), (10, 2), (0.03, 2), (1234.56, 6), (7, 3)]:
        shares, mine = calc.equal(total, n)
        check(f"{total} / {n+1} sums exactly",
              round(sum(shares) + mine, 2) == round(total, 2),
              f"{shares} + {mine} != {total}")

def test_initiator_absorbs_the_remainder():
    shares, mine = calc.equal(100, 2)
    check("100 three ways is 33.33/33.33/33.34", shares == [33.33, 33.33] and mine == 33.34,
          f"{shares} {mine}")
    check("the payer takes the extra fil, not a participant", mine >= max(shares))

def test_over_allocation_is_refused():
    try:
        calc.by_amount(100, [70, 50])
        check("over-allocated amounts refused", False, "no exception")
    except ValueError:
        check("over-allocated amounts refused", True)
    try:
        calc.by_percentage(100, [60, 60])
        check("over-100% refused", False, "no exception")
    except ValueError:
        check("over-100% refused", True)

def test_a_split_needs_someone_else():
    try:
        calc.equal(100, 0)
        check("a split of one is refused", False, "no exception")
    except ValueError:
        check("a split of one is refused", True)


# ── direction: split is not send ─────────────────────────────────────────────────────────
def test_split_does_not_move_money_from_the_customer():
    a = ask("split my last payment with alex", want_second_opinion=False, session="d1",
            pick={"intent": "split_payment"})
    check("a split proposes a split, not a transfer",
          a.proposal and a.proposal.get("kind") == "split", a.proposal)
    check("no amount is debited in the proposal",
          a.proposal and "payable" not in a.proposal)

def test_send_is_still_send():
    a = ask("send 50 to alex morgan", want_second_opinion=False, session="d2")
    check("send_money is untouched by the split work", a.intent == "send_money", a.intent)


# ── what can be split ────────────────────────────────────────────────────────────────────
def test_only_merchant_payments_are_splittable():
    a = ask("split my last payment with alex", want_second_opinion=False, session="m1",
            pick={"intent": "split_payment"})
    if a.proposal:
        import data
        txn = data.find_txn(a.proposal["referenceTxnId"])
        check("the target is a merchant payment", txn and txn.is_merchant,
              f"{txn.payee if txn else None} is_merchant={txn.is_merchant if txn else None}")
        check("the participant is not also the merchant",
              all(p["name"].split()[0].lower() not in a.proposal["merchant"].lower()
                  for p in a.proposal["participants"]),
              f"{a.proposal['merchant']} vs {[p['name'] for p in a.proposal['participants']]}")

def test_cannot_split_more_than_the_bill():
    a = ask("split 5000 with alex", want_second_opinion=False, session="m2",
            pick={"intent": "split_payment"})
    check("splitting more than the bill is refused",
          a.outcome == "NEED_INFO" and not a.proposal, f"{a.outcome}")


# ── participants ─────────────────────────────────────────────────────────────────────────
def test_unresolvable_participant_is_not_invented():
    """The roster comes from the mandate config and carries no userId, so being on the
    roster does NOT mean there is a OpenWallet account to split with. A guessed id would send
    somebody else a bill."""
    s = extract("split this with rahul", "split_payment")
    rahul = next((p for p in (s.participants or []) if "Rahul" in p["name"]), None)
    if rahul:
        check("no id is invented for a non-OpenWallet roster name", rahul.get("userId") is None,
              rahul)
    a = ask("split this with rahul", want_second_opinion=False, session="p1",
            pick={"intent": "split_payment"})
    check("asks rather than building a split with no id",
          a.outcome == "NEED_INFO" and not a.proposal, a.outcome)

def test_every_named_person_is_captured():
    s = extract("split the bill with alex and imran", "split_payment")
    names = {p["name"] for p in (s.participants or [])}
    check("both people captured, not just the first",
          names == {"Alex Morgan", "Imran Sheikh"}, names)

def test_participants_only_from_the_customers_own_list():
    s = extract("split this with mallory", "split_payment")
    check("a stranger is not a participant", not s.participants, s.participants)


# ── slots ────────────────────────────────────────────────────────────────────────────────
def test_ways_is_not_an_amount():
    """Same collision as 'last 5 transactions': a quantity and a sum of money must never
    both own the same number."""
    s = extract("split this three ways", "split_payment")
    check("ways parsed", s.ways == 3, s.ways)
    check("three is NOT read as money", s.amount is None, s.amount)
    s = extract("split 300 three ways", "split_payment")
    check("both a real amount and a count survive together",
          s.amount == 300.0 and s.ways == 3, f"{s.amount} {s.ways}")

def test_split_type_from_words():
    for text, want in [("split it fifty fifty with alex", "SHARE"),
                       ("split equally with alex", "SHARE"),
                       ("alex pays 60 percent", "PERCENTAGE"),
                       ("custom amounts", "AMOUNT")]:
        check(f"{text[:28]!r} -> {want}", extract(text, "split_payment").split_type == want,
              extract(text, "split_payment").split_type)


# ── tokenisation: the bug that broke every Devanagari lookup ─────────────────────────────
def test_devanagari_tokenisation():
    r"""`re.split(r"\W+", "विकास")` returns ['व','क','स'] — Devanagari vowel signs are
    category Mc, which `\w` excludes, so every Hindi word was shredded into bare consonants.
    Nine call sites tokenised that way."""
    check("a Devanagari word stays whole", tokens("विकास के साथ बांटो") ==
          {"विकास", "के", "साथ", "बांटो"}, tokens("विकास के साथ बांटो"))
    check("Arabic stays whole", "فيكاس" in tokens("اقسم 100 مع فيكاس"),
          tokens("اقسم 100 مع فيكاس"))
    for text in ["विकास के साथ बांटो", "اقسم 100 مع فيكاس", "alex ke saath baanto"]:
        s = extract(text, "split_payment")
        check(f"participant resolved in {text[:20]!r}",
              any("Alex" in p["name"] for p in (s.participants or [])), s.participants)


# ── the real API ─────────────────────────────────────────────────────────────────────────
def test_real_api_reads():
    from split import api
    ppl = api.people()
    check("/people answers", isinstance(ppl, list), api.last_error())
    dash = api.dashboard()
    check("/split-dashboard answers", isinstance(dash, dict) and dash, api.last_error())
    if ppl:
        check("balances are signed from the customer's side",
              all(isinstance(p.get("netBalance", 0), (int, float)) for p in ppl))

def test_status_never_creates_anything():
    a = ask("who owes me money", want_second_opinion=False, session="s1",
            pick={"intent": "split_status"})
    check("a balance question makes no proposal", a.proposal is None, a.proposal)
    check("and offers nothing executable",
          all(o["kind"] in ("ask_text", "split_nudge", "label", "dismiss", "pick_txn")
              for o in a.options), [o["kind"] for o in a.options])


# ── the PIN gate ─────────────────────────────────────────────────────────────────────────
def test_pin_gate_is_off_by_default():
    from engine import pin
    os.environ.pop("CX_REQUIRE_PIN", None)
    check("off unless switched on", not pin.required())
    os.environ["CX_REQUIRE_PIN"] = "1"
    check("on when switched on", pin.required())
    os.environ.pop("CX_REQUIRE_PIN", None)

def test_pin_is_strict_about_what_a_pin_looks_like():
    from engine import pin
    # EXACTLY four digits — PinDTO carries @Pattern(REGEX_PIN, "Pin should be 4 digit").
    for good in ["1234", "12 34", "1234-"]:
        check(f"{good!r} is a PIN", pin.looks_like_pin(good))
    for bad in ["123", "12345", "123456", "1234567", "send 1234 to alex", "", "abcd", "50"]:
        check(f"{bad!r} is NOT a PIN", not pin.looks_like_pin(bad))

def test_pin_attempts_are_bounded():
    """Our limit must sit well below the WALLET's, which we cannot reset.

    `PinServiceImpl.pinValidate` counts failures on the user row and at the fifth sets
    UserStatus.FROZEN with an email, an SMS and a push. Stopping at two means a broken
    integration cannot walk somebody into a frozen wallet.
    """
    from engine import pin
    check("our limit is below the wallet's freeze threshold",
          pin.MAX_ATTEMPTS < pin.WALLET_FREEZE_AT,
          f"{pin.MAX_ATTEMPTS} vs {pin.WALLET_FREEZE_AT}")
    pin.challenge("pg", {"kind": "send", "amount": 10})
    check("the payment is held", pin.pending("pg") is not None)
    rec = pin.pending("pg")
    rec["tries"] = pin.MAX_ATTEMPTS - 1
    import wallet_pin
    real = wallet_pin.verify
    wallet_pin.verify = lambda p: (False, "WRONG_PIN")
    try:
        ok, msg = pin.validate("pg", "0000")
    finally:
        wallet_pin.verify = real
    check("exhausted attempts abandon the payment", not ok and pin.pending("pg") is None,
          msg)
    check("and say nothing left the wallet", "nothing has left" in msg.lower(), msg)

def test_an_integration_failure_is_not_a_wrong_pin():
    """The wallet's counter only moves when the request reaches the PIN check. Counting our
    own transport errors would walk the customer towards a freeze for our fault."""
    from engine import pin
    import wallet_pin
    pin.challenge("pf", {"kind": "bill_pay"})
    real = wallet_pin.verify
    wallet_pin.verify = lambda p: (_ for _ in ()).throw(ConnectionError("down"))
    try:
        ok, msg = pin.validate("pf", "1234")
    finally:
        wallet_pin.verify = real
    check("no attempt consumed", pin.pending("pf") and pin.pending("pf")["tries"] == 0,
          pin.pending("pf"))
    check("and it says nothing was taken", "nothing has been taken" in msg.lower(), msg)

def test_a_locked_wallet_stops_immediately():
    from engine import pin
    import wallet_pin
    pin.challenge("pl", {"kind": "bill_pay"})
    real = wallet_pin.verify
    wallet_pin.verify = lambda p: (False, "LOCKED")
    try:
        ok, msg = pin.validate("pl", "1234")
    finally:
        wallet_pin.verify = real
    check("retrying is not offered", pin.pending("pl") is None)
    check("and the real reason is given", "locked" in msg.lower(), msg)

def test_pin_is_never_echoed():
    from engine import pin
    pin.challenge("pe", {"kind": "send", "amount": 10})
    ok, msg = pin.validate("pe", "9137")
    check("the PIN never appears in the message", "9137" not in msg, msg)


# ── live settlement updates ──────────────────────────────────────────────────────────────
def test_watcher_announces_only_transitions():
    """A participant who was ALREADY paid when we first saw the split has not "just paid".

    The watcher reports a change of state, not a state. Without the baseline pass, opening
    the page would announce every settled share in every split as if it had happened that
    second.
    """
    from split import watcher, api
    got = []
    watcher._WATCHED.clear()
    watcher._EMIT = lambda stage, kind, detail, tag: got.append((kind, detail))
    fake = {"SP-T1": {"splitStatus": "COLLECTING", "participants": [
        {"userId": "u1", "name": "Alex U", "status": "PENDING", "shareAmount": 42.5,
         "initiator": False},
        {"userId": "me", "name": "Sagar", "status": "PAID", "shareAmount": 42.5,
         "initiator": True}]}}
    real_detail = api.detail
    api.detail = lambda sid: fake.get(sid)
    try:
        watcher.watch("SP-T1")
        watcher._tick()
        check("first look announces nothing", not got, got)
        fake["SP-T1"]["participants"][0]["status"] = "PAID"
        watcher._tick()
        check("the transition is announced",
              any(k == "split.paid" and "Alex" in d for k, d in got), got)
        check("and the settle", any(k == "split.settled" for k, d in got), got)
        check("the initiator is never announced",
              not any("Sagar" in d for _, d in got), got)
        check("a settled split stops being polled", "SP-T1" not in watcher._WATCHED)
    finally:
        api.detail = real_detail
        watcher._WATCHED.clear()

def test_watcher_survives_a_failed_read():
    from split import watcher, api
    watcher._WATCHED.clear()
    watcher._EMIT = lambda *a: None
    real_detail = api.detail
    api.detail = lambda sid: None          # the API is unreachable
    try:
        watcher.watch("SP-T2")
        watcher._tick()
        check("an unreachable API does not drop the watch", "SP-T2" in watcher._WATCHED)
    finally:
        api.detail = real_detail
        watcher._WATCHED.clear()

def test_groups_distinguishes_failure_from_empty():
    """`d or []` made a failed read look exactly like "this customer has no groups" — the
    same collapse that made every merchant unpayable during a VPN drop."""
    from split import api
    real_call = api.call
    api.call = lambda *a, **k: None
    try:
        check("a failed groups read returns None, not []", api.groups() is None)
    finally:
        api.call = real_call


# ── yes and no ───────────────────────────────────────────────────────────────────────────
def test_yes_confirms_the_standing_offer():
    """Reported: proposal -> nudge -> "yes" -> "Of course. I handle anything to do with your
    payments." The offer lived only inside one HTTP response, so a bare "yes" reached the
    classifier with no subject and scored as a greeting."""
    from engine import conversation
    a = ask("split my last payment with alex", want_second_opinion=False, session="y1")
    check("something is on offer", a.outcome == "NEED_CONFIRMATION" and a.proposal, a.outcome)
    check("and it survives the turn", conversation.proposal("y1") is not None)
    b = ask("yes", want_second_opinion=False, session="y1")
    check("yes confirms it", b.outcome == "CONFIRMED" and b.confirmed, f"{b.outcome}")
    check("and carries the same proposal",
          (b.proposal or {}).get("referenceTxnId") == a.proposal["referenceTxnId"])
    check("the engine still does not execute it itself", b.say == "")

def test_no_cancels_and_never_starts_another_flow():
    """Reported: "cancel" after a proposal talked about an unrelated payment and offered a
    refund. On their own these words have no subject, so the classifier invented one —
    cancel/no/stop landed in txn_failed, txn_history and txn_on_hold."""
    for word in ["cancel", "no", "not now", "nevermind", "stop", "forget it",
                 "nahi", "rehne do", "नहीं", "لا"]:
        ask("send 100 to alex", want_second_opinion=False, session="n" + word[:4])
        b = ask(word, want_second_opinion=False, session="n" + word[:4])
        check(f"{word!r} cancels", b.outcome == "CANCELLED", f"{b.intent}/{b.outcome}")
        check(f"{word!r} proposes nothing", not b.proposal)
        # Checked per LANGUAGE, not against English words. The reply is now in whatever
        # language the customer wrote in, so asserting "moved" appears made a correct Hindi
        # answer look like a failure.
        from engine import lang as _lang
        # BRD 2.1 T10 wording ("Payment cancelled. No money was sent.") is accepted alongside
        # the older "nothing has moved" phrasing, in each language.
        want = {"en": ("moved", "no problem", "no money was sent"), "hinglish": ("hua", "chod", "nahi gaya"),
                "hi": ("हुआ", "छोड़", "नहीं भेजा"), "ar": ("شيء", "تركت", "لم يتم إرسال")}[_lang.detect(word)]
        check(f"{word!r} confirms nothing moved, in its own language",
              any(w in (b.say or "").lower() for w in want), f"{b.say}")

def test_no_drops_the_offer_for_good():
    from engine import conversation
    ask("split my last payment with alex", want_second_opinion=False, session="n2")
    ask("no", want_second_opinion=False, session="n2")
    check("the offer is gone", conversation.proposal("n2") is None)
    b = ask("yes", want_second_opinion=False, session="n2")
    check("a later yes cannot resurrect it", not b.confirmed, b.outcome)

def test_yes_alone_does_not_invent_a_payment():
    from engine import conversation
    conversation.drop_proposal("y3")
    conversation.clear("y3")
    b = ask("yes", want_second_opinion=False, session="y3")
    check("yes with nothing on offer executes nothing",
          not b.confirmed and not b.proposal, f"{b.intent}/{b.outcome}")


# ── the nudge asks the right question ────────────────────────────────────────────────────
def test_nudge_asks_for_a_decision_not_a_slot():
    """Reported: the nudge asked "how much should Alex get?" about a split that already had
    its amount. A proposal is complete — we are waiting on yes or no, not a missing slot."""
    from engine import nudge, conversation
    ask("split my last payment with alex", want_second_opinion=False, session="nd1")
    nudge.reset("nd1")
    n = nudge.due("nd1", 60)
    check("it nudges", n is not None)
    check("it does NOT ask how much",
          n and "how much" not in n["say"].lower(), (n or {}).get("say"))
    check("it names what is waiting",
          n and "split" in n["say"].lower(), (n or {}).get("say"))
    check("and offers yes/no",
          n and {o["text"] for o in n["options"]} == {"yes", "no"},
          [o.get("text") for o in (n or {}).get("options", [])])
    frame = conversation.get("nd1")
    check("a complete proposal is not recorded as needing a slot",
          frame is None or frame.asked_for is None,
          frame.asked_for if frame else None)


# ── only the current split speaks ────────────────────────────────────────────────────────
def test_only_this_conversations_split_is_announced():
    """Reported: the chat filled with "Alex just paid their 90.00… their 40.00…" for every
    historical group. Old splits are still followed so a restart does not lose them, but
    they do not interrupt."""
    from split import watcher, api
    got = []
    watcher._WATCHED.clear()
    watcher._EMIT = lambda stage, kind, detail, tag: got.append((kind, detail))
    fake = {sid: {"splitStatus": "COLLECTING", "participants": [
        {"userId": "u1", "name": "Alex U", "status": "PENDING", "shareAmount": 90.0,
         "initiator": False},
        {"userId": "me", "name": "Sagar", "status": "PAID", "shareAmount": 10.0,
         "initiator": True}]} for sid in ("SP-OLD", "SP-NEW")}
    real = api.detail
    api.detail = lambda sid: fake.get(sid)
    try:
        watcher.watch("SP-OLD", announce=False)
        watcher.watch("SP-NEW")
        watcher._tick()
        for sid in fake:
            fake[sid]["participants"][0]["status"] = "PAID"
        watcher._tick()
        check("something was announced", got)
        check("and ONLY the current split",
              all("SP-NEW" in d for _, d in got), got)
    finally:
        api.detail = real
        watcher._WATCHED.clear()


# ── bills: never propose what cannot be paid ─────────────────────────────────────────────
def test_unroutable_biller_is_refused_before_the_pin():
    """Reported: entered a correct PIN, got "That didn't go through".

    The PIN was fine — the transaction was rejected with "Transaction cannot be carried out
    for this amount", because the biller's `product_service_provider_slab` row was PENDING
    and routing filters on ACTIVE. Finding that out AFTER taking somebody's PIN reads as the
    wallet being broken.

    The RULE is tested here, not a particular dev row. The first version asserted that
    product 50 was unroutable, which was true until the row was activated — a test that
    fails when the environment is fixed is testing the environment.
    """
    from bills import quote
    import devdb
    real_q = devdb.q
    cases = [
        ("no provider at all", [], False),
        ("provider not live", [{"status": "PENDING", "amount_from": 1, "amount_to": 10000,
                                "date_from": None, "date_to": None}], False),
        ("active but out of range", [{"status": "ACTIVE", "amount_from": 1,
                                      "amount_to": 100, "date_from": None,
                                      "date_to": None}], False),
        ("active and in range", [{"status": "ACTIVE", "amount_from": 1,
                                  "amount_to": 10000, "date_from": None,
                                  "date_to": None}], True),
        ("one active among several", [
            {"status": "INACTIVE", "amount_from": 1, "amount_to": 10, "date_from": None,
             "date_to": None},
            {"status": "ACTIVE", "amount_from": 1, "amount_to": 10000, "date_from": None,
             "date_to": None}], True),
    ]
    for name, rows, want in cases:
        quote._CACHE.clear()
        devdb.q = lambda sql, args=None, n=50, _r=rows: _r
        try:
            ok, why = quote.routable("test", 506)
        finally:
            devdb.q = real_q
            quote._CACHE.clear()
        check(f"routable: {name}", ok == want, f"got {ok} ({why})")
        if not want:
            check(f"  …with a reason a customer can read: {name}", bool(why), why)

def test_an_unroutable_bill_is_never_proposed():
    """Whatever the dev config happens to be, a bill the engine knows cannot route must not
    reach a proposal — because the next thing after a proposal is a PIN."""
    import bills.quote as quote
    from engine import engine as E
    real = quote.routable
    quote.routable = lambda pid, amt: (False, "its payment provider isn't live yet")
    try:
        a = ask("pay my home electricity bill", want_second_opinion=False, session="rtX",
                pick={"intent": "bill_pay"})
    finally:
        quote.routable = real
    check("refused, not proposed", a.outcome == "UNAVAILABLE" and not a.proposal,
          f"{a.outcome}")
    check("and the reason is given", "provider" in (a.say or "").lower(), a.say)

def test_provider_charge_is_clamped_like_the_backend():
    """RoutingMechanismImpl clamps: max(min, min(charge, max)). ADDC is 1% with a floor of
    2 and a ceiling of 10, so a 100 bill is charged 2, not 1."""
    from bills import quote
    import devdb
    real_q = devdb.q
    rows = [{"amount_from": 1, "amount_to": 10000, "charges": 1,
             "commission_type": "PERCENTAGE", "min_charges": 2, "max_charges": 10}]
    quote._CACHE.clear()
    devdb.q = lambda sql, args=None, n=50, _r=rows: (_r if "service_provider_slab" in sql
                                                     else [])
    try:
        lo = quote.explain("t", 100)["biller_charge"]      # 1% = 1.00 -> floored to 2
        mid = quote.explain("t", 506)["biller_charge"]     # 1% = 5.06 -> unchanged
        hi = quote.explain("t", 5000)["biller_charge"]     # 1% = 50   -> capped at 10
    finally:
        devdb.q = real_q
        quote._CACHE.clear()
    check("floored at the minimum", lo == 2.0, lo)
    check("left alone in between", mid == 5.06, mid)
    check("capped at the maximum", hi == 10.0, hi)


# ── a pending bill is not a failed bill ──────────────────────────────────────────────────
def test_pending_is_a_successful_bill_submission():
    """Reported: paid the internet bill, got "That didn't go through — unknown".

    It HAD gone through. `WalletToUtilityHandler` writes the ledger line as
    TransactionStatus.PENDING at submission, because a bill is handed to a third-party biller
    and settles asynchronously. Demanding SUCCESS denied a payment that had left the wallet
    and had a ledger row. Telling someone a payment failed when it did not is worse than
    saying nothing at all.
    """
    import bills.pay as bp
    import openwallet_api as api
    real = api.transact
    captured = {}

    def fake(amount, **kw):
        captured.update(kw)
        return {"status": "200", "data": {"txnId": "TXN-PENDING-1", "status": "PENDING"}}

    api.transact = fake
    live_before = bp.LIVE
    bp.LIVE = True
    try:
        out = bp.pay_now({"payMode": "WALLET", "total": 161.0,
                          "bills": [{"planId": "67ee331b2c647e6c6c55d91b",
                                     "productId": "81", "identifier": "42991140037",
                                     "nickname": "Home internet", "amount": 161.0}]},
                         lambda *a, **k: None)
    finally:
        api.transact = real
        bp.LIVE = live_before
    check("PENDING is accepted, not failed", out.get("ok"), out.get("error"))
    check("and the reference is given", "TXN-PENDING-1" in (out.get("say") or ""),
          out.get("say"))
    check("the wording says it is still settling",
          "biller" in (out.get("say") or "").lower(), out.get("say"))
    check("it routes as a bill, not a transfer",
          captured.get("product") == "BILL_PAYMENTS", captured)

def test_no_transaction_id_is_a_real_failure():
    """The opposite error: a response with no transaction id created nothing, and must not
    be dressed up as "sent"."""
    import bills.pay as bp
    import openwallet_api as api
    real = api.transact
    api.transact = lambda amount, **kw: {"status": "500",
                                         "message": "No Service Provider Found"}
    live_before = bp.LIVE
    bp.LIVE = True
    try:
        out = bp.pay_now({"payMode": "WALLET", "total": 161.0,
                          "bills": [{"planId": "p", "productId": "81", "identifier": "i",
                                     "nickname": "Home internet", "amount": 161.0}]},
                         lambda *a, **k: None)
    finally:
        api.transact = real
        bp.LIVE = live_before
    check("no txnId means failure", not out.get("ok"), out)
    check("and the real reason is quoted, never 'unknown'",
          "No Service Provider" in (out.get("error") or "")
          and "unknown" not in (out.get("error") or "").lower(), out.get("error"))


# ── the conversation remembers its subject ───────────────────────────────────────────────
def test_the_subject_carries_across_turns():
    """Reported: quoted "AED 506.00 for Home electricity", then said "split kardo", and got
    "Who are you splitting AED 450 with?" — an unrelated merchant payment.

    The frame remembered the open QUESTION and the topic remembered the last INTENT; neither
    remembered WHAT the conversation was about. A person would have remembered the bill.
    """
    from engine import conversation as conv
    sid = "subj1"
    conv.clear(sid)
    conv.forget_subject(sid, "bill")
    a = ask("pay my home electricity bill", want_second_opinion=False, session=sid)
    check("the bill is resolved", "Home electricity" in (a.say or ""), a.say)
    check("and remembered", conv.subject(sid, "bill") == "Home electricity",
          conv.subject(sid, "bill"))
    b = ask("split kardo", want_second_opinion=False, session=sid)
    check("a bare follow-up knows which bill",
          "Home electricity" in (b.say or ""), b.say)
    check("and it is a BILL split, not a merchant one",
          b.intent == "split_payment" and "450" not in (b.say or ""), b.say)

def test_a_named_subject_overrides_the_remembered_one():
    """Memory fills GAPS. It must never beat what the customer just said, or they can never
    change the subject."""
    sid = "subj2"
    from engine import conversation as conv
    conv.clear(sid)
    conv.forget_subject(sid, "bill")
    ask("pay my home electricity bill", want_second_opinion=False, session=sid)
    b = ask("split my internet bill", want_second_opinion=False, session=sid)
    check("the newly named bill wins",
          "internet" in (b.say or "").lower() and "electricity" not in (b.say or "").lower(),
          b.say)

def test_memory_does_not_leak_between_sessions():
    from engine import conversation as conv
    conv.forget_subject("subj3", "bill")
    ask("pay my home electricity bill", want_second_opinion=False, session="subj4")
    check("another session knows nothing of it", conv.subject("subj3", "bill") is None)

def test_the_subject_goes_stale():
    """Yesterday's bill must not answer today's "split it"."""
    from engine import conversation as conv
    conv.note_subject("subj5", "bill", "Home electricity")
    check("fresh", conv.subject("subj5", "bill") == "Home electricity")
    conv._SUBJECT["subj5"]["bill"] = ("Home electricity",
                                      __import__("time").time() - conv.SUBJECT_TTL - 1)
    check("stale is forgotten", conv.subject("subj5", "bill") is None)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{'all green' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)} "
          f"({len(FAILED)} failing)")
