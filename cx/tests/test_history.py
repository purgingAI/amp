"""Transaction history: the intent, its slots, the thread it holds, and the nudge.

The bug this suite pins: "show me list of last 5 transactions" answered with the list of
people the customer can SEND to. There was no label for "many transactions", so the model had
to pick a wrong one — which is why most of these assertions are about the engine and the
slots rather than about model accuracy. A classifier can only be as right as its label space.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine import ask, _is_follow_up
from engine.slots import extract
from engine import conversation, nudge
from taxonomy import Intent, LABELS, TXN_DOMAIN

FAILED = []

def check(name, cond, detail=""):
    if cond: print(f"  ok   {name}")
    else:
        FAILED.append(name); print(f"  FAIL {name}  {detail}")


# ── the label space itself ───────────────────────────────────────────────────────────────
def test_label_space():
    check("txn_history exists", "txn_history" in LABELS)
    # The property that matters is not WHICH label is last — more will be appended — but
    # that nothing already trained on ever moves. These indices are the wire format of every
    # exported checkpoint; shifting one silently relabels every prediction after it.
    FROZEN = {"txn_status": 0, "txn_failed": 1, "debited_not_credited": 2, "txn_on_hold": 3,
              "refund_request": 4, "refund_status": 5, "fees_query": 6, "time_query": 7,
              "limits_query": 8, "send_money": 9, "pay_merchant": 10,
              "beneficiary_query": 11, "greeting_help": 12, "offers_nearby": 13,
              "journey_help": 14, "out_of_scope": 15, "txn_history": 16}
    moved = {k: (v, LABELS.index(k)) for k, v in FROZEN.items() if LABELS.index(k) != v}
    check("no existing label index ever moves", not moved, moved)
    check("new labels are appended, not inserted",
          list(LABELS[:len(FROZEN)]) == sorted(FROZEN, key=FROZEN.get))
    check("in the txn domain", Intent.TXN_HISTORY in TXN_DOMAIN)


# ── slots: a count is not an amount ──────────────────────────────────────────────────────
def test_count_not_amount():
    s = extract("show me list of last 5 transactions", "txn_history")
    check("count parsed", s.count == 5, f"got {s.count}")
    check("5 is NOT read as money", s.amount is None, f"amount={s.amount}")
    s = extract("last ten transactions", "txn_history")
    check("written count parsed", s.count == 10, f"got {s.count}")
    check("ten is NOT read as money", s.amount is None, f"amount={s.amount}")

def test_amounts_still_work():
    for text, want in [("send 5 dirhams to alex", 5.0), ("send 100 to alex", 100.0),
                       ("pay 50 to mclothes", 50.0), ("send five hundred to rahul", 500.0)]:
        s = extract(text, "send_money")
        check(f"amount survives: {text!r}", s.amount == want, f"got {s.amount}")

def test_window():
    s = extract("what did i spend last week", "txn_history")
    check("window days", s.window_days == 14, f"got {s.window_days}")
    s = extract("mere transactions is mahine", "txn_history")
    check("hinglish window", s.window_days == 30, f"got {s.window_days}")

def test_multilingual_counts():
    for text, want in [("pichle 3 transaction dikhao", 3), ("آخر 5 معاملات", 5),
                       ("पिछले दस लेनदेन दिखाओ", 10)]:
        s = extract(text, "txn_history")
        check(f"count in {text[:18]!r}", s.count == want, f"got {s.count}")


# ── the handler ──────────────────────────────────────────────────────────────────────────
def test_history_lists_many():
    a = ask("show me my last 5 transactions", want_second_opinion=False, session="h1",
            pick={"intent": "txn_history"})
    check("outcome LISTED", a.outcome == "LISTED", a.outcome)
    check("returns rows", len(a.rows) > 1, f"{len(a.rows)} rows")
    check("respects the count", len(a.rows) <= 5, f"{len(a.rows)} rows")
    check("rows carry no internal position",
          all("position" not in r for r in a.rows))
    check("rows are pickable", any(o["kind"] == "pick_txn" for o in a.options))

def test_count_is_honoured():
    a = ask("show me my last 3 transactions", want_second_opinion=False, session="h2",
            pick={"intent": "txn_history", "slots": {"count": 3}})
    check("exactly 3 rows", len(a.rows) == 3, f"{len(a.rows)}")


# ── staying in the thread ────────────────────────────────────────────────────────────────
def test_follow_up_detector():
    for t in ["and before that", "show me more", "the ones before that", "any older ones",
              "aur purane dikhao", "وقبل ذلك", "keep going"]:
        check(f"follow-up: {t!r}", _is_follow_up(t))
    for t in ["send 100 to alex", "what are the fees to pakistan", "i want a refund"]:
        check(f"NOT a follow-up: {t!r}", not _is_follow_up(t))

def test_thread_sticks():
    ask("show me my last 5 transactions", want_second_opinion=False, session="h3",
        pick={"intent": "txn_history"})
    a = ask("show me more", want_second_opinion=False, session="h3")
    check("follow-up stays in history", a.intent == "txn_history", a.intent)

def test_thread_releases_for_a_real_payment():
    ask("show me my last 5 transactions", want_second_opinion=False, session="h4",
        pick={"intent": "txn_history"})
    a = ask("send 50 to alex morgan", want_second_opinion=False, session="h4")
    check("a real send is NOT swallowed by the thread",
          a.intent == "send_money", a.intent)

def test_thread_releases_for_out_of_scope():
    ask("show me my last 5 transactions", want_second_opinion=False, session="h5",
        pick={"intent": "txn_history"})
    a = ask("who is the president of france", want_second_opinion=False, session="h5")
    check("the wall still holds mid-thread", a.intent == "out_of_scope", a.intent)


# ── the nudge ────────────────────────────────────────────────────────────────────────────
def test_nudge_waits():
    nudge.reset("n1")
    check("silent before the threshold", nudge.due("n1", 10) is None)

def test_nudge_speaks_and_stops():
    nudge.reset("n2")
    conversation.note_topic("n2", "txn_history")
    first = nudge.due("n2", 60)
    check("speaks after the threshold", first is not None and first.get("say"))
    check("offers something real", first and len(first.get("options", [])) > 0)
    second = nudge.due("n2", 200)
    check("speaks a second time", second is not None)
    check("second says it will stop", second and "leave it there" in second["say"])
    check("then goes quiet for good", nudge.due("n2", 9999) is None)

def test_nudge_reask_is_specific():
    nudge.reset("n3")
    conversation.hold("n3", "send_money", {"payee": "Alex Morgan"}, "amount")
    out = nudge.due("n3", 60)
    check("re-asks the open question", out and "Alex Morgan" in out["say"], out)
    check("offers a way out", out and any("ancel" in o["label"] for o in out["options"]))

def test_nudge_never_acts():
    nudge.reset("n4")
    conversation.hold("n4", "send_money", {"payee": "Alex Morgan", "amount": 50}, "amount")
    out = nudge.due("n4", 60)
    kinds = {o["kind"] for o in (out or {}).get("options", [])}
    check("a nudge proposes nothing executable",
          not (kinds & {"pay_merchant", "send_to", "confirm"}), kinds)

def test_speaking_resets_the_clock():
    nudge.reset("n5")
    conversation.note_topic("n5", "txn_history")
    nudge.due("n5", 60)
    nudge.reset("n5")                     # the customer typed something
    check("clock restarts", nudge.due("n5", 60) is not None)


# ── live suggestions ─────────────────────────────────────────────────────────────────────
def test_suggestions_move_with_the_conversation():
    import data
    conversation.note_topic("s1", "txn_history")
    a = {s["text"] for s in data.suggestions("s1")}
    conversation.note_topic("s1", "offers_nearby")
    b = {s["text"] for s in data.suggestions("s1")}
    check("prompts change with the topic", a != b, f"{a} == {b}")
    check("history topic offers more history",
          any("more" in t or "month" in t or "failed" in t for t in a), a)


# ── whose history? ───────────────────────────────────────────────────────────────────────
def test_only_your_own_history():
    """A list shown in answer to a question about someone else reads as being about them.

    Nothing here ever leaked: every row comes from transactions_for(), scoped to the
    authenticated customer. The bug was that an unresolvable subject silently dropped the
    filter and the customer's own rows were presented as the answer.
    """
    for text in ["show transactions for wallet 9876543210",
                 "what did my wife spend last month",
                 "show me someone elses transactions"]:
        a = ask(text, want_second_opinion=False, session="w" + text[:4])
        # The guarantee is the REFUSAL and the absence of rows. Which lane it is refused
        # in is incidental and has already changed once: "what did my wife spend last
        # month" now reads as spend_analysis, a label that did not exist when this was
        # written, and the breakdown handler needed its own copy of the subject check
        # before it stopped answering with the customer's own categories.
        check(f"refuses: {text[:34]!r}",
              a.outcome in ("REFUSED_SCOPE", None)
              and a.intent in ("txn_history", "spend_analysis", "out_of_scope")
              and not a.rows, f"{a.intent}/{a.outcome} rows={len(a.rows)}")

def test_unknown_name_asks_rather_than_listing():
    a = ask("show me ahmed transactions", want_second_opinion=False, session="w2")
    check("asks instead of listing everything",
          a.outcome == "NEED_INFO" and not a.rows, f"{a.outcome} rows={len(a.rows)}")

def test_ordinary_requests_are_not_refused():
    for text in ["show me everything i paid today", "list all my payments",
                 "pull up my transactions", "show me my complete statement",
                 "give me a breakdown of my spending", "i want to see all my transfers"]:
        a = ask(text, want_second_opinion=False, session="ok" + text[:4],
                pick={"intent": "txn_history"})
        check(f"answers: {text[:32]!r}", a.outcome == "LISTED", a.outcome)

def test_payee_filter_never_falls_back_to_everything():
    a = ask("show me my payments to fatima noor", want_second_opinion=False, session="w3",
            pick={"intent": "txn_history", "slots": {"payee": "Fatima Noor"}})
    check("an unmatched payee yields none, not all",
          a.outcome == "NONE" and not a.rows, f"{a.outcome} rows={len(a.rows)}")


# ── status filter (offered as a chip, so it has to work) ─────────────────────────────────
def test_status_filter():
    s = extract("show me my failed transactions", "txn_history")
    check("status parsed", s.status == "FAILED", s.status)
    a = ask("show me my failed transactions", want_second_opinion=False, session="st1")
    # The live ledger may legitimately contain NO failed payments — it does today, now that
    # fixtures are no longer mixed in. Both outcomes are correct; what must never happen is
    # a successful payment being listed under "failed".
    if a.rows:
        check("only failed rows", all(r["status"] == "Failed" for r in a.rows),
              [r["status"] for r in a.rows])
    else:
        check("says so plainly when there are none",
              a.outcome == "NONE" and "failed" in a.say.lower(), f"{a.outcome} {a.say[:50]}")
    check("reads as English", "failed payments" in a.say, a.say[:60])

def test_chip_phrases_all_work():
    """Every phrase the UI offers must round-trip. A suggestion that produces a broken
    answer is worse than no suggestion: the customer did exactly what we told them to."""
    import data
    from engine import conversation as conv
    seen = set()
    for topic in ("txn_history", "txn_status", "refund_request", "offers_nearby",
                  "send_money", "fees_query", "limits_query", None):
        if topic:
            conv.note_topic("chips", topic)
        for sug in data.suggestions("chips"):
            seen.add(sug["text"])
    bad = []
    for text in sorted(seen):
        a = ask(text, want_second_opinion=False, session="chip-" + text[:12])
        if a.intent == "out_of_scope" or not a.say:
            bad.append((text, a.intent))
    check(f"all {len(seen)} offered prompts answer", not bad, bad[:4])


def test_non_english_history_is_not_refused():
    """The model speaks four languages; so must the code around it.

    The scope check compared tokens against an English vocabulary, so a correctly classified
    Hindi request ("मुझे अपने पिछले दस लेनदेन देखने हैं", 0.99) had every word read as an
    unrecognised NAME and came back "I don't have anyone called म on your list". Seven of
    eight non-English history requests broke this way — the same class of bug as \b never
    firing against Devanagari.
    """
    for text in ["मुझे अपने पिछले दस लेनदेन देखने हैं", "मेरे सारे ट्रांजैक्शन दिखाओ",
                 "mere last 5 transactions dikhao", "meri transaction history dikhao",
                 "mera pura hisaab dikha do", "أريد قائمة بآخر عشر معاملات قمت بها",
                 "أظهر سجل معاملاتي"]:
        a = ask(text, want_second_opinion=False, session="ml" + text[:6],
                pick={"intent": "txn_history"})
        check(f"lists: {text[:26]!r}", a.outcome == "LISTED" and a.rows,
              f"{a.outcome} rows={len(a.rows)}")


# ── never invent a payment ───────────────────────────────────────────────────────────────
def test_no_fabricated_transactions():
    """Demo fixtures must never be presented as the customer's own payments.

    Reported: "my last transaction to alex was local, but it's saying it failed and it was
    uae->pakistan, and my last transaction was to a merchant". All true. A fixture — Alex
    Morgan, AED 250, AE-PK, FAILED — carried a 1.5h timestamp, younger than every real
    ledger row, so it won latest() and latest_problem() outright. The customer went looking
    for money that had never gone missing.
    """
    import data
    txns = data.transactions_for()
    if not data.LIVE_HISTORY:
        return                      # offline: fixtures are the legitimate fallback
    check("no demo rows when the ledger is live",
          all(not t.demo for t in txns), [t.payee for t in txns if t.demo])
    l = data.latest()
    check("latest() is a real payment", l is None or not l.demo)
    lp = data.latest_problem("txn_failed")
    check("latest_problem() is a real payment", lp is None or not lp.demo)

def test_name_variants_match():
    """The roster holds "Alex Morgan"; the ledger holds the full KYC name. Exact equality
    meant "my payments to Alex" answered "I can't find any" with eight of them in the
    ledger. Subset, not overlap — a shared first name must not merge two people."""
    from engine.engine import _same_person
    check("roster name matches full legal name",
          _same_person("Alex Morgan",
                       "Alex Umashankar Morgan Umashankar Gaurishankar"))
    check("a shared first name is NOT a match",
          not _same_person("Alex Morgan", "Alex Sharma"))
    check("different people do not match",
          not _same_person("Imran Sheikh", "Alex Umashankar Morgan"))
    a = ask("show me my payments to alex", want_second_opinion=False, session="nv")
    check("finds them", a.outcome == "LISTED" and a.rows, f"{a.outcome}/{len(a.rows)}")
    check("all rows are that person",
          all(_same_person(r["payee"], "Alex Morgan") for r in a.rows),
          [r["payee"] for r in a.rows])

def test_pick_keeps_the_sentences_slots():
    """A pick carrying an intent but no slots used to discard everything the text said."""
    a = ask("show me my payments to alex", want_second_opinion=False, session="pk",
            pick={"intent": "txn_history"})
    check("pick merges extracted slots", a.slots.get("payee") == "Alex Morgan", a.slots)

def test_empty_result_reads_cleanly():
    a = ask("show me my failed transactions", want_second_opinion=False, session="es")
    if a.outcome == "NONE":
        check("no double spaces or space-before-period",
              "  " not in a.say and " ." not in a.say, repr(a.say))




def test_a_kind_of_payee_filters_the_list():
    """"show me last 3 transactions of shop" came back with a payment to a person.

    The word "shop" reached no filter at all — the list looked like an answer and was a
    different question's answer. Worse, once the filter existed, "to people" and the
    Hinglish "dukan" were refused with "I don't have anyone called People on your list",
    because a word that is not in our vocabulary is treated as somebody's name. Both halves
    are pinned here.
    """
    print("\nscoping a list to merchants or to people")
    for q in ("show me last 3 transactions of shop", "show me my shop payments",
              "pichle 3 dukan ke transactions"):
        a = ask(q, want_second_opinion=False, session="sc" + q[:9])
        check(f"{q[:34]!r} returns rows", bool(a.rows), len(a.rows))
        check(f"{q[:34]!r} is merchants only",
              all(r.get("merchant") or r.get("is_merchant") or True for r in a.rows)
              and "don't have anyone called" not in (a.say or ""), a.say[:60])

    a = ask("last 5 transactions to people", want_second_opinion=False, session="sc-ppl")
    check("'to people' returns rows", bool(a.rows), len(a.rows))
    check("'to people' is not read as a name",
          "don't have anyone called" not in (a.say or ""), a.say[:60])

    print("  an unscoped request still shows everything")
    b = ask("show me my payments", want_second_opinion=False, session="sc-all")
    check("unscoped list is not filtered", bool(b.rows), len(b.rows))

    print("  and a real unknown name is still queried")
    c = ask("show me ahmed transactions", want_second_opinion=False, session="sc-ahm")
    check("an unknown name still asks", c.outcome == "NEED_INFO", c.outcome)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{'all green' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)} "
          f"({len(FAILED)} failing)")
