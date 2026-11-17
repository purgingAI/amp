#!/usr/bin/env python3
"""The bug: being told somebody paid, and offering to pay them instead.

Reported from a real session. The customer split an electricity bill with Alex, Alex
paid, and:

* the chat never noticed the money had arrived, and
* "alex ne paise bhej diye hai" was answered with "alex ko kitne bhejne hai?" — an offer
  to send money to the man who had just sent it.

Three separate defects produced that, and each gets a test here, because each would come
back on its own:

1. the direction of the sentence (`_is_payment_claim`),
2. which earlier thing the sentence is ABOUT (`conversation.find_task`),
3. whether one wallet credit can settle more than one share (`claim_credit`).

Run: ~/amp/.venv/bin/python tests/test_payment_claim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, wanted {want!r}")
        print(f"  FAIL  {name}  -> {got!r} (wanted {want!r})")
    else:
        print(f"  ok    {name}")


# ── 1. direction: a report of money IN vs a request to send money OUT ────────────────────
def test_direction():
    from engine.engine import _is_payment_claim
    print("\ndirection — reports of money that arrived")
    for s in ["alex ne paise bhej diye hai", "alex ne paise bhej diye", "he paid",
              "alex has paid me", "paisa aa gaya", "alex ne pay kar diya",
              "he sent it", "paise mil gaye", "विकास ने भेज दिया",
              "alex ne bhej diya hai bhai", "asad ne apna share bhej diya"]:
        check(f"claim: {s}", _is_payment_claim(s), True)

    print("\ndirection — requests to SEND, which must never be read as reports")
    for s in ["alex ko paise bhejne hai", "paise bhejdo", "send 50 to asad",
              "alex ko bhej do", "paise bhejna hai", "send him 100",
              "विकास को भेजना है", "pay asad 20", "alex ko uska hissa bhejna hai"]:
        check(f"send:  {s}", _is_payment_claim(s), False)


# ── 2. reaching back past several topic changes ──────────────────────────────────────────
def test_recall_across_topics():
    from engine import conversation as C
    print("\nrecall — which earlier task is this about?")
    S = "test-recall"
    C._TASKS.pop(S, None)
    C.open_task(S, "bill_split", "split Home electricity with Alex",
                {"bill": "Home electricity", "payee": "Alex Morgan",
                 "split_id": "BS-1"}, "waiting")
    C.open_task(S, "send", "send AED 50 to Asad", {"payee": "Asad Khan"}, "done")
    C.open_task(S, "bill_pay", "Internet bill payment", {"bill": "Internet"}, "done")
    C.open_task(S, "pay_merchant", "pay MClothes", {"merchant": "MClothes"}, "done")

    # Three topics have gone by since the split. The single-subject memory has been
    # overwritten twice; the ledger still knows.
    for said, want in [
        ("alex ne paise bhej diye hai", "bill_split"),
        ("electricity wala kya hua", "bill_split"),
        ("asad ko bheja tha wo ho gaya", "send"),
        ("internet bill ka status", "bill_pay"),
        ("mclothes wala", "pay_merchant"),
    ]:
        t = C.find_task(S, said)
        check(f"recall: {said}", t.kind if t else None, want)

    # Ambiguity is answered by asking, not by guessing the most recent one.
    check("recall: bare 'that one' with 1 live task",
          (C.find_task(S, "what about that one") or type("x", (), {"kind": None})).kind,
          "bill_split")
    check("recall: unrelated question", C.find_task(S, "what is the weather"), None)


# ── 3. one credit cannot settle two shares ───────────────────────────────────────────────
def test_credit_claimed_once():
    from bills import billsplit
    print("\ncredits — one payment, one share")
    txn = "TEST-CREDIT-ONCE-1"
    try:
        import sqlite3
        from bills import schedule as store
        if not store.DSN.startswith("mysql://"):
            with store._conn() as c:
                c.cursor().execute("DELETE FROM bill_split_credit WHERE txn_id = ?",
                                   (txn,))
                c.commit()
    except Exception:
        pass

    check("first claim wins", billsplit.claim_credit(txn, "BS-A", "u1", 253.0), True)
    check("second claim refused", billsplit.claim_credit(txn, "BS-B", "u1", 253.0), False)
    owner = billsplit.credit_claimed_by(txn)
    check("owner recorded", (owner or {}).get("split_id"), "BS-A")

    try:
        from bills import schedule as store
        with store._conn() as c:
            c.cursor().execute(f"DELETE FROM bill_split_credit WHERE txn_id = "
                               f"{store._ph()}", (txn,))
            if not store.DSN.startswith("mysql://"):
                c.commit()
    except Exception:
        pass


# ── 4. the credit must actually be from that person ──────────────────────────────────────
def test_sender_identity():
    from bills import billsplit_watch as W
    print("\nsenders — a credit only counts if it came from that participant")
    # No decryptable sender and no id match: must NOT be accepted. The old code accepted
    # any row whose decryption returned something, for everyone.
    check("unknown sender rejected",
          W._is_them({"sender": "not-a-real-token"}, "uid-1", "Alex Morgan"), False)
    check("id match accepted",
          W._is_them({"sender": "wallet-uid-1-abc"}, "uid-1", "Alex Morgan"), True)
    check("empty sender rejected",
          W._is_them({"sender": ""}, "uid-1", "Alex Morgan"), False)


# ── 5. one bill is never paid twice ──────────────────────────────────────────────────────
def test_no_double_payment():
    """The duplicate-collection bug left FOUR READY collections for one AED 506 bill.

    Nothing was paid only because the biller happened to be timing out. A healthy biller
    would have been charged 2,024 AED for a 506 AED bill, from the customer's own wallet.
    """
    from bills import billsplit
    print("\nduplicates — one bill, one payment")
    import uuid
    plan, ident = "PLAN-TEST", "IDENT-" + uuid.uuid4().hex[:6]
    bill = {"planId": plan, "identifier": ident, "nickname": "Test bill",
            "biller": "TestCo", "productId": "1"}
    a = billsplit.create("u-test", bill, 100.0, [{"userId": "p1", "name": "Pat Tester"}])
    b = billsplit.create("u-test", bill, 100.0, [{"userId": "p1", "name": "Pat Tester"}])

    check("no twin before either pays",
          billsplit.already_paid_for(plan, ident, a["id"]), None)

    # a pays
    rec = billsplit.get(a["id"])
    rec["status"], rec["paid_txn"] = billsplit.PAID, "TXN-PAID-A"
    billsplit._save(rec)

    twin = billsplit.already_paid_for(plan, ident, b["id"])
    check("b can see a already paid", (twin or {}).get("id"), a["id"])
    check("b is superseded, not paid", billsplit.supersede(b["id"], a["id"]), True)
    check("b's final status", (billsplit.get(b["id"]) or {}).get("status"),
          billsplit.CANCELLED)
    check("b paid nothing", (billsplit.get(b["id"]) or {}).get("paid_txn"), None)

    for sid in (a["id"], b["id"]):          # tidy up after ourselves
        try:
            from bills import schedule as store
            with store._conn() as c:
                c.cursor().execute(f"DELETE FROM bill_split WHERE id = {store._ph()}",
                                   (sid,))
                if not store.DSN.startswith("mysql://"):
                    c.commit()
        except Exception:
            pass




def test_a_negated_arrival_is_not_an_arrival():
    """"not credited" contains "credited", and that was enough to invert the answer.

    Reported: "My last transaction got deducted but the receiver didn't get money" came
    back "Yes — Alex's 253.00 came in. That's every share — I can pay Home electricity
    now." A customer reporting money lost in transit was told it had arrived. That is the
    worst shape of wrong answer available here: it is false, it is reassuring, and it stops
    them chasing the money.

    Two separate holes. The arrival words are matched as substrings, so the negation was
    never read; and once the negation WAS read, the classifier still reached payment_claim
    on its own at full confidence, because "alex ne paise nahi bheje" is one short token
    away from the sentence that means the opposite.
    """
    from engine.engine import ask
    print("\nmoney that did NOT arrive")
    for q in ("amount debited but not credited to receiver",
              "My last transaction got deducted but the receiver didnt get money",
              "i sent money but he has not received it",
              "the money never reached him",
              "alex ne paise nahi bheje",
              "usne paise nahi diye",
              "paisa kat gaya par nahi pahuncha",
              "विकास ने पैसे नहीं भेजे"):
        a = ask(q, want_second_opinion=False, session="neg-" + q[:12])
        check(f"{q[:42]!r} is not read as money in",
              a.intent == "payment_claim", False)

    print("  and money that DID arrive still is")
    for q in ("alex paid me", "asad sent me his share", "the money came in",
              "alex ne paise bhej diye", "paisa aa gaya", "उसने पैसे भेज दिए",
              "فيكاس دفع لي"):
        a = ask(q, want_second_opinion=False, session="pos-" + q[:12])
        check(f"{q[:42]!r} is money in", a.intent, "payment_claim")


if __name__ == "__main__":
    test_direction()
    test_recall_across_topics()
    test_credit_claimed_once()
    test_sender_identity()
    test_no_double_payment()
    test_a_negated_arrival_is_not_an_arrival()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED:")
        for f in FAILED:
            print("  -", f)
        sys.exit(1)
    print("all good")
