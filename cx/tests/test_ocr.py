#!/usr/bin/env python3
"""Reading a payment screenshot, and refusing to read too much into it.

Two real screenshots drove this, and the second one is why the file is careful:

  * a Transaction Details screen for a payment — reference, wallets, amount all read
  * a REFUND screen, whose reference belongs to the MERCHANT, not to the customer

The second was answered "the reference on it isn't one of the payments from your wallet",
about a refund the customer had genuinely received. OCR was never the problem — it read
the number perfectly. The lookup was.

Most of what is pinned here is what the screenshot is NOT allowed to do.

Run: ~/amp/.venv/bin/python cx/tests/test_ocr.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILED = []


def check(name, ok, got=""):
    if ok:
        print(f"  ok    {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got}")


#: Exactly what our own Transaction Details screen puts on the page. Kept as text so the
#: extraction is testable without an OCR engine installed and without a 400KB fixture.
SCREEN = """Transaction Details
Alex Umashankar Morgan Umashankar Gaurishankar
- 135.70
Type  Mobile Payment
Paid From  Sagar Uniyal Apoorv Uniyal
Wallet (544244401)
Paid To  Alex Umashankar Morgan Umashankar Gaurishankar
Wallet (501245908)
Remarks  PayI voice payment
Amount  100.00
Fees (incl VAT)  35.70
Transaction ID  1184361013111816
Date  13/10/26, 11:18 AM"""


def test_it_reads_the_right_number_off_the_screen():
    import ocr
    print("\nwhat a transaction screen is worth reading")
    f = ocr.fields(SCREEN)
    check("the LABELLED reference wins", f["labelled"] == "1184361013111816", f["labelled"])
    check("both wallet numbers are found",
          f["wallets"] == ["544244401", "501245908"], f["wallets"])
    check("the amount is the amount, not the total with fees", f["amount"] == 100.0,
          f["amount"])
    # The screen also contains a wallet id, a date and a battery reading. Picking "the
    # longest digit run" finds the wrong one often enough that the label is not optional.
    check("the reference is not confused with a wallet number",
          f["labelled"] not in f["wallets"], f["labelled"])


def test_ocr_misreads_are_tolerated_but_real_differences_are_not():
    import ocr
    print("\ndigit confusion vs a different payment")
    real = "1184361013111816"
    for bad, why in (("1184361013111816", "exact"),
                     ("118436IOI3111816", "1 read as I, 0 as O"),
                     ("1184361013111B16", "8 read as B"),
                     ("II8436IOI3III8I6", "every 1 read as I")):
        check(f"{why}: matches", ocr._loose_equal(bad, real), bad)
    print("  but a genuinely different reference must NOT match")
    for bad, why in ((real[:-1] + "7", "last digit differs"),
                     (real[:-2] + real[-1] + real[-2], "two digits transposed"),
                     (real[:-1], "one digit missing")):
        check(f"{why}: no match", not ocr._loose_equal(bad, real), bad)


def test_a_screenshot_can_only_reach_the_customers_own_payments():
    """The security property. Everything else here is about accuracy; this is about scope."""
    import ocr
    import data
    print("\nscope")
    txns = data.transactions_for(limit=60)
    if not txns:
        return check("SKIPPED: no ledger to match against", True)

    r = ocr.resolve("Transaction ID 9999888877776666", txns)
    check("a reference that is not theirs resolves to nothing", r["match"] is None,
          getattr(r["match"], "ref", None))

    mine = str(txns[0].ref)
    r2 = ocr.resolve(f"Transaction ID {mine}", txns)
    check("one of their own resolves", getattr(r2["match"], "ref", None) == mine,
          getattr(r2["match"], "ref", None))


def test_a_refund_screenshot_resolves_through_the_original():
    """The reported bug. The refund's reference is the merchant's; the payment it refunded
    is the customer's, and that link is what makes it answerable."""
    import history
    print("\na refund screenshot")
    link = history.refunded_link("1006861008155738")
    if link is None:
        return check("SKIPPED: that refund is not in this ledger", True)
    check("it finds the original", bool(link["original_ref"]), link)
    check("and the original is a different transaction",
          link["original_ref"] != link["refund_ref"], link)
    check("the amount comes from the customer's own row", link["amount"] > 0, link["amount"])

    print("  an ORDINARY payment must not be reported as refunded")
    # Every payment carries an originalTxnId pointing at itself, so the field's presence
    # proves nothing. Without the REFUNDED check, every screenshot would be announced as
    # already refunded — a far worse bug than the one being fixed.
    import data
    txns = [t for t in data.transactions_for(limit=20)]
    plain = next((t for t in txns if not history.refunded_link(str(t.ref))), None)
    check("at least one ordinary payment is not seen as a refund", plain is not None,
          "every payment looked refunded")


def test_it_says_so_when_it_cannot_read():
    import ocr
    print("\nno reader, or nothing legible")
    check("backend is reported honestly", ocr.backend() in (None, "tesseract", "rapidocr"),
          ocr.backend())
    r = ocr.resolve("", [])
    check("empty text resolves to nothing", r["match"] is None and not r["candidates"], r)
    check("and it says why", bool(r["why"]), r["why"])


def test_the_upload_is_remembered():
    """A picture is a thing the customer did, and they refer back to it turns later."""
    from engine import conversation
    from engine.engine import _means_the_upload, _note_upload
    print("\nreferring back to the picture")
    S = "ocr-recall"
    _note_upload(S, "1234567890123456", "a payment")
    check("it lands in the durable ledger",
          any(t.kind == "upload" for t in conversation.tasks(S)),
          [t.kind for t in conversation.tasks(S)])
    check("and is recallable as a subject",
          conversation.subject(S, "uploaded_ref") == "1234567890123456",
          conversation.subject(S, "uploaded_ref"))

    for q in ("im talking about the one i uploaded", "what about the screenshot",
              "the picture i sent", "jo maine bheja tha", "الصورة التي أرسلتها"):
        check(f"{q[:34]!r} points at the upload", _means_the_upload(q), q)
    for q in ("my last transaction", "send 100 to alex", "what are my fees"):
        check(f"{q[:28]!r} does not", not _means_the_upload(q), q)


if __name__ == "__main__":
    test_it_reads_the_right_number_off_the_screen()
    test_ocr_misreads_are_tolerated_but_real_differences_are_not()
    test_a_screenshot_can_only_reach_the_customers_own_payments()
    test_a_refund_screenshot_resolves_through_the_original()
    test_it_says_so_when_it_cannot_read()
    test_the_upload_is_remembered()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
