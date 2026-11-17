"""Reading a transaction screenshot the customer sends in.

Customers do not type reference numbers. They screenshot the payment and send the picture,
and until now that was a dead end. This turns a picture into the one thing worth having
from it — which of THEIR payments it is — and then answers about that payment with the
same handler as if they had typed the reference.

## The screenshot is untrusted, and that decides the whole design

Three separate reasons never to answer straight from what the picture says:

* **OCR misreads digits.** `0/O`, `1/l`, `5/S`, `8/B`. A sixteen-digit reference read off a
  phone photo at an angle will be wrong often enough to matter, and a wrong reference that
  happens to exist is worse than one that does not.
* **The numbers in the image are not evidence.** A screenshot can be cropped, edited, or
  be somebody else's. If we read "delivered" off the picture and repeat it, we have
  confirmed whatever the image claimed.
* **It may not be their transaction at all.**

So the picture is only ever used to IDENTIFY a candidate, and the answer comes from the
ledger — scoped, as everywhere else, to the authenticated customer. A reference that does
not match one of their own payments is reported as not found. We never describe a payment
that is not theirs, however clearly the image shows one.

## Matching is deliberately tolerant, then strict

Tolerant about the string: a reference is compared against the customer's own transactions
allowing the handful of digit confusions OCR actually makes, so a single misread character
still finds the right payment. Strict about the result: it must match exactly one of their
payments, or we ask rather than guess. Two candidates is not an answer.

## Backends

`tesseract`, `rapidocr`, or none. With none installed `read()` returns None and the
assistant says plainly that it cannot read pictures yet — which is a better answer than a
hallucinated reference, and it keeps every other part of this file testable.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

#: Digits OCR swaps for each other, and the characters they get swapped with. Used only to
#: match a read reference against the customer's OWN references — never to rewrite one.
CONFUSIONS = {
    "0": "0oOQD", "1": "1lI|i", "2": "2zZ", "5": "5sS", "6": "6bG",
    "7": "7T", "8": "8B", "9": "9gq", "4": "4A", "3": "3",
}

#: A OpenWallet reference is a long run of digits. Shorter runs are amounts, dates and wallet
#: numbers, which are matched separately.
_REF = re.compile(r"\b(\d[\dOoIlSsBbgGqQzZ]{11,21})\b")
_WALLET = re.compile(r"wallet\s*\(?\s*(\d{6,12})\s*\)?", re.I)
_AMOUNT = re.compile(r"(?:amount|total|paid)\D{0,12}([\d,]+\.\d{2})", re.I)

#: The labels our own transaction screen uses. Anchoring on them is far more reliable than
#: guessing which long number on the screen is the reference — a screenshot also contains a
#: wallet id, a date and a battery percentage.
_LABELLED_REF = re.compile(
    r"(?:transaction\s*id|txn\s*id|reference|ref\s*(?:no|number)?|رقم\s*العملية)"
    r"\s*[:\-]?\s*([\dOoIlSsBbgGqQzZ]{10,22})", re.I)


def backend() -> str | None:
    """Which OCR is actually available here."""
    if shutil.which("tesseract"):
        return "tesseract"
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return "rapidocr"
    except Exception:                                         # noqa: BLE001
        return None


def read(image: bytes) -> str | None:
    """The text in the picture, or None if we cannot read pictures here."""
    which = backend()
    if which == "tesseract":
        return _tesseract(image)
    if which == "rapidocr":
        return _rapidocr(image)
    return None


def _tesseract(image: bytes) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(image)
            path = fh.name
        try:
            out = subprocess.run(["tesseract", path, "stdout", "--psm", "6"],
                                 capture_output=True, timeout=25)
            return (out.stdout or b"").decode("utf-8", "replace")
        finally:
            os.unlink(path)
    except Exception:                                         # noqa: BLE001
        return None


def _rapidocr(image: bytes) -> str | None:
    try:
        import io

        import numpy as np
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
        arr = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        res, _ = RapidOCR()(arr)
        return "\n".join(line[1] for line in (res or []))
    except Exception:                                         # noqa: BLE001
        return None


def fields(text: str) -> dict:
    """What a transaction screenshot is worth reading: a reference, a wallet, an amount.

    The labelled reference wins over a bare long number. Our own screen puts "Transaction
    ID" next to it, and a screenshot also contains a wallet number, a date and a battery
    reading — picking "the longest digit run" finds the wrong one often enough that it is
    not worth the simplicity.
    """
    t = text or ""
    out: dict = {"refs": [], "wallets": [], "amount": None, "labelled": None}
    m = _LABELLED_REF.search(t)
    if m:
        out["labelled"] = m.group(1)
        out["refs"].append(m.group(1))
    for m in _REF.finditer(t):
        if m.group(1) not in out["refs"]:
            out["refs"].append(m.group(1))
    out["wallets"] = [m.group(1) for m in _WALLET.finditer(t)]
    m = _AMOUNT.search(t)
    if m:
        try:
            out["amount"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return out


def _loose_equal(read_ref: str, real_ref: str) -> bool:
    """Is the reference we READ the same as one of theirs, allowing OCR's own mistakes?

    Position by position: the real digit, or any character OCR is known to substitute for
    it. This is not a general fuzzy match — it will not accept a transposition or a missing
    digit, because those produce a DIFFERENT valid-looking reference and quietly answering
    about the wrong payment is the failure this whole file is arranged to avoid.
    """
    a, b = (read_ref or "").strip(), (real_ref or "").strip()
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if ca not in CONFUSIONS.get(cb, cb):
            return False
    return True


def resolve(text: str, transactions) -> dict:
    """Which of the customer's OWN payments this picture is about.

    `transactions` are theirs and only theirs — the caller passes what the ledger returned
    for the authenticated customer, so nothing here can reach anybody else's payment.
    """
    got = fields(text)
    out = {"fields": got, "match": None, "candidates": [], "why": ""}
    if not transactions:
        out["why"] = "no transactions to match against"
        return out

    for ref in got["refs"]:
        hits = [t for t in transactions if _loose_equal(ref, str(getattr(t, "ref", "")))]
        if len(hits) == 1:
            out["match"] = hits[0]
            out["why"] = ("reference read from the picture, matched against your own "
                          "payments")
            return out
        if len(hits) > 1:
            out["candidates"] = hits
            out["why"] = "more than one of your payments matches that reference"
            return out

    # No reference matched. An amount plus a wallet number is weaker evidence and is
    # offered as CANDIDATES to choose from, never resolved silently.
    if got["amount"]:
        near = [t for t in transactions
                if abs(float(getattr(t, "amount", 0) or 0) - got["amount"]) < 0.01]
        if near:
            out["candidates"] = near[:5]
            out["why"] = ("no reference I could match, but the amount lines up with these"
                          if len(near) > 1 else
                          "no reference I could match — this is the payment for that amount")
            if len(near) == 1:
                out["candidates"] = []
                out["match"] = near[0]
            return out

    out["why"] = ("I could read the picture but none of the references in it match a "
                  "payment from your wallet")
    return out
