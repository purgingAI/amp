"""Spelling pre-filter: fix the customer's words BEFORE the intent model reads them.

Voice and typing both garble the same few words. Whisper hears "Sen money", a thumb types
"senmoney", a repeated clip gives "senmoneysenmony". The intent model should not have to
learn every misspelling. This filter turns them back into the words it already knows:

    "sen money to alex"   -> "send money to alex"
    "senmoney"             -> "send money"
    "senmoneysenmony"      -> "send money"            (split, corrected, repeat dropped)
    "pls pey my dewa bil"  -> "pls pay my dewa bill"
    "send100aed"           -> "send 100 aed"

It only corrects onto a SHORT list of payment words (VOCAB), and only when:
  * the word is not a real word already (the English list + every word seen in our own
    training corpus, which covers Hinglish and the payee names) — "son", "end", "sent",
    "kar", "Alex" are never touched;
  * it is one edit away (two for long words) and starts with the same letter.
Digits are never changed, names are never invented, and anything it is unsure of is left
exactly as written. Arabic and Devanagari pass straight through.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWN_FILE = os.path.join(HERE, "prefilter_known.txt")

#: The words a garbled token may be corrected TO. Payment words only.
VOCAB = {
    "send", "money", "pay", "payment", "payments", "transfer", "recharge", "balance", "dirhams",
    "dirham", "refund", "bill", "bills", "split", "wallet", "withdraw", "remind", "reminder",
    "cancel", "retry", "transaction", "transactions", "beneficiary", "history", "status",
    "fees", "charges", "limit", "account", "bank", "merchant", "received", "failed", "pending",
    "deducted", "amount", "mobile", "spend", "spent", "check", "show", "dewa", "etisalat",
    "please", "transfer", "statement", "cashback", "offers",
}
#: Small words that may appear inside a glued token ("paymybill"), exact only.
GLUE = VOCAB | {"to", "my", "me", "the", "for", "of", "a", "i", "is", "on", "in", "and",
                "top", "up", "add", "last", "month", "now", "all", "what", "how", "much", "aed"}


#: Everyday Hinglish words the fixer once "corrected" ("dawai" -> "dewa i"). Never touched.
_HINGLISH_KEEP = frozenset({"dawai", "dawa", "dawaai", "sugar", "badh", "kaunsi", "lun", "doctor", "dard",
                            "bimaar", "tabiyat", "khana", "pani", "ghar", "kaam", "naukri"})


@lru_cache(maxsize=1)
def known() -> frozenset:
    words = set()
    for path in (KNOWN_FILE, "/usr/share/dict/words"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                # lowercase entries only: the English list's proper nouns ("Sen", "Penn")
                # would otherwise shield garbles from correction
                words |= {w.strip() for w in fh if w.strip().isalpha() and w.strip().islower()}
        except OSError:
            continue
    return frozenset(words | VOCAB | _HINGLISH_KEEP)


def _dist(a: str, b: str, cap: int = 3) -> int:
    """Damerau-Levenshtein (adjacent swaps count as one edit), stopped early past `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def _fix(word: str) -> str | None:
    """The payment word this garbled word was meant to be, or None."""
    w = word.lower()
    if len(w) < 3 or w in VOCAB:
        return None
    allow = 1 if len(w) <= 6 else 2
    best = None
    for v in VOCAB:
        if v[0] != w[0] or abs(len(v) - len(w)) > allow:
            continue
        d = _dist(w, v, allow)
        if d <= allow and (best is None or d < best[0]):
            best = (d, v)
        elif best and d == best[0] and v != best[1]:
            best = (d, None)                   # two equally close words: don't guess
    return best[1] if best and best[1] else None


def _split(word: str) -> list[str] | None:
    """A glued token ("senmoneysenmony") as the words in it, or None if it isn't one."""
    w = word.lower()
    n = len(w)
    # best[i] = (cost, pieces) covering w[:i]
    best: list = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + 14) + 1):
            piece = w[i:j]
            if piece in GLUE:
                cand, c = piece, 0
            else:
                cand, c = (_fix(piece), 1) if len(piece) >= 3 else (None, 0)
            if not cand:
                continue
            cost = best[i][0] + c
            pieces = best[i][1] + [cand]
            if best[j] is None or (cost, len(pieces)) < (best[j][0], len(best[j][1])):
                best[j] = (cost, pieces)
    got = best[n]
    if not got or len(got[1]) < 2:
        return None
    cost, pieces = got
    if not any(p in VOCAB and len(p) >= 3 for p in pieces) or cost > len(pieces) // 2 + 1:
        return None
    return pieces


def _collapse(tokens: list[str]) -> list[str]:
    """ "send money send money" -> "send money". Only runs that contain a payment word."""
    out = list(tokens)
    for size in (4, 3, 2, 1):
        i = 0
        while i + 2 * size <= len(out):
            a, b = out[i:i + size], out[i + size:i + 2 * size]
            if [x.lower() for x in a] == [x.lower() for x in b] and any(x.lower() in VOCAB for x in a) \
                    and all(x.isalpha() for x in a):
                del out[i + size:i + 2 * size]
            else:
                i += 1
    return out


_CURRENCY = {"aed", "dhs", "dh", "dirham", "dirhams", "rs", "inr", "usd"}


def _split_glued(m: re.Match) -> str:
    pre, num, post = m.group(1), m.group(2), m.group(3)
    if (pre + num).lower() == "openwallet":
        return m.group(0)                        # our own name, not "pay 10" dirhams
    ok_pre = pre.lower() in GLUE or bool(_fix(pre))
    ok_post = not post or post.lower() in _CURRENCY or post.lower() in GLUE or bool(_fix(post))
    if ok_pre and ok_post:
        return " ".join(x for x in (pre, num, post) if x)
    return m.group(0)


#: Short words the edit-distance fixer cannot reach (it only corrects ONTO payment words),
#: and two-word slips: "in need tro send money" is "i need to send money".
_TYPOS = {"tro": "to", "ot": "to", "tp": "to", "teh": "the", "nd": "and", "plz": "please", "pls": "please",
          "u": "you", "ur": "your", "mney": "money", "monye": "money"}
_PHRASE_TYPOS = ((re.compile(r"^(\W*)in\s+need\s+(?=to\b|tro\b)", re.I), r"\1i need "),)


def correct(text: str) -> str:
    """The sentence with garbled payment words fixed. Unchanged if nothing was garbled."""
    if not text or not re.search(r"[A-Za-z]", text):
        return text
    fixed = re.sub(r"\b[A-Za-z]+\b", lambda m: _TYPOS.get(m.group(0).lower(), m.group(0))
                   if m.group(0).lower() in ("tro", "ot", "tp", "teh", "mney", "monye") else m.group(0), text)
    for rx, rep in _PHRASE_TYPOS:
        fixed = rx.sub(rep, fixed)
    if fixed != text:
        return correct(fixed) if fixed.lower() != text.lower() else fixed
    # letters glued to a number: "send100aed" -> "send 100 aed" — but ONLY when the letters
    # are a payment word. "Heuwoa1234" (a keyboard mash) used to become "Heuwoa 1234", the
    # amount parser — which rightly refuses a number glued to letters — then saw a clean
    # 1234, and an open "how much?" was answered with AED 1,234. Garbage stays glued.
    t = re.sub(r"\b([A-Za-z]+)(\d+(?:[.,]\d+)?)([A-Za-z]*)\b", _split_glued, text)
    t = re.sub(r"\b(\d+(?:[.,]\d+)?)(aed|dhs|dh|dirhams?|rs|inr|usd)\b", r"\1 \2", t, flags=re.I)  # "100aed"
    words = known()
    out: list[str] = []
    for chunk in t.split():
        # Only a plain word (with punctuation around it) is ever looked at. Anything with a
        # symbol or digit inside — "f*cking", "1,000", "25th", "o'clock", an email — passes
        # through exactly as written; the abuse rule and the amount parser read those.
        m = re.fullmatch(r"([\"'(]*)([A-Za-z]+)([.,!?;:)\"']*)", chunk)
        if not m or m.group(2).lower() in words:
            out.append(chunk)
            continue
        pre, core, post = m.groups()
        fixed = _fix(core)
        pieces = [fixed] if fixed else ((_split(core) if len(core) >= 5 else None) or [core])
        pieces[0], pieces[-1] = pre + pieces[0], pieces[-1] + post
        out.extend(pieces)
    s = _echo(" ".join(_collapse(out)))
    return s if s.lower() != " ".join(text.split()).lower() else text


def _echo(s: str) -> str:
    """A clip that repeats itself across a full stop: "Send money. Send money" and the cut-off
    "Send money. Send" are one request. Only when the repeated part has a payment word."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?,])\s+", s) if p.strip()]
    if len(parts) < 2:
        return s
    keep = [parts[0]]
    for p in parts[1:]:
        prev = re.findall(r"[a-z0-9]+", keep[-1].lower())
        cur = re.findall(r"[a-z0-9]+", p.lower())
        if cur and prev[:len(cur)] == cur and any(w in VOCAB for w in prev):
            continue                                     # the same words again, or the start of them
        keep.append(p)
    return " ".join(keep).rstrip(",") if len(keep) < len(parts) else s


def build_known(corpus_files: list[str], min_count: int = 8) -> int:
    """Write prefilter_known.txt: the English list plus every word our own corpus uses often
    (Hinglish, payee names, billers). Run once when the corpus changes; the file ships with
    the engine so production does not depend on /usr/share/dict."""
    import collections
    import json
    c = collections.Counter()
    for f in corpus_files:
        for line in open(f, encoding="utf-8"):
            try:
                c.update(re.findall(r"[a-z]+", json.loads(line)["text"].lower()))
            except (ValueError, KeyError):
                continue
    english = set()
    try:
        english = {w.strip() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")
                   if w.strip().isalpha() and w.strip().islower()}
    except OSError:
        pass
    # The corpus has deliberate typos in it ("sned", "mony") so the model learns to read
    # them. Those must NOT become "known" words, or the filter would leave them alone.
    corpus = {w for w, k in c.items() if k >= min_count and len(w) >= 2 and not _fix(w)}
    words = english | corpus
    with open(KNOWN_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(words)) + "\n")
    known.cache_clear()
    return len(words)
