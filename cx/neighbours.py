"""Finding the nearest known question, for the ones nobody trained on.

Two problems, one mechanism.

**The typo.** "yes show me the detials" was answered "I don't have anyone called Detials on
your list". One transposed pair turned a common word into a name, and nothing in the
pipeline could tell that `detials` is `details`: the classifier tokenises on words, and a
word it has never seen contributes nothing.

**The question nobody wrote down.** A taxonomy of 26 intents and a bank of 5,445 questions
is still a finite list, and customers are not. When the classifier is unsure and no rule
fires, the honest options are to ask or to guess — unless there is a third: find the
closest thing we DO know and see what it was.

## Why character n-grams

Because they solve both. `details` and `detials` share almost every 3-gram they contain, so
one is near the other without anybody listing the misspelling. And they need no tokeniser,
which matters here more than usual: this assistant handles Devanagari and Arabic, where
word splitting has already caused two separate production bugs (`\\W+` shredding Hindi into
bare consonants, and Arabic diacritics breaking substring matches).

Word embeddings would place "send" near "transfer", which n-grams cannot do. That is a real
limitation and it is the right trade here: a 421M-parameter encoder needs ~850MB and this
runs on a box with about 4GB free, and the gap is covered by the phrase lists the engine
already has. Nothing stops a vector backend being added behind the same interface later.

## What it is NOT allowed to do

**It can never select an intent that moves money.** A nearest neighbour is a similarity, not
an understanding, and "send 100 to alex" being the closest known sentence to something
ambiguous must never turn that something into a payment. Money-moving intents are reachable
only through the classifier and the explicit rules, both of which are followed by the
guards. This is enforced here rather than at the call site, so a future caller cannot
forget.

Built lazily on first use and cached to disk, keyed on a hash of the corpus.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus", "questions.jsonl")
CACHE = os.path.join(HERE, "corpus", ".neighbours.pkl")

#: Intents a similarity match may never reach. See the note above.
FORBIDDEN = frozenset({"send_money", "pay_merchant", "bill_pay", "bill_schedule",
                       "split_payment", "refund_request"})

#: Matching is on 3, 4 and 5 character grams. Three alone is too forgiving across short
#: words; five alone loses the robustness to a typo that is the whole point.
NGRAMS = (3, 4, 5)

#: Below this the nearest thing is not near enough to mean anything.
FLOOR = 0.34
#: And the winner must lead the runner-up by this much of the total vote, or the
#: neighbourhood is mixed and the honest answer is that we do not know.
MARGIN = 0.18

_TASHKEEL = re.compile(r"[ً-ْٰـ]")
_SQUASH = re.compile(r"\s+")

_INDEX: dict | None = None


def flat(text: str | None) -> str:
    """The normal form. Same normalisation the engine's phrase lists use."""
    return _SQUASH.sub(" ", _TASHKEEL.sub("", (text or "").lower())).strip()


def grams(text: str) -> list[str]:
    s = " " + flat(text) + " "
    out = []
    for n in NGRAMS:
        out.extend(s[i:i + n] for i in range(max(0, len(s) - n + 1)))
    return out


def _load_rows() -> list[dict]:
    if not os.path.exists(CORPUS):
        return []
    with open(CORPUS) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _build(rows: list[dict]) -> dict:
    """An inverted index with tf-idf weights, L2 normalised per document.

    Inverted rather than a dense matrix because the query only touches the features it
    actually contains — a few hundred out of a hundred thousand — so scoring is proportional
    to the question, not to the corpus.
    """
    df: dict[str, int] = defaultdict(int)
    doc_grams: list[dict[str, int]] = []
    for r in rows:
        counts: dict[str, int] = defaultdict(int)
        for g in grams(r["text"]):
            counts[g] += 1
        doc_grams.append(counts)
        for g in counts:
            df[g] += 1

    n = max(1, len(rows))
    idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}
    inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, counts in enumerate(doc_grams):
        weights = {g: (1 + math.log(c)) * idf[g] for g, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        for g, w in weights.items():
            inverted[g].append((i, w / norm))
    return {"rows": rows, "idf": idf, "inverted": dict(inverted),
            "sig": _signature(rows)}


def _signature(rows: list[dict]) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(str(len(rows)).encode())
    for r in rows[:: max(1, len(rows) // 200)]:
        h.update(r["text"].encode("utf-8"))
    return h.hexdigest()


def index() -> dict:
    """Built once, then cached on disk so a restart does not pay for it again."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    rows = _load_rows()
    sig = _signature(rows)
    if os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as fh:
                got = pickle.load(fh)
            if got.get("sig") == sig:
                _INDEX = got
                return _INDEX
        except Exception:                                      # noqa: BLE001
            pass
    _INDEX = _build(rows)
    try:
        with open(CACHE, "wb") as fh:
            pickle.dump(_INDEX, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:                                          # noqa: BLE001
        pass
    return _INDEX


def nearest(text: str, k: int = 8) -> list[dict]:
    """The k most similar known questions, best first."""
    ix = index()
    if not ix["rows"]:
        return []
    counts: dict[str, int] = defaultdict(int)
    for g in grams(text):
        counts[g] += 1
    idf, inverted = ix["idf"], ix["inverted"]
    q = {g: (1 + math.log(c)) * idf.get(g, 0.0) for g, c in counts.items()}
    qn = math.sqrt(sum(w * w for w in q.values())) or 1.0

    scores: dict[int, float] = defaultdict(float)
    for g, w in q.items():
        for i, dw in inverted.get(g, ()):
            scores[i] += (w / qn) * dw
    best = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for i, s in best:
        r = ix["rows"][i]
        out.append({"text": r["text"], "intent": r["intent"], "lang": r.get("lang"),
                    "score": round(s, 4)})
    return out


def suggest(text: str, k: int = 8) -> dict | None:
    """The intent of the neighbourhood, or None when it is not clear enough.

    A weighted vote rather than the single best match: one close neighbour can be a
    coincidence of spelling, while five of the eight nearest agreeing is a neighbourhood.
    """
    near = nearest(text, k)
    if not near or near[0]["score"] < FLOOR:
        return None
    vote: dict[str, float] = defaultdict(float)
    for r in near:
        vote[r["intent"]] += r["score"]
    total = sum(vote.values()) or 1.0
    ranked = sorted(vote.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if (top_score - second) / total < MARGIN:
        return None                      # a mixed neighbourhood is not an answer
    if top in FORBIDDEN:
        return None                      # similarity may not reach for money
    return {"intent": top,
            "confidence": round(min(0.95, near[0]["score"]), 3),
            "share": round(top_score / total, 3),
            "neighbours": near[:4]}
