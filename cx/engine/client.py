"""Talk to the model sidecar, and work without it.

The fallback is not a stub. If the sidecar is down the assistant still classifies, still
refuses out-of-scope questions and still runs the refund matrix — it is just less accurate at
picking the lane, and the UI says which classifier answered. A customer-support surface that
goes dark because a model process died is a worse product than one that degrades.
"""
from __future__ import annotations

from collections import OrderedDict

import json
import os
import urllib.error
import urllib.request

SIDECAR = os.environ.get("CX_MODEL_URL", "http://127.0.0.1:8450")
TIMEOUT = float(os.environ.get("CX_MODEL_TIMEOUT", "2.5"))


def _post(path: str, payload: dict, timeout: float = TIMEOUT) -> dict | None:
    try:
        req = urllib.request.Request(
            SIDECAR + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return None


def health() -> dict:
    try:
        with urllib.request.urlopen(SIDECAR + "/health", timeout=1.5) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return {"ready": False}


#: The same sentences come round constantly — "my bills", "where is my payment", a tapped
#: suggestion chip — and classifying one costs a round trip to the sidecar and a forward
#: pass. The model is deterministic for a given string, so the result is worth keeping.
#:
#: What is cached is the CLASSIFICATION, never the answer. The answer depends on the
#: session, the ledger and the time of day, and caching that would serve one customer's
#: balance to the next person who typed the same four words. This holds nothing but a
#: sentence and a label.
#:
#: Keyed on the normalised form, so "My Bills", "my bills" and "my  bills" are one entry.
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
CACHE_MAX = 4096
_CACHE_STATS = {"hit": 0, "miss": 0}


def _key(text: str) -> str:
    return " ".join((text or "").lower().split())


def classify(text: str) -> dict:
    k = _key(text)
    got = _CACHE.get(k)
    if got is not None:
        _CACHE_STATS["hit"] += 1
        _CACHE.move_to_end(k)
        return dict(got)
    _CACHE_STATS["miss"] += 1
    out = _post("/classify", {"text": text})
    if not (out and out.get("intent")):
        out = _fallback(text)
        # A degraded answer is the sidecar being down, not a fact about the sentence. Cache
        # it and the assistant keeps answering from keyword rules long after the model is
        # back up.
        if out.get("degraded"):
            return out
    _CACHE[k] = dict(out)
    if len(_CACHE) > CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def cache_stats() -> dict:
    total = _CACHE_STATS["hit"] + _CACHE_STATS["miss"]
    return dict(_CACHE_STATS, size=len(_CACHE),
                rate=round(_CACHE_STATS["hit"] / total, 3) if total else 0.0)


def second_opinion(text: str, intent: str) -> dict:
    # Laya is slow on CPU (p50 ~860ms, tail several seconds) and it is advisory, so it gets a
    # longer timeout than the classifier but is never allowed to hold up the answer.
    out = _post("/second-opinion", {"text": text, "intent": intent}, timeout=7.0)
    return out or {"available": False, "agrees": None, "disagrees_on": [],
                   "reason": "sidecar unreachable"}


# ---------------------------------------------------------------------------------------
# The fallback classifier: keyword scoring over the same closed label set.
# ---------------------------------------------------------------------------------------

CUES: dict[str, tuple[tuple[str, float], ...]] = {
    "debited_not_credited": (("debited", 3), ("deducted", 3), ("not credited", 4),
                             ("not received", 3), ("cut from", 3), ("did not get", 3),
                             ("money gone", 3), ("balance dropped", 3), ("charged", 2),
                             ("not delivered", 2), ("disappeared", 2)),
    "txn_failed":          (("failed", 3), ("failure", 2), ("declined", 3), ("rejected", 3),
                            ("error", 2), ("unsuccessful", 3), ("did not go through", 3),
                            ("not working", 2), ("bounced", 2)),
    "txn_on_hold":         (("on hold", 4), ("pending", 3), ("under review", 4),
                            ("stuck", 3), ("processing", 2), ("frozen", 3),
                            ("compliance", 3), ("still spinning", 3)),
    "txn_status":          (("status", 3), ("where is", 2), ("did it", 2), ("has it", 2),
                            ("go through", 2), ("track", 3), ("update on", 2),
                            ("delivered", 2), ("received the money", 2), ("did it land", 3)),
    "refund_request":      (("refund", 3), ("money back", 4), ("return my", 3),
                            ("reverse", 3), ("give back", 3), ("claim", 2), ("cancel", 2)),
    "refund_status":       (("refund status", 5), ("my refund", 3), ("refund not", 4),
                            ("where is my refund", 5), ("refund has not", 4),
                            ("chasing my refund", 4), ("reversal", 2)),
    "fees_query":          (("fee", 3), ("fees", 3), ("charge", 3), ("charges", 3),
                            ("cost", 3), ("commission", 3), ("markup", 3), ("free", 2)),
    "time_query":          (("how long", 3), ("how much time", 3), ("when will", 3),
                            ("instant", 2), ("same day", 3), ("how quick", 3),
                            ("how fast", 3), ("duration", 3), ("arrive", 2)),
    "limits_query":        (("limit", 4), ("maximum", 3), ("cap", 3), ("allowed to send", 3),
                            ("in one go", 2), ("per transaction", 3), ("ceiling", 3)),
    "send_money":          (("send", 3), ("transfer to", 3), ("pay ", 2), ("remit", 3),
                            ("wire", 3), ("move ", 2), ("shoot", 2), ("same again", 3)),
    "pay_merchant":        (("pay the", 3), ("merchant", 3), ("shop", 3), ("cafe", 3),
                            ("till", 3), ("checkout", 4), ("bill", 3), ("store", 2),
                            ("pharmacy", 3), ("grocery", 3), ("restaurant", 3)),
    "beneficiary_query":   (("beneficiar", 4), ("my list", 3), ("saved", 3),
                            ("recipients", 4), ("who can i send", 5), ("who's on my", 4),
                            ("people i can pay", 4)),
    "offers_nearby":       (("near me", 5), ("nearby", 5), ("offer", 3), ("discount", 3),
                            ("cashback", 4), ("deals", 3), ("around me", 4),
                            ("close to me", 4), ("around here", 4)),
    "journey_help":        (("how do i", 4), ("how can i", 4), ("how to", 3), ("steps", 3),
                            ("walk me through", 5), ("guide me", 4), ("where do i", 3),
                            ("explain", 2), ("process to", 3), ("first time", 2)),
}


def _fallback(text: str) -> dict:
    low = " ".join((text or "").lower().split())
    if not low:
        return {"intent": "out_of_scope", "confidence": 1.0, "top": [],
                "model": "fallback-keyword", "ms": 0, "degraded": True}
    scores: dict[str, float] = {}
    for intent, cues in CUES.items():
        total = sum(w for cue, w in cues if cue in low)
        if total:
            scores[intent] = total
    if not scores:
        return {"intent": "out_of_scope", "confidence": 0.5, "top": [],
                "model": "fallback-keyword", "ms": 0, "degraded": True}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    pool = sum(s for _, s in ranked) or 1
    return {"intent": ranked[0][0], "confidence": round(top / pool, 3),
            "top": [{"intent": i, "p": round(s / pool, 3)} for i, s in ranked[:3]],
            "model": "fallback-keyword", "ms": 0, "degraded": True}
