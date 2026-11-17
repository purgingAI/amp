"""Is a knowledge agent's answer actually supported by its documents?

Every answer gets the cheap check; an LLM judge reads a sample (and every answer the cheap check flags):

  cheap   each sentence vs the passages it cites ([n]) — or all retrieved passages when it cites none:
            · meaning: best cosine against the passages' own sentences (same multilingual encoder)
            · words: share of its content words found in those passages
            · numbers: every number in the sentence (days, fees, limits…) must appear in the source —
              a wrong number is the most harmful thing a policy answer can say
          → supported / partial / unsupported, and the answer's score = supported share
  judge   Claude reads passages + answer and marks each sentence supported / partial / not supported

Results go into the ledger (faith, faith_score) and show in the console, the X-ray and the inbox.
"""
from __future__ import annotations

import json
import random
import re

import encoder
import llm

_SENT = re.compile(r"(?<=[.!?।؟])\s+|\n+")
_NUM = re.compile(r"(?<![\w\[])\d+(?:[.,]\d+)?(?!\])")
_WORD = re.compile(r"[^\W\d_]{3,}", re.U)
META = re.compile(r"(?i)don'?t have that information|contact (openwallet )?support|not in (the|my) (documents|notes)|I can'?t find")
STOP = set("the and for with that this from are was were have has not you your can will what when where which how about into than then they them their there also only".split())


def _split(text: str) -> list[str]:
    out = []
    for s in _SENT.split(text or ""):
        s = re.sub(r"^[\s\-*•\d.)]+", "", s).strip()
        if len(s) >= 18 and not s.rstrip("*_ ").endswith(":"):          # a lead-in ("…depends on:") claims nothing
            out.append(s)
    return out


def _cites(s: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", s)}


def _norm_num(n: str) -> str:
    return n.replace(",", "").rstrip("0").rstrip(".") if "." in n else n.replace(",", "")


def _foreign(s: str) -> bool:
    import router
    return router.language(s) != "en"


def check(answer: str, passages: list[dict]) -> dict:
    """passages: [{"n": 1, "text": "…"}, …] as the agent saw them."""
    sents = _split(answer)
    if not sents or not passages:
        return {"sentences": [], "score": None, "unsupported": 0, "checked": 0}
    by_n = {p["n"]: p["text"] for p in passages}
    psents = {n: [x for x in _split(t)] or [t] for n, t in by_n.items()}
    enc = encoder.get(None)
    flat = [(n, x) for n, xs in psents.items() for x in xs]
    PV = enc.encode([x for _, x in flat], max_len=128)
    SV = enc.encode([re.sub(r"\[\d+\]", "", s) for s in sents], max_len=128)
    out = []
    for i, s in enumerate(sents):
        if META.search(s):
            out.append({"text": s, "verdict": "meta", "why": "says what it does not know"})
            continue
        cited = _cites(s) & set(by_n) or set(by_n)
        idx = [k for k, (n, _) in enumerate(flat) if n in cited]
        sims = (PV[idx] @ SV[i]) if idx else []
        best = float(max(sims)) if len(sims) else 0.0
        src = " ".join(by_n[n] for n in cited).lower()
        words = [w for w in _WORD.findall(s.lower()) if w not in STOP]
        overlap = sum(w in src for w in words) / max(1, len(words))
        src_nums = {_norm_num(n) for n in _NUM.findall(src)}
        missing = [n for n in _NUM.findall(re.sub(r"\[\d+\]", "", s)) if _norm_num(n) not in src_nums]
        if missing:
            verdict, why = "unsupported", f"number(s) not in the source: {', '.join(missing[:4])}"
        elif _foreign(s):
            # Hinglish / Hindi / Arabic answer over an English document: words cannot overlap, so judge by
            # meaning only (cross-language similarity runs lower — measured ~0.45–0.6 for faithful sentences)
            verdict = "supported" if best >= 0.5 else "partial" if best >= 0.38 else "unsupported"
            why = f"meaning {best:.2f} (cross-language)"
        elif best >= 0.62 or overlap >= 0.6:
            verdict, why = "supported", f"meaning {best:.2f} · words {overlap:.0%}"
        elif best >= 0.45 or overlap >= 0.4:
            verdict, why = "partial", f"meaning {best:.2f} · words {overlap:.0%}"
        else:
            verdict, why = "unsupported", f"meaning {best:.2f} · words {overlap:.0%}"
        out.append({"text": s, "verdict": verdict, "why": why, "meaning": round(best, 3), "words": round(overlap, 2),
                    "cited": sorted(_cites(s))})
    scored = [x for x in out if x["verdict"] != "meta"]
    sup = sum(1 if x["verdict"] == "supported" else .5 if x["verdict"] == "partial" else 0 for x in scored)
    return {"sentences": out, "score": round(sup / len(scored), 3) if scored else None,
            "unsupported": sum(x["verdict"] == "unsupported" for x in scored), "checked": len(scored)}


JUDGE = ("You check whether a customer-support answer is supported by the source passages. For EACH numbered answer "
         "sentence output one line: <number> ||| supported|partial|unsupported|unknown ||| <short reason>. A sentence is unsupported "
         "if it states anything (especially a number, time, fee or condition) the passages do not say. Say unknown only when "
         "you genuinely cannot tell from the passages. Output only those lines.")
VERDICTS = ("supported", "partial", "unsupported", "unknown")      # unknown → a person decides (judges.py); never cached


def judge(answer: str, passages: list[dict]) -> dict | None:
    if not llm.api_key():
        return None
    sents = _split(answer)
    if not sents:
        return None
    prompt = "PASSAGES\n\n" + "\n\n".join(f"[{p['n']}] {p['text']}" for p in passages) + \
             "\n\nANSWER SENTENCES\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    try:
        raw = llm.complete(JUDGE, prompt, max_tokens=1400)
    except Exception as exc:                                # noqa: BLE001
        return {"error": llm.error_text(exc)}
    verdicts = {}
    for line in raw.splitlines():
        parts = [x.strip() for x in line.split("|||")]
        if len(parts) >= 2 and parts[0].rstrip(".").isdigit() and parts[1] in VERDICTS:
            verdicts[int(parts[0].rstrip(".")) - 1] = {"verdict": parts[1], "why": parts[2] if len(parts) > 2 else ""}
    out = [{"text": s, **verdicts.get(i, {"verdict": "?", "why": ""})} for i, s in enumerate(sents)]
    known = [x for x in out if x["verdict"] in ("supported", "partial", "unsupported")]
    score = sum(1 if x["verdict"] == "supported" else .5 if x["verdict"] == "partial" else 0 for x in known) / len(known) if known else None
    import judges
    return {"sentences": out, "score": round(score, 3) if score is not None else None, "model": llm.QUESTION_MODEL, "judge": judges.pin("faith")["id"],
            "unknown": sum(x["verdict"] == "unknown" for x in out)}


def assess(did: int, answer: str, passages: list[dict], sample: float = 0.25, cache: dict | None = None) -> dict:
    """Cheap check always; the judge on a sample and on anything the cheap check flags. Stored in the ledger.
    cache: {cid, question, lang} — a candidate for the verified answer cache: the judge always runs, and a
    passing answer is admitted (answer_cache.py)."""
    import ledger
    res = {"cheap": check(answer, passages)}
    flagged = (res["cheap"]["unsupported"] or 0) > 0 or (res["cheap"]["score"] is not None and res["cheap"]["score"] < 0.75)
    if flagged or cache or random.random() < sample:
        res["judge"] = judge(answer, passages)
    j = res.get("judge") or {}
    score = j.get("score") if j.get("score") is not None else res["cheap"]["score"]
    ledger.set_faith(did, res, score)
    if cache:
        try:
            import answer_cache
            res["cached"] = answer_cache.admit(cache["cid"], cache["question"], cache["lang"], answer, passages, res, did)
        except Exception:                                   # noqa: BLE001 — caching is a bonus, never a failure
            pass
    return res
