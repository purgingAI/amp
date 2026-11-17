"""Context-aware routing: a message is judged inside its conversation, not on its own.

The engine has always been context-aware (engine/conversation.py): an open question ("How much?")
makes "100 dirhams" an answer, a sticky topic makes "show me more" mean more of the same list, and a
follow-up with nothing to follow is refused. The Orchestrator has to respect all of that, and do the
same for its own agents:

  1  guardrails run first, always — a follow-up is never a way around them (router.route)
  2  the engine is WAITING (open question or a payment proposal awaiting yes/no) and this reads like
     the answer → the engine keeps it
  3  this follows up the Orchestrator's own last answer (a knowledge agent, within THREAD_TTL)
     → stay with that agent, and search its documents with a standalone rewrite of the question
     ("and for Pakistan?" → "What is the fee refund policy for failed transfers to Pakistan?")
  4  this follows up the engine's thread → the engine keeps it
  5  the sentence names a new subject of its own (confident reflex, real content, no follow-up cue)
     → normal routing: people change topic, and a sticky router is as wrong as a forgetful one

Everything decided here is written to the trace ("context"), so the X-ray can show why.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque

THREAD_TTL = 600.0          # a follow-up 10 minutes later is a new question
ENGINE_FRAME_TTL = 1800.0   # the engine's own open-question lifetime (conversation.TTL)
SWITCH_CONF = 0.85          # how sure the reflex must be about ANOTHER agent to leave a thread
MAX_TURNS = 6

# ── conversation threads (per session, in memory) ────────────────────────────────────
_threads: dict[str, deque] = {}
_lock = threading.Lock()


def note(session: str, text: str, agent: str | None, kind: str | None, standalone: str | None = None,
         answer: str = "", lang: str | None = None) -> None:
    """Remember one turn. agent = the Studio agent that ANSWERED it, or "engine" when the live engine did."""
    if not session:
        return
    with _lock:
        d = _threads.setdefault(session, deque(maxlen=MAX_TURNS))
        d.append({"text": text[:400], "standalone": (standalone or "")[:400], "agent": agent, "kind": kind,
                  "answer": (answer or "")[:600], "lang": lang, "at": time.time()})
        if len(_threads) > 5000:                                   # bound memory: drop the stalest sessions
            for k in sorted(_threads, key=lambda k: _threads[k][-1]["at"] if _threads[k] else 0)[:500]:
                _threads.pop(k, None)


def thread(session: str) -> list[dict]:
    with _lock:
        d = _threads.get(session)
        if not d:
            return []
        live = [t for t in d if time.time() - t["at"] <= THREAD_TTL]
        return [dict(t) for t in live]


def forget(session: str) -> None:
    with _lock:
        _threads.pop(session, None)


# ── is this a follow-up? (multilingual cues, each one named in the trace) ─────────────
_CONNECTIVE = re.compile(
    r"^(?:and|but|or|also|so|then|plus|what about|how about|same for|same with|and what about|and if|what if|"
    r"aur|or phir|phir|toh|to phir|aur agar|lekin|par|"
    r"और|तो|फिर|लेकिन|और अगर|"
    r"و|وماذا عن|ماذا عن|وكيف|وهل|طيب|ولو)(?:\s|$|\b)", re.IGNORECASE)
_ANAPHORA = re.compile(
    r"\b(?:it|its|that|this|those|these|them|they|there|same|such|the one|that one|"
    r"uska|uski|uske|iska|iski|iske|woh|wo|ye|yeh|wahan|waha|usme|isme|usmein|ismein|usse|isse)\b"
    # Devanagari and Arabic: no \b — Python treats Hindi vowel signs (matras) as non-word characters, so
    # \b never fires after "उसमें"; match on whitespace/edges instead
    r"|(?:^|\s)(?:उसका|उसकी|उसके|इसका|इसकी|इसके|वो|वह|ये|यह|वहां|उसमें|इसमें|उससे|इससे)(?=\s|$|[?.!,।])"
    r"|(?:ذلك|هذا|هذه|تلك|نفس|عنه|عنها|فيه|فيها)", re.IGNORECASE)
_ARABIC_WA = re.compile(r"^و[؀-ۿ]")       # و joined to the next word: "وباكستان؟"
_MINE = re.compile(r"\b(?:my|mine|mera|meri|mere|mujhe|hamara|hamari|apna)\b|(?:मेरा|मेरी|मेरे|मुझे|हमारा)|(?:\bلي\b|عندي|حسابي|فلوسي|مبلغي)", re.IGNORECASE)
#: words that ask without naming anything — a question built only from these needs a subject from before
_GENERIC = set("""how much many long when where what which who why time fee fees charge charges cost costs rate limit limits
take takes taking does it there any free days hours minutes kitna kitni kitne kab kahan kaise kyun kaun lagta lagti lagte
lagega lagenge hota hoti hote time din ghante paisa paise charge kya कितना कितनी कितने कब कहां कैसे समय दिन शुल्क लगता लगेगा
كم متى أين كيف رسوم الرسوم وقت يستغرق تستغرق يأخذ تأخذ ياخذ تاخذ المدة يوم أيام""".split())
#: a question about HOW things work (policy) rather than about one's own money (status)
_POLICY_Q = re.compile(r"\b(?:how|policy|policies|rule|rules|allowed|possible|can i|can you|process|days|hours|time|"
                       r"kitna|kitne|kitni|kab|kaise|niyam|allowed|din)\b|(?:कितने|कितना|कब|कैसे|नियम|समय|दिन)|(?:كم|كيف|سياسة|متى|مدة)",
                       re.IGNORECASE)
_ACK = re.compile(r"^(?:ok|okay|okk|k|thanks|thank you|thx|got it|alright|right|cool|great|fine|hmm+|achha|acha|accha|"
                  r"theek hai|thik hai|ok ji|shukriya|ठीक है|अच्छा|धन्यवाद|حسنا|طيب|شكرا|تمام)\b[\s,.!-]*", re.IGNORECASE)
_WORD = re.compile(r"[\w\u0621-\u064A\u0660-\u0669\u0671-\u06D3\u0900-\u097F]+")
_FILLER = set("""a an the is are was were be to of for in on at by with my me i you your we our please pls plz can could
would will do does did hai hain ka ki ke ko se me mein kya kaise mera meri mere mujhe""".split())


def followup(text: str) -> dict:
    """{score 0..1, cues [...], words, content} — a score ≥ 0.5 reads as a follow-up."""
    low = " ".join((text or "").lower().split())
    acked = _ACK.match(low)
    if acked and acked.end() < len(low):
        low = low[acked.end():].lstrip(" ,.!-")              # "ok and how many days?" is "and how many days?"
    words = _WORD.findall(low)
    content = [w for w in words if w not in _FILLER]
    cues, s = [], 0.0
    if _CONNECTIVE.match(low):
        cues.append("starts like a continuation"); s += 0.55
    elif _ARABIC_WA.match(low):
        cues.append("starts with و (and)"); s += 0.3
    if _MINE.search(low):
        cues.append("about their own account"); mine = True
    else:
        mine = False
    if _ANAPHORA.search(low):
        cues.append("points back (it / that / uska / ذلك)"); s += 0.35
    if len(words) <= 5:
        cues.append(f"only {len(words)} word{'s' if len(words) != 1 else ''}"); s += 0.25 if len(words) <= 4 else 0.2
    if len(content) <= 2:
        cues.append("almost no subject of its own"); s += 0.15
    if content and all(w in _GENERIC for w in content):
        cues.append("asks without naming a subject"); s += 0.3
    if len(content) >= 7:
        s -= 0.3                                                   # a full sentence carries its own subject
    subject = [w for w in content if w not in _GENERIC and not _ANAPHORA.fullmatch(w) and not _MINE.fullmatch(w)
               and not _CONNECTIVE.fullmatch(w)]
    return {"score": round(max(0.0, min(1.0, s)), 2), "cues": cues, "words": len(words), "content": len(content),
            "subject": len(subject), "mine": mine, "policy_q": bool(_POLICY_Q.search(low))}


def _answer_like(text: str, frame: dict) -> bool:
    """Does this read like the answer to the engine's open question? Short, or carrying a number/amount."""
    low = (text or "").lower()
    if re.search(r"\d", low) or re.search(r"[٠-٩०-९]", low):
        return True
    if len(_WORD.findall(low)) <= 6:
        return True
    return False


_YESNO = re.compile(r"^(?:yes|yeah|yep|ok|okay|sure|confirm|go ahead|do it|no|nope|cancel|stop|haan|han|ha|nahi|nahin|"
                    r"theek hai|thik hai|kar do|हां|हाँ|नहीं|ठीक है|نعم|لا|تمام|موافق|الغاء|إلغاء)\b", re.IGNORECASE)


# ── the decision ─────────────────────────────────────────────────────────────────────
def consider(text: str, rx: dict, ctx: dict | None, g: dict, cands: list[dict]) -> dict:
    """Returns the context step for the trace. Keys: applied (bool), agent, rule, why, standalone_needed, followup."""
    ctx = ctx or {}
    ids = {c["id"]: c for c in cands}
    engine_agent = next((c["id"] for c in cands if c.get("kind") == "action"), None)
    turns = ctx.get("thread") or []
    eng = ctx.get("engine") or {}
    fu = followup(text)
    out = {"applied": False, "followup": fu, "turns": len(turns), "engine": {k: eng.get(k) for k in ("topic", "frame", "proposal") if eng.get(k)}}

    # 2 · the engine is waiting for an answer
    frame = eng.get("frame") or None
    if frame and (frame.get("age") or 0) <= ENGINE_FRAME_TTL and _answer_like(text, frame) and engine_agent:
        what = (frame.get("asked_for") or "a detail").replace("_", " ")
        return dict(out, applied=True, agent=engine_agent, rule="engine-waiting",
                    why=f"the engine asked for {what} ({(frame.get('intent') or '').replace('_', ' ')}) and this reads like the answer")
    if eng.get("proposal") and (_YESNO.match(" ".join(text.lower().split())) or fu["words"] <= 4) and engine_agent:
        return dict(out, applied=True, agent=engine_agent, rule="engine-confirming",
                    why="a payment is waiting for yes or no — this is the reply to it")

    last = turns[-1] if turns else None
    is_fu = fu["score"] >= 0.5
    if not is_fu:
        return out

    # 5 · a confident, substantial sentence about another agent leaves the thread
    prev = (last or {}).get("agent")
    if rx.get("available") and rx.get("choice") and rx.get("choice") != prev and rx.get("confidence", 0) >= SWITCH_CONF \
            and fu["subject"] >= 2 and "starts like a continuation" not in fu["cues"]:
        return dict(out, rule="new-subject", why=f"names a new subject of its own (reflex {rx['confidence']:.0%} sure of {rx['choice']})")

    # 3 · following up the Orchestrator's own answer
    if last and prev and prev != "engine" and prev in ids:
        n = next((x for x in g["nodes"] if x["id"] == prev), {})
        if n.get("kind") == "knowledge" and fu["mine"] and not fu["policy_q"] and engine_agent:
            # "and my refund?" after a policy answer is about THEIR money — the engine's job, not the documents'
            return dict(out, applied=True, agent=engine_agent, rule="personal-turn",
                        why=f"asks about their own account after a policy answer — that needs the engine, not {n.get('name', prev)}")
        return dict(out, applied=True, agent=prev, rule="studio-thread", standalone_needed=n.get("kind") == "knowledge",
                    why=f"follows up the last answer from {n.get('name', prev)} (“{last['text'][:60]}”)")

    # 4 · following up the engine's thread
    if engine_agent and ((last and prev == "engine") or (not last and eng.get("topic"))):
        topic = (eng.get("topic") or "").replace("_", " ")
        return dict(out, applied=True, agent=engine_agent, rule="engine-thread",
                    why=f"follows up the engine's conversation{(' about ' + topic) if topic else ''}")
    return dict(out, rule="nothing-to-follow", why="reads like a follow-up, but there is no recent conversation to follow")


# ── a follow-up as a standalone question (for searching the documents) ───────────────
_SA_CACHE: dict[str, str] = {}
REWRITE = ("You rewrite a customer's follow-up message as ONE standalone question in English, using the conversation "
           "so far to fill in what it refers to. Keep product names, countries, amounts and numbers. Do not answer it. "
           "Output only the rewritten question.")


def standalone(text: str, turns: list[dict]) -> tuple[str, str]:
    """(question, how) — how is 'rewritten' (Claude) or 'joined' (no key / failure: previous question + this one)."""
    prev = [t for t in turns if t.get("text")][-2:]
    if not prev:
        return text, "as-is"
    joined = " ".join([(prev[-1].get("standalone") or prev[-1]["text"]), text])
    import llm
    if not llm.api_key():
        return joined, "joined"
    convo = "\n".join(f"Customer: {t.get('standalone') or t['text']}\nAssistant: {(t.get('answer') or '')[:300]}" for t in prev)
    key = convo + "\n>>" + text
    if key not in _SA_CACHE:
        try:
            q = llm.complete(REWRITE, f"CONVERSATION\n{convo}\n\nFOLLOW-UP\n{text}", max_tokens=120, fast=True).strip().strip('"')[:300]
            _SA_CACHE[key] = q if len(q) > 3 else ""
        except Exception:                                   # noqa: BLE001
            _SA_CACHE[key] = ""
        if len(_SA_CACHE) > 2000:
            _SA_CACHE.pop(next(iter(_SA_CACHE)))
    return (_SA_CACHE[key], "rewritten") if _SA_CACHE[key] else (joined, "joined")
