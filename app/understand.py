"""AMP understanding layer — feelings, trust, and recommendations over the message + event firehose.

Deliberately token-frugal:
  * per-message SENTIMENT is lexical (no LLM) -> agent_events(kind='sentiment')
  * TRUST/relationship is a FORMULA over ClickHouse (volume, reciprocity, sentiment,
    recency) -> no LLM
  * deep INSIGHT (summary / topics / strengths / recommendations) uses Haiku and is
    CACHED as agent_events(kind='insight') so repeat calls don't re-spend tokens.

Degrades gracefully when ClickHouse (chdb.enabled()) or the Anthropic key is absent.
"""
from __future__ import annotations
import os, json, math, time
from . import chdb, audit

try:
    from . import orchestrator as _orc          # reuse the shared Anthropic client
    _client = _orc._client
except Exception:                                # pragma: no cover
    _client = None

INSIGHT_MODEL = os.environ.get("AMP_INSIGHT_MODEL", "claude-haiku-4-5-20251001")
INSIGHT_TTL = int(os.environ.get("AMP_INSIGHT_TTL", "600"))     # seconds

# ---------------------------------------------------------------- feelings
_POS = ["thank", "thanks", "great", "good", "nice", "awesome", "perfect", "love",
        "agree", "cool", "appreciate", "well done", "happy", "glad", "excellent",
        "ready", "done", "works", "fixed", "success", "👍", "🙏", "🎉", "😊", "❤"]
_NEG = ["bug", "broken", "fail", "error", "issue", "problem", "angry", "hate", "bad",
        "wrong", "blocked", "stuck", "confused", "annoyed", "sorry", "can't", "cant",
        "won't", "delay", "late", "urgent", "critical", "frustrat", "😡", "😞", "😠", "👎"]


def sentiment(text: str) -> dict:
    """Cheap lexical sentiment. Returns {score:-1..1, label, pos, neg}."""
    t = (text or "").lower()
    pos = sum(t.count(w) for w in _POS)
    neg = sum(t.count(w) for w in _NEG)
    if pos + neg == 0:
        return {"score": 0.0, "label": "neutral", "pos": 0, "neg": 0}
    s = round((pos - neg) / (pos + neg), 3)
    lab = "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
    return {"score": s, "label": lab, "pos": pos, "neg": neg}


def record_sentiment(env: dict):
    """Called on every outbound message (best-effort). Stores a 'sentiment' event."""
    if not chdb.enabled():
        return
    text = (env.get("body", {}) or {}).get("text", "")
    if not text or len(text.strip()) < 2:
        return
    frm, to = env.get("from", ""), env.get("to", "")
    if not frm or to in ("", "broadcast") and not frm:
        pass
    s = sentiment(text)
    chdb.insert_event(frm, to or "broadcast", "sentiment",
                      {"score": s["score"], "label": s["label"], "thread": env.get("thread", "")})


# ---------------------------------------------------------------- trust / relationship
def _one(sql: str, default=0):
    try:
        v = chdb._q(sql).strip()
        return type(default)(v) if v != "" else default
    except Exception:
        return default


def _recency_factor(last_ts_epoch: float) -> float:
    if not last_ts_epoch:
        return 0.0
    days = max(0.0, (time.time() - last_ts_epoch) / 86400.0)
    return round(math.exp(-days / 14.0), 3)          # ~2-week half-ish life


def relationship(a: str, b: str) -> dict:
    """Trust + rapport between two agents, derived from the firehose. 0-100 trust."""
    zero = {"a": a, "b": b, "trust": 0, "messages": 0, "reciprocity": 0.0,
            "sentiment": 0.0, "recency": 0.0, "connected": False}
    if not chdb.enabled() or a == b:
        return zero
    ea, eb = chdb._esc(a), chdb._esc(b)
    a2b = _one(f"SELECT count() FROM amp.messages WHERE frm='{ea}' AND too='{eb}'")
    b2a = _one(f"SELECT count() FROM amp.messages WHERE frm='{eb}' AND too='{ea}'")
    total = a2b + b2a
    last = _one(
        f"SELECT toUnixTimestamp(max(ts)) FROM amp.messages "
        f"WHERE (frm='{ea}' AND too='{eb}') OR (frm='{eb}' AND too='{ea}')", 0)
    sent = 0.0
    try:
        r = chdb._rows(
            f"SELECT avg(JSONExtractFloat(data,'score')) AS s FROM amp.agent_events "
            f"WHERE kind='sentiment' AND ((actor='{ea}' AND peer='{eb}') OR (actor='{eb}' AND peer='{ea}'))")
        sent = round(float(r[0]["s"]), 3) if r and r[0].get("s") is not None else 0.0
    except Exception:
        sent = 0.0
    connected = _one(
        f"SELECT count() FROM amp.connections WHERE (a='{ea}' AND b='{eb}') OR (a='{eb}' AND b='{ea}')") > 0

    recip = (min(a2b, b2a) / max(a2b, b2a)) if max(a2b, b2a) else 0.0
    vol = min(1.0, math.log1p(total) / math.log(30))          # ~30 msgs saturates
    rec = _recency_factor(last)
    sent01 = (sent + 1) / 2
    base = 0.35 * vol + 0.25 * recip + 0.25 * sent01 + 0.15 * rec
    if connected and total == 0:
        base = max(base, 0.10)                                # a handshake => a little trust
    return {"a": a, "b": b, "trust": round(100 * base), "messages": total,
            "reciprocity": round(recip, 2), "sentiment": sent, "recency": rec,
            "connected": connected, "a_to_b": a2b, "b_to_a": b2a}


def top_peers(agent_id: str, limit: int = 6) -> list:
    """Agents this one talks with most, each scored with trust."""
    if not chdb.enabled():
        return []
    e = chdb._esc(agent_id)
    rows = chdb._rows(
        f"SELECT peer, count() AS n FROM ("
        f"  SELECT too AS peer FROM amp.messages WHERE frm='{e}' AND too NOT IN ('','broadcast') "
        f"  UNION ALL SELECT frm AS peer FROM amp.messages WHERE too='{e}' AND frm!='' "
        f") GROUP BY peer ORDER BY n DESC LIMIT {int(limit)}")
    out = []
    for r in rows:
        rel = relationship(agent_id, r["peer"])
        out.append({"agent_id": r["peer"], "messages": int(r["n"]), "trust": rel["trust"],
                    "sentiment": rel["sentiment"]})
    return out


def mood(agent_id: str) -> dict:
    """Recent emotional trend of an agent from its sentiment events."""
    if not chdb.enabled():
        return {"label": "neutral", "score": 0.0, "samples": 0}
    e = chdb._esc(agent_id)
    r = chdb._rows(
        f"SELECT avg(JSONExtractFloat(data,'score')) AS s, count() AS n FROM amp.agent_events "
        f"WHERE kind='sentiment' AND actor='{e}' AND ts > now() - INTERVAL 7 DAY")
    if not r or r[0].get("n", 0) in (0, None):
        return {"label": "neutral", "score": 0.0, "samples": 0}
    s = round(float(r[0]["s"] or 0), 3)
    lab = "positive" if s > 0.12 else "negative" if s < -0.12 else "neutral"
    return {"label": lab, "score": s, "samples": int(r[0]["n"])}


# ---------------------------------------------------------------- deep insight (LLM, cached)
def _recent_context(agent_id: str, n: int = 14) -> list:
    if not chdb.enabled():
        return []
    e = chdb._esc(agent_id)
    return chdb._rows(
        f"SELECT frm, too, intent, substring(text,1,240) AS text FROM amp.messages "
        f"WHERE frm='{e}' OR too='{e}' ORDER BY ts DESC LIMIT {int(n)}")[::-1]


def _cached_insight(agent_id: str):
    e = chdb._esc(agent_id)
    r = chdb._rows(
        f"SELECT data, toUnixTimestamp(ts) AS ts FROM amp.agent_events "
        f"WHERE kind='insight' AND actor='{e}' ORDER BY ts DESC LIMIT 1")
    if r and (time.time() - int(r[0]["ts"])) < INSIGHT_TTL:
        try:
            return json.loads(r[0]["data"])
        except Exception:
            return None
    return None


def insight(agent_id: str, force: bool = False) -> dict:
    """Full understanding card for an agent: mood + top peers (trust) + LLM digest.

    The LLM digest is cached (INSIGHT_TTL) so repeated opens are token-free.
    """
    card = {"agent_id": agent_id, "mood": mood(agent_id), "peers": top_peers(agent_id)}
    digest = None if force else _cached_insight(agent_id)
    if digest is None:
        digest = _llm_digest(agent_id, card)
        if chdb.enabled() and digest.get("_generated"):
            chdb.insert_event(agent_id, "", "insight", digest)
    card["digest"] = digest
    return card


def _llm_digest(agent_id: str, card: dict) -> dict:
    ctx = _recent_context(agent_id)
    fallback = {"summary": "Not enough conversation yet to profile this agent.",
                "topics": [], "strengths": [], "recommendations": [], "_generated": False}
    if not ctx or _client is None:
        # cheap non-LLM recommendation: nudge toward the highest-trust peer
        if card["peers"]:
            p = card["peers"][0]
            fallback["recommendations"] = [
                {"to": p["agent_id"], "why": f"Your strongest working relationship (trust {p['trust']})."}]
        return fallback
    lines = [f"{m['frm']}->{m['too']} [{m.get('intent','chat')}]: {m['text']}" for m in ctx]
    peers = ", ".join(f"{p['agent_id']}(trust {p['trust']})" for p in card["peers"]) or "none yet"
    prompt = (
        f"You analyse an agent messaging network. Agent '{agent_id}' — recent messages:\n"
        + "\n".join(lines)
        + f"\n\nKnown collaborators: {peers}. Current mood: {card['mood']['label']}.\n\n"
        "Return ONLY compact JSON: {\"summary\": one sentence on what this agent works on and their "
        "communication style, \"topics\": [up to 4 short tags], \"strengths\": [up to 3], "
        "\"recommendations\": [up to 3 {\"to\": agent_id or a role, \"why\": short reason to connect/act]}. "
        "Base it strictly on the messages.")
    try:
        msg = _client.messages.create(
            model=INSIGHT_MODEL, max_tokens=320,
            messages=[{"role": "user", "content": prompt}])
        audit.llm(INSIGHT_MODEL, msg.usage, "insight", agent_id)
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1]
        data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        data["_generated"] = True
        for k in ("topics", "strengths", "recommendations"):
            data.setdefault(k, [])
        data.setdefault("summary", "")
        return data
    except Exception as ex:
        fallback["summary"] = "Profile unavailable (analysis error)."
        audit.log("error", agent_id, where="insight", err=str(ex)[:200])
        return fallback
