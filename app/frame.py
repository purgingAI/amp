"""AMP compact wire codec — the 'fast & low-token' framing.

A full JSON envelope is verbose. For high-frequency chat and for embedding messages
into LLM context cheaply, AMP defines a terse single-line frame:

    A1 <T><intent> <from> <to> <thread> <sig8> :: <body>

- T is a 1-char type (Q R E K X). intent is a short code (see INTENT_CODES) or raw.
- ts is omitted (the hub stamps it). sig8 = first 8 hex of the HMAC (cheap integrity hint;
  full sig still travels on the JSON path for strong auth).
- <body> is raw text if the payload is just text, else compact JSON.

Typically ~55-70% smaller than JSON => fewer bytes on the wire AND fewer tokens when an
agent reads a transcript of frames in-context.
"""
from __future__ import annotations
import json

TYPE_TO = {"REQUEST": "Q", "RESPONSE": "R", "EVENT": "E", "ACK": "K", "ERROR": "X"}
TO_TYPE = {v: k for k, v in TYPE_TO.items()}
INTENT_CODES = {"chat": "c", "query.knowledge": "qk", "plan.feature": "pf",
                "implement.feature": "if", "collision.check": "cc", "claim.files": "cf",
                "release.claim": "rc", "presence": "ps", "typing": "ty", "suggest": "sg"}
CODE_INTENT = {v: k for k, v in INTENT_CODES.items()}


def encode(env: dict) -> str:
    t = TYPE_TO.get(env.get("type", "EVENT"), "E")
    ic = INTENT_CODES.get(env.get("intent", "chat"), env.get("intent", "chat"))
    body = env.get("body", {})
    if isinstance(body, dict) and set(body.keys()) <= {"text"}:
        payload = body.get("text", "")
    else:
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    sig8 = (env.get("sig", "") or "")[:8]
    head = f"A1 {t}{ic} {env.get('from','?')} {env.get('to','?')} {env.get('thread','-')} {sig8}"
    return f"{head} :: {payload}"


def decode(line: str) -> dict:
    head, _, body = line.partition(" :: ")
    f = head.split(" ")
    ti = f[1] if len(f) > 1 else "Ec"
    typ = TO_TYPE.get(ti[0], "EVENT")
    intent = CODE_INTENT.get(ti[1:], ti[1:])
    env = {"amp": "1", "type": typ, "intent": intent,
           "from": f[2] if len(f) > 2 else "?", "to": f[3] if len(f) > 3 else "?",
           "thread": f[4] if len(f) > 4 else None, "sig8": f[5] if len(f) > 5 else ""}
    body = body.strip()
    if body.startswith("{") or body.startswith("["):
        try:
            env["body"] = json.loads(body)
        except Exception:
            env["body"] = {"text": body}
    else:
        env["body"] = {"text": body}
    return env


def savings(env: dict) -> dict:
    j = json.dumps(env, separators=(",", ":"), ensure_ascii=False)
    f = encode(env)
    return {"json_bytes": len(j), "frame_bytes": len(f),
            "reduction_pct": round((1 - len(f) / max(1, len(j))) * 100),
            "approx_tokens_json": round(len(j) / 4), "approx_tokens_frame": round(len(f) / 4)}
