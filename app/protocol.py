"""AMP/1 protocol primitives: envelope, canonical signing, and prompt-form render/parse."""
from __future__ import annotations
import hmac, hashlib, json, uuid, re, datetime as dt

VERSION = "1"
TYPES = {"REQUEST", "RESPONSE", "EVENT", "ACK", "ERROR"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix="msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _canonical(env: dict) -> str:
    """Deterministic string over the signed fields (everything except token+sig)."""
    keys = ["amp", "id", "thread", "ts", "from", "to", "type", "intent", "body"]
    return json.dumps({k: env.get(k) for k in keys}, sort_keys=True, separators=(",", ":"), default=str)


def sign(env: dict, secret: str) -> str:
    return hmac.new(secret.encode(), _canonical(env).encode(), hashlib.sha256).hexdigest()


def verify(env: dict, secret: str) -> bool:
    return hmac.compare_digest(env.get("sig", ""), sign(env, secret))


def make(frm, to, intent, body, secret, type="REQUEST", thread=None, token=None) -> dict:
    env = {
        "amp": VERSION, "id": new_id(), "thread": thread or new_id("thr"),
        "ts": now_iso(), "from": frm, "to": to, "type": type,
        "intent": intent, "body": body if isinstance(body, dict) else {"text": str(body)},
    }
    env["sig"] = sign(env, secret)
    if token:
        env["token"] = token
    return env


# ---- prompt form ----
def to_prompt(env: dict) -> str:
    b = env.get("body", {})
    hints = "\n".join(f"{k}: {v}" for k, v in b.items() if k != "text")
    head = (f"⟦AMP/1 {env['type']} {env['intent']}⟧\n"
            f"id: {env['id']}   thread: {env['thread']}   "
            f"from: {env['from']} → {env['to']}   ts: {env['ts']}\n"
            f"sig: {env.get('sig','')[:8]}…")
    mid = (("\n" + hints) if hints else "") + "\n---\n" + str(b.get("text", ""))
    return head + mid + "\n⟦/AMP⟧"


_FENCE = re.compile(r"⟦AMP/1\s+(\w+)\s+([\w.]+)⟧(.*?)⟦/AMP⟧", re.S)


def parse_prompt(text: str) -> list[dict]:
    """Extract AMP messages an agent embedded in free text (lossy: header + body text)."""
    out = []
    for typ, intent, inner in _FENCE.findall(text):
        body_text = inner.split("---", 1)[1].strip() if "---" in inner else inner.strip()
        out.append({"type": typ, "intent": intent, "body": {"text": body_text}})
    return out
