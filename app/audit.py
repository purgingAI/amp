"""AMP/ACE service audit log — application-level events + Claude token/cost accounting.

Writes each event to BOTH a rotating JSONL file (data/audit.log) and a queryable `audit` table.
Secrets/tokens are always redacted. This is service-specific (not uvicorn's HTTP access log).
"""
from __future__ import annotations
import json, datetime, threading
from pathlib import Path
from . import store

LOG = Path(__file__).resolve().parent.parent / "data" / "audit.log"
LOG.parent.mkdir(exist_ok=True)
MAX_BYTES = 5_000_000
_lock = threading.Lock()
_REDACT = {"token", "secret", "password", "authorization", "api_key", "apikey",
           "sig", "access_token", "client_secret", "bearer"}

# $ per 1M tokens (input, output) — approx list prices; tune as needed.
_PRICES = {"opus": (15.0, 75.0), "sonnet": (3.0, 15.0), "haiku": (0.80, 4.0)}


def _clean(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if k.lower() in _REDACT:
            out[k] = "***"
        elif isinstance(v, str) and len(v) > 600:
            out[k] = v[:600] + "…"
        else:
            out[k] = v
    return out


def log(event: str, actor: str = "", **fields):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fields = _clean(fields)
    rec = {"ts": ts, "event": event, "actor": actor, **fields}
    line = json.dumps(rec, default=str)
    with _lock:
        try:
            if LOG.exists() and LOG.stat().st_size > MAX_BYTES:
                LOG.replace(LOG.with_name("audit.log.1"))
            with open(LOG, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
    try:
        store.add_audit(ts, event, actor, json.dumps(fields, default=str))
    except Exception:
        pass
    return rec


def llm(model: str, usage, purpose: str = "", actor: str = ""):
    """Record a Claude call's token usage + estimated cost."""
    it = int(getattr(usage, "input_tokens", 0) or 0)
    ot = int(getattr(usage, "output_tokens", 0) or 0)
    tier = "opus" if "opus" in model else "haiku" if "haiku" in model else "sonnet"
    ip, op = _PRICES[tier]
    cost = round(it / 1e6 * ip + ot / 1e6 * op, 6)
    log("llm.usage", actor=actor, model=model, purpose=purpose,
        in_tokens=it, out_tokens=ot, cost_usd=cost)
    return cost
