"""ClickHouse client for AMP — accounts (admin-gated), message firehose, agent events.

Uses the HTTP interface (localhost:8123). Degrades gracefully: if ClickHouse is unreachable
(e.g. on a dev laptop), enabled() is False and callers fall back to the SQLite path.
"""
from __future__ import annotations
import os, json, urllib.request, datetime

CH_URL = os.environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123")
CH_USER = os.environ.get("CLICKHOUSE_USER", "default")
CH_PASS = os.environ.get("CLICKHOUSE_PASSWORD", "")
_ok = None


def _headers():
    h = {"X-ClickHouse-User": CH_USER}
    if CH_PASS:
        h["X-ClickHouse-Key"] = CH_PASS
    return h


def _q(sql: str, timeout=6) -> str:
    req = urllib.request.Request(CH_URL + "/", data=sql.encode(), headers=_headers())
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def enabled() -> bool:
    global _ok
    if _ok is None:
        try:
            _ok = _q("SELECT 1", timeout=3).strip() == "1"
        except Exception:
            _ok = False
    return _ok


def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace("'", "\\'")


def _rows(sql: str):
    try:
        out = _q(sql + " FORMAT JSONEachRow")
    except Exception:
        return []
    res = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                res.append(json.loads(line))
            except Exception:
                pass
    return res


# ---------------- users / accounts ----------------
def count_users() -> int:
    try:
        return int(_q("SELECT count() FROM amp.users").strip() or 0)
    except Exception:
        return 0


def get_user_by_email(email: str):
    r = _rows(f"SELECT * FROM amp.users FINAL WHERE email='{_esc(email.lower())}' LIMIT 1")
    return r[0] if r else None


def get_user(agent_id: str):
    r = _rows(f"SELECT * FROM amp.users FINAL WHERE agent_id='{_esc(agent_id)}' LIMIT 1")
    return r[0] if r else None


def upsert_user(u: dict):
    row = {
        "agent_id": u["agent_id"], "email": u.get("email", "").lower(), "name": u.get("name", ""),
        "company": u.get("company", ""), "token": u.get("token", ""), "secret": u.get("secret", ""),
        "role": u.get("role", "member"), "approved": 1 if u.get("approved") else 0,
        "version": int(datetime.datetime.now().timestamp()),
    }
    body = "INSERT INTO amp.users (agent_id,email,name,company,token,secret,role,approved,version) FORMAT JSONEachRow\n" + json.dumps(row)
    _q(body)
    return row


def approve(agent_id: str):
    u = get_user(agent_id)
    if not u:
        return None
    u["approved"] = 1
    return upsert_user(u)


def set_role(agent_id: str, role: str):
    u = get_user(agent_id)
    if u:
        u["role"] = role
        upsert_user(u)


def list_users():
    return _rows("SELECT agent_id,email,name,company,role,approved,created FROM amp.users FINAL ORDER BY created DESC LIMIT 500")


# ---------------- firehose / events ----------------
def insert_message(env: dict):
    row = {"id": env["id"], "thread": env.get("thread", ""), "frm": env.get("from", ""),
           "too": env.get("to", ""), "type": env.get("type", ""), "intent": env.get("intent", ""),
           "text": (env.get("body", {}) or {}).get("text", "")[:8000]}
    try:
        _q("INSERT INTO amp.messages (id,thread,frm,too,type,intent,text) FORMAT JSONEachRow\n" + json.dumps(row))
    except Exception:
        pass


def insert_event(actor: str, peer: str, kind: str, data: dict | None = None):
    row = {"actor": actor, "peer": peer, "kind": kind, "data": json.dumps(data or {})[:4000]}
    try:
        _q("INSERT INTO amp.agent_events (actor,peer,kind,data) FORMAT JSONEachRow\n" + json.dumps(row))
    except Exception:
        pass


def record_connection(a: str, b: str, status: str = "connected"):
    try:
        _q(f"INSERT INTO amp.connections (a,b,status) FORMAT JSONEachRow\n"
           + json.dumps({"a": a, "b": b, "status": status}))
    except Exception:
        pass
