"""ACE-side authentication — deliberately SEPARATE from AMP's login/user store.

ACE gets its own users table (data/ace_users.db) so ACE and AMP no longer share accounts:
signing into ACE never touches AMP's ClickHouse users and vice-versa. Simple email login with
a token; the first user becomes admin, others are auto-approved (flip ACE_ADMIN_GATE=1 to require
admin approval). This is intentionally small and dependency-free — the heavy company knowledge lives
in the (read-only) knowledge graph, not here.
"""
from __future__ import annotations
import os, sqlite3, secrets, time
from pathlib import Path

DB = Path(os.environ.get("ACE_USERS_DB", str(Path(__file__).resolve().parent.parent / "data" / "ace_users.db")))
ADMIN_GATE = os.environ.get("ACE_ADMIN_GATE", "0") == "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ace_users (
  agent_id TEXT PRIMARY KEY, email TEXT UNIQUE, name TEXT, token TEXT,
  role TEXT DEFAULT 'member', approved INTEGER DEFAULT 0, created REAL, last_seen REAL
);
"""

def _c():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10); c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=6000"); c.executescript(_SCHEMA); return c

def _aid(name: str, c) -> str:
    base = "".join(ch for ch in (name or "user").lower().replace(" ", ".") if ch.isalnum() or ch == ".")[:24] or "user"
    aid = base; i = 1
    while c.execute("SELECT 1 FROM ace_users WHERE agent_id=?", (aid,)).fetchone():
        i += 1; aid = f"{base}{i}"
    return aid

def by_token(agent_id: str, token: str):
    if not agent_id or not token:
        return None
    with _c() as c:
        r = c.execute("SELECT * FROM ace_users WHERE agent_id=? AND token=?", (agent_id, token)).fetchone()
        if r and r["approved"]:
            c.execute("UPDATE ace_users SET last_seen=? WHERE agent_id=?", (time.time(), agent_id)); c.commit()
            return dict(r)
    return None

def login(email: str, name: str = ""):
    """Return a user dict (with token) on success, or {status:'pending'} if admin-gated & unapproved."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    name = (name or email.split("@")[0]).strip()
    with _c() as c:
        r = c.execute("SELECT * FROM ace_users WHERE email=?", (email,)).fetchone()
        if r:
            if not r["approved"]:
                return {"status": "pending", "agent_id": r["agent_id"],
                        "message": "Your ACE access is awaiting admin approval."}
            c.execute("UPDATE ace_users SET last_seen=? WHERE agent_id=?", (time.time(), r["agent_id"])); c.commit()
            return {**dict(r), "returning": True}
        first = c.execute("SELECT COUNT(*) n FROM ace_users").fetchone()["n"] == 0
        aid = _aid(name, c); token = "ace_" + secrets.token_hex(16)
        approved = 1 if (first or not ADMIN_GATE) else 0
        c.execute("INSERT INTO ace_users(agent_id,email,name,token,role,approved,created,last_seen) "
                  "VALUES(?,?,?,?,?,?,?,?)",
                  (aid, email, name, token, "admin" if first else "member", approved, time.time(), time.time()))
        c.commit()
        if not approved:
            return {"status": "pending", "agent_id": aid, "message": "Your ACE access is awaiting admin approval."}
        return {"agent_id": aid, "email": email, "name": name, "token": token,
                "role": "admin" if first else "member", "approved": 1, "returning": False}

def list_users():
    with _c() as c:
        return [dict(r) for r in c.execute(
            "SELECT agent_id,email,name,role,approved,last_seen FROM ace_users ORDER BY created").fetchall()]

def approve(agent_id: str):
    with _c() as c:
        c.execute("UPDATE ace_users SET approved=1 WHERE agent_id=?", (agent_id,)); c.commit()
        return c.total_changes > 0
