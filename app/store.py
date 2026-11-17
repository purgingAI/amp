"""AMP hub persistence: agents, messages, claims."""
from __future__ import annotations
import sqlite3, json, threading, time
from contextlib import contextmanager
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "amp.db"
DB.parent.mkdir(exist_ok=True)
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY, name TEXT, token TEXT, secret TEXT,
  capabilities TEXT, created TEXT, last_seen TEXT, cursor INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, thread TEXT, ts TEXT, frm TEXT, too TEXT,
  type TEXT, intent TEXT, body TEXT, delivered INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, repo TEXT, paths TEXT,
  intent TEXT, status TEXT, ts TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS workorders (
  id TEXT PRIMARY KEY, request TEXT, build TEXT, tasks TEXT, requester TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event TEXT, actor TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, agent_id TEXT,
  role TEXT, content TEXT, ts TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit(id DESC);
CREATE INDEX IF NOT EXISTS ix_audit_event ON audit(event);
CREATE INDEX IF NOT EXISTS ix_msg_to ON messages(too, delivered);
CREATE INDEX IF NOT EXISTS ix_msg_thread ON messages(thread);
CREATE INDEX IF NOT EXISTS ix_conv_session ON conversations(session, id);
"""


@contextmanager
def conn():
    """Commit on success and ALWAYS close — `with conn() as c` used to leak the fd
    (a raw connection commits but doesn't close), exhausting fds under load."""
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=8000")
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with _lock, conn() as c:
        c.executescript(SCHEMA)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(agents)").fetchall()]
        if "cursor" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN cursor INTEGER DEFAULT 0")
            maxr = c.execute("SELECT COALESCE(MAX(rowid),0) m FROM messages").fetchone()["m"]
            c.execute("UPDATE agents SET cursor=?", (maxr,))   # existing agents start 'now'
        if "auto" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN auto INTEGER DEFAULT 0")
        if "role" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN role TEXT DEFAULT ''")
        if "skills" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN skills TEXT DEFAULT '[]'")
        if "email" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN email TEXT DEFAULT ''")
        if "company" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN company TEXT DEFAULT ''")


def add_agent(agent_id, name, token, secret, caps):
    with _lock, conn() as c:
        maxr = c.execute("SELECT COALESCE(MAX(rowid),0) m FROM messages").fetchone()["m"]
        c.execute("""INSERT OR REPLACE INTO agents(agent_id,name,token,secret,capabilities,created,last_seen,cursor)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (agent_id, name, token, secret, json.dumps(caps), _now(), _now(), maxr))


def get_agent(agent_id):
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        return dict(r) if r else None


def touch(agent_id):
    with _lock, conn() as c:
        c.execute("UPDATE agents SET last_seen=? WHERE agent_id=?", (_now(), agent_id))


def set_auto(agent_id, on: bool):
    with _lock, conn() as c:
        c.execute("UPDATE agents SET auto=? WHERE agent_id=?", (1 if on else 0, agent_id))


def get_agent_by_email(email):
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM agents WHERE email=? AND email!='' LIMIT 1", (email,)).fetchone()
        return dict(r) if r else None


def set_identity(agent_id, email=None, company=None):
    with _lock, conn() as c:
        if email is not None:
            c.execute("UPDATE agents SET email=? WHERE agent_id=?", (email, agent_id))
        if company is not None:
            c.execute("UPDATE agents SET company=? WHERE agent_id=?", (company, agent_id))


def set_profile(agent_id, role=None, skills=None):
    with _lock, conn() as c:
        if role is not None:
            c.execute("UPDATE agents SET role=? WHERE agent_id=?", (role, agent_id))
        if skills is not None:
            c.execute("UPDATE agents SET skills=? WHERE agent_id=?", (json.dumps(skills), agent_id))


def roster():
    """Agents as hireable workers: id, name, role, skills, online."""
    with _lock, conn() as c:
        rows = c.execute("SELECT agent_id,name,role,skills,last_seen FROM agents WHERE agent_id!='orchestrator'").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["skills"] = json.loads(d.get("skills") or "[]")
        except Exception:
            d["skills"] = []
        d["online"] = (time.time() - _epoch(d["last_seen"])) < 90
        out.append(d)
    return out


# ---- work orders ----
def add_workorder(wo_id, request, build, tasks, requester):
    with _lock, conn() as c:
        c.execute("INSERT OR REPLACE INTO workorders(id,request,build,tasks,requester,ts) VALUES(?,?,?,?,?,?)",
                  (wo_id, request, build, json.dumps(tasks), requester, _now()))


def list_workorders(limit=30):
    with _lock, conn() as c:
        rows = c.execute("SELECT * FROM workorders ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r); d["tasks"] = json.loads(d["tasks"] or "[]"); out.append(d)
    return out


def set_task_status(wo_id, task_index, status):
    with _lock, conn() as c:
        r = c.execute("SELECT tasks FROM workorders WHERE id=?", (wo_id,)).fetchone()
        if not r:
            return
        tasks = json.loads(r["tasks"] or "[]")
        if 0 <= task_index < len(tasks):
            tasks[task_index]["status"] = status
            c.execute("UPDATE workorders SET tasks=? WHERE id=?", (json.dumps(tasks), wo_id))


def directory():
    with _lock, conn() as c:
        rows = c.execute("SELECT agent_id,name,capabilities,last_seen FROM agents ORDER BY last_seen DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r); d["capabilities"] = json.loads(d["capabilities"] or "[]")
        d["online"] = (time.time() - _epoch(d["last_seen"])) < 90
        out.append(d)
    return out


def store_message(env: dict):
    with _lock, conn() as c:
        c.execute("""INSERT OR REPLACE INTO messages(id,thread,ts,frm,too,type,intent,body,delivered)
                     VALUES(?,?,?,?,?,?,?,?,0)""",
                  (env["id"], env["thread"], env["ts"], env["from"], env["to"],
                   env["type"], env["intent"], json.dumps(env.get("body", {}))))


def fetch_inbox(agent_id, mark=True):
    """Per-agent cursor delivery: each agent gets every matching message (incl. broadcast) once."""
    with _lock, conn() as c:
        row = c.execute("SELECT cursor FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        cur = (row["cursor"] if row and row["cursor"] is not None else 0)
        maxr = c.execute("SELECT COALESCE(MAX(rowid),0) m FROM messages").fetchone()["m"]
        rows = c.execute("""SELECT rowid,* FROM messages
                            WHERE rowid>? AND rowid<=? AND (too=? OR too='broadcast')
                            ORDER BY rowid""", (cur, maxr, agent_id)).fetchall()
        msgs = [_row_to_env(r) for r in rows]
        if mark:
            c.execute("UPDATE agents SET cursor=? WHERE agent_id=?", (maxr, agent_id))
    return msgs


def thread(thread_id):
    with _lock, conn() as c:
        rows = c.execute("SELECT * FROM messages WHERE thread=? ORDER BY ts", (thread_id,)).fetchall()
    return [_row_to_env(r) for r in rows]


# ---- ask conversation sessions (per-user chat memory the agent reasons over) ----
def add_turn(session: str, agent_id: str, role: str, content: str):
    """Record one turn (role='user'|'assistant') of an Ask conversation."""
    content = (content or "")[:8000]
    with _lock, conn() as c:
        c.execute("INSERT INTO conversations(session,agent_id,role,content,ts) VALUES(?,?,?,?,?)",
                  (session, agent_id, role, content, _now()))


def conversation(session: str, limit: int = 12) -> list:
    """Last `limit` turns for a session, oldest-first, as Anthropic messages [{role,content}].
    Consecutive same-role turns are merged so the sequence always alternates user/assistant."""
    with _lock, conn() as c:
        rows = c.execute("SELECT role,content FROM conversations WHERE session=? ORDER BY id DESC LIMIT ?",
                         (session, limit)).fetchall()
    hist, out = list(reversed([dict(r) for r in rows])), []
    for h in hist:
        role = "assistant" if h["role"] == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + h["content"]
        else:
            out.append({"role": role, "content": h["content"]})
    while out and out[0]["role"] != "user":   # a valid history must open with a user turn
        out.pop(0)
    while out and out[-1]["role"] != "assistant":   # ...and end on assistant, so the next
        out.pop()                                   # user question appended by the caller alternates
    return out


def drop_last_turns(session: str, n: int = 2) -> int:
    """Remove the most recent n turns of a session (used to regenerate a 👎-rejected answer)."""
    with _lock, conn() as c:
        n = c.execute("DELETE FROM conversations WHERE id IN "
                      "(SELECT id FROM conversations WHERE session=? ORDER BY id DESC LIMIT ?)",
                      (session, n)).rowcount
    return n


def reset_conversation(session: str) -> int:
    with _lock, conn() as c:
        n = c.execute("DELETE FROM conversations WHERE session=?", (session,)).rowcount
    return n


def list_sessions(agent_id: str, limit: int = 20) -> list:
    """Recent Ask sessions for a user with a preview + turn count."""
    with _lock, conn() as c:
        rows = c.execute(
            """SELECT session, COUNT(*) turns, MAX(ts) last_ts,
                      (SELECT content FROM conversations WHERE session=x.session AND role='user' ORDER BY id LIMIT 1) first_q
               FROM conversations x WHERE agent_id=? GROUP BY session ORDER BY last_ts DESC LIMIT ?""",
            (agent_id, limit)).fetchall()
    return [dict(r) for r in rows]


# ---- claims ----
def add_claim(agent, repo, paths, intent, note=""):
    with _lock, conn() as c:
        c.execute("""INSERT INTO claims(agent,repo,paths,intent,status,ts,note)
                     VALUES(?,?,?,?, 'active', ?, ?)""",
                  (agent, repo, json.dumps(paths), intent, _now(), note))


def release_claims(agent, repo=None):
    with _lock, conn() as c:
        if repo:
            c.execute("UPDATE claims SET status='released' WHERE agent=? AND repo=? AND status='active'", (agent, repo))
        else:
            c.execute("UPDATE claims SET status='released' WHERE agent=? AND status='active'", (agent,))


def active_claims():
    with _lock, conn() as c:
        rows = c.execute("SELECT agent,repo,paths,intent,ts,note FROM claims WHERE status='active' ORDER BY ts DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r); d["paths"] = json.loads(d["paths"] or "[]"); out.append(d)
    return out


def add_audit(ts, event, actor, data):
    with _lock, conn() as c:
        c.execute("INSERT INTO audit(ts,event,actor,data) VALUES(?,?,?,?)", (ts, event, actor, data))


def list_audit(limit=100, event=None, actor=None):
    sql = "SELECT ts,event,actor,data FROM audit WHERE 1=1"
    params = []
    if event:
        sql += " AND event=?"; params.append(event)
    if actor:
        sql += " AND actor=?"; params.append(actor)
    sql += " ORDER BY id DESC LIMIT ?"; params.append(int(limit))
    with _lock, conn() as c:
        rows = c.execute(sql, tuple(params)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["data"] = json.loads(d["data"] or "{}")
        except Exception:
            pass
        out.append(d)
    return out


def audit_stats():
    with _lock, conn() as c:
        by = [dict(r) for r in c.execute("SELECT event, COUNT(*) n FROM audit GROUP BY event ORDER BY n DESC").fetchall()]
        tok = c.execute("""SELECT
              COALESCE(SUM(json_extract(data,'$.in_tokens')),0) it,
              COALESCE(SUM(json_extract(data,'$.out_tokens')),0) ot,
              COALESCE(SUM(json_extract(data,'$.cost_usd')),0) cost
              FROM audit WHERE event='llm.usage'""").fetchone()
    return {"by_event": by, "tokens_in": tok["it"], "tokens_out": tok["ot"],
            "est_cost_usd": round(tok["cost"] or 0, 4)}


def _row_to_env(r):
    return {"amp": "1", "id": r["id"], "thread": r["thread"], "ts": r["ts"], "from": r["frm"],
            "to": r["too"], "type": r["type"], "intent": r["intent"], "body": json.loads(r["body"] or "{}")}


def _now():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _epoch(iso):
    import datetime as dt
    try:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0
