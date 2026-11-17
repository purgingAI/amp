"""Company knowledge graph for AMP/ACE — a READ-ONLY structured graph over git history + notes.

This is a self-contained copy of the TeamLens retrieval layer (people / projects / commits /
techs / notes / updates). We never touch TeamLens: its ingester (which needs the git repos on the
laptop) builds `teamlens.db`, and a sync job ships that file here as `data/knowledge.db`. The hub
only reads it.

Pluggable + tenant-scoped by design: today one DB = OUR company (COMPANY). When sold, each
customer gets their own knowledge.db (keyed by tenant) and the messaging layer stays generic.
Degrades gracefully: if the DB is missing, enabled() is False and callers skip grounding.
"""
from __future__ import annotations
import os, sqlite3, datetime, json
from pathlib import Path

from . import tenancy
COMPANY = os.environ.get("AMP_COMPANY", "our company")


def _dbp() -> Path:
    """Per-tenant knowledge DB (falls back to the shared one). Env override wins for single-tenant."""
    if os.environ.get("AMP_KNOWLEDGE_DB"):
        return Path(os.environ["AMP_KNOWLEDGE_DB"])
    return tenancy.db_path("knowledge")


def enabled() -> bool:
    return _dbp().exists()


def _rows(sql: str, params=()):
    dbp = _dbp()
    if not dbp.exists():
        return []
    c = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, timeout=8)
    c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception:
        return []
    finally:
        c.close()


def freshness() -> dict:
    r = _rows("SELECT v FROM meta WHERE k='last_sync'")
    ls = json.loads(r[0]["v"]) if r else {}
    synced = ls.get("synced_at")
    age_min = None
    if synced:
        try:
            t = datetime.datetime.fromisoformat(synced)
            age_min = round((datetime.datetime.now(t.tzinfo) - t).total_seconds() / 60)
        except Exception:
            pass
    return {"synced_at": synced, "age_minutes": age_min, "company": COMPANY,
            "people": ls.get("people"), "projects": ls.get("projects"), "commits": ls.get("commits")}


# ---------------- retrieval (mirrors TeamLens queries.py) ----------------
def person_techs(name, limit=8):
    return _rows("SELECT tech, weight FROM techs WHERE person=? ORDER BY weight DESC LIMIT ?", (name, limit))


ACTIVE_DAYS = int(os.environ.get("ACE_ACTIVE_DAYS", "14"))   # people idle longer than this = departed


def _active_names(days=ACTIVE_DAYS) -> set:
    return {r["name"] for r in _rows("SELECT name FROM people WHERE last_seen >= date('now', ?)",
                                     (f"-{int(days)} days",))}


def list_people(limit=100, active_days=ACTIVE_DAYS):
    where, params = "", []
    if active_days:                                   # only people who committed within the window
        where = "WHERE last_seen >= date('now', ?)"; params.append(f"-{int(active_days)} days")
    people = _rows(f"SELECT name,email,commits,first_seen,last_seen FROM people {where} "
                   "ORDER BY commits DESC LIMIT ?", (*params, limit))
    for p in people:
        p["top_tech"] = [t["tech"] for t in person_techs(p["name"], 5)]
    return people


def latest_update(person):
    r = _rows("SELECT text,focus,status,eta,ts FROM updates WHERE person=? ORDER BY ts DESC LIMIT 1", (person,))
    return r[0] if r else None


def get_person(name):
    row = _rows("SELECT * FROM people WHERE name LIKE ? ORDER BY commits DESC LIMIT 1", (f"%{name}%",))
    if not row:
        return {"error": f"No person matching '{name}'"}
    p = row[0]
    p["techs"] = person_techs(p["name"], 12)
    p["projects"] = _rows("SELECT project, COUNT(*) c, MAX(ts) last FROM commits WHERE author=? "
                          "GROUP BY project ORDER BY c DESC LIMIT 12", (p["name"],))
    p["recent"] = _rows("SELECT ts,subject,project,repo FROM commits WHERE author=? ORDER BY ts DESC LIMIT 12", (p["name"],))
    p["current_update"] = latest_update(p["name"])
    return p


def list_projects(active_days=None, limit=60):
    sql = "SELECT name,repo,kind,ticket,last_activity,commit_count,contributors FROM projects"
    params = ()
    if active_days:
        sql += " WHERE last_activity >= date('now', ?)"; params = (f"-{int(active_days)} days",)
    sql += " ORDER BY last_activity DESC LIMIT ?"
    return _rows(sql, (*params, limit))


def get_project(name):
    row = _rows("SELECT * FROM projects WHERE name LIKE ? OR ticket LIKE ? ORDER BY commit_count DESC LIMIT 1",
                (f"%{name}%", f"%{name}%"))
    if not row:
        return {"error": f"No project matching '{name}'"}
    p = row[0]
    p["recent"] = _rows("SELECT ts,author,subject,repo FROM commits WHERE project=? ORDER BY ts DESC LIMIT 15", (p["name"],))
    p["top_authors"] = _rows("SELECT author, COUNT(*) c FROM commits WHERE project=? GROUP BY author ORDER BY c DESC LIMIT 8", (p["name"],))
    return p


def search_commits(query="", author=None, project=None, days=None, limit=25):
    sql = "SELECT ts,author,subject,project,repo FROM commits WHERE 1=1"
    params = []
    if query:   sql += " AND subject LIKE ?"; params.append(f"%{query}%")
    if author:  sql += " AND author LIKE ?"; params.append(f"%{author}%")
    if project: sql += " AND project LIKE ?"; params.append(f"%{project}%")
    if days:    sql += " AND ts >= date('now', ?)"; params.append(f"-{int(days)} days")
    sql += " ORDER BY ts DESC LIMIT ?"; params.append(int(limit))
    return _rows(sql, tuple(params))


def get_activity(days=14, limit=40):
    return _rows("SELECT ts,author,subject,project,repo FROM commits WHERE ts >= date('now', ?) "
                 "ORDER BY ts DESC LIMIT ?", (f"-{int(days)} days", int(limit)))


def search_notes(query="", limit=6):
    if query:
        rows = _rows("SELECT slug,title,body FROM notes WHERE body LIKE ? OR title LIKE ? LIMIT ?",
                     (f"%{query}%", f"%{query}%", limit))
    else:
        rows = _rows("SELECT slug,title,body FROM notes LIMIT ?", (limit,))
    for r in rows:
        r["body"] = r["body"][:1500]
    return rows


def search_updates(query="", limit=15):
    if not query:
        return _rows("SELECT person,text,focus,status,eta,source,ts FROM updates ORDER BY ts DESC LIMIT ?", (limit,))
    return _rows("SELECT person,text,focus,status,eta,ts FROM updates WHERE text LIKE ? OR focus LIKE ? "
                 "OR person LIKE ? ORDER BY ts DESC LIMIT ?", (f"%{query}%", f"%{query}%", f"%{query}%", limit))


def find_experts(topic, limit=6):
    topic = (topic or "").strip()
    if not topic:
        return []
    scores = {}
    for r in _rows("SELECT person, SUM(weight) w FROM techs WHERE tech LIKE ? GROUP BY person", (f"%{topic}%",)):
        scores[r["person"]] = scores.get(r["person"], 0) + r["w"] * 3
    for r in _rows("SELECT author, COUNT(*) c FROM commits WHERE subject LIKE ? GROUP BY author", (f"%{topic}%",)):
        scores[r["author"]] = scores.get(r["author"], 0) + r["c"]
    active = _active_names()                          # don't recommend people who have left the company
    scores = {k: v for k, v in scores.items() if k in active}
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
    out = []
    for name, sc in ranked:
        p = _rows("SELECT commits,last_seen FROM people WHERE name=?", (name,))
        upd = latest_update(name)
        out.append({"name": name, "score": sc, "commits": p[0]["commits"] if p else 0,
                    "top_tech": [t["tech"] for t in person_techs(name, 4)],
                    "current": upd["focus"] if upd else None})
    return out


def overview():
    counts = _rows("SELECT (SELECT COUNT(*) FROM people) p, (SELECT COUNT(*) FROM projects) j, "
                   "(SELECT COUNT(*) FROM commits) c, (SELECT COUNT(*) FROM notes) n")
    top_tech = _rows("SELECT tech, SUM(weight) w FROM techs GROUP BY tech ORDER BY w DESC LIMIT 14")
    return {"counts": counts[0] if counts else {}, "top_tech": top_tech, "freshness": freshness()}


# ---------------- code-level ownership (from file_changes: who owns which code) ----------------
def module_owners(module, limit=8):
    """Top authors of a module/service by files changed (code ownership, not just commit count)."""
    return _rows("SELECT author, COUNT(*) changes, SUM(insertions+deletions) churn, MAX(ts) last "
                 "FROM file_changes WHERE module LIKE ? GROUP BY author ORDER BY changes DESC LIMIT ?",
                 (f"%{module}%", limit))


def person_modules(name, limit=15):
    """Which modules/services a person actually writes code in."""
    return _rows("SELECT module, repo, COUNT(*) changes, SUM(insertions+deletions) churn, MAX(ts) last "
                 "FROM file_changes WHERE author=? GROUP BY module ORDER BY changes DESC LIMIT ?",
                 (name, limit))


def file_owners(path, limit=6):
    """Who has touched a specific file the most (bus-factor / who to ask)."""
    return _rows("SELECT author, COUNT(*) changes, MAX(ts) last FROM file_changes "
                 "WHERE path LIKE ? GROUP BY author ORDER BY changes DESC LIMIT ?", (f"%{path}%", limit))


def code_map(limit=40):
    """Top modules across the codebase with their primary owner + churn — the code knowledge graph."""
    mods = _rows("SELECT module, repo, COUNT(DISTINCT author) contributors, COUNT(*) changes, "
                 "SUM(insertions+deletions) churn, MAX(ts) last FROM file_changes "
                 "WHERE module != '' GROUP BY module, repo ORDER BY changes DESC LIMIT ?", (limit,))
    for m in mods:
        top = _rows("SELECT author, COUNT(*) c FROM file_changes WHERE module=? AND repo=? "
                    "GROUP BY author ORDER BY c DESC LIMIT 3", (m["module"], m["repo"]))
        m["owners"] = [t["author"] for t in top]
    return mods


def code_enabled() -> bool:
    """True once the synced knowledge DB carries the file-level ownership table."""
    return bool(_rows("SELECT name FROM sqlite_master WHERE type='table' AND name='file_changes'"))


# ---------------- AMP glue: map an AMP agent -> a KB person ----------------
def person_for(agent: dict):
    """Best-effort: match an AMP agent (email/name) to a person in the graph."""
    if not agent:
        return None
    email = (agent.get("email") or "").lower()
    if email:
        local = email.split("@")[0]
        r = _rows("SELECT name FROM people WHERE lower(email)=? OR lower(email) LIKE ? LIMIT 1",
                  (email, f"{local}@%"))
        if r:
            return r[0]["name"]
    name = (agent.get("name") or "").strip()
    if name:
        r = _rows("SELECT name FROM people WHERE name LIKE ? ORDER BY commits DESC LIMIT 1", (f"%{name}%",))
        if r:
            return r[0]["name"]
    return None


def context_for(question: str, as_person: str | None = None, budget: int = 5) -> str:
    """Compact, retrieval-only company context for grounding a single-shot reply (token-frugal)."""
    if not enabled():
        return ""
    parts = []
    if as_person:
        p = get_person(as_person)
        if p and not p.get("error"):
            techs = ", ".join(t["tech"] for t in p.get("techs", [])[:6])
            projs = ", ".join(f"{x['project']}({x['c']})" for x in p.get("projects", [])[:5])
            upd = p.get("current_update")
            parts.append(f"YOU are {p['name']} — expertise: {techs}. Projects: {projs}."
                         + (f" Currently: {upd.get('focus') or upd.get('text','')}" if upd else ""))
            rec = "; ".join(x["subject"][:70] for x in p.get("recent", [])[:4])
            if rec:
                parts.append(f"Your recent commits: {rec}")
    experts = find_experts(question, limit=budget)
    if experts:
        parts.append("People who know this: " + "; ".join(
            f"{e['name']} ({', '.join(e['top_tech'][:2])}{', now: '+e['current'] if e.get('current') else ''})"
            for e in experts))
    hits = search_commits(question, limit=6)
    if hits:
        parts.append("Related work: " + "; ".join(f"{h['author']}: {h['subject'][:70]}" for h in hits))
    notes = search_notes(question, limit=2)
    if notes:
        parts.append("Notes: " + " | ".join(f"{n['title']}: {n['body'][:200]}" for n in notes))
    return "\n".join(parts)[:3500]
