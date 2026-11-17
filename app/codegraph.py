"""Read-only query layer over the CODE-UNDERSTANDING knowledge graph (data/code_knowledge.db).

This is the distilled understanding of the codebase built by codegraph_build.py (which reads the
actual code, not just commits, and re-tunes hourly on what changed). ask.py answers from THIS —
module purposes, capabilities/features, endpoints, key classes, owners — instead of re-scanning
raw commits. Degrades gracefully: if the DB is absent, enabled() is False.
"""
from __future__ import annotations
import os, sqlite3, json, datetime
from pathlib import Path
from . import tenancy


def _dbp() -> Path:
    """Per-tenant code-graph DB (falls back to the shared one). Env override wins for single-tenant."""
    if os.environ.get("AMP_CODEGRAPH_DB"):
        return Path(os.environ["AMP_CODEGRAPH_DB"])
    return tenancy.db_path("code_knowledge")


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


def _j(v, default):
    try:
        return json.loads(v) if v else default
    except Exception:
        return default


def _mod(r: dict, full=False) -> dict:
    m = {"id": r["id"], "repo": r["repo"], "module": r["module"], "purpose": r.get("purpose"),
         "capabilities": _j(r.get("capabilities"), []), "owners": _j(r.get("owners"), []),
         "language": r.get("language")}
    if full:
        m.update({"summary": r.get("summary"), "endpoints": _j(r.get("endpoints"), []),
                  "key_classes": _j(r.get("key_classes"), []), "keywords": _j(r.get("keywords"), []),
                  "path": r.get("path"), "files": r.get("files"), "last_analyzed": r.get("last_analyzed")})
    return m


def freshness() -> dict:
    r = _rows("SELECT v FROM meta WHERE k='last_build'")
    d = _j(r[0]["v"], {}) if r else {}
    at = d.get("at"); age = None
    if at:
        try:
            t = datetime.datetime.fromisoformat(at)
            age = round((datetime.datetime.now(t.tzinfo) - t).total_seconds() / 60)
        except Exception:
            pass
    return {"built_at": at, "age_minutes": age, "modules": d.get("modules"),
            "last_reanalyzed": d.get("analyzed"), "unchanged": d.get("skipped")}


def list_modules(limit: int = 200, repo: str | None = None) -> list:
    if repo:
        rows = _rows("SELECT * FROM modules WHERE repo=? ORDER BY files DESC LIMIT ?", (repo, limit))
    else:
        rows = _rows("SELECT * FROM modules ORDER BY files DESC LIMIT ?", (limit,))
    return [_mod(r) for r in rows]


def get_module(name: str) -> dict:
    if not name:
        return {"error": "name required"}
    like = f"%{name}%"
    rows = _rows("SELECT * FROM modules WHERE module LIKE ? OR id LIKE ? ORDER BY files DESC LIMIT 1", (like, like))
    if not rows:
        return {"error": f"No module matching '{name}'"}
    return _mod(rows[0], full=True)


def search_capabilities(query: str, limit: int = 12) -> list:
    """Find modules whose distilled understanding matches the query (capabilities / keywords /
    summary / purpose). Ranks by how many query terms hit. This is the primary 'what/where is X
    built' retrieval — over the CODE understanding, not commits."""
    q = (query or "").strip().lower()
    if not q:
        return []
    terms = [t for t in __import__("re").split(r"[^a-z0-9]+", q) if len(t) > 2]
    rows = _rows("SELECT * FROM modules")
    scored = []
    for r in rows:
        hay = " ".join([str(r.get("module", "")), str(r.get("purpose", "")), str(r.get("summary", "")),
                        str(r.get("capabilities", "")), str(r.get("keywords", ""))]).lower()
        score = sum(hay.count(t) for t in terms)
        if score:
            caps = _j(r.get("capabilities"), [])
            hitcaps = [c for c in caps if any(t in c.lower() for t in terms)]
            scored.append((score, {**_mod(r), "matched_capabilities": hitcaps[:5],
                                   "summary": r.get("summary")}))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored[:limit]]


def overview() -> dict:
    counts = _rows("SELECT COUNT(*) n, COUNT(DISTINCT repo) r FROM modules")
    repos = _rows("SELECT repo, COUNT(*) n FROM modules GROUP BY repo ORDER BY n DESC")
    return {"modules": counts[0]["n"] if counts else 0, "repos": counts[0]["r"] if counts else 0,
            "by_repo": {x["repo"]: x["n"] for x in repos}, "freshness": freshness()}
