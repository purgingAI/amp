"""The AMP Orchestrator: a built-in agent with live repo access + merge-safety.

Handles query.knowledge / plan.feature / collision.check / claim.files. It grounds
plans in the REAL code (grep, read, git history) and cross-checks the claims ledger
so parallel work merges cleanly. Phase 2 will extend plan.feature -> implement.feature
(apply in a worktree + run tests).
"""
from __future__ import annotations
import os, subprocess, json
from pathlib import Path
import anthropic
from . import store, protocol, audit

HOME = Path.home()
MODEL = os.environ.get("AMP_MODEL", "claude-sonnet-4-6")


def _load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for f in (HOME / "creator-os/.env", HOME / "trading-ai/.env"):
        try:
            for line in f.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY"):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        os.environ["ANTHROPIC_API_KEY"] = v; return v
        except FileNotFoundError:
            pass
    return None


_client = anthropic.Anthropic(api_key=_load_key()) if _load_key() else None


def discover_repos():
    repos = {}
    for parent in (HOME / "Documents", HOME / "Downloads", HOME):
        if not parent.exists():
            continue
        for c in parent.iterdir():
            if (c / ".git").is_dir() and c.name not in ("flutter", ".pyenv", "amp", "team-lens"):
                repos.setdefault(c.name, c)
    return repos


REPOS = discover_repos()


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"(git error: {e})"


# ---------------- tools ----------------
def t_list_repos(a):
    return [{"repo": n, "path": str(p).replace(str(HOME), "~")} for n, p in REPOS.items()]


def t_search_code(a):
    q = a.get("query", ""); repo = a.get("repo")
    targets = [REPOS[repo]] if repo in REPOS else list(REPOS.values())
    hits = []
    for p in targets[:3]:
        try:
            out = subprocess.run(["grep", "-rInE", "--include=*.java", "--include=*.ts",
                                  "--include=*.py", "--include=*.sql", q, str(p)],
                                 capture_output=True, text=True, timeout=12).stdout
        except Exception:
            out = ""
        for line in out.splitlines()[:12]:
            parts = line.split(":", 2)
            if len(parts) == 3:
                hits.append({"repo": p.name, "file": parts[0].replace(str(p) + "/", ""),
                             "line": parts[1], "text": parts[2].strip()[:160]})
        if len(hits) > 30:
            break
    return hits[:30]


def t_read_file(a):
    repo, rel = a.get("repo"), a.get("path", "")
    if repo not in REPOS:
        return {"error": "unknown repo"}
    f = REPOS[repo] / rel
    try:
        txt = f.read_text()
        return {"repo": repo, "path": rel, "lines": txt.count("\n") + 1, "content": txt[:6000]}
    except Exception as e:
        return {"error": str(e)}


def t_git_recent(a):
    repo = a.get("repo"); path = a.get("path"); days = a.get("days", 60)
    if repo not in REPOS:
        return {"error": "unknown repo"}
    args = ["log", f"--since={days} days ago", "--pretty=%an|%ad|%s", "--date=short", "-20"]
    if path:
        args += ["--", path]
    out = _git(REPOS[repo], *args)
    return [dict(zip(("author", "date", "subject"), l.split("|", 2)))
            for l in out.splitlines() if "|" in l]


def t_check_collision(a):
    repo = a.get("repo"); paths = a.get("paths", [])
    warnings = []
    for cl in store.active_claims():
        if cl["repo"] == repo:
            overlap = set(cl["paths"]) & set(paths)
            if overlap:
                warnings.append({"kind": "active-claim", "agent": cl["agent"],
                                 "paths": sorted(overlap), "since": cl["ts"]})
    # recent commits touching those paths (last 21d)
    if repo in REPOS:
        for pth in paths[:8]:
            out = _git(REPOS[repo], "log", "--since=21 days ago", "--pretty=%an|%ad", "--date=short", "--", pth)
            authors = {l.split("|")[0] for l in out.splitlines() if "|" in l}
            if authors:
                warnings.append({"kind": "recent-commits", "path": pth, "authors": sorted(authors)})
    return {"repo": repo, "requested": paths, "collisions": warnings,
            "verdict": "CONFLICTS" if warnings else "clear"}


def t_active_claims(a):
    return store.active_claims()


SNAP = Path(__file__).resolve().parent.parent / "data" / "commits.json"


def recent_commits(limit=40, repo=None):
    """Aggregate recent commits across all repos — the 'check all the commits' feed.

    If this host has the repos (e.g. the dev laptop), read them live AND refresh a snapshot
    file. If it doesn't (e.g. the EC2 deployment), fall back to that snapshot so the feed still
    works. Push a fresh snapshot with: orchestrator.write_snapshot() then scp data/commits.json.
    """
    targets = {repo: REPOS[repo]} if repo in REPOS else REPOS
    out = []
    for name, p in list(targets.items())[:12]:
        log = _git(p, "log", "--all", "--no-merges", "-20",
                   "--pretty=%h%x01%an%x01%aI%x01%s")
        for line in log.splitlines():
            parts = line.split("\x01")
            if len(parts) == 4:
                out.append({"repo": name, "sha": parts[0], "author": parts[1],
                            "date": parts[2], "subject": parts[3]})
    if out:
        out.sort(key=lambda c: c["date"], reverse=True)
        try:
            SNAP.parent.mkdir(exist_ok=True)
            json.dump(out[:400], open(SNAP, "w"))
        except Exception:
            pass
        return out[:limit]
    # no repos here -> serve the snapshot
    try:
        snap = json.load(open(SNAP))
        if repo:
            snap = [c for c in snap if c.get("repo") == repo]
        return snap[:limit]
    except Exception:
        return []


def write_snapshot():
    recent_commits(400)
    return {"snapshot": str(SNAP), "count": len(json.load(open(SNAP))) if SNAP.exists() else 0}


TOOLS = [
    {"name": "list_repos", "description": "List all company repos the orchestrator can access.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_code", "description": "Grep source across repos.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "repo": {"type": "string"}}, "required": ["query"]}},
    {"name": "read_file", "description": "Read a file from a repo.",
     "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}, "path": {"type": "string"}}, "required": ["repo", "path"]}},
    {"name": "git_recent", "description": "Recent commits (optionally for a path) — who touched what.",
     "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}, "path": {"type": "string"}, "days": {"type": "integer"}}, "required": ["repo"]}},
    {"name": "check_collision", "description": "Check requested file paths against active claims + recent commits (merge-safety).",
     "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}}, "required": ["repo", "paths"]}},
    {"name": "active_claims", "description": "List everyone's active file claims.",
     "input_schema": {"type": "object", "properties": {}}},
]
DISPATCH = {"list_repos": t_list_repos, "search_code": t_search_code, "read_file": t_read_file,
            "git_recent": t_git_recent, "check_collision": t_check_collision, "active_claims": t_active_claims}

SYSTEM = """You are the AMP ORCHESTRATOR — a company-wide senior engineer agent with live access to
ALL repos and to the shared claims ledger of who is editing what. Engineers' personal agents call
you over AMP instead of working alone, so you must:
- Ground every plan in the REAL code (use search_code/read_file/git_recent). Cite files.
- For any implementation ask, produce: (1) a concrete step-by-step plan with exact files to change,
  (2) the FILE PATHS you'd touch, (3) a merge-safety check via check_collision, calling out anyone
  else already in those files, (4) a test plan.
- Be decisive and concrete. Never invent files — verify they exist.
Reply in clean markdown. End implementation plans with a '### Files to claim' list and a
'### Merge safety' note."""


def handle(env: dict) -> dict:
    """Process an inbound REQUEST envelope -> RESPONSE envelope (body.text = answer)."""
    intent = env.get("intent", "chat")
    body = env.get("body", {})
    frm = env.get("from")
    secret = store.get_agent("orchestrator")["secret"]

    # deterministic intents (no LLM)
    if intent == "claim.files":
        store.add_claim(frm, body.get("repo", "?"), body.get("paths", []), "claim", body.get("text", ""))
        text = f"✅ Claimed {len(body.get('paths', []))} path(s) in `{body.get('repo')}` for **{frm}**."
        return protocol.make("orchestrator", frm, intent, {"text": text}, secret, type="RESPONSE", thread=env["thread"])
    if intent == "release.claim":
        store.release_claims(frm, body.get("repo"))
        return protocol.make("orchestrator", frm, intent, {"text": f"🔓 Released claims for {frm}."}, secret, type="RESPONSE", thread=env["thread"])
    if intent == "collision.check":
        res = t_check_collision(body)
        return protocol.make("orchestrator", frm, intent, {"text": _fmt_collision(res), "data": res}, secret, type="RESPONSE", thread=env["thread"])

    # LLM intents
    if not _client:
        return protocol.make("orchestrator", frm, intent, {"text": "⚠️ Orchestrator has no ANTHROPIC_API_KEY."}, secret, type="RESPONSE", thread=env["thread"])
    question = body.get("text", "")
    ctx = f"Requesting agent: {frm}. Intent: {intent}. Active claims: {json.dumps(store.active_claims())[:1200]}"
    messages = [{"role": "user", "content": f"{ctx}\n\nTask:\n{question}"}]
    answer = _run_llm(messages)
    return protocol.make("orchestrator", frm, intent, {"text": answer}, secret, type="RESPONSE", thread=env["thread"])


def _run_llm(messages) -> str:
    collected = []
    for _ in range(10):
        msg = _client.messages.create(model=MODEL, max_tokens=1800, system=SYSTEM, tools=TOOLS, messages=messages)
        audit.llm(MODEL, msg.usage, "orchestrator")
        messages.append({"role": "assistant", "content": msg.content})
        collected += [b.text for b in msg.content if b.type == "text"]
        if msg.stop_reason != "tool_use":
            return "".join(b.text for b in msg.content if b.type == "text") or "\n".join(collected)
        results = []
        for b in msg.content:
            if b.type == "tool_use":
                try:
                    out = DISPATCH[b.name](b.input if isinstance(b.input, dict) else {})
                except Exception as e:
                    out = {"error": str(e)}
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, default=str)[:8000]})
        messages.append({"role": "user", "content": results})
    # ran out of tool rounds — force a final written answer without tools
    try:
        messages.append({"role": "user", "content": "Stop researching now and write your final answer/plan."})
        final = _client.messages.create(model=MODEL, max_tokens=1800, system=SYSTEM, messages=messages)
        return "".join(b.text for b in final.content if b.type == "text") or "\n".join(collected)
    except Exception:
        return "\n".join(collected) or "(no answer produced)"


def _fmt_collision(res) -> str:
    if res["verdict"] == "clear":
        return f"🟢 No collisions in `{res['repo']}` for {res['requested']}. Safe to proceed."
    lines = [f"🔴 **Possible merge collisions in `{res['repo']}`:**"]
    for w in res["collisions"]:
        if w["kind"] == "active-claim":
            lines.append(f"- **{w['agent']}** already claimed {w['paths']} (since {w['since']})")
        else:
            lines.append(f"- `{w['path']}` recently changed by {', '.join(w['authors'])}")
    return "\n".join(lines)


def _fmt(x):
    return _fmt_collision(x)
