#!/usr/bin/env python3
"""Code-understanding knowledge graph builder (runs on the LAPTOP where the repos live).

Instead of indexing commits, this READS THE CODE and distills, per module, what it actually
does — purpose, capabilities/features, endpoints, key classes, searchable keywords — using an
LLM. It is INCREMENTAL: each module has a content hash, and only modules whose code changed
since the last run are re-analyzed (so the hourly "auto-tune" is cheap). The result is written
to data/code_knowledge.db, which is synced to the EC2 hub; ask.py then answers from this
distilled understanding rather than re-scanning commits.

Run:  ./.venv/bin/python codegraph_build.py            (all repos, incremental)
      ./.venv/bin/python codegraph_build.py --repo core-api   (one repo)
      ./.venv/bin/python codegraph_build.py --full     (ignore hashes, re-analyze everything)
"""
from __future__ import annotations
import os, sys, re, json, sqlite3, hashlib, subprocess, time, argparse
from pathlib import Path

HOME = Path.home()
DB = Path(os.environ.get("AMP_CODEGRAPH_DB", HOME / "amp" / "data" / "code_knowledge.db"))
MODEL = os.environ.get("AMP_CODEGRAPH_MODEL", "claude-haiku-4-5-20251001")
# Never analyze: SDKs/tooling, this app + its siblings, and personal projects / wallet worktrees —
# the company knowledge graph must contain only the company's product repos.
DENY = {"flutter", ".pyenv", "team-lens", "amp", "node_modules", "go", "sdk", "build", "target",
        "trading-ai", "creator-os", "vantage", "employee-tracker", "ace_app", "amp-ios",
        "cba", "citybank", "team-lens",
        "wallet-rmc", "core-api", "core-api", "core-api"}
SCAN_PARENTS = [HOME / "company-repos", HOME / "Documents", HOME / "Downloads", HOME]
CODE_EXT = {".java", ".kt", ".ts", ".tsx", ".js", ".py", ".go", ".sql"}


def _load_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for f in (HOME / "creator-os/.env", HOME / "trading-ai/.env"):
        try:
            for line in f.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY"):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        os.environ["ANTHROPIC_API_KEY"] = v
                        return v
        except FileNotFoundError:
            pass
    return None


import anthropic
_client = anthropic.Anthropic(api_key=_load_key()) if _load_key() else None


# ---------------- storage ----------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
  id TEXT PRIMARY KEY, repo TEXT, module TEXT, path TEXT, language TEXT,
  purpose TEXT, summary TEXT, capabilities TEXT, endpoints TEXT, key_classes TEXT,
  keywords TEXT, owners TEXT, files INTEGER, code_hash TEXT, last_analyzed TEXT
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS ix_mod_repo ON modules(repo);
"""


def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


# ---------------- repo / module discovery ----------------
def discover_repos(only: str | None = None):
    seen = {}
    for parent in SCAN_PARENTS:
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            if not (child / ".git").is_dir() or child.name in DENY or child.name.startswith("."):
                continue
            if only and child.name != only:
                continue
            seen.setdefault(child.name, child)   # first wins (Documents before Downloads)
    return seen


def find_modules(repo_path: Path):
    """A 'module' = a dir containing src/main (Java/Kotlin) or a top-level app dir (TS/py)."""
    mods = []
    # Java/Kotlin multi-module: */src/main -> the module dir is two levels up from src/main
    for src in repo_path.glob("*/src/main"):
        mods.append(src.parent.parent)
    # single-module java repo
    if not mods and (repo_path / "src/main").exists():
        mods.append(repo_path)
    # frontend/other: treat src/app or the repo root as one module
    if not mods:
        for cand in (repo_path / "src", repo_path / "app", repo_path):
            if cand.exists():
                mods.append(cand if cand != repo_path else repo_path)
                break
    return mods


def _code_files(mod: Path, limit=4000):
    out = []
    for p in mod.rglob("*"):
        if p.is_file() and p.suffix in CODE_EXT and "/target/" not in str(p) and "/build/" not in str(p) and "/node_modules/" not in str(p):
            out.append(p)
            if len(out) >= limit:
                break
    return out


def content_hash(files):
    h = hashlib.sha256()
    for p in sorted(files):
        try:
            st = p.stat()
            h.update(str(p).encode()); h.update(str(int(st.st_mtime)).encode()); h.update(str(st.st_size).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


# ---------------- per-module code digest (no LLM) ----------------
def build_digest(mod: Path, files) -> dict:
    endpoints, controllers, services, entities, enums, configs = [], [], [], [], [], []
    pkgtree = set()
    ep_re = re.compile(r'@(?:Request|Post|Get|Put|Delete|Patch)Mapping\((?:value\s*=\s*)?["\']([^"\']+)["\']')
    cls_re = re.compile(r'\b(?:class|interface|enum|record)\s+([A-Z]\w+)')
    for p in files:
        rel = str(p.relative_to(mod))
        parts = rel.split("/")
        if len(parts) > 2:
            pkgtree.add("/".join(parts[:-1]))
        name = p.stem
        if p.suffix in (".java", ".kt"):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            if name.endswith("Controller"):
                paths = ep_re.findall(txt)
                controllers.append(name)
                for pa in paths[:8]:
                    endpoints.append(pa)
            elif "Service" in name and "Impl" not in name:
                services.append(name)
            elif name.endswith(("DTO", "Entity")) or "/entity/" in rel:
                entities.append(name)
            elif name.endswith("Enum") or "/enums/" in rel:
                enums.append(name)
        elif p.name in ("application.yml", "application.yaml"):
            try:
                configs += re.findall(r'^\s*([a-zA-Z][\w.-]+):', p.read_text(errors="ignore"), re.M)[:20]
            except Exception:
                pass
    def cap(lst, n): return sorted(set(lst))[:n]
    return {
        "endpoints": cap(endpoints, 25), "controllers": cap(controllers, 20),
        "services": cap(services, 30), "entities": cap(entities, 25), "enums": cap(enums, 15),
        "packages": cap(pkgtree, 25), "configs": cap(configs, 20), "n_files": len(files),
    }


def recent_commits(mod: Path, repo: Path, n=15):
    try:
        rel = mod.relative_to(repo)
        out = subprocess.run(["git", "-C", str(repo), "log", "-n", str(n), "--pretty=%an|%s", "--", str(rel)],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return [l for l in out if l]
    except Exception:
        return []


def owners(mod: Path, repo: Path, top=6):
    # Bounded to the last 400 commits touching the module (shortlog over full 20k+ history is slow).
    try:
        from collections import Counter
        rel = mod.relative_to(repo)
        out = subprocess.run(["git", "-C", str(repo), "log", "-n", "400", "--format=%an", "--", str(rel)],
                             capture_output=True, text=True, timeout=20).stdout.splitlines()
        cnt = Counter(a.strip() for a in out if a.strip())
        return [{"name": nm, "commits": c} for nm, c in cnt.most_common(top)]
    except Exception:
        return []


# ---------------- LLM: understand the module from its code ----------------
SYS = ("You analyse a software module from a compact digest of its real code (endpoints, class "
       "names, package layout, config keys, recent commit subjects). Infer what the module actually "
       "DOES and the concrete FEATURES/CAPABILITIES it implements — go beyond restating commit "
       "messages; reason about the code structure. Output ONLY compact JSON.")


def analyze(repo_name, mod_name, digest, commits) -> dict:
    fallback = {"purpose": f"{mod_name} module of {repo_name}.", "summary": "", "capabilities": [],
                "endpoints": digest.get("endpoints", [])[:15], "key_classes": (digest.get("controllers", []) + digest.get("services", []))[:12],
                "keywords": [mod_name.lower()], "_generated": False}
    if _client is None:
        return fallback
    prompt = (
        f"Repo: {repo_name}\nModule: {mod_name}\n"
        f"Endpoints: {digest.get('endpoints')}\nControllers: {digest.get('controllers')}\n"
        f"Services: {digest.get('services')}\nEntities/DTOs: {digest.get('entities')}\n"
        f"Enums: {digest.get('enums')}\nPackages: {digest.get('packages')}\n"
        f"Config keys: {digest.get('configs')}\nRecent commits:\n" + "\n".join(commits[:15]) +
        "\n\nReturn ONLY JSON: {\"purpose\": one sentence on what this module is for, "
        "\"summary\": 2-3 sentences on what it does and how, \"capabilities\": [up to 8 concrete "
        "features/things it implements — nouns/verbs a teammate would search for], "
        "\"endpoints\": [notable API paths], \"key_classes\": [up to 10 important classes], "
        "\"keywords\": [up to 12 lowercase search terms incl domain concepts]}.")
    try:
        msg = _client.messages.create(model=MODEL, max_tokens=700,
                                      system=SYS, messages=[{"role": "user", "content": prompt}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1]
        data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        for k in ("capabilities", "endpoints", "key_classes", "keywords"):
            data.setdefault(k, [])
        data.setdefault("purpose", fallback["purpose"]); data.setdefault("summary", "")
        data["_generated"] = True
        data["_usage"] = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
        return data
    except Exception as e:
        print(f"    ! analyze failed for {mod_name}: {e}")
        return fallback


def _lang(repo_name):
    if any(x in repo_name for x in ("shield", "ui", "app", "customer")):
        return "TypeScript/Angular"
    return "Java/Spring"


def run(only=None, full=False):
    c = db()
    repos = discover_repos(only)
    print(f"codegraph: {len(repos)} repo(s): {list(repos)}")
    analyzed = skipped = 0; tin = tout = 0
    for repo_name, repo_path in repos.items():
        for mod in find_modules(repo_path):
            mod_name = mod.name if mod != repo_path else repo_name
            mid = f"{repo_name}::{mod_name}"
            files = _code_files(mod)
            if len(files) < 3:      # skip trivial/empty modules
                continue
            chash = content_hash(files)
            row = c.execute("SELECT code_hash FROM modules WHERE id=?", (mid,)).fetchone()
            if row and row["code_hash"] == chash and not full:
                skipped += 1
                continue
            digest = build_digest(mod, files)
            commits = recent_commits(mod, repo_path)
            print(f"  ~ analyzing {mid} ({len(files)} files, {len(digest['endpoints'])} endpoints)…")
            data = analyze(repo_name, mod_name, digest, commits)
            u = data.pop("_usage", None)
            if u:
                tin += u["in"]; tout += u["out"]
            c.execute("""INSERT INTO modules
                (id,repo,module,path,language,purpose,summary,capabilities,endpoints,key_classes,keywords,owners,files,code_hash,last_analyzed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET repo=excluded.repo,module=excluded.module,path=excluded.path,
                language=excluded.language,purpose=excluded.purpose,summary=excluded.summary,
                capabilities=excluded.capabilities,endpoints=excluded.endpoints,key_classes=excluded.key_classes,
                keywords=excluded.keywords,owners=excluded.owners,files=excluded.files,code_hash=excluded.code_hash,
                last_analyzed=excluded.last_analyzed""",
                (mid, repo_name, mod_name, str(mod.relative_to(repo_path)) if mod != repo_path else ".",
                 _lang(repo_name), data["purpose"], data["summary"], json.dumps(data["capabilities"]),
                 json.dumps(data["endpoints"] or digest["endpoints"]), json.dumps(data["key_classes"]),
                 json.dumps(data["keywords"]), json.dumps(owners(mod, repo_path)), digest["n_files"],
                 chash, time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            c.commit()
            analyzed += 1
    c.execute("INSERT INTO meta(k,v) VALUES('last_build',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
              (json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "analyzed": analyzed, "skipped": skipped,
                           "modules": c.execute("SELECT COUNT(*) FROM modules").fetchone()[0]}),))
    c.commit()
    est = tin / 1e6 * 0.80 + tout / 1e6 * 4.0
    print(f"codegraph done: analyzed={analyzed} skipped(unchanged)={skipped} "
          f"total_modules={c.execute('SELECT COUNT(*) FROM modules').fetchone()[0]} "
          f"tokens in={tin} out={tout} ~${est:.3f}")
    c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run(only=a.repo, full=a.full)
