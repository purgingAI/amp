#!/usr/bin/env python3
"""Auto-discover the company's GitLab repos and clone any new ones into ~/company-repos, so the
hourly code-graph re-tune automatically starts covering repos added after the initial seed.

Credentials come from ~/.amp_gitlab (chmod 600), which sets EITHER:
    GITLAB_TOKEN=glpat-...          # a read_api Personal Access Token  (preferred, durable)
  or
    GITLAB_USER=... / GITLAB_PASS=...   # falls back to the OAuth password grant (token expires ~2h)

Only repos under the company group (global/**) are considered; personal projects are ignored.
Repos already cloned anywhere (company-repos, Documents, Downloads, HOME) are skipped by name.
"""
from __future__ import annotations
import os, re, json, subprocess, urllib.request, urllib.parse
from pathlib import Path

HOME = Path.home()
BASE = "https://git.example.com"
CLONE_DIR = HOME / "company-repos"
COMPANY_GROUP_PREFIX = "global/"          # only clone repos under this GitLab group
# don't re-clone what we already have (by basename); also skip our own tooling
KNOWN_PARENTS = [CLONE_DIR, HOME / "Documents", HOME / "Downloads", HOME]
SKIP_NAMES = {"amp", "team-lens", "flutter"}


def _creds():
    f = HOME / ".amp_gitlab"
    env = {}
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _api_token(c: dict) -> str | None:
    if c.get("GITLAB_TOKEN"):
        return c["GITLAB_TOKEN"]
    if c.get("GITLAB_USER") and c.get("GITLAB_PASS"):
        try:
            data = urllib.parse.urlencode({"grant_type": "password", "username": c["GITLAB_USER"],
                                           "password": c["GITLAB_PASS"]}).encode()
            r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/oauth/token", data=data), timeout=25))
            return r.get("access_token")
        except Exception as e:
            print("discover: oauth password grant failed:", e)
    return None


def _list_projects(token: str, is_pat: bool) -> list:
    hdr = {"PRIVATE-TOKEN": token} if is_pat else {"Authorization": "Bearer " + token}
    out, page = [], 1
    while page <= 10:
        url = f"{BASE}/api/v4/projects?membership=true&simple=true&per_page=100&page={page}"
        try:
            batch = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=25))
        except Exception as e:
            print("discover: list projects failed:", e); break
        if not batch:
            break
        out += batch
        page += 1
    return out


def _already_cloned() -> set:
    have = set()
    for parent in KNOWN_PARENTS:
        if parent.exists():
            for c in parent.iterdir():
                if (c / ".git").is_dir():
                    have.add(c.name)
    return have


def run():
    c = _creds()
    token = _api_token(c)
    if not token:
        print("discover: no GitLab credentials (~/.amp_gitlab) — skipping.")
        return
    is_pat = bool(c.get("GITLAB_TOKEN"))
    projects = _list_projects(token, is_pat)
    have = _already_cloned()
    CLONE_DIR.mkdir(exist_ok=True)
    cloned = 0
    for p in projects:
        path = p.get("path_with_namespace", "")
        name = p.get("path", "")
        if not path.startswith(COMPANY_GROUP_PREFIX):   # company repos only
            continue
        if name in have or name in SKIP_NAMES:
            continue
        url = p.get("http_url_to_repo") or f"{BASE}/{path}.git"
        dest = CLONE_DIR / name
        print(f"discover: NEW repo -> cloning {path}")
        rc = subprocess.run(["git", "clone", "--depth", "200", "-q", url, str(dest)],
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}).returncode
        if rc == 0:
            cloned += 1; have.add(name)
        else:
            print(f"discover: clone failed for {path}")
    print(f"discover: {len(projects)} accessible projects, cloned {cloned} new company repo(s).")


if __name__ == "__main__":
    run()
