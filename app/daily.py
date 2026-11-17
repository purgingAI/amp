"""Daily briefing — 'what happened today' across the agentic company, spoken on ACE login.

Assembles real signals: commits landed today (grouped by author), work orders + tasks dispatched,
per-person capabilities (inverted from the code-understanding graph owners), and activity counts.
An LLM writes a short, TTS-friendly spoken narration. Cached per-day so re-opens are token-free.
"""
from __future__ import annotations
import os, re, json, datetime, collections
from . import store, codegraph, audit

try:
    from . import orchestrator as _orc
    from . import knowledge
except Exception:                      # pragma: no cover
    _orc = None; knowledge = None
_client = getattr(_orc, "_client", None)

BRIEF_MODEL = os.environ.get("AMP_BRIEF_MODEL", "claude-haiku-4-5-20251001")
BRIEF_TTL = int(os.environ.get("AMP_BRIEF_TTL", "600"))   # 10 min (matches push_commits cadence)
_cache: dict = {}   # date-iso -> (epoch, payload)


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _today():
    return _now().date()


def _is_today(iso: str | None) -> bool:
    if not iso:
        return False
    try:
        return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).date() == _today()
    except Exception:
        return False


def _capabilities_by_person(limit_mods: int = 400) -> dict:
    """Invert code-graph module owners → person → the modules & capabilities they own."""
    if not codegraph.enabled():
        return {}
    people: dict = collections.defaultdict(lambda: {"modules": [], "caps": set(), "commits": 0})
    for m in codegraph.list_modules(limit=limit_mods):
        for o in (m.get("owners") or [])[:3]:
            name = o.get("name") if isinstance(o, dict) else o
            if not name:
                continue
            p = people[name]
            p["modules"].append(m["module"])
            p["commits"] += int((o.get("commits") if isinstance(o, dict) else 0) or 0)
            for c in (m.get("capabilities") or [])[:3]:
                p["caps"].add(c)
    return {n: {"modules": v["modules"][:6], "capabilities": sorted(v["caps"])[:6], "weight": v["commits"]}
            for n, v in people.items()}


def _canon_key(name: str) -> str:
    """Collapse author-name variants to one key: split camelCase, strip separators, lowercase.
    'SamLee'/'Sam Lee' → 'samlee'; 'alexmorgan'/'Alex Morgan' → 'alexmorgan'."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name or "")
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def _display_name(variants: set) -> str:
    """Pick the most human display form among an author's spelling variants."""
    best = None
    for v in variants:
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", v)
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", s) if t]
        if not toks:
            continue
        cand = " ".join(t[:1].upper() + t[1:] for t in toks)
        score = (len(toks) >= 2) * 2 + sum(1 for t in toks if t[:1].isupper())
        if best is None or score > best[0] or (score == best[0] and len(cand) > len(best[1])):
            best = (score, cand)
    return best[1] if best else "someone"


# the user's PERSONAL projects — excluded from the company briefing (mirrors codegraph_build.DENY)
_NON_COMPANY = set(filter(None, (os.environ.get("AMP_NONCOMPANY_REPOS") or
    "trading-ai,creator-os,vantage,employee-tracker,ace_app,amp,amp-ios,team-lens,cba,citybank,"
    "citybank-cbt,wallet-rmc").split(",")))


def _company_commits() -> list:
    """The commit feed with personal projects filtered out — only company product repos."""
    return [c for c in (_orc.recent_commits(limit=400) if _orc else [])
            if (c.get("repo") or "") not in _NON_COMPANY]


def _commits_on(day_iso: str) -> dict:
    """Commits on a given YYYY-MM-DD, grouped by person with author-alias variants merged."""
    groups = collections.defaultdict(list)
    variants = collections.defaultdict(set)
    for c in _company_commits():
        if not (c.get("date") or "").startswith(day_iso):
            continue
        author = c.get("author") or "someone"
        key = _canon_key(author) or author.lower()
        if len(key) < 2:                      # skip junk authors like a lone "s"
            continue
        groups[key].append({"repo": c.get("repo"), "subject": c.get("subject")})
        variants[key].add(author)
    return {_display_name(variants[k]): v[:8] for k, v in groups.items()}


def _latest_commit_day() -> str | None:
    days = [(c.get("date") or "")[:10] for c in _company_commits() if c.get("date")]
    return max(days) if days else None


def _workorders_today() -> list:
    out = []
    for w in store.list_workorders(limit=40):
        if not _is_today(w.get("ts")):
            continue
        tasks = w.get("tasks") or []
        out.append({"request": (w.get("request") or "")[:140], "requester": w.get("requester"),
                    "tasks": [{"title": t.get("title"), "assignee": t.get("assignee_name") or t.get("assignee"),
                               "status": t.get("status", "assigned")} for t in tasks][:8]})
    return out[:8]


def _activity_today() -> dict:
    ev = store.list_audit(limit=400)
    today = [e for e in ev if _is_today(e.get("ts"))]
    c = collections.Counter(e.get("event") for e in today)
    return {"asks": c.get("ask", 0), "messages": c.get("message", 0),
            "dispatches": c.get("ace.dispatch", 0), "logins": c.get("login", 0) + c.get("login.new", 0),
            "total_events": len(today)}


# commit-verb → where a feature sits in its lifecycle (a hint; the LLM makes the final call)
_STAGE = [
    (("release", "deploy", "ship", "rollout", "go-live", "golive", "bump", "tag"), "shipping"),
    (("test", "qa", "smoke", "verify", "coverage", "e2e"), "testing"),
    (("fix", "bug", "edge", "hotfix", "patch", "revert", "issue", "error", "fail", "regression"), "stabilizing"),
    (("refactor", "cleanup", "clean", "polish", "rename", "tidy", "restructure"), "refining"),
    (("add", "implement", "new", "initial", "scaffold", "introduce", "create", "build", "wip", "feat", "support"), "building"),
]


def _stage(subjects: list) -> str:
    text = " ".join(subjects).lower()
    counts = {}
    for kws, label in _STAGE:
        c = sum(text.count(k) for k in kws)
        if c:
            counts[label] = counts.get(label, 0) + c
    return max(counts, key=counts.get) if counts else "in progress"


def _features_for(subjects: list) -> list:
    """Ask the CODE-UNDERSTANDING graph what feature/module today's commits are advancing."""
    if not codegraph.enabled() or not subjects:
        return []
    mods = codegraph.search_capabilities(" ".join(subjects)[:400], limit=3)
    feats = []
    for m in mods:
        feats.append({"module": m["module"], "purpose": (m.get("purpose") or "")[:160],
                      "capabilities": (m.get("matched_capabilities") or m.get("capabilities") or [])[:3]})
    return feats


def collect() -> dict:
    """Structured 'today' snapshot — ONLY people who actually shipped work today (real commits),
    each enriched with the part of the company's tech they own. Deliberately excludes anything that
    came from the ACE console (dispatched work orders, questions asked) — that is not 'work done'."""
    today = _today().isoformat()
    commits = _commits_on(today)
    as_of = today
    if not commits:                            # nothing committed yet today → most recent working day
        ld = _latest_commit_day()
        if ld:
            commits = _commits_on(ld)
            as_of = ld
    caps = _capabilities_by_person()
    people = []
    for name in sorted(commits, key=lambda n: -len(commits[n])):
        ci = commits[name]
        subs = [c["subject"] for c in ci]
        cap = caps.get(name) or _fuzzy_caps(name, caps)
        people.append({
            "name": name,
            "shipped_today": subs[:8],
            "repos": sorted({c["repo"] for c in ci if c.get("repo")})[:4],
            "commit_count": len(ci),
            "capabilities": (cap or {}).get("capabilities", []),
            "owns_modules": (cap or {}).get("modules", []),
            "features": _features_for(subs),   # what the graph says these commits are advancing
            "stage": _stage(subs),             # lifecycle hint from commit verbs
        })
    return {"date": today, "as_of": as_of, "is_today": as_of == today, "generated_at": _now().isoformat(),
            "people": people, "active_count": len(people), "team_size": len(caps)}


_BUILD_FALLBACK = ["Add a /health endpoint with a smoke test",
                   "Add end-to-end tests for the most active module",
                   "Surface a recent backend change in the admin UI"]


def build_suggestions() -> list:
    """Live 'what to build next' — grounded in the team's RECENT commits + key modules (in-flight work)."""
    import time
    key = "buildsug:" + _today().isoformat()
    if key in _cache and time.time() - _cache[key][0] < BRIEF_TTL:
        return _cache[key][1]
    subs = [c.get("subject") for c in _company_commits()[:26] if c.get("subject")]
    mods = []
    if codegraph.enabled():
        for m in codegraph.list_modules(limit=10):
            mods.append(f"{m['module']}: {(m.get('purpose') or '')[:90]}")
    out = _BUILD_FALLBACK
    if subs and _client is not None:
        prompt = ("An engineering team's RECENT commits (what they are building right now):\n"
                  + "\n".join("- " + s for s in subs[:26])
                  + (("\n\nKey modules in the codebase:\n" + "\n".join("- " + m for m in mods)) if mods else "")
                  + "\n\nPropose exactly 3 concrete, specific NEXT tasks that would advance or complement this "
                    "in-progress work. Each: one imperative line, <=14 words, buildable by one engineer "
                    "(e.g. 'Add duplicate-IBAN validation to the ENBD settlement batch'). Return ONLY the 3 lines, "
                    "no numbering or extra text.")
        try:
            msg = _client.messages.create(model=BRIEF_MODEL, max_tokens=220,
                                          messages=[{"role": "user", "content": prompt}])
            audit.llm(BRIEF_MODEL, msg.usage, "build_suggest", "ace")
            txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            lines = [re.sub(r"^[\s\-\d\.\)•]+", "", l).strip() for l in txt.splitlines() if l.strip()]
            out = lines[:3] or _BUILD_FALLBACK
        except Exception as ex:
            audit.log("error", "ace", where="build_suggest", err=str(ex)[:150])
    _cache[key] = (time.time(), out)
    return out


def _fuzzy_caps(name: str, caps: dict) -> dict:
    """Commit author names vary (e.g. 'dev' vs 'Sagar Uniyal') — best-effort match."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    for n, v in caps.items():
        k2 = "".join(ch for ch in n.lower() if ch.isalnum())
        if key and (key in k2 or k2 in key):
            return v
    return {}


def _spoken(txt: str) -> str:
    """Strip markdown so a TTS voice reads clean prose (no '#', '*', bullets, etc.)."""
    import re
    txt = re.sub(r"^#{1,6}\s*", "", txt, flags=re.M)     # headings
    txt = re.sub(r"[*_`]{1,3}", "", txt)                   # bold/italic/code
    txt = re.sub(r"^\s*[-•]\s*", "", txt, flags=re.M)      # bullet markers
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _narrate(data: dict) -> str:
    people = data["people"]
    if not people:
        return ("No commits have landed today yet, so there's no engineering work to report. "
                "I'll brief you the moment the team starts pushing code.")
    if _client is None:
        # deterministic fallback narration (no LLM)
        n = len(people)
        parts = [f"{n} {'person has' if n == 1 else 'people have'} shipped work today."]
        for p in people[:8]:
            what = "; ".join(p["shipped_today"][:2]) or "code changes"
            area = ", ".join(p["capabilities"][:2]) or ", ".join(p["owns_modules"][:2])
            parts.append(f"{p['name']} made {p['commit_count']} commit"
                         f"{'s' if p['commit_count'] != 1 else ''}: {what}." + (f" Their area is {area}." if area else ""))
        return " ".join(parts)
    lines = []
    for p in people[:9]:
        feat = "; ".join(f"{f['module']} ({f['purpose']})" for f in p.get("features", [])[:2]) or "—"
        lines.append(
            f"- {p['name']}: {p['commit_count']} commits, stage-signal '{p['stage']}'.\n"
            f"    commits: {'; '.join(p['shipped_today'][:6])}\n"
            f"    feature/module context (from the code-understanding graph): {feat}\n"
            + (f"    owns/leads: {', '.join(p['owns_modules'][:3])}" if p["owns_modules"] else "    owns/leads: —"))
    when = ("today" if data.get("is_today", True)
            else f"nothing has landed yet today, so this is the most recent working day, {data.get('as_of')}")
    prompt = (
        "You are the voice of ACE, an agentic engineering company that UNDERSTANDS its own codebase — it knows "
        "each module's purpose and who owns it. Write a SPOKEN daily briefing read aloud to a manager. "
        f"Timeframe: {when}. {data['active_count']} of {data['team_size']} people shipped work.\n"
        + ("" if data.get("is_today", True) else
           "Open by briefly noting there are no commits yet today so you're recapping the most recent day.\n") + "\n"
        "For each person you get their real commits, a lifecycle 'stage-signal' inferred from the commit verbs "
        "(building / testing / stabilizing / refining / shipping), and the FEATURE/MODULE their commits map to "
        "in the code graph (with what that module actually does).\n\n"
        "DATA:\n" + "\n".join(lines) + "\n\n"
        "Write ~180 words of warm, concise, PLAIN SPOKEN PROSE (no markdown, no bullets, no emojis). "
        "Open with a one-line headline. Then ONE-TO-TWO sentences per person that DO NOT just list commits — "
        "instead SYNTHESISE: (1) name the FEATURE they're advancing, described using the code graph's "
        "understanding of that module (what it does), not the raw commit words; (2) give a PROGRESS read — how "
        "far along it looks — grounded in the stage-signal and the number/nature of today's commits (e.g. "
        "'just kicking off', 'actively building it out', 'stabilising with fixes and tests', 'polishing', or "
        "'shipping it'). Be specific and show you understand the tech. "
        "STRICT: cover ONLY the people listed. Never mention work orders, dispatched tasks, console questions, "
        "or activity counts. Do not invent features not supported by the data.")
    try:
        msg = _client.messages.create(model=BRIEF_MODEL, max_tokens=560,
                                      messages=[{"role": "user", "content": prompt}])
        audit.llm(BRIEF_MODEL, msg.usage, "daily_brief", "ace")
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return _spoken(txt)
    except Exception as ex:
        audit.log("error", "ace", where="daily_brief", err=str(ex)[:200])
        n = len(people)   # deterministic fallback if the LLM call fails
        return (f"{n} {'person' if n == 1 else 'people'} shipped work today. "
                + " ".join(f"{p['name']} made {p['commit_count']} commits." for p in people[:8]))


def briefing(force: bool = False) -> dict:
    key = _today().isoformat()
    import time
    if not force and key in _cache and time.time() - _cache[key][0] < BRIEF_TTL:
        return _cache[key][1]
    data = collect()
    data["narration"] = _narrate(data)
    _cache[key] = (time.time(), data)
    return data
