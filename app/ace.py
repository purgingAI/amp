"""ACE — Agentic Communication Engine: the routing/dispatch brain on top of AMP.

Give it a request; it decides WHAT needs to get built, breaks it into tasks, picks WHO to call
(matching each task to the agent/person whose skills fit), flags cross-task coordination, creates
a work order, and dispatches AMP `task.assign` messages so the team's agents build it together.
"""
from __future__ import annotations
import json
from . import store, protocol, orchestrator, audit, codegraph

MODEL = orchestrator.MODEL

# Fallback org expertise so ACE can name the right specialist even if they're not online yet.
ORG_EXPERTS = {
    "Sagar":     "HBL/UBL cross-border payouts, Incentive Engine, provider_bank_coverage, backend",
    "Alex":     "Transaction core, settlement legs, treasury, backend",
    "Udita":     "Angular frontend, merchant onboarding UI",
    "Riley":   "Refund maker-checker, frontend, RBAC",
    "Randhir":   "Transaction validators, cross-border fees, backend",
    "Chinmay":   "TransactionUtility, Kafka, event pipelines",
    "Divya":     "User service, KYC, ClickHouse user-segmentation",
    "Aaqib":     "Platform / DevOps / ECS / proxies",
    "Pratiksha": "Incentive reports frontend, campaign ledger",
}

DEFAULT_SKILLS = {
    "sagar": ["backend", "cross-border", "hbl", "incentive-engine"],
    "alex": ["backend", "transaction", "settlement", "treasury"],
    "udita": ["frontend", "angular", "ui"],
    "riley": ["frontend", "refund", "rbac"],
    "randhir": ["backend", "validators", "fees"],
    "chinmay": ["backend", "kafka", "pipelines"],
    "divya": ["user-service", "kyc", "clickhouse"],
    "aaqib": ["devops", "infra", "ecs"],
    "pratiksha": ["frontend", "reports"],
}


def _roster_text():
    live = store.roster()
    lines, live_ids = [], set()
    for a in live:
        live_ids.add(a["agent_id"])
        sk = ", ".join(a["skills"]) or "(no skills set)"
        lines.append(f"- agent_id `{a['agent_id']}` — {a['name']} [{a.get('role') or 'member'}] "
                     f"· skills: {sk} · {'ONLINE' if a['online'] else 'offline'}")
    # org experts not currently registered as live agents
    for name, exp in ORG_EXPERTS.items():
        if not any(name.lower() in a["name"].lower() or name.lower() in a["agent_id"] for a in live):
            lines.append(f"- person `person:{name}` — {name} (not yet on ACE) · expertise: {exp}")
    return "\n".join(lines) or "(no one available)"


def _ownership_text(request: str) -> str:
    """Who actually owns the code for this request (from the code-understanding graph)."""
    if not codegraph.enabled():
        return ""
    mods = codegraph.search_capabilities(request, limit=4)
    if not mods:
        return ""
    lines = []
    for m in mods:
        owners = ", ".join(o["name"] for o in m.get("owners", [])[:3])
        lines.append(f"- `{m['module']}` ({(m.get('purpose') or '')[:80]}) — owned by: {owners or 'unknown'}")
    return "CODE OWNERSHIP (who wrote the relevant modules — strongly prefer assigning to these people):\n" + "\n".join(lines)


ROUTER_SYS = """You are ACE, an Agentic Communication Engine that coordinates a team of agents to build
software together. Given a work request and the available roster (live agents + known experts), you:
1) state WHAT needs to get built (one tight paragraph),
2) break it into 2-6 concrete tasks,
3) assign each task to the BEST-matching worker — STRONGLY prefer whoever OWNS the relevant code
   (see CODE OWNERSHIP), then fall back to skills; use their exact `agent_id` if they have a live
   agent, else `person:Name` for a known owner/expert, else "UNASSIGNED",
4) give each task a clear deliverable and note dependencies + any cross-task coordination risk.
Return STRICT JSON only:
{"build": "...", "tasks": [{"title": "...", "assignee": "agent_id|person:Name|UNASSIGNED",
"assignee_name": "...", "skills": ["..."], "deliverable": "...", "depends_on": -1, "coordinate": "" }]}"""


def plan(request: str) -> dict:
    if not orchestrator._client:
        return {"build": "(no ANTHROPIC_API_KEY)", "tasks": []}
    own = _ownership_text(request)
    content = f"ROSTER:\n{_roster_text()}\n\n" + (own + "\n\n" if own else "") + f"REQUEST:\n{request}"
    msg = orchestrator._client.messages.create(
        model=MODEL, max_tokens=1200, system=ROUTER_SYS,
        messages=[{"role": "user", "content": content}])
    audit.llm(MODEL, msg.usage, "ace.plan")
    txt = "".join(b.text for b in msg.content if b.type == "text")
    s, e = txt.find("{"), txt.rfind("}")
    try:
        data = json.loads(txt[s:e + 1])
    except Exception:
        data = {"build": txt[:400], "tasks": []}
    data.setdefault("build", "")
    for t in data.get("tasks", []):
        t.setdefault("status", "assigned")
        t.setdefault("coordinate", "")
        t.setdefault("depends_on", -1)
    return data


def dispatch(request: str, requester: str = "ace") -> dict:
    """Plan + create a work order + send AMP task.assign messages to live assignees."""
    data = plan(request)
    wo_id = protocol.new_id("wo")
    sec = store.get_agent("orchestrator")["secret"]
    store.add_workorder(wo_id, request, data.get("build", ""), data.get("tasks", []), requester)

    # broadcast the plan, then assign each task to its live agent
    _send(sec, "broadcast", "ace.plan",
          {"text": f"🧩 ACE work order {wo_id}: {data.get('build','')[:280]}",
           "workorder": wo_id})
    for i, t in enumerate(data.get("tasks", [])):
        who = t.get("assignee", "")
        if who and not who.startswith("person:") and who != "UNASSIGNED" and store.get_agent(who):
            _send(sec, who, "task.assign",
                  {"text": f"📌 Task for you: {t.get('title')}\nDeliverable: {t.get('deliverable','')}"
                           + (f"\n⚠ Coordinate: {t['coordinate']}" if t.get('coordinate') else ""),
                   "workorder": wo_id, "task_index": i})
    return {"workorder": wo_id, **data}


def _send(sec, to, intent, body):
    env = protocol.make("ace", to, intent, body, sec, thread="thr_ace")
    store.store_message(env)
