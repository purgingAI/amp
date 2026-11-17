"""AMP hub — registry + signed message bus + orchestrator driver + test-client host."""
from __future__ import annotations
import secrets, threading, time, json, os, random
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import store, protocol, orchestrator, suggest, frame, handshake, mdwire, council, autoreply, ace, audit, chdb, understand, knowledge, ask, learn, route, tenancy, insights, daily, tts, ace_auth, palm, handoff


def _set_tenant(frm):
    """Bind this request/worker to the caller's tenant (company). Default -> shared graph."""
    try:
        a = store.get_agent(frm)
        tenancy.set_tenant(tenancy.tenant_for_company((a or {}).get("company")))
    except Exception:
        tenancy.set_tenant(None)


def _mirror(env):
    if chdb.enabled():
        threading.Thread(target=chdb.insert_message, args=(env,), daemon=True).start()
        threading.Thread(target=understand.record_sentiment, args=(env,), daemon=True).start()
import asyncio

AUTO_DELAY = int(os.environ.get("AMP_AUTO_DELAY", "10"))   # seconds of silence before an agent answers
AUTO_CAP = int(os.environ.get("AMP_AUTO_CAP", "6"))        # max auto-turns per thread (stops runaway)
AUTO_COUNT: dict = {}

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="AMP Hub")


@app.on_event("startup")
def _startup():
    store.init()
    if not store.get_agent("orchestrator"):
        store.add_agent("orchestrator", "AMP Orchestrator",
                        "tok_" + secrets.token_hex(12), "sec_" + secrets.token_hex(20),
                        ["query.knowledge", "plan.feature", "implement.feature", "collision.check", "claim.files"])
    suggest.set_client(orchestrator._client)
    autoreply.set_client(orchestrator._client)


def _auth(agent_id: str, token: str):
    a = store.get_agent(agent_id)
    if not a or a["token"] != token:
        raise HTTPException(401, "bad agent/token")
    return a


@app.post("/amp/register")
async def register(req: Request):
    b = await req.json()
    if chdb.enabled():
        raise HTTPException(403, "Open registration is disabled — sign in with your work email; an admin approves access.")
    name = (b.get("name") or "agent").strip()
    base = "".join(ch for ch in name.lower().replace(" ", ".") if ch.isalnum() or ch == ".")[:24] or "agent"
    agent_id = base
    i = 1
    while store.get_agent(agent_id):
        i += 1; agent_id = f"{base}{i}"
    token = "tok_" + secrets.token_hex(12)
    secret = "sec_" + secrets.token_hex(20)
    store.add_agent(agent_id, name, token, secret, b.get("capabilities", ["chat"]))
    # auto-assign skills so ACE can route to this worker immediately
    skills = b.get("skills")
    if not skills:
        for k, v in ace.DEFAULT_SKILLS.items():
            if k in agent_id or k in name.lower():
                skills = v; break
    store.set_profile(agent_id, role=b.get("role", ""), skills=skills)
    return {"agent_id": agent_id, "token": token, "secret": secret, "skills": skills or [],
            "hub": "amp/1", "orchestrator": "orchestrator"}


def _slug(name):
    return "".join(ch for ch in name.lower().replace(" ", ".") if ch.isalnum() or ch == ".")[:24] or "agent"


def _skills_for(agent_id, name):
    for k, v in ace.DEFAULT_SKILLS.items():
        if k in agent_id or k in name.lower():
            return v
    return []


def _unique_agent_id(name):
    base = _slug(name); aid = base; i = 1
    while store.get_agent(aid) or (chdb.enabled() and chdb.get_user(aid)):
        i += 1; aid = f"{base}{i}"
    return aid


def _ensure_store_agent(u):
    """Mirror an approved ClickHouse user into the SQLite store so AMP messaging works."""
    if not store.get_agent(u["agent_id"]):
        store.add_agent(u["agent_id"], u.get("name", ""), u["token"], u["secret"], ["chat"])
        store.set_identity(u["agent_id"], email=u.get("email", ""), company=u.get("company", ""))
        store.set_profile(u["agent_id"], role=u.get("company", ""), skills=_skills_for(u["agent_id"], u.get("name", "")))


@app.post("/amp/login")
async def login(req: Request):
    """Email login. With ClickHouse enabled, access is ADMIN-GATED: unknown users become
    'pending' until an admin approves them (the very first user is auto-admin). Without
    ClickHouse it falls back to open self-signup."""
    b = await req.json()
    email = (b.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "valid email required")
    name = (b.get("name") or email.split("@")[0]).strip()
    company = (b.get("company") or "").strip()

    if chdb.enabled():
        u = chdb.get_user_by_email(email)
        if u:
            if not u.get("approved"):
                return {"status": "pending", "agent_id": u["agent_id"],
                        "message": "Your access is awaiting admin approval."}
            _ensure_store_agent(u); store.touch(u["agent_id"])
            audit.log("login", actor=u["agent_id"], email_domain=email.split("@")[-1])
            return {"agent_id": u["agent_id"], "token": u["token"], "secret": u["secret"],
                    "name": u.get("name", ""), "email": email, "company": u.get("company", ""),
                    "role": u.get("role", "member"), "skills": _skills_for(u["agent_id"], u.get("name", "")),
                    "approved": True, "returning": True}
        first = chdb.count_users() == 0
        agent_id = _unique_agent_id(name)
        token = "tok_" + secrets.token_hex(12); secret = "sec_" + secrets.token_hex(20)
        u = {"agent_id": agent_id, "email": email, "name": name, "company": company,
             "token": token, "secret": secret, "role": "admin" if first else "member",
             "approved": first}
        chdb.upsert_user(u)
        audit.log("login.request", actor=agent_id, email_domain=email.split("@")[-1], approved=bool(first))
        if first:
            _ensure_store_agent(u)
            return {"agent_id": agent_id, "token": token, "secret": secret, "name": name,
                    "email": email, "company": company, "role": "admin",
                    "skills": _skills_for(agent_id, name), "approved": True, "returning": False}
        return {"status": "pending", "agent_id": agent_id,
                "message": "Access requested — an admin needs to approve you."}

    # ---- fallback: no ClickHouse -> open self-signup (dev) ----
    ex = store.get_agent_by_email(email)
    if ex:
        store.touch(ex["agent_id"])
        try: skills = json.loads(ex.get("skills") or "[]")
        except Exception: skills = []
        return {"agent_id": ex["agent_id"], "token": ex["token"], "secret": ex["secret"],
                "name": ex["name"], "email": email, "company": ex.get("company", ""),
                "skills": skills, "returning": True}
    agent_id = _unique_agent_id(name)
    token = "tok_" + secrets.token_hex(12); secret = "sec_" + secrets.token_hex(20)
    store.add_agent(agent_id, name, token, secret, ["chat"])
    store.set_identity(agent_id, email=email, company=company)
    store.set_profile(agent_id, role=company, skills=_skills_for(agent_id, name))
    return {"agent_id": agent_id, "token": token, "secret": secret, "name": name,
            "email": email, "company": company, "skills": _skills_for(agent_id, name), "returning": False}


def _require_admin(b):
    a = _auth(b.get("from"), b.get("token"))
    u = chdb.get_user(a["agent_id"]) if chdb.enabled() else None
    if not u or u.get("role") != "admin":
        raise HTTPException(403, "admin only")
    return a


@app.post("/amp/admin/users")
async def admin_users(req: Request):
    b = await req.json(); _require_admin(b)
    return {"users": chdb.list_users()}


@app.post("/amp/admin/tenants")
async def admin_tenants(req: Request):
    """Admin view of tenants (companies): which have their own graphs + how many users."""
    b = await req.json(); _require_admin(b)
    users = chdb.list_users() if chdb.enabled() else []
    counts: dict = {}
    for u in users:
        counts[tenancy.tenant_for_company(u.get("company"))] = counts.get(tenancy.tenant_for_company(u.get("company")), 0) + 1
    ts = []
    for t in tenancy.list_tenants():
        base = tenancy.DATA if t == tenancy.DEFAULT else tenancy.DATA / "tenants" / t
        ts.append({"tenant": t, "name": tenancy.company_name(t),
                   "knowledge": (base / "knowledge.db").exists(),
                   "code_knowledge": (base / "code_knowledge.db").exists(),
                   "users": counts.get(t, 0), "shared": t == tenancy.DEFAULT})
    return {"tenants": ts}


@app.post("/amp/admin/learnings")
async def admin_learnings(req: Request):
    """Admin view of the feedback-learnings that fine-tune Ask answers."""
    b = await req.json(); _require_admin(b)
    return {"learnings": learn.list_all(200), "stats": learn.stats()}


@app.post("/amp/admin/learnings/delete")
async def admin_learnings_delete(req: Request):
    b = await req.json(); _require_admin(b)
    ok = learn.delete(b.get("id"))
    return {"ok": ok, "stats": learn.stats()}


@app.post("/amp/admin/approve")
async def admin_approve(req: Request):
    b = await req.json(); _require_admin(b)
    u = chdb.get_user(b.get("agent_id"))
    if not u:
        raise HTTPException(404, "no such user")
    chdb.approve(u["agent_id"]); _ensure_store_agent({**u, "approved": 1})
    audit.log("admin.approve", actor=b.get("from"), target=u["agent_id"])
    return {"ok": True, "agent_id": u["agent_id"]}


@app.post("/amp/admin/adduser")
async def admin_adduser(req: Request):
    b = await req.json(); _require_admin(b)
    email = (b.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "valid email required")
    ex = chdb.get_user_by_email(email)
    if ex:
        chdb.approve(ex["agent_id"]); _ensure_store_agent({**ex, "approved": 1})
        return {"ok": True, "agent_id": ex["agent_id"], "existing": True}
    name = (b.get("name") or email.split("@")[0]).strip()
    agent_id = _unique_agent_id(name)
    token = "tok_" + secrets.token_hex(12); secret = "sec_" + secrets.token_hex(20)
    u = {"agent_id": agent_id, "email": email, "name": name, "company": b.get("company", ""),
         "token": token, "secret": secret, "role": b.get("role", "member"), "approved": 1}
    chdb.upsert_user(u); _ensure_store_agent(u)
    audit.log("admin.adduser", actor=b.get("from"), target=agent_id, role=u["role"])
    return {"ok": True, "agent_id": agent_id, "role": u["role"]}


@app.post("/amp/chat")
async def api_chat(req: Request):
    """Streaming AI assistant chat (chunked text/plain). Contextual per-thread; stores both sides."""
    b = await req.json()
    frm = b.get("from"); a = _auth(frm, b.get("token"))
    text = (b.get("text") or "").strip()
    thread = b.get("thread") or protocol.new_id("thr")
    store.store_message(protocol.make(frm, "assistant", "chat", {"text": text}, a["secret"], thread=thread))
    hist = store.thread(thread)[-12:]
    msgs = []
    for m in hist:
        role = "assistant" if m["from"] in ("assistant", "orchestrator", "ace") else "user"
        t = m["body"].get("text", "")
        if t:
            msgs.append({"role": role, "content": t})
    if not msgs:
        msgs = [{"role": "user", "content": text}]

    async def gen():
        loop = asyncio.get_event_loop(); q: asyncio.Queue = asyncio.Queue()

        def worker():
            full = []
            try:
                if not orchestrator._client:
                    raise RuntimeError("no API key")
                with orchestrator._client.messages.stream(
                        model=orchestrator.MODEL, max_tokens=900,
                        system=("You are ACE, a sharp, friendly company engineering assistant. Answer the "
                                "user's messages directly and specifically; keep it concise. Use the "
                                "conversation for context."),
                        messages=msgs) as s:
                    for tok in s.text_stream:
                        full.append(tok); loop.call_soon_threadsafe(q.put_nowait, tok)
                    fm = s.get_final_message()
                    try: audit.llm(orchestrator.MODEL, fm.usage, "chat", actor=frm)
                    except Exception: pass
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, f"\n⚠️ {e}")
            finally:
                sec = store.get_agent("orchestrator")["secret"]
                store.store_message(protocol.make("assistant", frm, "chat", {"text": "".join(full)}, sec, thread=thread))
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            tok = await q.get()
            if tok is None:
                break
            yield tok

    return StreamingResponse(gen(), media_type="text/plain",
                             headers={"X-Thread": thread, "Cache-Control": "no-cache"})


@app.get("/amp/route")
def api_route(topic: str, me: str = ""):
    """Who-to-call: rank the best people to handle a topic/request (code owners + experts + trust)."""
    _set_tenant(me)
    return route.route_expert(topic, requester=me or None)


@app.get("/amp/nudges")
def api_nudges(agent_id: str):
    """Proactive suggestions for an agent: who to connect with / who to ask."""
    _set_tenant(agent_id)
    return {"nudges": insights.nudges(agent_id)}


@app.get("/amp/network")
def api_network():
    """Trust-graph: nodes (agents) + weighted trust edges, for a network view."""
    return insights.network()


@app.get("/amp/knowledge")
def api_knowledge():
    """Company knowledge-graph status + a small overview (freshness, people/projects/tech)."""
    if not knowledge.enabled():
        return {"enabled": False}
    ov = knowledge.overview()
    return {"enabled": True, "freshness": knowledge.freshness(), "counts": ov.get("counts", {}),
            "top_tech": [t["tech"] for t in ov.get("top_tech", [])]}


# ============================================================================
# ACE 2.0 — its OWN login (separate from AMP) + committer-people + code graph.
# The GraphRAG (/amp/ask) is unchanged; the ACE UI just calls it after /ace login.
# ============================================================================
def _ace_user(req: Request):
    """Authenticate an ACE request from X-Ace-Agent/X-Ace-Token headers (own session)."""
    return ace_auth.by_token(req.headers.get("X-Ace-Agent"), req.headers.get("X-Ace-Token"))


@app.post("/ace/login")
async def ace_login(req: Request):
    """ACE's own email login — never touches AMP's user store."""
    b = await req.json()
    try:
        res = ace_auth.login(b.get("email", ""), b.get("name", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return res


@app.get("/ace/me")
async def ace_me(req: Request):
    u = _ace_user(req)
    if not u:
        raise HTTPException(401, "sign in to ACE")
    return {"agent_id": u["agent_id"], "name": u["name"], "email": u["email"], "role": u["role"]}


@app.get("/ace/people")
async def ace_people(req: Request):
    """People = all git committers from the knowledge graph (not who-logged-in)."""
    if not _ace_user(req):
        raise HTTPException(401, "sign in to ACE")
    if not knowledge.enabled():
        return {"enabled": False, "people": []}
    return {"enabled": True, "people": knowledge.list_people(200), "freshness": knowledge.freshness()}


@app.get("/ace/codemap")
async def ace_codemap(req: Request):
    """Code knowledge graph: modules/services with their owners + churn (understands the code)."""
    if not _ace_user(req):
        raise HTTPException(401, "sign in to ACE")
    return {"enabled": knowledge.code_enabled(), "modules": knowledge.code_map(60) if knowledge.code_enabled() else []}


@app.get("/ace/person/{name}")
async def ace_person(name: str, req: Request):
    if not _ace_user(req):
        raise HTTPException(401, "sign in to ACE")
    p = knowledge.get_person(name)
    if isinstance(p, dict) and not p.get("error"):
        p["modules"] = knowledge.person_modules(p.get("name", name))
    return p


@app.get("/ace/overview")
async def ace_overview(req: Request):
    if not _ace_user(req):
        raise HTTPException(401, "sign in to ACE")
    ov = knowledge.overview()
    return {"counts": ov.get("counts", {}), "top_tech": [t["tech"] for t in ov.get("top_tech", [])],
            "freshness": knowledge.freshness(), "code": knowledge.code_enabled()}


@app.get("/ace/daily")
async def ace_daily(req: Request, force: bool = False):
    """Spoken briefing for the mesh core — who's working on what today (narration is TTS-ready).
    Same daily engine as AMP, under ACE auth."""
    u = _ace_user(req)
    if not u:
        raise HTTPException(401, "sign in to ACE")
    _set_tenant(u["agent_id"])
    return daily.briefing(force=force)


@app.post("/ace/tts")
async def ace_tts(req: Request):
    """ElevenLabs voice for the ACE mesh-core briefing (ACE auth) — same provider as AMP."""
    from fastapi.responses import Response
    if not _ace_user(req):
        raise HTTPException(401, "sign in to ACE")
    b = await req.json()
    audio, mime, reason = tts.synth(b.get("text") or "")
    if not audio:
        raise HTTPException(503, reason or "no TTS provider")
    return Response(content=audio, media_type=mime, headers={"Cache-Control": "no-store"})


@app.post("/ace/admin/users")
async def ace_admin_users(req: Request):
    u = _ace_user(req)
    if not u or u.get("role") != "admin":
        raise HTTPException(403, "admin only")
    return {"users": ace_auth.list_users()}


@app.post("/ace/admin/approve")
async def ace_admin_approve(req: Request):
    u = _ace_user(req)
    if not u or u.get("role") != "admin":
        raise HTTPException(403, "admin only")
    b = await req.json()
    return {"ok": ace_auth.approve(b.get("agent_id", ""))}


@app.post("/amp/ask")
async def api_ask(req: Request):
    """Ask-the-company: agentic Q&A grounded in the knowledge graph (streamed text/plain).
    improve=true → the previous answer was 👎-rejected; regenerate a better one and save it as a
    human-validated learning (fine-tunes future answers)."""
    b = await req.json()
    frm = b.get("from"); a = _auth(frm, b.get("token"))
    question = (b.get("text") or b.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "empty question")
    improve = bool(b.get("improve"))
    improve_note = (b.get("reason") or "") if improve else None
    session = (b.get("session") or frm).strip()
    if improve:
        store.drop_last_turns(session, 2)   # forget the rejected Q/A so we regenerate cleanly
    history = store.conversation(session, limit=12)   # prior turns → the agent has context
    store.add_turn(session, frm, "user", question)
    audit.log("ask", actor=frm, q=question[:200], improve=improve, session=session)

    async def gen():
        loop = asyncio.get_event_loop(); q: asyncio.Queue = asyncio.Queue()
        collected: list = []

        def worker():
            _set_tenant(frm)   # bind this worker thread to the caller's tenant graph
            try:
                for ev in ask.run(question, actor=frm, history=history, improve_note=improve_note):
                    if ev["type"] == "text":
                        collected.append(ev["text"])
                        loop.call_soon_threadsafe(q.put_nowait, ev["text"])
                    elif ev["type"] == "tool":
                        loop.call_soon_threadsafe(q.put_nowait, f"\x1f{ev['name']}\x1f")
                    elif ev["type"] == "reset":
                        collected.clear()   # keep only the clean final answer (post-narration)
                        loop.call_soon_threadsafe(q.put_nowait, "\x1e")
                    elif ev["type"] == "error":
                        loop.call_soon_threadsafe(q.put_nowait, f"\n⚠️ {ev['text']}")
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, f"\n⚠️ {e}")
            finally:
                ans = "".join(collected).strip()
                if ans:
                    store.add_turn(session, frm, "assistant", ans)   # remember what the agent said
                    if improve:
                        # the regenerated (better) answer becomes a human-validated learning
                        learn.add(question, ans, "corrected", improve_note or "", frm, validated=1)
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            tok = await q.get()
            if tok is None:
                break
            yield tok

    return StreamingResponse(gen(), media_type="text/plain", headers={"Cache-Control": "no-cache"})


@app.post("/ace/ask")
async def ace_ask(req: Request):
    """ACE 'ask the company' — identical GraphRAG to /amp/ask (ask.run), but authenticated with
    ACE's own login and scoped to an ACE session. The learning flow is unchanged."""
    u = _ace_user(req)
    if not u:
        raise HTTPException(401, "sign in to ACE")
    b = await req.json()
    frm = u["agent_id"]
    question = (b.get("text") or b.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "empty question")
    improve = bool(b.get("improve"))
    improve_note = (b.get("reason") or "") if improve else None
    session = (b.get("session") or ("ace:" + frm)).strip()
    if improve:
        store.drop_last_turns(session, 2)
    history = store.conversation(session, limit=12)
    store.add_turn(session, frm, "user", question)
    audit.log("ace.ask", actor=frm, q=question[:200], improve=improve, session=session)

    async def gen():
        loop = asyncio.get_event_loop(); q: asyncio.Queue = asyncio.Queue()
        collected: list = []

        def worker():
            _set_tenant(frm)
            try:
                for ev in ask.run(question, actor=frm, history=history, improve_note=improve_note):
                    if ev["type"] == "text":
                        collected.append(ev["text"]); loop.call_soon_threadsafe(q.put_nowait, ev["text"])
                    elif ev["type"] == "tool":
                        loop.call_soon_threadsafe(q.put_nowait, f"\x1f{ev['name']}\x1f")
                    elif ev["type"] == "reset":
                        collected.clear(); loop.call_soon_threadsafe(q.put_nowait, "\x1e")
                    elif ev["type"] == "error":
                        loop.call_soon_threadsafe(q.put_nowait, f"\n⚠️ {ev['text']}")
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, f"\n⚠️ {e}")
            finally:
                ans = "".join(collected).strip()
                if ans:
                    store.add_turn(session, frm, "assistant", ans)
                    if improve:
                        learn.add(question, ans, "corrected", improve_note or "", frm, validated=1)
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            tok = await q.get()
            if tok is None:
                break
            yield tok

    return StreamingResponse(gen(), media_type="text/plain", headers={"Cache-Control": "no-cache"})


@app.post("/ace/ask/feedback")
async def ace_ask_feedback(req: Request):
    u = _ace_user(req)
    if not u:
        raise HTTPException(401, "sign in to ACE")
    b = await req.json()
    rating = (b.get("rating") or "").lower()
    if rating not in ("up", "down"):
        raise HTTPException(400, "rating must be up or down")
    learn.add((b.get("question") or "").strip(), (b.get("answer") or "").strip(),
              rating, b.get("reason") or "", u["agent_id"], validated=1 if rating == "up" else 0)
    audit.log("ace.ask.feedback", actor=u["agent_id"], rating=rating)
    return {"ok": True, "rating": rating, "regenerate": rating == "down", "stats": learn.stats()}


@app.post("/amp/ask/feedback")
async def api_ask_feedback(req: Request):
    """Rate an Ask answer. 👍 (up) saves it as a human-validated learning that grounds future
    answers; 👎 (down) records the miss (the client then re-asks with improve=true to regenerate)."""
    b = await req.json()
    frm = b.get("from"); _auth(frm, b.get("token"))
    rating = (b.get("rating") or "").lower()
    if rating not in ("up", "down"):
        raise HTTPException(400, "rating must be up or down")
    question = (b.get("question") or "").strip()
    answer = (b.get("answer") or "").strip()
    reason = b.get("reason") or ""
    learn.add(question, answer, rating, reason, frm, validated=1 if rating == "up" else 0)
    audit.log("ask.feedback", actor=frm, rating=rating, q=question[:120])
    return {"ok": True, "rating": rating, "regenerate": rating == "down", "stats": learn.stats()}


@app.get("/amp/ask/history")
def api_ask_history(agent_id: str, token: str = "", session: str = ""):
    """The remembered chat history the agent uses for context. Defaults to the user's rolling session."""
    a = _auth(agent_id, token)
    sid = (session or agent_id).strip()
    return {"session": sid, "turns": store.conversation(sid, limit=100)}


@app.get("/amp/ask/sessions")
def api_ask_sessions(agent_id: str, token: str = ""):
    a = _auth(agent_id, token)
    return {"sessions": store.list_sessions(agent_id)}


@app.post("/amp/ask/reset")
async def api_ask_reset(req: Request):
    """Start a fresh conversation (clears the remembered turns for this session)."""
    b = await req.json()
    frm = b.get("from"); _auth(frm, b.get("token"))
    sid = (b.get("session") or frm).strip()
    n = store.reset_conversation(sid)
    audit.log("ask.reset", actor=frm, session=sid, cleared=n)
    return {"ok": True, "session": sid, "cleared": n}


@app.get("/amp/tts/status")
def api_tts_status():
    """Is a rich 3rd-party voice configured? (client uses this to decide real audio vs browser fallback)"""
    return tts.status()


@app.post("/amp/tts")
async def api_tts(req: Request):
    """Synthesize speech with the configured 3rd-party provider (British voice). Returns audio/mpeg,
    or 503 with a reason when no provider is configured (client falls back to the browser voice)."""
    from fastapi.responses import Response
    b = await req.json()
    _auth(b.get("from") or b.get("agent_id"), b.get("token"))
    audio, mime, reason = tts.synth(b.get("text") or "")
    if not audio:
        raise HTTPException(503, reason or "no TTS provider")
    return Response(content=audio, media_type=mime, headers={"Cache-Control": "no-store"})


@app.post("/amp/tts/timed")
async def api_tts_timed(req: Request):
    """Rich voice + character-level timestamps (ElevenLabs) so the client can light each person's
    node exactly when their name is spoken. 503 → client falls back to /amp/tts + proportional sync."""
    b = await req.json()
    _auth(b.get("from") or b.get("agent_id"), b.get("token"))
    data, reason = tts.synth_timed(b.get("text") or "")
    if not data:
        raise HTTPException(503, reason or "no timed tts")
    return data


@app.get("/amp/build/suggestions")
def api_build_suggestions(agent_id: str, token: str = ""):
    """Live 'what to build next' chips for the ACE console — grounded in features being built today."""
    _auth(agent_id, token)
    _set_tenant(agent_id)
    return {"suggestions": daily.build_suggestions()}


@app.get("/amp/daily")
def api_daily(agent_id: str, token: str = "", force: bool = False):
    """The spoken daily briefing shown/read on ACE login: what happened today, per-person
    capabilities + what they shipped, dispatched work orders, and activity. `narration` is TTS-ready."""
    a = _auth(agent_id, token)
    _set_tenant(agent_id)
    d = daily.briefing(force=force)
    audit.log("daily.brief", actor=agent_id, people=len(d.get("people", [])))
    return d


TYPING: dict = {}   # thread -> {agent_id: ts}


@app.post("/amp/typing")
async def api_typing(req: Request):
    b = await req.json()
    a = _auth(b.get("from"), b.get("token"))
    thread = b.get("thread") or "thr_lobby"
    TYPING.setdefault(thread, {})[a["agent_id"]] = time.time()
    return {"ok": True}


@app.get("/amp/typing")
def get_typing(thread: str = "thr_lobby", me: str = ""):
    now = time.time()
    d = TYPING.get(thread, {})
    who = [aid for aid, ts in d.items() if now - ts < 4 and aid != me]
    return {"typing": who}


@app.get("/amp/agent")
def api_agent(id: str):
    """Validate/lookup an agent by id (for 'connect by agent id')."""
    a = store.get_agent(id)
    if not a:
        return {"exists": False}
    approved = True
    if chdb.enabled():
        u = chdb.get_user(id)
        approved = bool(u and u.get("approved"))
    online = (time.time() - store._epoch(a.get("last_seen", ""))) < 90
    return {"exists": True, "approved": approved, "name": a.get("name", id),
            "agent_id": id, "online": online}


@app.post("/amp/connect")
async def api_connect(req: Request):
    """Connect to another user by agent id -> a private 1:1 thread. Records the link in ClickHouse."""
    b = await req.json()
    a = _auth(b.get("from"), b.get("token"))
    to = (b.get("to") or "").strip()
    tgt = store.get_agent(to)
    if not tgt:
        raise HTTPException(404, "No agent with that id")
    if chdb.enabled():
        u = chdb.get_user(to)
        if not (u and u.get("approved")):
            raise HTTPException(403, "That user isn't approved")
        chdb.record_connection(a["agent_id"], to)
        chdb.insert_event(a["agent_id"], to, "connect", {"to_name": tgt.get("name", to)})
    thread = "dm_" + "_".join(sorted([a["agent_id"], to]))
    return {"ok": True, "agent_id": to, "name": tgt.get("name", to), "thread": thread,
            "online": (time.time() - store._epoch(tgt.get("last_seen", ""))) < 90}


@app.get("/amp/insight")
def api_insight(agent_id: str, force: bool = False):
    """Understanding card for an agent: mood + top peers (trust) + cached LLM digest."""
    return understand.insight(agent_id, force=force)


@app.get("/amp/relationship")
def api_relationship(a: str, b: str):
    """Trust + rapport between two agents, derived from the message/event firehose."""
    return understand.relationship(a, b)


@app.get("/amp/whoami")
def whoami(agent_id: str, token: str):
    """Validate a stored session. When ClickHouse gating is on, the user must exist AND be approved
    (agents created via the old open /amp/register aren't in ClickHouse -> invalid)."""
    a = store.get_agent(agent_id)
    if not a or a["token"] != token:
        return {"ok": False, "reason": "bad session"}
    if chdb.enabled():
        u = chdb.get_user(agent_id)
        if not u or not u.get("approved"):
            return {"ok": False, "reason": "not approved", "approved": bool(u and u.get("approved"))}
        return {"ok": True, "approved": True, "role": u.get("role", "member"), "agent_id": agent_id}
    return {"ok": True, "approved": True, "role": "member", "agent_id": agent_id}


@app.get("/amp/commits")
def api_commits(limit: int = 40, repo: str = None):
    return {"commits": orchestrator.recent_commits(limit, repo),
            "repos": list(orchestrator.REPOS.keys())}


@app.post("/amp/send")
async def send(req: Request):
    env = await req.json()
    frm = env.get("from"); token = env.get("token")
    a = _auth(frm, token)
    if not protocol.verify(env, a["secret"]):
        raise HTTPException(403, "signature verification failed")
    env.pop("token", None)                    # hub strips bearer before storage/delivery
    store.store_message(env); _mirror(env)
    store.touch(frm)
    # drive orchestrator asynchronously so slow LLM work doesn't block the sender
    if env.get("to") == "orchestrator" and env.get("type") == "REQUEST":
        threading.Thread(target=_run_orchestrator, args=(env,), daemon=True).start()
    _schedule_autoreply(env)
    return {"ok": True, "id": env["id"], "thread": env["thread"], "routed_to": env["to"]}


def _run_orchestrator(env):
    try:
        resp = orchestrator.handle(env)
    except Exception as e:
        audit.log("error", actor="orchestrator", intent=env.get("intent"), error=str(e))
        sec = store.get_agent("orchestrator")["secret"]
        resp = protocol.make("orchestrator", env["from"], env.get("intent", "chat"),
                             {"text": f"⚠️ Orchestrator error: {e}"}, sec, type="ERROR", thread=env["thread"])
    store.store_message(resp)


# ---------------- agent auto-pilot ----------------
def _schedule_autoreply(env):
    """Auto-answer for agents in auto mode. Works for a DIRECT 1:1 recipient AND for LOBBY
    (broadcast): every other auto agent may reply. Capped per thread to avoid runaway."""
    if env.get("intent") != "chat" or not AUTO_ENABLED:
        return
    to = env.get("to")
    if to == "orchestrator":
        return
    if to == "broadcast":
        for d in store.directory():
            rid = d["agent_id"]
            if rid in (env["from"], "orchestrator") or not d.get("online"):
                continue                       # only currently-online agents auto-reply in the lobby
            ag = store.get_agent(rid)
            if ag and ag.get("auto"):
                # jitter so agents don't all fire at once -> first to wake replies, others stand down
                threading.Timer(AUTO_DELAY + random.uniform(0, 2.5), _do_autoreply, args=(env, rid)).start()
    else:
        ag = store.get_agent(to)
        if ag and ag.get("auto"):
            threading.Timer(AUTO_DELAY, _do_autoreply, args=(env, to)).start()


def _do_autoreply(trigger, responder):
    thread = trigger["thread"]
    a = store.get_agent(responder)
    if not a or not a.get("auto"):
        return
    msgs = store.thread(thread)
    if not msgs or msgs[-1]["id"] != trigger["id"]:    # thread already moved on (someone replied) -> stand down
        return                                          # => in the lobby, exactly ONE agent answers per message
    if AUTO_COUNT.get(thread, 0) >= AUTO_CAP:           # stop runaway agent↔agent chatter
        return
    text = _grounded_autoreply(trigger, a) or autoreply.generate(thread, responder, a.get("name"))
    if not text:                                        # natural end / nothing to add
        return
    dest = "broadcast" if trigger.get("to") == "broadcast" else trigger["from"]
    env2 = protocol.make(responder, dest, "chat", {"text": text, "auto": True}, a["secret"], thread=thread)
    store.store_message(env2); store.touch(responder)
    AUTO_COUNT[thread] = AUTO_COUNT.get(thread, 0) + 1
    _schedule_autoreply(env2)                           # let others respond (capped)


def _grounded_autoreply(trigger, responder_agent):
    """Auto-answer AS the employee, grounded in the company knowledge graph. Returns None to
    fall back to the plain (context-only) auto-reply when the graph can't help."""
    if not knowledge.enabled():
        return None
    q = (trigger.get("body", {}) or {}).get("text", "").strip()
    if not q or len(q) < 3:
        return None
    _set_tenant(responder_agent.get("agent_id", ""))   # answer from the employee's tenant graph
    person = knowledge.person_for(responder_agent)
    if not person:
        return None
    me = responder_agent.get("agent_id", "")
    # give the auto-agent the running DM thread so it understands follow-ups, not just this line
    hist, tid = [], trigger.get("thread") or ""
    for m in store.thread(tid):
        if m.get("id") == trigger.get("id"):
            continue   # the current question is appended by run(); don't duplicate it
        txt = ((m.get("body") or {}).get("text") or "").strip()
        if not txt:
            continue
        role = "assistant" if m.get("from") == me else "user"
        if hist and hist[-1]["role"] == role:
            hist[-1]["content"] += "\n\n" + txt
        else:
            hist.append({"role": role, "content": txt})
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    while hist and hist[-1]["role"] != "assistant":
        hist.pop()
    try:
        ans = ask.answer_text(q, as_person=person, actor=me, history=hist[-10:])
        return (ans or None)
    except Exception as e:
        audit.log("error", actor=responder_agent.get("agent_id", ""), where="grounded_autoreply", err=str(e)[:200])
        return None


@app.post("/amp/mode")
async def api_mode(req: Request):
    b = await req.json()
    a = _auth(b.get("from"), b.get("token"))
    store.set_auto(a["agent_id"], bool(b.get("auto")))
    return {"ok": True, "agent_id": a["agent_id"], "auto": bool(b.get("auto"))}


AUTO_ENABLED = True   # global kill switch for all auto-pilot (token saver)


@app.get("/amp/config")
def get_config():
    return {"delay": AUTO_DELAY, "cap": AUTO_CAP, "enabled": AUTO_ENABLED}


@app.post("/amp/config")
async def set_config(req: Request):
    """Live-tune auto-pilot to control token spend (no restart needed)."""
    global AUTO_DELAY, AUTO_CAP, AUTO_ENABLED
    b = await req.json()
    if "delay" in b:   AUTO_DELAY = max(1, min(300, int(b["delay"])))
    if "cap" in b:     AUTO_CAP = max(0, min(50, int(b["cap"])))
    if "enabled" in b: AUTO_ENABLED = bool(b["enabled"])
    return {"delay": AUTO_DELAY, "cap": AUTO_CAP, "enabled": AUTO_ENABLED}


@app.post("/amp/send-lite")
async def send_lite(req: Request):
    """Convenience for the browser test client: token-authed; the hub signs with the
    agent's stored secret (real agents sign themselves via /amp/send)."""
    b = await req.json()
    frm = b.get("from"); token = b.get("token")
    a = _auth(frm, token)
    to_ = b.get("to", "orchestrator")
    thr = b.get("thread") or ("thr_lobby" if to_ == "broadcast" else None)   # one shared lobby thread
    env = protocol.make(frm, to_, b.get("intent", "chat"),
                        b.get("body", {"text": b.get("text", "")}), a["secret"],
                        type=b.get("type", "REQUEST"), thread=thr)
    store.store_message(env); store.touch(frm); _mirror(env)
    audit.log("message", actor=frm, to=env["to"], intent=env["intent"], thread=env["thread"])
    if env["intent"] == "chat" and env["to"] != "orchestrator":
        AUTO_COUNT[env["thread"]] = 0          # human re-engaged -> refresh the auto-turn budget
    if env["to"] == "orchestrator" and env["type"] == "REQUEST":
        threading.Thread(target=_run_orchestrator, args=(env,), daemon=True).start()
    _schedule_autoreply(env)
    return {"ok": True, "id": env["id"], "thread": env["thread"]}


@app.get("/amp/poll")
def poll(agent_id: str, token: str, wait: float = 0):
    _auth(agent_id, token)
    store.touch(agent_id)
    deadline = time.time() + min(float(wait), 25)
    while True:
        msgs = store.fetch_inbox(agent_id)
        if msgs or time.time() >= deadline:
            return {"messages": msgs}
        time.sleep(0.6)


@app.get("/amp/directory")
def directory():
    agents = store.directory()
    if chdb.enabled():
        approved = {u["agent_id"] for u in chdb.list_users() if u.get("approved")}
        # only real, approved users who are currently online (orchestrator is a system agent, hidden)
        agents = [a for a in agents if a.get("online") and a["agent_id"] in approved]
    return {"agents": agents, "claims": store.active_claims()}


@app.get("/amp/thread/{thread_id}")
def thread(thread_id: str):
    return {"thread": thread_id, "messages": store.thread(thread_id)}


@app.get("/amp/spec")
def spec():
    return FileResponse(Path(__file__).resolve().parent.parent / "AMP-SPEC.md")


@app.get("/amp/suggest")
def api_suggest(thread: str, agent: str):
    """Reply recommendations from the whole thread (token-cheap, Haiku)."""
    return {"suggestions": suggest.suggest_replies(thread, agent)}


import base64


@app.get("/amp/demo/handshake")
def demo_handshake():
    """Run a full 3-way handshake between two sample agents and report the wire metrics."""
    A = handshake.AgentCard("sagar.laptop", "claude-" + secrets.token_hex(4), "backend eng",
                            ["plan.feature", "chat"],
                            ["cross-border payout", "IncentiveEngine", "provider_bank_coverage",
                             "FetchAccountTitle", "merge collision", "campaign reserve"])
    B = handshake.AgentCard("alex.laptop", "claude-" + secrets.token_hex(4), "backend eng",
                            ["plan.feature", "chat"],
                            ["cross-border payout", "settlement leg", "provider_bank_coverage",
                             "treasury override", "merge collision"])
    syn, stA = handshake.hello(A)
    synack, sessB = handshake.on_hello(syn, B)
    ack, sessA = handshake.on_hello_ack(synack, stA)
    sessB = handshake.on_ack(ack, sessB)
    # now a real message over the established session, packed with the codebook
    msg = ("# Plan · add /health to HBL\n"
           "Re: the **cross-border payout** work — I'll wire `provider_bank_coverage` into "
           "FetchAccountTitle and check for any **merge collision** on the campaign reserve path.")
    payload, flags = mdwire.pack(msg, sessA["codebook"])
    seg = mdwire.segment(payload, flags, mdwire.T_DATA, sid=bytes.fromhex(sessA["sid"]), seq=1)
    recovered = mdwire.unpack(mdwire.parse(seg)["payload"], mdwire.parse(seg)["flags"], sessB["codebook"])
    return {
        "handshake": {
            "SYN_bytes": len(syn), "SYN_ACK_bytes": len(synack), "ACK_bytes": len(ack),
            "session_id": sessA["sid"], "established_both": sessA["established"] and sessB["established"],
            "codebook_size": len(sessA["codebook"]),
            "codebook_sample": dict(list(sessA["codebook"].items())[:6]),
        },
        "message_roundtrip": {
            "lossless": recovered == msg,
            **mdwire.benchmark(msg, sessA["codebook"]),
            "wire_preview_hex": seg[:24].hex(),
        },
    }


@app.post("/amp/translate")
async def translate(req: Request):
    """Orchestrator-as-broker: interpret a message from one agent's dialect for another."""
    b = await req.json()
    if not orchestrator._client:
        return {"translation": "(no API key)"}
    prompt = (f"You are AMP's language broker between two agents.\n"
              f"SENDER card:\n{b.get('from_card','(unknown)')}\n\nRECIPIENT card:\n{b.get('to_card','(unknown)')}\n\n"
              f"Message from sender:\n{b.get('message','')}\n\n"
              f"Rewrite the message in terms the RECIPIENT will unambiguously understand; expand any "
              f"sender-specific jargon. Keep it short, Markdown.")
    m = orchestrator._client.messages.create(model=orchestrator.MODEL, max_tokens=500,
                                             messages=[{"role": "user", "content": prompt}])
    return {"translation": "".join(x.text for x in m.content if x.type == "text")}


@app.post("/amp/council")
async def api_council(req: Request):
    """Convene a multi-agent deliberation. Turns + conclusion are BROADCAST to every connected
    agent (shared spectacle) — anyone can convene, everyone watches via /amp/poll."""
    b = await req.json()
    topic = (b.get("topic") or "").strip()
    personas = b.get("personas")
    rounds = int(b.get("rounds", 2))
    if not topic:
        return {"ok": False, "error": "topic required"}

    def worker():
        for _ in council.run_council(topic, personas, rounds):
            pass   # run_council broadcasts each event to all agents

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "broadcast": True, "by": b.get("by")}


@app.post("/amp/profile")
async def api_profile(req: Request):
    b = await req.json()
    a = _auth(b.get("from"), b.get("token"))
    store.set_profile(a["agent_id"], role=b.get("role"), skills=b.get("skills"))
    return {"ok": True}


@app.get("/amp/roster")
def api_roster():
    r = store.roster()
    if chdb.enabled():
        approved = {u["agent_id"] for u in chdb.list_users() if u.get("approved")}
        r = [x for x in r if x["agent_id"] in approved]   # only real approved users (online or not)
    return {"roster": r}


@app.post("/amp/ace")
async def api_ace(req: Request):
    """ACE dispatch: understand the request -> what to build + who to call -> assign over AMP."""
    b = await req.json()
    request = (b.get("request") or "").strip()
    requester = b.get("from", "ace")
    res = ace.dispatch(request, requester)
    audit.log("ace.dispatch", actor=requester, workorder=res.get("workorder"),
              request=request, tasks=len(res.get("tasks", [])),
              assignees=[t.get("assignee") for t in res.get("tasks", [])])
    return res


@app.get("/amp/workorders")
def api_workorders():
    return {"workorders": store.list_workorders()}


@app.get("/amp/logs")
def api_logs(limit: int = 100, event: str = None, actor: str = None):
    """Service audit log (login / message / ace.dispatch / task.update / council / llm.usage / error)."""
    return {"events": store.list_audit(limit, event, actor), "stats": store.audit_stats()}


@app.post("/amp/task")
async def api_task(req: Request):
    """Advance a work-order task status (assigned → in-progress → done → blocked)."""
    b = await req.json()
    store.set_task_status(b.get("wo"), int(b.get("index", -1)), b.get("status", "assigned"))
    audit.log("task.update", workorder=b.get("wo"), index=b.get("index"), status=b.get("status"))
    sec = store.get_agent("orchestrator")["secret"]
    store.store_message(protocol.make("ace", "broadcast", "task.update",
                        {"text": f"task #{int(b.get('index',0))+1} → {b.get('status')}",
                         "workorder": b.get("wo")}, sec, thread="thr_ace"))
    return {"ok": True}


@app.post("/amp/admin/skills")
async def api_admin_skills(req: Request):
    """Set a worker's skills (org/skill editor)."""
    b = await req.json()
    store.set_profile(b.get("agent_id"), role=b.get("role"), skills=b.get("skills"))
    return {"ok": True}


MANIFEST = {
    "name": "ACE — Agentic Communication Engine", "short_name": "ACE",
    "start_url": "/ace", "display": "standalone", "background_color": "#0a0b16",
    "theme_color": "#8b7bff", "icons": [
        {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"}]}
MANIFEST_MSG = {**MANIFEST, "name": "AMP Messenger", "short_name": "AMP",
                "start_url": "/", "theme_color": "#37f5a5", "background_color": "#04100c"}
ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#31e6c8"/><stop offset="1" stop-color="#8b7bff"/></linearGradient></defs>'
            '<rect width="100" height="100" rx="22" fill="#0a0b16"/>'
            '<circle cx="50" cy="50" r="26" fill="none" stroke="url(#g)" stroke-width="8"/>'
            '<circle cx="50" cy="24" r="7" fill="#31e6c8"/><circle cx="74" cy="63" r="7" fill="#8b7bff"/>'
            '<circle cx="26" cy="63" r="7" fill="#ff6ad5"/></svg>')
SW_JS = ("self.addEventListener('install',e=>self.skipWaiting());"
         "self.addEventListener('activate',e=>self.clients.claim());"
         "self.addEventListener('fetch',()=>{});")


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse(MANIFEST)


@app.get("/manifest.msg.webmanifest")
def manifest_msg():
    return JSONResponse(MANIFEST_MSG)


@app.get("/icon.svg")
def icon():
    from fastapi.responses import Response
    return Response(ICON_SVG, media_type="image/svg+xml")


@app.get("/sw.js")
def sw():
    from fastapi.responses import Response
    return Response(SW_JS, media_type="application/javascript")


# ============================================================================
# Jarvis knowledge handoff — a capsule carried between machines on your palm.
# Hand geometry says WHO is presenting; the ACE session says which agent is
# signed in here. Both are required: the biometric's measured false-accept rate
# is ~7% (tools/palm_bench.py), which identifies a person far too loosely to be
# the only thing standing between two agents' memories.
# ============================================================================

def _handoff_secret() -> str:
    """Signing key for handoff envelopes.

    From the environment so it survives a restart — a key regenerated on boot would
    invalidate every capsule still in flight, and worse, would make `verify` fail in a way
    that reads as tampering. The fallback exists so a dev box runs without setup; it is
    per-process, so capsules do not survive a restart there, which is the honest behaviour
    for a key nobody configured.
    """
    return os.environ.get("ACE_HANDOFF_SECRET") or _EPHEMERAL_HANDOFF_SECRET


_EPHEMERAL_HANDOFF_SECRET = "eph_" + secrets.token_hex(20)


# The palm pages get their OWN login path, not /ace/login. On the public box Caddy sends
# everything except these palm routes to the Go ACE, which has its own account store and its
# own X-Ace-Token. Sharing /ace/login would mean the page authenticating against one server
# and calling another, and — because the Go console keys its session on "ace2.session" too —
# signing into the palm tool would have silently logged the user out of ACE itself.
@app.post("/palm/login")
async def palm_login(req: Request):
    b = await req.json()
    try:
        return ace_auth.login(b.get("email", ""), b.get("name", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/palm/me")
async def palm_me(req: Request):
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    return {"agent_id": u["agent_id"], "name": u.get("name"), "role": u.get("role"),
            "approved": u.get("approved", 1)}


@app.get("/palm")
def palm_console():
    """Enrolment and measurement UI. Same MediaPipe pipeline the product uses, so the
    landmarks enrolled here are the landmarks the gesture layer will see."""
    return FileResponse(WEB / "palm.html")


@app.post("/ace/palm/enrol")
async def ace_palm_enrol(req: Request):
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    subject = (b.get("subject") or u.get("name") or u.get("agent_id") or "").strip()
    samples = b.get("samples") or []
    if not subject:
        return JSONResponse({"error": "subject required"}, status_code=400)
    try:
        return handoff.enrol_palm(subject, u["agent_id"], samples, meta=b.get("meta"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ace/palm/identify")
async def ace_palm_identify(req: Request):
    """Who is this hand? Returns the numbers — distance, margin, runners-up — not a verdict,
    so the measurement page can plot the distributions rather than count yes/no."""
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    try:
        return handoff.identify(b.get("landmarks") or [], threshold=float(b.get("threshold") or palm.DEFAULT_THRESHOLD))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/ace/palm/enrolments")
async def ace_palm_enrolments(req: Request):
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    return {"enrolments": handoff.enrolments_summary(), "threshold": palm.DEFAULT_THRESHOLD,
            "min_margin": handoff.MIN_MARGIN, "min_samples": palm.MIN_SAMPLES}


@app.post("/ace/palm/forget")
async def ace_palm_forget(req: Request):
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    subject = (b.get("subject") or "").strip()
    if not subject:
        return JSONResponse({"error": "subject required"}, status_code=400)
    return handoff.forget_palm(subject)


@app.post("/ace/palm/score")
async def ace_palm_score(req: Request):
    """Distance to every enrolment — what the FAR/FRR measurement needs."""
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    try:
        return handoff.score_all(b.get("landmarks") or [])
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ace/handoff/grab")
async def ace_handoff_grab(req: Request):
    """Pinch and pull: lift a topic out of this agent and hold it on the palm."""
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    try:
        r = handoff.grab(u["agent_id"], b.get("topic") or "", b.get("landmarks") or [], _handoff_secret())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return r if r.get("ok") else JSONResponse(r, status_code=409)


@app.post("/ace/handoff/release")
async def ace_handoff_release(req: Request):
    """Open palm on the other machine: deliver to the agent signed in there."""
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    b = await req.json()
    try:
        r = handoff.release(u["agent_id"], b.get("landmarks") or [], _handoff_secret())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return r if r.get("ok") else JSONResponse(r, status_code=409)


@app.get("/ace/handoff/in-flight")
async def ace_handoff_in_flight(req: Request):
    u = _ace_user(req)
    if not u:
        return JSONResponse({"error": "login required"}, status_code=401)
    return {"in_flight": handoff.in_flight(req.query_params.get("subject"))}


@app.get("/ace")
def ace_console():
    return FileResponse(WEB / "ace.html")


@app.get("/ace2")
def ace2_console():
    return FileResponse(WEB / "ace2.html")


@app.get("/db")
def db_map_page():
    return FileResponse(WEB / "db.html")


@app.get("/amp/dbmap")
def api_dbmap(agent_id: str = "", token: str = ""):
    """Table → product-feature map of APP_DB_dev (+ discard candidates). Manager view."""
    import json as _json
    p = Path(__file__).resolve().parent.parent / "data" / "db_map.json"
    if not p.exists():
        return {"database": None, "tables": [], "features": [], "unused": [], "infra": [],
                "error": "map not built yet"}
    try:
        return _json.load(open(p))
    except Exception as e:
        raise HTTPException(500, f"db_map read error: {e}")


@app.get("/get")
def get_page():
    """Simple install page — Chrome blocks .apk over http, so we also offer .zip (rename to .apk)."""
    html = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Install ACE</title>
<style>body{background:#0a0b16;color:#eef1ff;font-family:system-ui;text-align:center;padding:44px 20px}
a{display:block;margin:14px auto;max-width:340px;background:#8b7bff;color:#fff;text-decoration:none;
padding:16px;border-radius:14px;font-weight:600}a.msg{background:#37f5a5;color:#04100c}
small{color:#9aa0cc;display:block;margin-top:10px}h2{letter-spacing:2px}.logo{width:56px;height:56px;border-radius:16px;margin:0 auto 14px;background:conic-gradient(from 200deg,#31e6c8,#8b7bff,#ff6ad5,#31e6c8)}</style>
<div class=logo></div><h2>Install</h2>
<a href="https://localhost/ACE.apk">⬇ ACE — business app</a>
<a class=msg href="https://localhost/AMP-Messenger.apk">⬇ AMP Messenger</a>
<small>Tap a button → download → open the file → allow "install unknown apps" → Install.<br>
Served over HTTPS, so Chrome installs it directly. 🔒</small>
"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(html)


@app.get("/ACE.zip")
def dl_ace():
    return FileResponse(WEB / "ACE.apk", media_type="application/zip", filename="ACE.zip")


@app.get("/AMP-Messenger.zip")
def dl_msg():
    return FileResponse(WEB / "AMP-Messenger.apk", media_type="application/zip", filename="AMP-Messenger.zip")


@app.get("/")
def root():
    return FileResponse(WEB / "client.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
