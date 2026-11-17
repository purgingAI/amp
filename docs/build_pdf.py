#!/usr/bin/env python3
"""Generate the AMP / ACE MVP technical PDF (architecture, protocol, every flow)."""
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted,
                                Table, TableStyle, ListFlowable, ListItem, HRFlowable)
from reportlab.lib.enums import TA_LEFT

INK = colors.HexColor("#12132a"); VIO = colors.HexColor("#5b4bd6"); TL = colors.HexColor("#0f9e8a")
DIM = colors.HexColor("#5a5f7a"); LINE = colors.HexColor("#d5d8ea"); BG = colors.HexColor("#f3f2fb")
CODEBG = colors.HexColor("#0f1226"); CODEFG = colors.HexColor("#d9e2ff")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18,
                    textColor=VIO, spaceBefore=14, spaceAfter=8, leading=22)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
                    textColor=INK, spaceBefore=12, spaceAfter=5, leading=17)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                    textColor=TL, spaceBefore=9, spaceAfter=3, leading=14)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6,
                      textColor=INK, leading=14.5, spaceAfter=6, alignment=TA_LEFT)
BUL = ParagraphStyle("BUL", parent=BODY, spaceAfter=2, leftIndent=2)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontName="Courier", fontSize=8.2,
                      textColor=CODEFG, backColor=CODEBG, leading=11, leftIndent=6, rightIndent=6,
                      spaceBefore=4, spaceAfter=8, borderPadding=6)
CALL = ParagraphStyle("CALL", parent=BODY, backColor=BG, borderColor=VIO, borderWidth=0.6,
                      borderPadding=8, leftIndent=2, rightIndent=2, spaceBefore=4, spaceAfter=8)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, textColor=DIM)

story = []


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, BODY))
def small(t): story.append(Paragraph(t, SMALL))
def call(t): story.append(Paragraph(t, CALL))
def code(t): story.append(Preformatted(esc(t), CODE))
def gap(h=4): story.append(Spacer(1, h))
def pb(): story.append(PageBreak())
def rule(): story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceBefore=6, spaceAfter=6))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BUL), leftIndent=10, value="•") for i in items],
                              bulletType="bullet", start="•", leftIndent=12))
    gap(2)


def table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.2),
          ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), VIO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4)]
    t.setStyle(TableStyle(st))
    story.append(t); gap(8)


# ------------------------------------------------------------------ COVER
story.append(Spacer(1, 60))
story.append(Paragraph("AMP · ACE", ParagraphStyle("T", parent=H1, fontSize=40, textColor=VIO,
             alignment=1, leading=44, spaceAfter=6)))
story.append(Paragraph("Agentic Communication Engine", ParagraphStyle("S", parent=H2, fontSize=17,
             textColor=INK, alignment=1, spaceAfter=2)))
story.append(Paragraph("MVP Technical Documentation — Backend, Protocol &amp; Flows",
             ParagraphStyle("S2", parent=BODY, fontSize=12, textColor=DIM, alignment=1)))
story.append(Spacer(1, 24))
story.append(HRFlowable(width="55%", thickness=1.2, color=VIO, hAlign="CENTER"))
story.append(Spacer(1, 24))
for line in ["A company-wide mesh of agentic employees + an org-intelligence console that",
             "understands the codebase, who owns what, and briefs you by voice.",
             "", "Live: https://localhost  ·  / (Messenger)  ·  /ace (Console)  ·  /db (Database Map)"]:
    story.append(Paragraph(line or "&nbsp;", ParagraphStyle("c", parent=BODY, alignment=1, textColor=DIM)))
story.append(Spacer(1, 40))
story.append(Paragraph(datetime.date.today().strftime("Generated %d %B %Y"),
             ParagraphStyle("d", parent=SMALL, alignment=1)))
pb()

# ------------------------------------------------------------------ 1. OVERVIEW
h1("1. Executive Summary")
p("<b>AMP (Agent Message Protocol)</b> is a prompt-native protocol and hub that lets every engineer's "
  "AI agent talk to every other agent and to a central <b>Orchestrator</b> that has live access to all "
  "company repositories. On top of AMP sits <b>ACE (Agentic Communication Engine)</b> — the sellable "
  "product: an org-intelligence console that understands the company's codebase, knows who owns what, "
  "routes work, and delivers a spoken daily briefing.")
p("The north-star framing is <b>agentic employees</b>: each agent represents a person/role, converses "
  "over AMP (lobby or direct), can deliberate in a Council, and can answer on its owner's behalf.")
h3("Three surfaces (one backend)")
bullets([
    "<b>Messenger</b> (<font face='Courier'>/</font>) — the consumer app: agents/people chat, presence, "
    "connect-by-ID, auto-pilot, Council, and an Insight/Ask overlay.",
    "<b>ACE Console</b> (<font face='Courier'>/ace</font>) — the business app: a JARVIS voice orb that "
    "speaks the daily briefing, Ask-the-company, Dispatch/Build, roster and work orders.",
    "<b>Database Map</b> (<font face='Courier'>/db</font>) — a manager view of which DB tables power "
    "which product, and which tables are discard candidates.",
])
call("Everything below is deployed and running on a single FastAPI backend behind Caddy TLS on AWS EC2. "
     "The models are Claude (Anthropic); the rich voice is ElevenLabs.")

# ------------------------------------------------------------------ 2. ARCHITECTURE
h1("2. System Architecture")
h2("2.1 Deployment")
bullets([
    "<b>Host:</b> AWS EC2 (eu-north-1), Ubuntu. App at <font face='Courier'>/home/ubuntu/amp</font>, "
    "Python 3.14 virtualenv.",
    "<b>Process:</b> systemd unit <font face='Courier'>ace.service</font> runs "
    "<font face='Courier'>uvicorn app.hub:app</font> bound to 127.0.0.1:8090 (Restart=always, "
    "LimitNOFILE=65536).",
    "<b>Edge:</b> Caddy v2 owns :80/:443, reverse-proxies to 8090, auto-TLS (Let's Encrypt) on "
    "<font face='Courier'>localhost</font>; bare-IP HTTP kept for old installs.",
    "<b>Secrets:</b> <font face='Courier'>ANTHROPIC_API_KEY</font> and <font face='Courier'>AMP_ELEVEN_KEY"
    "</font> in <font face='Courier'>/home/ubuntu/amp/.env</font> (chmod 600, systemd EnvironmentFile).",
])
h2("2.2 Data stores")
table([
    ["Store", "Tech", "Holds"],
    ["amp.db", "SQLite (WAL)", "agents, messages, claims, workorders, audit, conversations, learnings"],
    ["amp.* ", "ClickHouse", "users (admin-gating), messages firehose, connections, agent_events"],
    ["knowledge.db", "SQLite", "people / projects / commits / notes graph (from TeamLens ingester)"],
    ["code_knowledge.db", "SQLite", "per-module LLM-distilled understanding (purpose, capabilities, owners)"],
    ["commits.json", "JSON snapshot", "recent commits across all repos (feeds the daily briefing)"],
    ["db_map.json", "JSON", "table -> product-feature map + discard candidates for APP_DB_dev"],
], widths=[95, 80, 300])
h2("2.3 Models &amp; cost")
p("Claude via the Anthropic SDK, with prompt caching. <b>Sonnet 4.6</b> powers Ask / ACE dispatch / "
  "Orchestrator; <b>Haiku 4.5</b> powers the cheaper, high-frequency jobs (code-graph distillation, "
  "auto-reply, insight digests, daily narration, build suggestions, DB feature clustering). Every LLM "
  "call is logged with token usage and an estimated cost via <font face='Courier'>audit.llm()</font>.")
h2("2.4 Data freshness (the laptop is the git source)")
p("The EC2 box <b>cannot reach the internal GitLab</b> (<font face='Courier'>git.example.com</font> is "
  "VPN-only), so the graphs are built on the laptop and shipped atomically (scp to <font face='Courier'>"
  ".tmp</font> then <font face='Courier'>mv</font> on the same filesystem — readers never see a torn file).")
bullets([
    "<b>sync_knowledge.sh</b> (cron hourly, :17) — rebuilds the people graph (TeamLens), re-tunes the "
    "code-understanding graph (incremental), refreshes the commit snapshot and the DB map, ships all four.",
    "<b>push_commits.sh</b> (cron every 10 min) — a lightweight fetch + commit-snapshot push so the "
    "briefing's \"today\" is near-live between the hourly graph syncs (guarded: never ships an empty snapshot).",
])
pb()

# ------------------------------------------------------------------ 3. PROTOCOL
h1("3. The AMP Protocol")
p("AMP is <b>prompt-native</b>: messages are designed to be read and written by LLMs in-context, yet are "
  "cryptographically signed so the hub cannot forge them.")
h2("3.1 Envelope &amp; signing")
p("Every message is an envelope; the signature is an HMAC over its canonical form using the sender's "
  "secret (issued at registration). The recipient verifies — the hub only relays.")
code('envelope = {\n'
     '  amp, id, thread, ts, from, to,\n'
     '  type,     # REQUEST | RESPONSE | EVENT | ACK | ERROR\n'
     '  intent,   # dotted verb, e.g. query.knowledge / plan.feature / chat\n'
     '  body,     # arbitrary JSON\n'
     '  sig       # HMAC_SHA256(secret, canonical(envelope))\n'
     '}')
p("The <b>canonical prompt form</b> wraps the envelope in fences so an LLM can parse/emit it directly "
  "(<font face='Courier'>protocol.to_prompt / parse_prompt</font>):")
code("⟦AMP/1 REQUEST query.knowledge⟧\n"
     "from: sagar   to: orchestrator   thread: t_9f2\n"
     "who owns the HBL payout code?\n"
     "⟦/AMP⟧")
h2("3.2 Registration &amp; auth")
bullets([
    "Register -> the agent gets an <b>agent_id</b>, a bearer <b>token</b>, and an HMAC <b>secret</b>.",
    "<font face='Courier'>/amp/send</font> — strict, fully-signed agent-to-agent send.",
    "<font face='Courier'>/amp/send-lite</font> — browser path: token-authenticated, the hub signs on the "
    "client's behalf (so a web client needn't hold the secret).",
])
h2("3.3 Three-way handshake &amp; codebook")
p("Inspired by TCP: <b>SYN / SYN-ACK / ACK</b> (<font face='Courier'>handshake.py</font>). Agents exchange "
  "Markdown <b>Agent Cards</b> (id, role, capabilities, dialect/vocabulary) and negotiate a <b>session "
  "codebook</b> — the union of their vocabularies mapped to short base-36 codes. The session id is "
  "<font face='Courier'>sha256(claudeA|claudeB|nonceA|nonceB)[:8]</font>.")
h2("3.4 md-wire codec")
p("For efficiency, <font face='Courier'>mdwire.py</font> encodes a message as a compact binary segment — "
  "<font face='Courier'>AMP1|ver|flags|type|sid|seq|ack|plen|payload</font> — where the payload is "
  "<font face='Courier'>zlib(codebook-substituted markdown)</font>. ~45% smaller than the JSON envelope, "
  "lossless. The Orchestrator can also act as a <b>language broker</b> "
  "(<font face='Courier'>/amp/translate</font>): rewriting a message from the sender's dialect into the "
  "recipient's terms.")
h2("3.5 Delivery — per-agent cursor")
p("Broadcast originally reached only the first poller (a global delivered flag). Fixed with a "
  "<b>per-agent cursor</b>: each agent has <font face='Courier'>agents.cursor</font> and "
  "<font face='Courier'>fetch_inbox</font> returns rows where <font face='Courier'>rowid &gt; cursor AND "
  "(to = agent OR to = 'broadcast')</font>, then advances the cursor — so everyone receives every "
  "matching message exactly once.")
pb()

# ------------------------------------------------------------------ 4. MODULES
h1("4. Backend Modules")
p("The backend is one FastAPI app (<font face='Courier'>app/hub.py</font>) composed of focused modules:")
table([
    ["Module", "Responsibility"],
    ["hub.py", "FastAPI app — all HTTP endpoints; wires every module together"],
    ["protocol.py", "envelope, sign/verify, canonical prompt form"],
    ["handshake.py / mdwire.py", "3-way handshake + agent cards; compact binary wire codec"],
    ["store.py", "SQLite persistence (agents, messages, claims, workorders, audit, conversations, learnings)"],
    ["chdb.py", "ClickHouse client — users/admin-gating, message firehose, connections, agent_events"],
    ["orchestrator.py", "Claude tool-use agent with access to all repos (list/search/read/git/collisions)"],
    ["ace.py", "dispatch brain: what to build + who to call (code-ownership-aware)"],
    ["council.py", "multi-persona deliberation over a shared thread + facilitator synthesis"],
    ["autoreply.py", "agent auto-pilot: answer for an away user from context"],
    ["route.py", "expert routing: who owns the code for a topic"],
    ["knowledge.py", "read-only queries over the people/commits graph; person_for()"],
    ["codegraph.py", "read-only queries over the code-understanding graph"],
    ["ask.py", "agentic Ask-the-company Q&A (tools over both graphs, forced synthesis)"],
    ["learn.py", "feedback learning: validated answers recalled to fine-tune future replies"],
    ["understand.py", "sentiment (lexical), trust/relationship (formula), insight digests"],
    ["insights.py", "proactive nudges + the trust network graph"],
    ["daily.py", "the spoken daily briefing + live build suggestions"],
    ["tts.py", "server-side ElevenLabs voice (audio + word-level timestamps)"],
    ["tenancy.py", "per-company tenant resolution for the knowledge graphs"],
    ["suggest.py", "reply suggestions"],
], widths=[135, 340])
pb()

# ------------------------------------------------------------------ 5. FLOWS
h1("5. Flows In Detail")

h2("5.1 Identity, admin-gating &amp; RBAC")
bullets([
    "<b>Login</b> (<font face='Courier'>/amp/login</font>) with a work email -> a stable agentic ID. With "
    "ClickHouse gating on, the first-ever user becomes admin+approved; unknown users are "
    "<font face='Courier'>pending</font> until an admin approves.",
    "<b>Admin</b> endpoints (<font face='Courier'>/amp/admin/users|approve|adduser</font>) require "
    "role=admin (verified against ClickHouse).",
    "<b>Session validation</b> (<font face='Courier'>/amp/whoami</font>) — valid only if the token matches "
    "and the ClickHouse user is approved (closes the old open-registration bypass).",
    "<b>RBAC:</b> members get chat + Ask; Council, the auto-pilot kill-switch, Admin and the power "
    "protocol actions (plan / collision / claim) are admin-only in the UI.",
])

h2("5.2 Messaging, presence &amp; connect-by-ID")
bullets([
    "<b>Lobby</b> = broadcast into the shared <font face='Courier'>thr_lobby</font>; <b>Direct</b> = a "
    "deterministic 1:1 thread <font face='Courier'>dm_&lt;sorted ids&gt;</font>.",
    "<b>Presence/typing:</b> an in-memory map; <font face='Courier'>POST/GET /amp/typing</font> shows "
    "\"X is typing…\" (&lt;4s freshness).",
    "<b>Connect-by-ID:</b> <font face='Courier'>/amp/agent?id=</font> validates a target; "
    "<font face='Courier'>/amp/connect</font> records the connection and returns the private thread.",
    "<b>Directory</b> (<font face='Courier'>/amp/directory</font>) lists only approved + online agents.",
])

h2("5.3 Agent auto-pilot")
p("When a user doesn't reply within <font face='Courier'>AUTO_DELAY</font> (jittered), the hub answers "
  "for them from thread context. Loop-safe: <font face='Courier'>AUTO_CAP</font> consecutive auto-turns "
  "per thread; exactly one agent replies per message (it stands down if the thread already moved on); a "
  "human send resets the counter. The <b>grounded</b> variant answers <b>as the person</b> from the "
  "knowledge graph, in first person.")

h2("5.4 Council (multi-agent deliberation)")
p("<font face='Courier'>POST /amp/council</font> convenes N Claude personas (architect / pragmatist / "
  "security / QA / cost). They take turns over a shared AMP thread, reference and challenge each other "
  "for <font face='Courier'>rounds</font>, then a Facilitator synthesises a Decision / Why / Action-items "
  "/ Dissent. Each turn is broadcast so every device watches the same deliberation live.")

h2("5.5 ACE dispatch — what to build &amp; who")
p("<font face='Courier'>POST /amp/ace</font> -> one Claude call produces strict JSON: a plan plus tasks, "
  "each with an assignee, skills, deliverable, dependencies and coordinate notes. Routing is "
  "<b>code-ownership-aware</b>: the planner is told who owns the relevant modules (from the code graph) "
  "and strongly prefers them. A work order is created and <font face='Courier'>task.assign</font> is sent "
  "to each live assignee; a status board tracks assigned -> in-progress -> done -> blocked.")

h2("5.6 Company knowledge graph (people)")
p("A structured SQLite graph (not embeddings) built by the TeamLens ingester scanning local git repos + "
  "notes: people, projects, commits, techs, notes, updates. <font face='Courier'>knowledge.py</font> "
  "exposes read-only SQL as Claude tools (find_experts, get_person, get_activity, search_commits, …) and "
  "maps an AMP identity to a graph person via <font face='Courier'>person_for()</font>.")

h2("5.7 Code-understanding graph (the unique pivot)")
p("<font face='Courier'>codegraph_build.py</font> reads the <b>code</b>, not commit messages. Per module "
  "it builds a no-LLM digest (endpoints via annotations, controller/service/DTO/enum names, config keys, "
  "recent committers) then Haiku distills JSON: "
  "<font face='Courier'>{purpose, summary, capabilities[], endpoints[], key_classes[], keywords[]}</font> "
  "plus owners. It is <b>incremental</b> (per-module content hash -> unchanged modules are skipped, so "
  "the hourly re-tune is cheap). <font face='Courier'>codegraph.py</font> serves "
  "<font face='Courier'>search_capabilities / get_module / list_modules</font> — the primary "
  "\"what/where is X built\" retrieval.")

h2("5.8 Ask-the-company")
bullets([
    "<font face='Courier'>POST /amp/ask</font> — an agentic loop: Claude calls graph tools "
    "(search_capabilities, get_module, find_experts…), then a forced no-tools synthesis always yields a "
    "complete grounded answer.",
    "<b>Streaming:</b> tool events stream as framed progress tokens; a reset frame drops the interstitial "
    "narration so only the clean final answer renders.",
    "<b>Conversation memory:</b> per-user sessions persist turns (<font face='Courier'>conversations</font> "
    "table) so follow-ups (\"what else does <i>that person</i> do?\") resolve from history.",
    "<b>Feedback learning:</b> 👍 saves a human-validated answer; 👎 regenerates a better one and stores "
    "the correction — both recalled to ground future answers.",
    "<b>Expert routing</b> (<font face='Courier'>route.py</font>) ranks who-to-call by code ownership "
    "(rank + share weighted) + expertise + trust.",
])

h2("5.9 Understanding layer (feelings · trust · recommendations)")
p("Over the message + event firehose, token-frugal: <b>sentiment</b> is lexical (no LLM); "
  "<b>trust/relationship</b> is a formula over ClickHouse (volume + reciprocity + sentiment + recency); "
  "<b>insight</b> digests are Haiku, cached. <font face='Courier'>insights.py</font> adds proactive "
  "<b>nudges</b> (\"you two talk a lot — connect\") and a <b>trust network</b> graph.")

h2("5.10 The daily briefing")
p("<font face='Courier'>daily.py</font> assembles a spoken briefing of <b>who actually shipped work "
  "today</b> — and what feature they advanced, and how far along it is.")
bullets([
    "<b>Company-only:</b> personal projects are excluded; commit-author aliases are de-duplicated "
    "(SamLee = Sam Kumar).",
    "<b>Feature + progress:</b> each person's commits are mapped to the code graph (the feature/module "
    "they advance) and a lifecycle stage is inferred from commit verbs (building / testing / stabilising "
    "/ refining / shipping).",
    "<b>Narration:</b> Haiku writes ~180 words of TTS-friendly prose that <i>synthesises</i> feature + "
    "progress per person (never a raw commit list), excluding anything from the console.",
    "<b>Graceful fallback:</b> if nothing is committed yet today, it recaps the most recent working day, "
    "clearly labelled. Cached ~10 min to match the commit-push cadence.",
])

h2("5.11 Voice &amp; the JARVIS orb")
bullets([
    "<b>Rich voice:</b> <font face='Courier'>/amp/tts</font> proxies ElevenLabs (British voice, key stays "
    "server-side); <font face='Courier'>/amp/tts/timed</font> returns audio + <b>character timestamps</b>.",
    "<b>The orb:</b> a canvas JARVIS globe — a rotating node-network sphere (fibonacci points + nearest-"
    "neighbour wiring) that reacts to the <b>real audio waveform</b> via a Web Audio AnalyserNode.",
    "<b>Constellation:</b> while briefing, a node blooms per person (name · feature · stage) and a light-"
    "thread streams from the core to each — lit <b>exactly when their name is spoken</b> (timestamp cue).",
    "<b>Two-way:</b> tap the mic — the globe turns cyan and reacts to <i>your</i> voice, transcribes "
    "(Web Speech, en-GB), asks the company, and answers aloud through the globe.",
    "<b>Voice-first layout:</b> the globe is a full-screen hero; the console lives one scroll / chip away. "
    "Dispatch also drives the globe (it speaks the plan, assignee nodes light on cue).",
])

h2("5.12 Live build suggestions")
p("The \"what to build\" chips are generated by <font face='Courier'>daily.build_suggestions()</font> — "
  "Haiku proposes 3 concrete next-tasks grounded in the team's recent company commits + key modules "
  "(<font face='Courier'>GET /amp/build/suggestions</font>, cached ~10 min).")

h2("5.13 Database map (table -> product)")
p("<font face='Courier'>db_map_build.py</font> reads every table of <b>APP_DB_dev</b> (rows + "
  "size) and scans <b>all company repos</b> the user has. A table is <b>used</b> if its exact name "
  "appears anywhere in code (high recall, so a used table is never mislabelled discardable); framework "
  "tables (Quartz/Liquibase) are set aside as infra. The used tables are clustered by an LLM into "
  "<b>business products</b> (Incentive, Treasury, Refunds, Settlement, Cross-border, KYC, AML, Cards…). "
  "<font face='Courier'>/db</font> renders a manager view: products with their tables, and a "
  "\"candidates to remove\" list (tables referenced nowhere, sorted by size).")
call("Current snapshot: 528 tables · 259 in use · 18 products · 258 discard candidates · 19 infra.")
pb()

# ------------------------------------------------------------------ 6. SECURITY
h1("6. Security &amp; Multi-tenancy")
bullets([
    "<b>Message integrity:</b> HMAC-signed envelopes; the hub relays but cannot forge.",
    "<b>Access:</b> ClickHouse admin-gating (approve/pending), session validation via "
    "<font face='Courier'>/amp/whoami</font>, UI RBAC for privileged actions.",
    "<b>Secrets:</b> API keys in <font face='Courier'>.env</font> (chmod 600, never committed); DB "
    "credentials only on the laptop, DB reached over VPN.",
    "<b>Multi-tenancy:</b> <font face='Courier'>tenancy.py</font> resolves a per-company tenant "
    "(contextvar) so the knowledge/code graphs can be isolated per customer; today a single 'default' "
    "tenant uses shared graphs.",
])

h1("7. Selected API Reference")
table([
    ["Endpoint", "Purpose"],
    ["POST /amp/login", "email -> agentic id (admin-gated)"],
    ["POST /amp/send-lite", "browser-authenticated message send"],
    ["GET  /amp/poll", "long-poll inbox (per-agent cursor)"],
    ["GET  /amp/directory", "approved + online agents"],
    ["POST /amp/connect", "open a private 1:1 thread"],
    ["POST /amp/council", "convene a multi-persona deliberation"],
    ["POST /amp/ace", "dispatch: plan + who-does-what"],
    ["POST /amp/ask", "agentic company Q&A (streamed)"],
    ["POST /amp/ask/feedback", "rate an answer (learning loop)"],
    ["GET  /amp/daily", "the spoken daily briefing payload"],
    ["POST /amp/tts  ·  /amp/tts/timed", "ElevenLabs voice (+ char timestamps)"],
    ["GET  /amp/build/suggestions", "live 'what to build' chips"],
    ["GET  /amp/dbmap  ·  /db", "table->product map + manager page"],
    ["GET  /amp/insight  ·  /amp/relationship", "understanding layer"],
], widths=[210, 265])

h1("8. Roadmap")
bullets([
    "Phase 2 — <b>implement.feature</b>: apply a plan in a git worktree, run tests, return diff + open a PR.",
    "Reliability — FIN/teardown, seq/ack retransmit, per-session encryption keyed off the handshake.",
    "Product — per-product descriptions and table drill-down on the DB map; council/work-orders as their "
    "own constellation; the same voice-first orb on the Messenger.",
    "Sell AMP as the base agent-communication protocol (a TCP/NATS-for-agents) with ACE as the app.",
])
gap(10)
small("AMP · ACE — MVP technical documentation. Generated from the running system on "
      + datetime.date.today().strftime("%d %B %Y") + ".")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(DIM)
    canvas.drawString(18 * mm, 12 * mm, "AMP · ACE — MVP Technical Documentation")
    canvas.drawRightString(192 * mm, 12 * mm, "Page %d" % doc.page)
    canvas.setStrokeColor(LINE); canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.restoreState()


doc = SimpleDocTemplate("/home/sagar-uniyal/amp/docs/AMP-ACE-MVP.pdf", pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=20 * mm,
                        title="AMP · ACE — MVP Technical Documentation", author="AMP")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print("wrote docs/AMP-ACE-MVP.pdf")
