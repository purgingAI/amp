#!/usr/bin/env python3
"""Agent Studio — the agent graph, Laya routing, document collections and training jobs.

Listens on 127.0.0.1 only. The Learn page's serve.py is the front door: it checks the sign-in and
forwards /api/studio/* here with X-Studio-User / X-Studio-Admin. Changing anything needs an admin;
looking and testing needs only a sign-in.

    ~/amp/laya/.venv/bin/python app.py --port 8467
"""
from __future__ import annotations

import argparse
import hashlib
import queue
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import actions                                                # noqa: E402
import inbox                                                  # noqa: E402
import learning                                               # noqa: E402
import shadow                                                 # noqa: E402
import space                                                  # noqa: E402
import faith                                                  # noqa: E402
import academy                                                # noqa: E402
import context                                               # noqa: E402
import changelog                                             # noqa: E402
import doctor                                                # noqa: E402
import rollout_guard                                         # noqa: E402
import tune                                                  # noqa: E402
import answer_cache                                          # noqa: E402
import drift                                                 # noqa: E402
import brain                                                 # noqa: E402
import bus                                                   # noqa: E402
import arena                                                  # noqa: E402
import autopilot                                              # noqa: E402
import calibrate                                              # noqa: E402
import flowver                                                # noqa: E402
import forge                                                  # noqa: E402
import gaps                                                   # noqa: E402
import gates                                                  # noqa: E402
import suites                                                 # noqa: E402
import judges                                                 # noqa: E402
import genesis                                                # noqa: E402
import constitution                                           # noqa: E402
import passport                                               # noqa: E402
import oracle                                                 # noqa: E402
import whynot                                                 # noqa: E402
import immune                                                 # noqa: E402
import convo                                                  # noqa: E402
import diagnose                                               # noqa: E402
import ledger                                                 # noqa: E402
import xray                                                   # noqa: E402
import jobs                                                   # noqa: E402
import rag                                                    # noqa: E402
import receipts                                               # noqa: E402
import redteam                                                # noqa: E402
import tts                                                    # noqa: E402
import recipes                                                # noqa: E402
import evolution                                              # noqa: E402
import insight                                                # noqa: E402
import timemachine                                            # noqa: E402
import router                                                 # noqa: E402
import store                                                  # noqa: E402


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    # ── helpers ──────────────────────────────────────────────────────────────────────
    def _json(self, code, obj):
        if self.command == "POST" and code == 200:
            try:
                self._announce(obj)
            except Exception:                               # noqa: BLE001 — telling others is never worth a failed request
                pass
        b = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self, limit=65536):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b""
        return raw

    def _jbody(self):
        if not hasattr(self, "_jb"):                        # the body can be read once; handlers and _announce share it
            try:
                self._jb = json.loads(self._body() or b"{}")
            except ValueError:
                self._jb = {}
        return self._jb

    def _announce(self, obj) -> None:
        """Every open Studio hears about a change the moment it succeeds (bus.py → GET /events)."""
        p = urlparse(self.path).path.rstrip("/")
        if "sandbox=1" in self.path:
            return
        b = getattr(self, "_jb", {}) if isinstance(getattr(self, "_jb", None), dict) else {}
        u = self.user
        if p == "/graph":
            v = (obj.get("graph") or {}).get("_version") or {}
            bus.publish("flow", u, text=f"saved the flow{' as v' + str(v.get('n')) if v.get('n') else ''}")
        elif p == "/versions/promote":
            bus.publish("promote", u, text=f"promoted {'retriever' if b.get('kind') == 'retriever' else 'reflex'} {b.get('id')}", model=b.get("kind"))
        elif p == "/learning":
            ch = ", ".join(f"{k} {'on' if v is True else 'off' if v is False else v}" for sec in b.values() if isinstance(sec, dict) for k, v in sec.items())
            if "rollout" in b:
                ch = f"set the customer-app rollout to {b['rollout'].get('percent')}%"
            bus.publish("settings", u, text=ch or "changed learning settings")
        elif p == "/calibration/apply":
            bus.publish("settings", u, text=f"turned calibrated confidence {'on' if b.get('on') else 'off'}")
        elif p == "/tune/apply":
            bus.publish("flow", u, text=f"set the reflex bar to {round(float(b.get('bar', 0)) * 100)}%")
        elif p == "/inbox/label":
            bus.publish("label", u, text=f"taught a sentence → {b.get('agent')}")
        elif p == "/golden":
            bus.publish("golden", u, text="added a golden sentence")
        elif p in ("/judges/referee/calibrate", "/judges/faith/calibrate"):
            bus.publish("settings", u, text=f"calibrated the {p.split('/')[2]} judge against people's decisions")
        elif re.fullmatch(r"/golden/[\w-]+/role", p):
            bus.publish("golden", u, text="graduated a case to the regression suite" if b.get("role") == "regression" else "returned a case to the capability suite")
        elif re.fullmatch(r"/gaps/\w+/publish", p):
            bus.publish("knowledge", u, text=f"published {b.get('name') or 'an FAQ'} to close a gap")
        elif p == "/actions/switch":
            bus.publish("promote", u, text=f"is switching the intent model to {b.get('run')}", model="intent")
        elif re.fullmatch(r"/flow/versions/\d+/restore", p):
            bus.publish("flow", u, text=f"restored flow v{p.split('/')[3]}")
        elif p == "/timemachine/apply":
            if (b.get("fix") or {}).get("kind") == "teach":
                bus.publish("golden", u, text="locked a fix in from the Time Machine")
            else:
                bus.publish("flow", u, text="rewound a decision and applied a fix from the Time Machine")
        elif p == "/labels/remove":
            bus.publish("label", u, text="removed a harmful label (Label Doctor)")
        elif p == "/redteam/generate":
            bus.publish("golden", u, text=f"armed the red team with fresh {b.get('family') or ''} attacks")
        elif p == "/autopilot/run":
            bus.publish("settings", u, text="launched an Autopilot mission")
        elif re.fullmatch(r"/autopilot/m_\d+/promote", p):
            pass                                            # autopilot.promote announces itself
        elif p == "/labels/add":
            bus.publish("label", u, text="added a training example (Dataset Doctor)")

    def _events(self):
        q = bus.subscribe()
        if q is None:
            return self._json(503, {"error": "too many open studios"})
        self._sse_start()
        try:
            self._sse({"kind": "hello", "at": time.time()})
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    ev = {"kind": "ping"}                   # keeps proxies from closing a quiet stream
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            bus.unsubscribe(q)

    @property
    def user(self):
        return self.headers.get("X-Studio-User") or "local"

    @property
    def admin(self):
        return self.headers.get("X-Studio-Admin") == "1"

    def _need_admin(self):
        if not self.admin:
            self._json(403, {"ok": False, "error": "only an admin can change the studio"})
            return True
        return False

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _sse(self, ev):
        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n".encode())
        self.wfile.flush()

    # ── routes ───────────────────────────────────────────────────────────────────────
    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path.rstrip("/"), parse_qs(u.query)
        try:
            if p == "/status":
                v = store.versions()
                return self._json(200, {"laya": router.laya_state(), "mem_gb": round(router._mem_available_gb(), 1),
                                        "router": v["router"]["active"],
                                        "collections": rag.collections(), "running": jobs.running(), "admin": self.admin})
            if p == "/academy":
                return self._json(200, academy.progress(self.user, self.admin))
            if p == "/arena":
                return self._json(200, arena.build((q.get("a") or [""])[0], (q.get("b") or [""])[0]))
            if p == "/calibration":
                return self._json(200, calibrate.get())
            if p == "/flow/versions":
                return self._json(200, {"versions": flowver.listing()})
            m = re.fullmatch(r"/flow/versions/(\d+)", p)
            if m:
                v = flowver.get(int(m.group(1)))
                return self._json(200 if v else 404, v or {"error": "no such version"})
            if p == "/graph":
                g = router.sandbox_graph(self.user) if (q.get("sandbox") or [""])[0] else router.graph()
                ready = {n["id"]: dict(zip(("ok", "why"), router.readiness(n))) for n in g["nodes"] if n["type"] == "agent"}
                secrets = {n["id"]: sorted(router.secret_headers(n["id"])) for n in g["nodes"] if n.get("kind") == "webhook"}
                return self._json(200, {"graph": g, "kinds": router.KINDS, "ready": ready, "secret_names": secrets})
            if p == "/collections":
                return self._json(200, {"collections": rag.collections()})
            m = re.fullmatch(r"/collections/([\w-]+)", p)
            if m:
                meta = rag.meta(m.group(1))
                if not meta:
                    return self._json(404, {"error": "no such collection"})
                docs = [{k: d[k] for k in ("id", "name", "bytes", "by", "added")} | {"chunks": len(d["chunks"]),
                        "pages": max([c.get("page") or 0 for c in d["chunks"]] or [0]) or None,
                        "preview": [c["text"][:150] for c in d["chunks"][:3]]} for d in meta["docs"]]
                qf = os.path.join(store.DATA, "collections", m.group(1), "questions.jsonl")
                nq = sum(len(json.loads(l)["questions"]) for l in open(qf, encoding="utf-8")) if os.path.exists(qf) else 0
                return self._json(200, {"id": m.group(1), "name": meta["name"], "docs": docs, "questions": nq})
            m = re.fullmatch(r"/collections/([\w-]+)/map", p)
            if m:
                return self._json(200, self._map(m.group(1)))
            if p == "/ledger":
                return self._json(200, {"rows": ledger.recent(int((q.get("limit") or [50])[0]), (q.get("source") or [None])[0]),
                                        "stats": ledger.stats()})
            m = re.fullmatch(r"/ledger/(\d+)", p)
            if m:
                r = ledger.get(int(m.group(1)))
                return self._json(200 if r else 404, r or {"error": "no such decision"})
            if p == "/receipts":
                return self._json(200, dict(receipts.listing(int((q.get("limit") or [60])[0])), coverage=receipts.coverage()))
            m = re.fullmatch(r"/receipts/([0-9a-f]{64})", p)
            if m:
                rc = receipts.get(m.group(1))
                return self._json(200 if rc else 404, rc or {"error": "no such receipt"})
            if p == "/learning":
                return self._json(200, dict(learning.status(), shadow=shadow.status()))
            if p == "/golden/threads":
                return self._json(200, gates.run_threads())
            m = re.fullmatch(r"/gaps/(\w+)/draft", p)
            if m:
                return self._json(200, gaps.draft(m.group(1)))
            if p == "/forge":
                return self._json(200, forge.snapshot())
            if p == "/redteam":
                return self._json(200, redteam.snapshot())
            if p == "/tts/status":
                return self._json(200, tts.status())
            if p == "/tts":                                 # mp3 for one pitch sentence; only admins may spend credits on a new one
                audio, how = tts.get((q.get("t") or [""])[0], (q.get("v") or [""])[0] or None, may_render=self.admin)
                if not audio:
                    return self._json(404 if how == "not-cached" else 503 if how == "not-configured" else 502, {"ok": False, "error": how})
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio)))
                self.send_header("X-TTS", how)
                self.end_headers()
                self.wfile.write(audio)
                return
            if p == "/lessons":                             # what the next reflex challenger will learn (live + lessons)
                return self._json(200, recipes.pending())
            if p == "/genesis":
                return self._json(200, {"current": genesis.current(self.user)})
            if p == "/immune":
                return self._json(200, immune.state())
            if p == "/oracle":
                return self._json(200, oracle.state())
            if p == "/passports":
                return self._json(200, {"items": passport.index()})
            m = re.fullmatch(r"/passport/(h\d+)", p)
            if m:
                return self._json(200, {"passport": passport.latest(m.group(1))})
            if p == "/constitution":
                d = constitution.load()
                g = router.graph()
                vs = store.versions()["router"]
                return self._json(200, {"articles": d["articles"], "amendments": d["amendments"][-60:], "card": constitution.card(),
                                        "agents": {n["id"]: {"name": n["name"], "kind": n.get("kind")} for n in g["nodes"] if n["type"] == "agent"},
                                        "heads": constitution.heads_default(), "live": vs.get("active"),
                                        "versions": {h["id"]: {"blind": (h.get("scores") or {}).get("blind")} for h in vs["list"]}})
            if p == "/boardroom":
                return self._json(200, insight.boardroom())
            if p == "/genome":
                return self._json(200, insight.genome())
            if p == "/multiverse":
                hs = [h for h in ((q.get("heads") or [""])[0]).split(",") if h]
                return self._json(200, insight.multiverse(int((q.get("n") or [120])[0]), hs or None))
            if p == "/evolution":
                return self._json(200, evolution.tree())
            if p == "/autopilot":
                return self._json(200, autopilot.status())
            m = re.fullmatch(r"/autopilot/(m_\d+)", p)
            if m:
                st = autopilot.get(m.group(1))
                return self._json(200 if st else 404, st or {"error": "no such mission"})
            m = re.fullmatch(r"/autopilot/(m_\d+)/events", p)
            if m:
                return self._tail_mission(m.group(1))
            if p == "/space":
                return self._json(200, space.build(int((q.get("n") or [120])[0]), refresh=bool((q.get("refresh") or [""])[0])))
            if p == "/brain":
                return self._json(200, brain.snapshot(int((q.get("catchup") or [40])[0])))
            if p == "/brain/recent":
                return self._json(200, {"rows": brain.recent(float((q.get("hours") or [24])[0]), int((q.get("limit") or [300])[0]))})
            if p == "/events":
                return self._events()
            if p == "/drift":
                return self._json(200, drift.report())
            if p == "/rollout/guard":
                return self._json(200, rollout_guard.status())
            if p == "/changelog":
                return self._json(200, {"events": changelog.timeline()})
            if p == "/tune":
                return self._json(200, tune.get())
            if p == "/cache":
                return self._json(200, answer_cache.status())
            if p == "/gaps":
                return self._json(200, gaps.get(refresh=bool((q.get("refresh") or [""])[0])))
            m = re.fullmatch(r"/jobs/([\w-]+)/timelapse", p)
            if m:
                tl = os.path.join(store.DATA, "jobs", m.group(1), "timelapse.json")
                return self._json(200, json.load(open(tl, encoding="utf-8")) if os.path.exists(tl) else {"stages": []})
            if p == "/inbox":
                return self._json(200, inbox.items(int((q.get("limit") or [60])[0])))
            if p == "/labels/health":
                return self._json(200, doctor.health())
            if p == "/labels/weakspots":
                return self._json(200, doctor.weakspots())
            if p == "/labels/balance":
                return self._json(200, {"plan": doctor.balance_plan()})
            if p == "/golden":
                g = router.graph()
                return self._json(200, {"items": gates.golden(), "agents": {n["id"]: n["name"] for n in g["nodes"] if n["type"] == "agent"}})
            if p == "/gates/compare":
                return self._gates((q.get("candidate") or [""])[0])
            if p == "/judges":                              # the judge court: pins, calibrations, corrected numbers
                return self._json(200, judges.state())
            if p == "/judges/faith/queue":
                return self._json(200, judges.faith_queue())
            if p == "/golden/suites":                       # regression vs capability, history per live generation (cached)
                return self._stream_gen(suites.compute(fresh=(q.get("fresh") or [""])[0] == "1"))
            if p == "/versions":
                v = store.versions()
                kbs = {c["id"]: dict(store.kb_versions(c["id"]), name=c["name"], docs=c["docs"], chunks=c["chunks"]) for c in rag.collections()}
                return self._json(200, {"router": v["router"], "kbs": kbs})
            if p == "/actions":
                return self._json(200, actions.models())
            if p == "/jobs":
                return self._json(200, {"jobs": jobs.listing(), "running": jobs.running()})
            m = re.fullmatch(r"/jobs/([\w-]+)", p)
            if m:
                s = jobs.state(m.group(1))
                return self._json(200 if s else 404, s or {"error": "no such job"})
            m = re.fullmatch(r"/jobs/([\w-]+)/events", p)
            if m:
                return self._tail(m.group(1))
            return self._json(404, {"error": "not found"})
        except Exception as exc:                            # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        self.__dict__.pop("_jb", None)                      # one request's body never leaks into the next
        u = urlparse(self.path)
        p, q = u.path.rstrip("/"), parse_qs(u.query)
        try:
            if p == "/route":
                b = self._jbody()
                ctx = b.get("ctx") if isinstance(b.get("ctx"), dict) else None     # tests: a whole conversation in one call
                return self._json(200, router.route(str(b.get("text") or "")[:600], ctx=ctx))
            if p == "/thread/reset":
                context.forget(self._thread_key(self._jbody()))
                return self._json(200, {"ok": True})
            if p == "/run":
                return self._run(self._jbody())
            if p == "/academy/event":
                m_ = str(self._jbody().get("mission") or "")
                if m_ in ("xray", "court"):                  # the two missions only the page can witness
                    academy.mark(self.user, m_)
                return self._json(200, academy.progress(self.user, self.admin))
            if p == "/graph" and (q.get("sandbox") or [""])[0]:
                # anyone signed in may edit their OWN sandbox copy (Academy) — never the live graph
                return self._json(200, {"graph": router.save_graph(self._jbody().get("graph") or {}, self.user, sandbox_user=self.user)})
            if p == "/sandbox/reset":
                store.write(router._sandbox_file(self.user), json.loads(json.dumps(router.graph())) | {"sandbox": self.user})
                return self._json(200, {"ok": True})
            if p == "/app/route":
                return self._json(200, self._app_take(self._jbody()))
            if p == "/xray":
                b = self._jbody()
                if b.get("id"):
                    row = ledger.get(int(b["id"]))
                    if not row:
                        return self._json(404, {"error": "no such decision"})
                    return self._json(200, dict(xray.explain(row["text"], row["trace"]), ledger=row["id"], feedback=row["feedback"], label=row["label"],
                                                answer=row.get("answer"), faith=row.get("faith"), faith_score=row.get("faith_score")))
                return self._json(200, xray.explain(str(b.get("text") or "")[:600]))
            m = re.fullmatch(r"/receipts/([0-9a-f]{64})/verify", p)
            if m:                                           # read-only: recompute the chain + replay the decision
                return self._json(200, receipts.verify(m.group(1)))
            if p == "/timemachine/diagnose":                # read-only: simulates fixes, changes nothing
                return self._tm_diag(self._jbody())
            m = re.fullmatch(r"/ledger/(\d+)/feedback", p)
            if m:
                ledger.set_feedback(int(m.group(1)), int(self._jbody().get("value") or 0))
                return self._json(200, {"ok": True})
            m = re.fullmatch(r"/collections/([\w-]+)/search", p)
            if m:
                b = self._jbody()
                hits = rag.search(m.group(1), str(b.get("q") or "")[:400], k=int(b.get("k", 6)))
                return self._json(200, {"hits": [{k: h.get(k) for k in ("id", "doc", "page", "heading", "score", "dense", "bm25")}
                                                 | {"text": h["text"][:500]} for h in hits]})
            if p == "/tts/plan":                            # read-only: what a pre-render would cost
                b = self._jbody()
                return self._json(200, tts.plan([str(t) for t in (b.get("texts") or [])][:400], str(b.get("voice") or "") or tts.DEFAULT_VOICE))
            if p == "/lessons/attribute":                   # read-only: which pending lesson does what (leave-one-out refits)
                return self._stream_gen(recipes.attribute())
            if p == "/evolution/battle":                    # read-only: re-scores versions on the benchmark, promotes nothing
                b = self._jbody()
                return self._stream_gen(evolution.battle(str(b.get("kind") or "reflex"), [x for x in (b.get("ids") or []) if isinstance(x, str)] or None, b.get("kb")))
            if p == "/redteam/wave":                        # attacks through the live route; ledger rows, nothing changed
                b = self._jbody()
                return self._redteam_wave(int(b.get("n") or 24), [f for f in (b.get("families") or []) if isinstance(f, str)] or None)
            m = re.fullmatch(r"/redteam/breach/([0-9a-f]{12})/retry", p)
            if m:
                return self._json(200, redteam.retry(m.group(1), self.user))
            if p == "/space/place":                        # read-only: where a sentence lands in the map
                return self._json(200, space.place(str(self._jbody().get("text") or "")))
            if p == "/forge/estimate":                      # read-only: how long a recipe would take
                return self._json(200, forge.estimate(forge.recipe(self._jbody(), self.user)))
            if p == "/jobs/estimate":                       # read-only: how long a job would take
                cfg = self._cfg(self._jbody())
                return self._json(200, forge.estimate(cfg) if cfg.get("forge") else jobs.estimate(cfg))
            if p == "/diagnose":                            # read-only: why a candidate failed the court, and what would fix it
                return self._stream_gen(diagnose.run(str(self._jbody().get("candidate") or ""), self.user))
            if p == "/convo/replay":                        # read-only: whole conversations through every version
                return self._stream_gen(convo.replay(str(self._jbody().get("which") or "both")))
            if p == "/whynot":                              # read-only; the natural rewrite (referee model) is for admins
                b = self._jbody()
                return self._json(200, whynot.search(str(b.get("text") or ""), str(b.get("target") or "") or None, bool(b.get("natural")) and self.admin))
            if p == "/passport/verify":                     # read-only: does this passport still verify?
                return self._json(200, passport.verify(self._jbody().get("passport") or {}))
            if p == "/constitution/check":                  # read-only: every ratified article through every chosen version
                return self._stream_gen(constitution.check([h for h in (self._jbody().get("heads") or []) if isinstance(h, str)] or None))
            if self._need_admin():
                return
            m = re.fullmatch(r"/passport/(h\d+)/issue", p)
            if m:
                return self._stream_gen(passport.issue(m.group(1), self.user))
            if p == "/diagnose/contrast":                   # admin: drafts counter-examples from a lesson (referee model)
                b = self._jbody()
                return self._json(200, diagnose.contrast(str(b.get("lesson") or ""), str(b.get("agent") or ""), int(b.get("n") or 6)))
            if p == "/diagnose/same-need":                  # admin: fresh examples with a golden sentence's need (never blind)
                b = self._jbody()
                return self._json(200, diagnose.same_need(str(b.get("text") or ""), str(b.get("agent") or ""), int(b.get("n") or 6)))
            if p == "/immune/respond":                      # admin: mutations + antibodies (referee model); changes no model
                return self._stream_gen(immune.respond(self.user, [x for x in (self._jbody().get("ids") or []) if isinstance(x, str)] or None))
            if p == "/immune/approve":
                b = self._jbody()
                return self._json(200, immune.approve([x for x in (b.get("ids") or []) if isinstance(x, str)], bool(b.get("on", True)), self.user))
            if p == "/immune/vaccinate":                    # trains a CHALLENGER only; promotion stays in Versions
                return self._stream_gen(immune.vaccinate(self.user))
            if p == "/immune/remember":
                return self._json(200, immune.remember([x for x in (self._jbody().get("ids") or []) if isinstance(x, str)], self.user))
            if p == "/immune/retest":
                return self._json(200, immune.retest(self.user))
            if p == "/oracle/forecast":                     # admin: spends the referee model; changes nothing
                return self._stream_gen(oracle.forecast(self.user))
            if p == "/constitution/compile":
                b = self._jbody()
                return self._json(200, constitution.compile_rule(str(b.get("text") or ""), [str(x) for x in (b.get("seeds") or [])][:6], bool(b.get("draft", True))))
            if p == "/constitution/save":
                return self._json(200, {"ok": True, "article": constitution.save(self._jbody().get("article") or {}, self.user)})
            m = re.fullmatch(r"/constitution/([0-9a-f]{10})/(ratify|repeal)", p)
            if m:
                b = self._jbody()
                a = constitution.ratify(m.group(1), self.user) if m.group(2) == "ratify" else constitution.repeal(m.group(1), self.user, str(b.get("reason") or ""))
                return self._json(200, {"ok": True, "article": a})
            m = re.fullmatch(r"/gaps/(\w+)/publish", p)
            if m:
                b = self._jbody()
                return self._json(200, gaps.publish(m.group(1), str(b.get("collection") or ""), str(b.get("name") or ""),
                                                    str(b.get("markdown") or "")[:60000], self.user))
            if p == "/redteam/generate":
                b = self._jbody()
                made = redteam.generate(str(b.get("family") or ""), max(3, min(24, int(b.get("n") or 12))))
                return self._json(200, {"ok": True, "made": made})
            m = re.fullmatch(r"/redteam/breach/([0-9a-f]{12})/(dismiss|reopen)", p)
            if m:
                return self._json(200, {"ok": True, "breach": redteam.set_status(m.group(1), "dismissed" if m.group(2) == "dismiss" else "open", self.user)})
            if p == "/recipes/reconstruct":                 # pin an older head's recipe — only if a refit reproduces it
                return self._json(200, recipes.reconstruct(str(self._jbody().get("id") or store.active("router"))))
            if p == "/tts/prerender":                       # admin: render (and cache) every pitch sentence once
                b = self._jbody()
                texts = list(dict.fromkeys(str(t).strip() for t in (b.get("texts") or [])[:400] if str(t).strip()))
                voice = str(b.get("voice") or "") or tts.DEFAULT_VOICE
                def gen():
                    done = spent = 0
                    yield {"type": "start", "n": len(texts), **tts.plan(texts, voice)}
                    for t in texts:
                        was = tts.cached(t, voice)
                        _, how = tts.get(t, voice, may_render=True)
                        done += 1
                        spent += 0 if was else len(t)
                        yield {"type": "line", "done": done, "n": len(texts), "how": how, "spent": spent}
                        if how.startswith("error"):
                            yield {"type": "error", "error": how}
                            return
                    yield {"type": "done", "spent": spent, "quota": tts.quota()}
                return self._stream_gen(gen())
            if p == "/genesis/build":                       # admin: builds into the caller's OWN sandbox, never the live flow
                b = self._jbody()
                return self._stream_gen(genesis.build(str(b.get("collection") or ""), str(b.get("company") or ""), str(b.get("assistant") or ""),
                                                      self.user, train=b.get("train", True) is not False))
            if p == "/genesis/remove":
                return self._json(200, genesis.remove(self.user))
            if p == "/autopilot/run":
                return self._json(200, autopilot.start(self.user, "started by hand"))
            m = re.fullmatch(r"/autopilot/(m_\d+)/(promote|adopt|discard)", p)
            if m:
                mid, act = m.groups()
                if act == "promote":
                    return self._json(200, autopilot.promote(mid, self.user))
                if act == "adopt":
                    return self._json(200, {"ok": True, "adopted": autopilot.adopt(mid, self.user)})
                return self._json(200, autopilot.discard(mid, self.user))
            if p == "/cache/clear":
                answer_cache.clear(str(self._jbody().get("collection") or "") or None)
                return self._json(200, {"ok": True, **answer_cache.status()})
            if p == "/tune":
                tune.start(self.user)
                return self._json(200, {"ok": True, "state": "running"})
            if p == "/tune/apply":
                bar = float(self._jbody().get("bar"))
                if not 0.2 <= bar <= 0.9:
                    return self._json(400, {"error": "bar out of range"})
                g = router.graph()
                g["settings"]["reflex_threshold"] = round(bar, 2)
                return self._json(200, {"ok": True, "graph": router.save_graph(g, self.user)})
            if p == "/rollout/guard/check":
                return self._json(200, {"tripped": rollout_guard.check(), **rollout_guard.status()})
            if p == "/graph":
                return self._json(200, {"graph": router.save_graph(self._jbody().get("graph") or {}, self.user)})
            if p == "/arena/label":
                b = self._jbody()
                return self._json(200, {"ok": True, "label": inbox.label_text(str(b.get("text") or ""), str(b.get("agent") or ""), self.user,
                                                                             golden=bool(b.get("golden")))})
            if p == "/timemachine/apply":
                b = self._jbody()
                return self._json(200, timemachine.apply(int(b.get("id") or 0), str(b.get("want") or ""), b.get("fix") or {}, self.user))
            if p == "/calibration/fit":
                calibrate.start(self.user)
                return self._json(200, {"ok": True})
            if p == "/calibration/apply":
                return self._json(200, calibrate.set_applied(bool(self._jbody().get("on")), self.user))
            m = re.fullmatch(r"/flow/versions/(\d+)/restore", p)
            if m:
                v = flowver.get(int(m.group(1)))
                if not v:
                    return self._json(404, {"error": "no such version"})
                g = router.save_graph(json.loads(json.dumps(v["graph"])), self.user)
                return self._json(200, {"ok": True, "graph": g, "restored": v["n"]})
            if p == "/learning":
                return self._json(200, dict(learning.save(self._jbody(), self.user) and learning.status(), shadow=shadow.status()))
            if p == "/learning/train":
                return self._json(200, learning.start(str(self._jbody().get("system") or ""), self.user, "started by hand"))
            if p == "/shadow/run":
                return self._json(200, shadow.run_once())
            if p == "/inbox/label":
                b = self._jbody()
                return self._json(200, {"ok": True, "label": inbox.label(str(b.get("key") or ""), [int(i) for i in b.get("ids") or []],
                                                                        b.get("agent") or None, b.get("intent") or None, self.user, bool(b.get("golden")))})
            if p == "/inbox/dismiss":
                b = self._jbody()
                inbox.dismiss(str(b.get("key") or ""), str(b.get("why") or "not a real question"), self.user)
                return self._json(200, {"ok": True})
            if p == "/labels/remove":
                b = self._jbody()
                n = inbox.remove(str(b.get("key") or ""), self.user)
                return self._json(200, {"ok": True, "removed": n, **doctor.health()})
            if p == "/labels/add":
                b = self._jbody()
                item = doctor.add_example(str(b.get("text") or ""), str(b.get("agent") or ""), self.user)
                return self._json(200, {"ok": True, "item": item, **doctor.health()})
            if p == "/labels/generate":
                b = self._jbody()
                return self._json(200, doctor.generate(str(b.get("agent") or ""), int(b.get("n", 6)), str(b.get("hint") or "")))
            m = re.fullmatch(r"/ledger/(\d+)/label", p)
            if m:
                b = self._jbody()
                row = ledger.get(int(m.group(1)))
                if not row:
                    return self._json(404, {"error": "no such decision"})
                ledger.set_label(row["id"], str(b.get("agent") or ""), self.user)
                item = gates.add(row["text"], str(b.get("agent") or ""), self.user, note=str(b.get("note") or ""), source=f"ledger:{row['id']}") if b.get("golden", True) else None
                return self._json(200, {"ok": True, "golden": item})
            if p == "/golden":
                b = self._jbody()
                return self._json(200, {"ok": True, "item": gates.add(str(b.get("text") or ""), str(b.get("agent") or ""), self.user, str(b.get("note") or ""))})
            m = re.fullmatch(r"/golden/([\w-]+)/delete", p)
            if m:
                return self._json(200, {"ok": gates.remove(m.group(1))})
            if p == "/judges/referee/calibrate":            # spends language-model calls: admin only, on a button
                return self._stream_gen(judges.calibrate_referee(self.user))
            if p == "/judges/faith/calibrate":
                return self._stream_gen(judges.calibrate_faith(self.user))
            if p == "/judges/faith/mark":
                b = self._jbody()
                return self._json(200, judges.mark_faith(int(b.get("did") or 0), int(b.get("i") or 0), str(b.get("verdict") or ""), self.user))
            m = re.fullmatch(r"/golden/([\w-]+)/role", p)
            if m:                                           # a person graduates (or returns) a case — never automatic
                return self._json(200, {"ok": True, "item": suites.set_role(m.group(1), str(self._jbody().get("role") or ""), self.user)})
            if p == "/actions/switch":
                return self._json(200, actions.switch(str(self._jbody().get("run") or ""), self.user))
            m = re.fullmatch(r"/agents/([\w-]+)/secret", p)
            if m:
                return self._json(200, {"ok": True, "names": router.set_secret_headers(m.group(1), str(self._jbody().get("headers") or ""))})
            m = re.fullmatch(r"/agents/([\w-]+)/test", p)
            if m:
                n = router.node(router.graph(), m.group(1))
                if not n or n.get("kind") != "webhook":
                    return self._json(404, {"ok": False, "error": "no such webhook agent (save the flow first)"})
                r = router.call_webhook(n, str(self._jbody().get("text") or "hello")[:600])
                return self._json(200, {"result": r})
            if p == "/collections":
                b = self._jbody()
                return self._json(200, rag.create_collection(str(b.get("name") or ""), str(b.get("description") or "")))
            m = re.fullmatch(r"/collections/([\w-]+)/delete", p)
            if m:
                users = [n["name"] for n in router.graph()["nodes"] if n.get("kind") == "knowledge" and n.get("collection") == m.group(1)]
                if users and not self._jbody().get("force"):
                    return self._json(409, {"ok": False, "error": f"still used by {', '.join(users)} — point those agents elsewhere first", "agents": users})
                return self._json(200, rag.delete_collection(m.group(1)))
            m = re.fullmatch(r"/collections/([\w-]+)/upload", p)
            if m:
                name = unquote((q.get("name") or ["upload.txt"])[0])
                data = self._body(limit=rag.MAX_UPLOAD + 1)
                return self._json(200, {"ok": True, **rag.add_document(m.group(1), name, data, self.user)})
            m = re.fullmatch(r"/collections/([\w-]+)/docs/([\w-]+)/delete", p)
            if m:
                return self._json(200, {"ok": rag.remove_document(m.group(1), m.group(2))})
            if p == "/versions/promote":
                b = self._jbody()
                kind, vid = b.get("kind"), b.get("id")
                if kind == "retriever":                     # within ONE knowledge base
                    cid = str(b.get("collection") or "")
                    if not rag.meta(cid):
                        return self._json(400, {"error": "which knowledge base?"})
                    kv = store.kb_versions(cid)
                    x = next((x for x in kv["list"] if x["id"] == vid), None)
                    if not x or (vid != "base" and not os.path.isdir(os.path.join(store.DATA, "retriever", vid))):
                        return self._json(400, {"error": "no such version on disk"})
                    kv["active"] = vid
                    kv.setdefault("history", []).append({"id": vid, "at": time.time(), "by": self.user})
                    store.save_kb_versions(cid, kv)
                    threading.Thread(target=rag.index, args=(cid, vid), daemon=True).start()
                    return self._json(200, {"ok": True, "active": vid, "collection": cid})
                v = store.versions()
                if kind != "router" or vid not in [x["id"] for x in v["router"]["list"]]:
                    return self._json(400, {"error": "no such version"})
                v["router"]["active"] = vid
                v["router"].setdefault("history", []).append({"id": vid, "at": time.time(), "by": self.user})
                store.save_versions(v)
                return self._json(200, {"ok": True, "active": vid})
            if p == "/jobs":
                return self._json(200, jobs.start(self._cfg(self._jbody())))
            m = re.fullmatch(r"/jobs/([\w-]+)/cancel", p)
            if m:
                return self._json(200, {"ok": jobs.cancel(m.group(1))})
            return self._json(404, {"error": "not found"})
        except (ValueError, RuntimeError) as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:                            # noqa: BLE001
            return self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    # ── pieces ───────────────────────────────────────────────────────────────────────
    def _cfg(self, b: dict) -> dict:
        kind = b.get("kind")
        if kind == "intent_tune":                              # the encoder: a Forge recipe
            return forge.recipe(b, self.user)
        if kind not in ("rag_tune", "router_tune"):
            raise ValueError("kind must be rag_tune, router_tune or intent_tune")
        return {"kind": kind, "collection": str(b.get("collection") or ""), "by": self.user,
                "target": max(30.0, min(99.0, float(b.get("target", 85)))), "max_rounds": max(1, min(5, int(b.get("max_rounds", 3)))),
                "epochs": max(1, min(4, int(b.get("epochs", 2)))), "questions_per_passage": max(2, min(6, int(b.get("questions_per_passage", 4)))),
                "auto_promote": bool(b.get("auto_promote", False)), "then_router": bool(b.get("then_router", True)), "threads": 4,
                "from_scratch": bool(b.get("from_scratch", False)),
                "balance_labels": bool(b.get("balance_labels", False)), "label_boost": max(1, min(3, int(b.get("label_boost", 3))))}

    def _run(self, b: dict):
        text = str(b.get("text") or "").strip()[:600]
        self._sse_start()
        if not text:
            self._sse({"type": "error", "error": "type a message"})
            return
        try:
            t0 = time.time()
            sandbox = bool(b.get("sandbox"))
            g = router.sandbox_graph(self.user) if sandbox else router.graph()
            tkey = self._thread_key(b)
            turns = context.thread(tkey)
            r = router.route(text, g, ctx={"thread": turns})
            did = ledger.record("sandbox" if sandbox else "studio", text, r, session=self.user)
            self._sse({"type": "route", **r, "ledger": did})
            if r.get("blocked"):
                self._sse({"type": "delta", "text": router.BLOCK_REPLY.get(r["blocked"], "I can't help with that.")})
            answer, mode, passages, hit = "", None, [], None
            if r.get("blocked"):
                answer = router.BLOCK_REPLY.get(r["blocked"], "I can't help with that.")
            elif r.get("chosen"):
                engine_session = "studio-" + hashlib.md5(tkey.encode()).hexdigest()[:12]    # the engine keeps ITS context too
                for ev in router.run_stream(r["chosen"], text, g, history=turns, query=r.get("query"), session=engine_session):
                    if ev.get("type") == "delta":
                        answer += ev.get("text", "")
                    elif ev.get("type") == "done":
                        mode = ev.get("mode")
                        hit = ev.get("cache")
                    elif ev.get("type") == "passages":
                        passages = ev.get("passages") or []
                    self._sse(ev)
            else:
                answer = r.get("error") or "No agent could take this."
                self._sse({"type": "delta", "text": answer})
            ledger.finish(did, answer, mode, int((time.time() - t0) * 1000))
            if not sandbox:
                bus.publish("decision", self.user, text=text[:80], chosen=r.get("chosen"), source="studio", ledger=did,
                            path=brain.path(r, mode, hit is not None, "studio"), ms=int((time.time() - t0) * 1000))
            if r.get("chosen") and not r.get("blocked"):
                context.note(tkey, text, r["chosen"], (router.node(g, r["chosen"]) or {}).get("kind"),
                             standalone=r.get("query"), answer=answer, lang=r.get("language"))
            self._after_answer(did, g, r, text, answer, passages, mode, hit)
            self._sse({"type": "done", "ledger": did})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                            # noqa: BLE001
            try:
                self._sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _after_answer(self, did, g, r, text, answer, passages, mode, hit) -> None:
        """A provenance receipt for EVERY decision (chained + tamper-evident), then faithfulness in the
        background. A cache hit was verified when it was admitted (its score is recorded); a fresh knowledge
        answer is a cache candidate, so the judge always reads it."""
        try:
            receipts.record(did, g, r, text, passages, mode)
        except Exception:                                   # noqa: BLE001 — a receipt is never worth a failed answer
            pass
        if not (passages and answer):
            return
        if mode == "cache" and hit:
            ledger.set_faith(did, {"cache": hit}, hit.get("score"))
            return
        node = router.node(g, r.get("chosen") or "") or {}
        cache = {"cid": node.get("collection"), "question": r.get("query") or text, "lang": r.get("language") or "en"} \
            if node.get("kind") == "knowledge" and mode == "llm" and node.get("collection") else None
        threading.Thread(target=faith.assess, args=(did, answer, passages), kwargs={"cache": cache}, daemon=True).start()

    def _thread_key(self, b: dict) -> str:
        """One console conversation: this person + the conversation id the page keeps (New conversation = new id)."""
        return f"studio:{self.user}:{str(b.get('thread') or 'default')[:40]}"

    def _app_take(self, b: dict) -> dict:
        """The customer app asks: the Orchestrator routes EVERY request and records it. It DELIVERS the answer
        only for a ready knowledge agent whose documents cover the question; everything else (payments, refunds,
        PIN, short replies, errors) it routes to the right agent and hands to the engine to carry out — so no
        request bypasses the Orchestrator, but the money path still executes in the engine behind its PIN."""
        import hashlib
        text, session = str(b.get("text") or "").strip()[:600], str(b.get("session") or "")
        pct = learning.settings()["rollout"]["percent"]
        if not text or pct <= 0:
            return {"handled": False, "why": "outside the rollout"}
        if int(hashlib.md5(session.encode()).hexdigest(), 16) % 100 >= pct:
            return {"handled": False, "why": "session outside the rollout percentage"}
        t0 = time.time()
        g = router.graph()
        tkey = "app:" + session
        turns = context.thread(tkey)
        eng = b.get("engine") if isinstance(b.get("engine"), dict) else {}
        r = router.route(text, g, ctx={"thread": turns, "engine": eng})
        did = ledger.record("app", text, r, session=session)                # every request is recorded by the Orchestrator
        node = router.node(g, r.get("chosen") or "") or {}
        covered = any(c["agent"] == r.get("chosen") and c["best"] >= c["need"] for c in r.get("coverage", []))
        followed = r.get("by") == "context"
        # an older engine hook sends no conversation state: then short messages (likely answers to the engine's
        # own questions) stay with the engine, as before — context can only be trusted when it is actually sent
        no_state = "engine" not in b
        if node.get("kind") != "knowledge" or not covered or r.get("blocked") or (len(text.split()) < 3 and (not followed or no_state)) \
                or (no_state and followed and (r.get("context") or {}).get("rule") != "studio-thread"):
            # routed to a non-knowledge agent — recorded here, carried out by the engine
            ledger.finish(did, "", f"engine:{r.get('chosen') or r.get('blocked') or '?'}", int((time.time() - t0) * 1000))
            context.note(tkey, text, "engine", None, lang=r.get("language"))
            bus.publish("decision", "app customer", text=ledger.mask(text)[:80], chosen=r.get("chosen"), source="app", ledger=did, cached=False,
                        path=brain.path(r, f"engine:{r.get('chosen') or r.get('blocked') or '?'}", False, "app"), ms=int((time.time() - t0) * 1000))
            why = (r.get("context") or {}).get("why") if followed else None
            return {"handled": False, "why": why or f"routed to {r.get('chosen')} — handed to the engine", "chosen": r.get("chosen"), "ledger": did}
        answer, passages, mode = "", [], None
        hit = None
        for ev in router.run_stream(r["chosen"], text, g, history=turns, query=r.get("query")):
            if ev.get("type") == "delta":
                answer += ev.get("text", "")
            elif ev.get("type") == "passages":
                passages = ev.get("passages") or []
            elif ev.get("type") == "done":
                mode, hit = ev.get("mode"), ev.get("cache")
        if not answer.strip():
            ledger.finish(did, "", f"engine:{r.get('chosen')}", int((time.time() - t0) * 1000))
            context.note(tkey, text, "engine", None, lang=r.get("language"))
            return {"handled": False, "why": "no answer", "ledger": did}
        ledger.finish(did, answer, mode, int((time.time() - t0) * 1000))
        context.note(tkey, text, r["chosen"], "knowledge", standalone=r.get("query"), answer=answer, lang=r.get("language"))
        bus.publish("decision", "app customer", text=ledger.mask(text)[:80], chosen=r["chosen"], source="app", ledger=did, cached=bool(hit),
                    path=brain.path(r, mode, bool(hit), "app"), ms=int((time.time() - t0) * 1000))
        self._after_answer(did, g, r, text, answer, passages, mode, hit)
        return {"handled": True, "cached": bool(hit), "followup": followed, "say": answer, "agent": r["chosen"], "agent_name": node.get("name"), "ledger": did,
                "sources": [{"doc": p.get("doc"), "page": p.get("page")} for p in passages[:3]], "ms": int((time.time() - t0) * 1000)}

    def _stream_gen(self, gen):
        self._sse_start()
        try:
            for ev in gen:
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                            # noqa: BLE001
            try:
                self._sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _redteam_wave(self, n: int, families):
        self._sse_start()
        try:
            for ev in redteam.wave(n, families, self.user):
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                            # noqa: BLE001
            try:
                self._sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _tm_diag(self, b: dict):
        """Stream the Time Machine's rewind + search for the smallest safe fix (changes nothing)."""
        self._sse_start()
        try:
            for ev in timemachine.diagnose(int(b.get("id") or 0), str(b.get("want") or "")):
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                            # noqa: BLE001
            try:
                self._sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _gates(self, candidate: str):
        self._sse_start()
        try:
            if not candidate:
                self._sse({"type": "error", "error": "which version?"})
                return
            for ev in gates.compare(candidate):
                self._sse(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                            # noqa: BLE001
            try:
                self._sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass

    def _tail_mission(self, mid: str):
        """Stream a mission's events: everything so far, then new ones while it flies."""
        self._sse_start()
        pos, idle = 0, 0
        try:
            while True:
                evs, pos = autopilot.events(mid, pos)
                for ev in evs:
                    self._sse(ev)
                s = autopilot._state(mid) or {}
                if not evs:
                    idle += 1
                    if idle % 15 == 0:
                        self._sse({"type": "ping", "status": s.get("status")})
                else:
                    idle = 0
                if s.get("status") != "running" and not evs:
                    self._sse({"type": "end", "status": s.get("status")})
                    return
                time.sleep(0.6)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _tail(self, jid: str):
        """Stream a job's events: everything so far, then new ones as they are written."""
        self._sse_start()
        pos, idle = 0, 0
        try:
            while True:
                evs, pos = jobs.events(jid, pos)
                for ev in evs:
                    self._sse(ev)
                s = jobs.state(jid) or {}
                if not evs:
                    idle += 1
                    if idle % 15 == 0:
                        self._sse({"type": "ping", "status": s.get("status")})
                else:
                    idle = 0
                if s.get("status") not in ("running", "queued") and not evs:
                    self._sse({"type": "end", "status": s.get("status")})
                    return
                time.sleep(0.6)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _map(self, cid: str) -> dict:
        """A 2-D picture of the collection: every passage placed by meaning (PCA of its vector)."""
        import numpy as np
        ver = store.kb_active(cid)
        ids, vecs = rag._load_index(cid, ver)
        chunks = {c["id"]: c for c in rag.all_chunks(cid)}
        if len(ids) < 3:
            return {"points": []}
        X = vecs - vecs.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        P = X @ vt[:3].T
        P = P / (np.abs(P).max(0) + 1e-9)
        docs = sorted({chunks[i]["doc"] for i in ids})
        return {"version": ver, "docs": docs, "points": [{"id": i, "x": round(float(P[k, 0]), 4), "y": round(float(P[k, 1]), 4),
                "z": round(float(P[k, 2]), 4), "doc": docs.index(chunks[i]["doc"]), "heading": chunks[i].get("heading", "")[:60],
                "text": chunks[i]["text"][:120]} for k, i in enumerate(ids)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8467)
    a = ap.parse_args()
    router.graph()
    try:
        ledger.purge()                                      # 30-day retention
    except Exception:                                       # noqa: BLE001
        pass
    learning.scheduler()                                    # nightly challengers (only when switched on)
    shadow.worker()                                         # mirrors real app turns (only when switched on)
    rollout_guard.worker()                                  # switches the app rollout off if real answers go wrong
    autopilot.scheduler()                                   # nightly missions (only when switched on; never promotes)
    if router.graph()["settings"].get("laya_enabled", True):
        router.warm_laya()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"agent studio on http://127.0.0.1:{a.port}  data {store.DATA}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
