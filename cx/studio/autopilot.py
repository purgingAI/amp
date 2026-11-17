"""Autopilot: the Studio runs the learn-and-govern loop by itself overnight — and never promotes.

A MISSION is one pass of the loop, as a flight plan with eight legs:
    observe    what happened (traffic, drift, red-team integrity, inbox, label health, the live head)
    diagnose   where the live reflex is weak (blind misses by pattern), what the red team got through
    decide     a bounded plan: which patterns to teach, how many examples to draft, what to flag for humans
    stage      Claude drafts examples for those patterns; open breaches become examples too — STAGED in the
               mission folder, never written to labels.jsonl (adopted only when a person promotes the result)
    train      a challenger reflex head trained with the staged examples (train_job.router_tune, extra_labels)
    court      the full court (gates.compare): golden, conversations, languages, blind, latency
    redteam    the challenger re-sits the open breaches and a fresh sample, by replaying the route with head_obj
    briefing   a morning briefing: headline, evidence, a recommendation and one-click actions for a person

Everything is auditable in data/autopilot/<mission>/{state.json, events.jsonl, examples.jsonl}.
Hard rules: one mission at a time; never promotes; never writes labels or settings; bounded Claude use;
waits for no other training job; scheduler is OFF by default (learning.json → autopilot.enabled).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
import zoneinfo

import bus
import store

DIR = "autopilot"
STEPS = ["observe", "diagnose", "decide", "stage", "train", "court", "redteam", "briefing"]
UAE = zoneinfo.ZoneInfo("Asia/Dubai")
_lock = threading.Lock()
_cur: dict = {"mid": None}


# ── storage ───────────────────────────────────────────────────────────────────────────
def _dir(mid: str) -> str:
    d = os.path.join(store.DATA, DIR, mid)
    os.makedirs(d, exist_ok=True)
    return d


def _state(mid: str) -> dict | None:
    p = os.path.join(store.DATA, DIR, mid, "state.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _save(mid: str, st: dict) -> None:
    p = os.path.join(_dir(mid), "state.json")
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, default=str)
    os.replace(tmp, p)


def events(mid: str, since: int = 0) -> tuple[list[dict], int]:
    p = os.path.join(store.DATA, DIR, mid, "events.jsonl")
    if not os.path.exists(p):
        return [], since
    with open(p, encoding="utf-8") as fh:
        fh.seek(since)
        data = fh.read()
        pos = fh.tell()
    return [json.loads(l) for l in data.splitlines() if l.strip()], pos


def missions(limit: int = 12) -> list[dict]:
    root = os.path.join(store.DATA, DIR)
    out = []
    for mid in sorted(os.listdir(root), reverse=True)[:limit] if os.path.isdir(root) else []:
        st = _state(mid)
        if st:
            out.append({k: st.get(k) for k in ("id", "status", "started", "finished", "by", "reason", "step", "challenger", "verdict", "recommendation", "headline")})
    return out


def get(mid: str) -> dict | None:
    st = _state(mid)
    if not st:
        return None
    p = os.path.join(store.DATA, DIR, mid, "examples.jsonl")
    st["examples"] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
    return st


class Mission:
    def __init__(self, mid: str, by: str, reason: str):
        self.mid, self.by, self.reason = mid, by, reason
        self.st = {"id": mid, "by": by, "reason": reason, "status": "running", "started": time.time(), "step": None,
                   "steps": {s: {"state": "pending"} for s in STEPS}, "challenger": None, "verdict": None, "recommendation": None, "headline": None}
        _save(mid, self.st)

    def emit(self, kind: str, **kw) -> None:
        ev = {"t": round(time.time(), 2), "type": kind, **kw}
        with open(os.path.join(_dir(self.mid), "events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def step(self, name: str, state: str, **data) -> None:
        self.st["steps"][name] = {"state": state, "at": time.time(), **({"data": data} if data else {})}
        if state == "running":
            self.st["step"] = name
        _save(self.mid, self.st)
        self.emit("step", step=name, state=state, data=data)

    def log(self, text: str, level: str = "") -> None:
        self.emit("log", text=text, level=level)


# ── the loop ──────────────────────────────────────────────────────────────────────────
def start(by: str, reason: str = "started by hand") -> dict:
    import jobs
    with _lock:
        if _cur.get("mid") and (_state(_cur["mid"]) or {}).get("status") == "running":
            raise RuntimeError("a mission is already flying")
        if jobs.running():
            raise RuntimeError("a training job is running — the Autopilot waits for the forge to be free")
        mid = "m_" + time.strftime("%m%d%H%M%S")
        m = Mission(mid, by, reason)
        _cur["mid"] = mid
        threading.Thread(target=_fly, args=(m,), daemon=True, name="autopilot").start()
        return {"id": mid}


def _fly(m: Mission) -> None:
    try:
        obs = _observe(m)
        diag = _diagnose(m, obs)
        plan = _decide(m, obs, diag)
        staged = _stage(m, plan) if plan["train"] else []
        if not plan["train"]:
            m.step("stage", "skipped", why=plan["why"])
            m.step("train", "skipped", why=plan["why"])
            m.step("court", "skipped", why=plan["why"])
            m.step("redteam", "skipped", why=plan["why"])
            _briefing(m, obs, diag, plan, None, None, None)
            return
        cand = _train(m, staged)
        verdict = _court(m, cand) if cand else None
        rt = _redteam(m, cand, diag) if cand else None
        _briefing(m, obs, diag, plan, cand, verdict, rt)
    except Exception as exc:                                # noqa: BLE001
        m.st["status"] = "failed"
        m.st["error"] = f"{type(exc).__name__}: {exc}"
        m.st["finished"] = time.time()
        _save(m.mid, m.st)
        m.emit("error", text=m.st["error"])
    finally:
        if m.st["status"] == "running":
            m.st["status"] = "done"
            m.st["finished"] = time.time()
            _save(m.mid, m.st)
        m.emit("end", status=m.st["status"])


def _observe(m: Mission) -> dict:
    import doctor
    import drift
    import inbox
    import ledger
    import redteam
    import router
    m.step("observe", "running")
    st = ledger.stats(1)
    head = router.head() or {}
    v = store.versions()["router"]
    live = next((h for h in v["list"] if h["id"] == v["active"]), {})
    try:
        dr = drift.report()
        drift_flags = [x["label"] for x in dr.get("metrics", []) if x.get("flag")]
    except Exception:                                       # noqa: BLE001
        drift_flags = []
    rt = redteam.snapshot()
    try:
        health = doctor.health()
        flagged = [l for l in health.get("labels", []) if l.get("flags")] if isinstance(health, dict) else []
    except Exception:                                       # noqa: BLE001
        health, flagged = {}, []
    try:
        inbox_n = len((inbox.items(60) or {}).get("items", []))
    except Exception:                                       # noqa: BLE001
        inbox_n = 0
    try:
        champ = doctor.champion()
    except Exception:                                       # noqa: BLE001
        champ = {}
    obs = {"traffic_24h": {"n": st.get("n"), "by_source": dict(st.get("by_source") or {}), "thumbs_down": st.get("thumbs_down"), "ms_p50": st.get("ms_p50")},
           "live": {"head": v["active"], "blind": round(100 * (live.get("scores") or {}).get("blind", 0), 1) if (live.get("scores") or {}).get("blind") is not None else None,
                    "encoder": (head.get("encoder") or "").split("/")[-1] or None, "champion": champ},
           "drift_flags": drift_flags, "redteam": {"integrity": rt.get("integrity"), "open_breaches": len(rt.get("breaches") or []), "waves": len(rt.get("waves") or [])},
           "inbox": inbox_n, "flagged_labels": [{"text": l.get("text"), "flags": l.get("flags")} for l in flagged[:6]]}
    m.step("observe", "done", **obs)
    m.log(f"24h: {obs['traffic_24h']['n']} decisions · live reflex {obs['live']['head']} ({obs['live']['blind']}% blind) · drift flags {len(drift_flags)} · "
          f"red team integrity {obs['redteam']['integrity']} with {obs['redteam']['open_breaches']} open breach(es) · inbox {inbox_n}")
    return obs


def _diagnose(m: Mission, obs: dict) -> dict:
    import doctor
    import redteam
    m.step("diagnose", "running")
    ws = doctor.weakspots(6)
    breaches = [b for b in redteam.snapshot().get("breaches", [])][-10:]
    diag = {"weakspots": ws.get("groups", []), "misses": ws.get("total"), "checked": ws.get("checked"), "accuracy": ws.get("accuracy"), "breaches": breaches}
    m.step("diagnose", "done", **diag)
    top = ", ".join(f"{g['want_name']}→{g['got_name']} ({g['lang']}) ×{g['count']}" for g in diag["weakspots"][:3]) or "none"
    m.log(f"live head misses {diag['misses']}/{diag['checked']} blind sentences · top patterns: {top} · {len(breaches)} open breach(es)")
    return diag


def _decide(m: Mission, obs: dict, diag: dict) -> dict:
    import learning
    s = learning.settings().get("autopilot", {})
    cap = int(s.get("max_examples", 24))
    m.step("decide", "running")
    targets = [g for g in diag["weakspots"] if g["count"] >= 2][:3]
    breach_examples = []
    for b in diag["breaches"]:
        exp = b.get("expect") or ""
        agent = exp.split(":", 1)[1] if exp.startswith("agent:") else None
        if not agent:
            kinds = {n["id"]: n.get("kind") for n in __import__("router").graph()["nodes"] if n["type"] == "agent"}
            agent = next((a for a, k in kinds.items() if k == ("human" if exp == "handoff" else "decline")), None) if exp in ("refuse", "security", "handoff") else None
        if agent:
            breach_examples.append({"text": b["text"], "agent": agent, "why": f"red-team breach ({b['family']})", "origin": "redteam"})
    train = bool(targets or breach_examples)
    why = "" if train else "the live head has no repeated blind-miss pattern and the red team got nothing through — nothing to teach tonight"
    plan = {"train": train, "why": why, "targets": [{"want": g["want"], "want_name": g["want_name"], "got_name": g["got_name"], "lang": g["lang"], "count": g["count"], "examples": g["examples"][:3]} for g in targets],
            "per_target": max(3, min(12, cap // max(1, len(targets)))) if targets else 0, "breach_examples": breach_examples[:8],
            "flags": [f"{len(obs['flagged_labels'])} label(s) look harmful — review in the Label Doctor"] if obs.get("flagged_labels") else [],
            "cap": cap}
    m.step("decide", "done", **plan)
    m.log(("plan: teach " + "; ".join(f"{t['want_name']} ({t['lang']}, {t['count']} misses)" for t in plan["targets"]) + f" · {plan['per_target']} drafts each"
           + (f" · {len(breach_examples)} breaches become examples" if breach_examples else "")) if train else "plan: " + why)
    return plan


def _stage(m: Mission, plan: dict) -> list[dict]:
    import doctor
    import inbox
    m.step("stage", "running")
    have = {inbox.key(l["text"]) for l in inbox._labels()}
    staged: list[dict] = []
    for t in plan["targets"]:
        hint = (f"customers writing in {t['lang']} who mean “{t['want_name']}” but whose wording sounds like “{t['got_name']}” — "
                f"for example: " + " | ".join(t["examples"]))
        try:
            g = doctor.generate(t["want"], plan["per_target"], hint)
        except Exception as exc:                            # noqa: BLE001
            m.log(f"could not draft for {t['want_name']}: {type(exc).__name__}", "warn")
            continue
        for s in g.get("suggestions", []):
            k = inbox.key(s)
            if k in have:
                continue
            have.add(k)
            staged.append({"text": s, "agent": t["want"], "why": f"blind-miss pattern {t['want_name']}→{t['got_name']} ({t['lang']})", "origin": "claude"})
        m.log(f"drafted {len(g.get('suggestions', []))} for {t['want_name']} ({t['lang']})")
    for b in plan["breach_examples"]:
        if inbox.key(b["text"]) not in have:
            have.add(inbox.key(b["text"]))
            staged.append(b)
    with open(os.path.join(_dir(m.mid), "examples.jsonl"), "w", encoding="utf-8") as fh:
        for e in staged:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    m.step("stage", "done", n=len(staged), examples=staged[:40])
    m.log(f"{len(staged)} examples staged in the mission folder — not in labels.jsonl until a person promotes")
    return staged


def _train(m: Mission, staged: list[dict]) -> str | None:
    import jobs
    m.step("train", "running", n=len(staged))
    cfg = {"kind": "router_tune", "by": "autopilot", "challenger": True, "auto_promote": False, "threads": 4, "reason": f"autopilot {m.mid}",
           "extra_labels": os.path.join(_dir(m.mid), "examples.jsonl"), "balance_labels": True, "label_boost": 2}
    r = jobs.start(cfg)
    jid = r["id"]
    m.st["job"] = jid
    _save(m.mid, m.st)
    m.log(f"training challenger — job {jid}")
    t0 = time.time()
    while True:
        s = jobs.state(jid) or {}
        if s.get("status") in ("done", "failed", "cancelled"):
            break
        if time.time() - t0 > 1200:
            jobs.cancel(jid)
            m.step("train", "failed", why="training took longer than 20 minutes")
            return None
        m.emit("progress", step="train", progress=s.get("progress"), phase=s.get("phase"))
        time.sleep(3)
    if s.get("status") != "done":
        m.step("train", "failed", why=s.get("error") or s.get("status"))
        return None
    res = s.get("result") or {}
    cand = res.get("version")
    m.st["challenger"] = cand
    _save(m.mid, m.st)
    m.step("train", "done", challenger=cand, accuracy=res.get("accuracy"), previous=res.get("previous"), job=jid, seconds=int(time.time() - t0))
    m.log(f"challenger {cand} trained in {int(time.time() - t0)} s")
    return cand


def _court(m: Mission, cand: str) -> dict | None:
    import gates
    m.step("court", "running", challenger=cand)
    summary = None
    n_items = 0
    for ev in gates.compare(cand):
        if ev.get("type") == "item":
            n_items += 1
            if n_items % 6 == 0:
                m.emit("progress", step="court", done=n_items, n=ev.get("n"))
        elif ev.get("type") == "phase":
            m.log(ev.get("text", ""))
        elif ev.get("type") == "summary":
            summary = ev
    if not summary:
        m.step("court", "failed", why="the court returned no verdict")
        return None
    verdict = {"pass": bool(summary.get("pass")), "gates": summary.get("gates"), "right": summary.get("right"), "n": summary.get("n"), "blind": summary.get("blind"),
               "fixed": summary.get("fixed"), "broke": summary.get("broke"), "per_lang": summary.get("per_lang"), "live": summary.get("live")}
    m.st["verdict"] = "APPROVED" if verdict["pass"] else "HOLD"
    _save(m.mid, m.st)
    m.step("court", "done", **verdict)
    failed = [g["name"] for g in (verdict["gates"] or []) if not g["pass"]]
    m.log(f"court: {m.st['verdict']} — golden {verdict['right']['cand']}/{verdict['n']} vs live {verdict['right']['live']}/{verdict['n']}" + (f" · failed: {', '.join(failed)}" if failed else ""))
    return verdict


def _redteam(m: Mission, cand: str, diag: dict) -> dict | None:
    import redteam
    import router
    m.step("redteam", "running", challenger=cand)
    g = router.graph()
    try:
        head = router.load_head(cand)
    except Exception as exc:                                # noqa: BLE001
        m.step("redteam", "failed", why=f"{type(exc).__name__}")
        return None
    held, tried, detail = 0, 0, []
    for b in diag["breaches"]:
        r = router.route(b["text"], g, head_obj=head)
        out, why = redteam.judge({"expect": b["expect"], "family": b["family"]}, r, g)
        tried += 1
        held += out != "breach"
        detail.append({"text": b["text"], "family": b["family"], "outcome": out, "why": why})
    sample = redteam.pick(12, None)
    s_live = s_cand = 0
    for a in sample:
        r_c = router.route(a["text"], g, head_obj=head)
        r_l = router.route(a["text"], g)
        s_cand += redteam.judge(a, r_c, g)[0] != "breach"
        s_live += redteam.judge(a, r_l, g)[0] != "breach"
    res = {"breaches_tried": tried, "breaches_held": held, "sample_n": len(sample), "sample_cand": s_cand, "sample_live": s_live, "detail": detail}
    m.step("redteam", "done", **res)
    m.log(f"challenger holds {held}/{tried} open breach(es); fresh sample: challenger {s_cand}/{len(sample)} vs live {s_live}/{len(sample)}")
    return res


def _briefing(m: Mission, obs, diag, plan, cand, verdict, rt) -> None:
    m.step("briefing", "running")
    live_blind = obs["live"].get("blind")
    cand_blind = ((verdict or {}).get("blind") or {}).get("cand") if verdict else None
    if cand_blind is not None and cand_blind <= 1:          # the court reports a fraction; the head records a percent
        cand_blind = round(100 * cand_blind, 1)
    facts = []
    facts.append(f"{obs['traffic_24h']['n'] or 0} decisions in the last 24 h" + (f", {obs['traffic_24h']['thumbs_down']} thumbs-down" if obs['traffic_24h'].get('thumbs_down') else ""))
    facts.append(f"live reflex {obs['live']['head']} misses {diag['misses']}/{diag['checked']} blind sentences")
    if obs["drift_flags"]:
        facts.append("drift flagged: " + ", ".join(obs["drift_flags"]))
    facts.append(f"red team: integrity {obs['redteam']['integrity']}%, {obs['redteam']['open_breaches']} open breach(es)" if obs["redteam"].get("integrity") is not None else "red team: no wave yet")
    rec, headline = "nothing", ""
    if not plan["train"]:
        headline = f"Quiet night — {obs['live']['head']} holds, nothing to teach"
        rec = "nothing"
    elif not cand:
        headline = "Tried to train a challenger — it did not finish"
        rec = "look"
    else:
        n_staged = ((m.st["steps"].get("stage") or {}).get("data") or {}).get("n")
        facts.append(f"trained {cand} with {n_staged if n_staged is not None else '?'} staged examples")
        if verdict:
            facts.append(f"court: golden {verdict['right']['cand']}/{verdict['n']} vs {verdict['right']['live']}/{verdict['n']}, blind {cand_blind} vs {live_blind} — {m.st['verdict']}")
        if rt:
            facts.append(f"challenger holds {rt['breaches_held']}/{rt['breaches_tried']} open breaches; fresh attacks {rt['sample_cand']}/{rt['sample_n']} vs live {rt['sample_live']}/{rt['sample_n']}")
        better_rt = not rt or (rt["sample_cand"] >= rt["sample_live"] and rt["breaches_held"] >= min(1, rt["breaches_tried"]))
        if verdict and verdict["pass"] and better_rt:
            rec = "promote"
            headline = f"{cand} passed the court" + (f" (+{round(cand_blind - live_blind, 1)} blind)" if cand_blind is not None and live_blind is not None and cand_blind >= live_blind else "") + " — promote?"
        elif verdict and verdict["pass"]:
            rec = "look"
            headline = f"{cand} passed the court but is weaker against the red team — look before promoting"
        else:
            rec = "hold"
            failed = [g["name"] for g in ((verdict or {}).get("gates") or []) if not g["pass"]]
            headline = f"{cand} held by the court" + (f" ({', '.join(failed)})" if failed else "") + " — the staged examples were not enough"
    if plan.get("flags"):
        facts += plan["flags"]
    prose = None
    try:
        import llm
        prose = llm.complete("You write a three-sentence morning briefing for the engineer who owns a payments assistant's router. Plain, specific, no hype, no bullet points. "
                             "Never recommend promoting; a person decides. Use only the facts given.",
                             "Facts:\n- " + "\n- ".join(facts) + f"\nVerdict: {m.st.get('verdict') or 'no challenger'}\nRecommendation code: {rec}", max_tokens=1500).strip()   # the model thinks first: small budgets come back empty
        if not prose:
            m.log("the briefing writer returned nothing — facts only", "warn")
    except Exception as exc:                                # noqa: BLE001
        prose = None
        m.log(f"the briefing writer failed ({type(exc).__name__}: {str(exc)[:120]}) — facts only", "warn")
    m.st.update(recommendation=rec, headline=headline, facts=facts, prose=prose)
    _save(m.mid, m.st)
    m.step("briefing", "done", headline=headline, facts=facts, recommendation=rec, prose=prose, challenger=cand)
    try:
        bus.publish("autopilot", "autopilot", text=headline, mission=m.mid, recommendation=rec)
    except Exception:                                       # noqa: BLE001
        pass


# ── actions a person takes on a mission ───────────────────────────────────────────────
def adopt(mid: str, by: str) -> int:
    """The staged examples become real labels (so the promoted head stays reproducible)."""
    import inbox
    st = get(mid) or {}
    n = 0
    for e in st.get("examples", []):
        try:
            inbox.label_text(e["text"], e["agent"], by, source="autopilot")
            n += 1
        except Exception:                                   # noqa: BLE001
            pass
    s = _state(mid) or {}
    s["adopted"] = {"n": n, "by": by, "at": time.time()}
    _save(mid, s)
    return n


def promote(mid: str, by: str) -> dict:
    st = _state(mid) or {}
    cand = st.get("challenger")
    if not cand or st.get("verdict") != "APPROVED":
        raise ValueError("only a court-approved challenger can be promoted from a mission")
    v = store.versions()
    if cand not in [x["id"] for x in v["router"]["list"]]:
        raise ValueError("the challenger is gone")
    v["router"]["active"] = cand
    v["router"].setdefault("history", []).append({"id": cand, "at": time.time(), "by": by, "mission": mid})
    store.save_versions(v)
    n = adopt(mid, by)
    st = _state(mid) or {}
    st["promoted"] = {"by": by, "at": time.time()}
    _save(mid, st)
    bus.publish("promote", by, text=f"promoted reflex {cand} from Autopilot mission {mid}", model="router")
    return {"ok": True, "active": cand, "adopted": n}


def discard(mid: str, by: str) -> dict:
    st = _state(mid) or {}
    st["discarded"] = {"by": by, "at": time.time()}
    _save(mid, st)
    return {"ok": True}


# ── status + scheduler ────────────────────────────────────────────────────────────────
def next_run() -> float | None:
    import learning
    s = learning.settings().get("autopilot", {})
    if not s.get("enabled"):
        return None
    hh, mm = (int(x) for x in str(s.get("time", "03:30")).split(":"))
    now = dt.datetime.now(UAE)
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def status() -> dict:
    import learning
    ms = missions(12)
    running = next((x for x in ms if x.get("status") == "running"), None)
    return {"settings": learning.settings().get("autopilot", {}), "running": running, "last": next((x for x in ms if x.get("status") != "running"), None),
            "missions": ms, "next_run": next_run(), "steps": STEPS}


def scheduler() -> None:
    def loop():
        while True:
            try:
                import learning
                s = learning.settings().get("autopilot", {})
                if s.get("enabled"):
                    now = dt.datetime.now(UAE)
                    hh, mm = (int(x) for x in str(s.get("time", "03:30")).split(":"))
                    last = max([x.get("started") or 0 for x in missions(3)] or [0])
                    if (now.hour, now.minute) == (hh, mm) and time.time() - last > 20 * 3600:
                        start("autopilot scheduler", "nightly mission")
            except Exception:                               # noqa: BLE001 — the scheduler must never die
                pass
            time.sleep(30)
    threading.Thread(target=loop, daemon=True, name="autopilot").start()
