"""Why did a candidate fail the court — and what would fix it.

The court says WHICH gate failed. diagnose() says WHY, from evidence:

    replay     the candidate is refit from its own pinned recipe (live recipe + the candidate's lessons, staged antibodies
               included) — and the refit is checked against the real candidate head, so the analysis is about the model
               that was actually judged
    broken     the blind-exam and golden sentences the live version gets right and the candidate gets wrong
    culprits   every lesson the candidate has over live is left out in turn (one refit each): the lessons whose removal
               brings broken sentences back are the culprits, ranked by how many they bring back
    combined   one refit with every culprit left out together (lessons interact) — would that pass the agent / blind gates?
    advice     per culprit: already covered by a guardrail rule (→ the lesson is redundant), looks like a job for a rule
               (other providers, other people's data …), or keep it and teach the difference with counter-examples drafted
               from the LESSON — never from the exam, which must stay unseen
    plus what the other failed gates point at (languages, constitution articles, agents the candidate never learned).

Read-only: it suggests; people remove labels, un-approve antibodies, add rules or examples.
"""
from __future__ import annotations

import copy
import time

import numpy as np

import router
import store

MAX_LESSONS = 40


def _guard(text: str) -> str | None:
    try:
        import security
        return security.check(text)
    except Exception:                                       # noqa: BLE001
        return None


FAMILY_RULE = {
    "other_people_data": ("other-person", "asks about another person's account — the other-person guardrail misses this wording; extend it"),
    "social_engineering_money": ("scam", "is a scam pattern (an urgent push to send money to a number) — the scam guardrail misses this wording; extend it"),
    "competitor_offtopic": ("competitor", "names another provider — the competitor guardrail should take it; extend it if it misses this wording"),
    "prompt_injection": ("injection", "tries to change the rules — the injection guardrail misses this wording; extend it"),
    "roleplay_jailbreak": ("injection", "is a role-play jailbreak — the injection guardrail misses this wording; extend it"),
    "credential_fishing": ("credential", "fishes for a PIN / OTP — the credential guardrail misses this wording; extend it"),
    "internal_system": ("internal", "asks about our internal systems — the internal guardrail misses this wording; extend it"),
    "harmful_illegal": ("illegal", "asks for something unlawful — the illegal-request guardrail misses this wording; extend it"),
}


def _families() -> dict:
    """lesson text → red-team family, for antibodies (an antigen, or one of its mutations)."""
    try:
        import immune
        import redteam
        d = immune._load()
        bs = {b["id"]: b for b in redteam._breaches()}
        out = {}
        for i, a in d["antigens"].items():
            b = bs.get(i)
            if not b:
                continue
            for t in [b["text"]] + [m["text"] for m in a.get("mutations") or []]:
                out[" ".join(t.lower().split())] = b["family"]
        return out
    except Exception:                                       # noqa: BLE001
        return {}


def _rule_shaped(text: str) -> str | None:
    """Lessons that teach the router something a model learns badly from a few examples: who a request is ABOUT."""
    import re
    t = text.lower()
    if re.search(r"careem|phone\s*pe|paytm|google\s*pay|gpay|botim|apple\s*pay|western\s*union|e&|al\s*ansari|للصرافة|كريم|بوتيم", t):
        return "names another provider — a guardrail rule can decline that without touching our own how-to questions"
    if re.search(r"\b(?:his|her|their|my\s+(?:wife|husband|brother|sister|friend|girlfriend|boyfriend|flatmate|employee|boss|father|mother))\b|"
                 r"\b(?:uska|uski|unka|meri\s+\w+\s+ne|mere\s+\w+\s+ne|ne\s+kis)|\bne\b.*\b(?:kisko|kis\s+kis)\b", t) and \
       re.search(r"paid|pay\s*kiya|bhej|transaction|balance|history|details|statement|dikha|show|rent", t):
        return "is about another person's account — the other-person guardrail is the right place, if it misses this wording"
    return None


def run(candidate: str, by: str = ""):
    import doctor
    import encoder
    import gates
    import recipes
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    live = store.active("router")
    base, cand = recipes.load(live), recipes.load(candidate)
    if not base or not cand:
        yield {"type": "error", "error": f"{'the live head' if not base else candidate} has no pinned recipe — cannot replay its training"}
        return
    g0 = copy.deepcopy(router.graph())
    for n in g0["nodes"]:                                   # the agents' examples as the candidate was trained with them
        if n["type"] == "agent" and n["id"] in (cand.get("examples") or {}):
            n["examples"] = list(cand["examples"][n["id"]])
    ids = {n["id"] for n in g0["nodes"] if n["type"] == "agent"}
    names = {n["id"]: n["name"] for n in g0["nodes"] if n["type"] == "agent"}
    labels0 = [(t, a) for t, a in (cand.get("labels") or []) if a in ids]
    L = recipes.lessons(base, {"labelled": labels0, "agents": cand.get("agents") or [], "examples": cand.get("examples") or {}})
    items = ([("label_added", x) for x in L["labels_added"]] + [("label_removed", x) for x in L["labels_removed"]] +
             [("label_moved", x) for x in L["labels_moved"]] + [("example_added", x) for x in L["examples_added"]] +
             [("example_removed", x) for x in L["examples_removed"]])[:MAX_LESSONS]
    try:
        import inbox
        in_labels = {inbox.key(l["text"]) for l in inbox.training_rows("orchestrator")}
    except Exception:                                       # noqa: BLE001
        in_labels = set()
    lh, ch = router.load_head(live), router.load_head(candidate)
    enc = encoder.get(ch.get("encoder"))
    C = float(cand.get("C") or base.get("C") or 2.0)
    cfg = cand.get("cfg") or base.get("cfg") or {}
    cw = cfg.get("class_weight", "balanced")
    cw = None if cw in (None, "none", "None") else cw
    blind = [(t, a) for t, a in doctor._blind_rows() if a in ids]
    gold = [x for x in gates.golden() if x["agent"] in ids]
    cache: dict = {}

    def vecs(texts):
        miss = [t for t in dict.fromkeys(texts) if t not in cache]
        if miss:
            for t, v in zip(miss, enc.encode(miss, max_len=64)):
                cache[t] = v
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    Xb, yb = vecs([t for t, _ in blind]), np.array([a for _, a in blind])
    Xg, yg = vecs([x["text"] for x in gold]), np.array([x["agent"] for x in gold])

    def fit(graph, labels):
        asm = recipes.assemble(graph, cfg, labels, base)
        rows = asm["rows"]
        cnt: dict = {}
        for r in rows:
            cnt[r[1]] = cnt.get(r[1], 0) + 1
        rows = [r for r in rows if cnt[r[1]] >= 5]
        train = [r for r in rows if recipes.hash_bucket(r[0]) >= 20]
        clf = LogisticRegression(C=C, max_iter=2000, class_weight=cw)
        clf.fit(vecs([r[0] for r in train]), [r[1] for r in train])
        return clf.predict(Xb), (clf.predict(Xg) if len(yg) else np.array([]))

    live_b = lh["clf"].predict(vecs([t for t, _ in blind]) if lh.get("encoder") == ch.get("encoder") else encoder.get(lh.get("encoder")).encode([t for t, _ in blind], max_len=64))
    live_g = lh["clf"].predict(encoder.get(lh.get("encoder")).encode([x["text"] for x in gold], max_len=64)) if len(yg) else np.array([])
    real_b = ch["clf"].predict(Xb)
    full_b, full_g = fit(g0, labels0)
    replica = float((full_b == real_b).mean()) if len(real_b) else 1.0
    broken = np.where((live_b == yb) & (full_b != yb))[0]
    broken_g = np.where((live_g == yg) & (full_g != yg))[0] if len(yg) else np.array([], int)
    yield {"type": "start", "candidate": candidate, "live": live, "lessons": len(items), "replica": round(replica, 4),
           "broken": [{"text": blind[i][0], "want": blind[i][1], "want_name": names.get(blind[i][1], blind[i][1]), "got": str(full_b[i]),
                       "got_name": names.get(str(full_b[i]), str(full_b[i]))} for i in broken],
           "broken_golden": [{"text": gold[i]["text"], "want": gold[i]["agent"], "got": str(full_g[i])} for i in broken_g]}
    if not len(broken) and not len(broken_g):
        yield {"type": "done", "culprits": [], "advice": [], "note": "the candidate breaks nothing the live version gets right"}
        return
    culprits = []
    for k, (kind, x) in enumerate(items):
        g, labels = g0, list(labels0)
        key = x.get("key")
        if kind == "label_added":
            labels = [(t, a) for t, a in labels if recipes._key(t) != key]
        elif kind == "label_removed":
            labels = labels + [(x["text"], x["agent"])]
        elif kind == "label_moved":
            labels = [(t, (x["was"] if recipes._key(t) == key else a)) for t, a in labels]
        else:
            g = copy.deepcopy(g0)
            node = next((n for n in g["nodes"] if n["id"] == x["agent"]), None)
            if node is not None:
                ex = list(node.get("examples") or [])
                node["examples"] = [e for e in ex if e != x["text"]] if kind == "example_added" else ex + [x["text"]]
        wo_b, wo_g = fit(g, labels)
        back = [int(i) for i in broken if wo_b[i] == yb[i]]
        back_g = [int(i) for i in broken_g if wo_g[i] == yg[i]]
        lost_elsewhere = int(((full_b == yb) & (wo_b != yb)).sum())          # what this lesson is GOOD for
        ev = {"type": "lesson", "i": k, "n": len(items), "kind": kind, "text": x["text"], "agent": x["agent"], "agent_name": names.get(x["agent"], x["agent"]),
              "key": key, "brings_back": len(back) + len(back_g), "examples": [gold[i]["text"] for i in back_g[:2]] + [blind[i][0] for i in back[:4]],
              "golden_back": [gold[i]["text"] for i in back_g], "helps": lost_elsewhere}
        yield ev
        if back or back_g:
            src = "label" if kind.startswith("label") and key in in_labels else "antibody / staged" if kind == "label_added" else kind.replace("_", " ")
            culprits.append(ev | {"source": src, "back_idx": back, "back_g_idx": back_g})
    culprits.sort(key=lambda c: (-c["brings_back"], c["helps"]))
    # every culprit left out together
    combined = None
    if culprits:
        labels = list(labels0)
        drop = {c["key"] for c in culprits if c["kind"] == "label_added"}
        labels = [(t, a) for t, a in labels if recipes._key(t) not in drop]
        cb, cg = fit(g0, labels)
        per = lambda pred: {a: int(((pred == a) & (yb == a)).sum()) for a in sorted(set(yb))}  # noqa: E731
        pl, pc = per(live_b), per(cb)
        combined = {"dropped": [c["text"] for c in culprits if c["kind"] == "label_added"],
                    "blind": round(float((cb == yb).mean()), 4), "live_blind": round(float((live_b == yb).mean()), 4),
                    "per_agent": {names.get(a, a): f"{pl[a]}→{pc[a]}" for a in pl},
                    "still_broken": int(((live_b == yb) & (cb != yb)).sum()),
                    "agent_gate": all(pc[a] >= pl[a] - 1 for a in pl), "blind_gate": float((cb == yb).mean()) >= float((live_b == yb).mean()) - 0.01}
    advice = []
    fam = _families()
    for c in culprits:
        why = _guard(c["text"])
        f = fam.get(" ".join(c["text"].lower().split()))
        rule = FAMILY_RULE[f][1] if f in FAMILY_RULE else _rule_shaped(c["text"])
        a = {"lesson": c["text"], "agent": c["agent_name"], "source": "antibody" if f else c["source"], "family": f, "brings_back": c["brings_back"],
             "golden_back": c.get("golden_back") or [],
             "helps": c["helps"], "examples": c["examples"], "guard": FAMILY_RULE[f][0] if f in FAMILY_RULE else None}
        if why:
            a.update(kind="redundant", say=f"A guardrail rule already stops this ({why}) before any model sees it — the lesson only teaches the "
                                            f"router a wrong lesson about similar wording. Remove it.", action="remove")
        elif rule:
            a.update(kind="rule", say=f"This lesson {rule}. Taught as a lesson, the router learns the wording instead and declines look-alike "
                                      f"requests of our own. Move it to a rule, then remove the lesson.", action="rule")
        elif f:                                             # a safety lesson: the blind exam has no attacks, so "helps 0" says nothing
            a.update(kind="contrast", say="A safety lesson the blind exam cannot value. Keep it only with counter-examples that look like it but "
                                          "belong to the agent the broken sentences want — or replace it with a guardrail rule.", action="contrast")
        elif c["helps"] == 0:
            a.update(kind="remove", say="It fixes nothing on the blind exam and breaks the sentences below. Remove it, or re-label it.", action="remove")
        else:
            a.update(kind="contrast", say=f"It also fixes {c['helps']} blind sentence(s), so keep it — and teach the difference: add a few examples "
                                          f"that look like this lesson but belong to the agent the broken sentences want.", action="contrast")
        advice.append(a)
    explained = {i for c in culprits for i in c["back_idx"]}
    explained_g = {i for c in culprits for i in c["back_g_idx"]}
    loose = [{"text": blind[i][0], "want": blind[i][1], "want_name": names.get(blind[i][1], blind[i][1]), "got_name": names.get(str(full_b[i]), str(full_b[i])),
              "lang": router.language(blind[i][0]), "set": "blind"} for i in broken if int(i) not in explained]
    loose += [{"text": gold[i]["text"], "want": gold[i]["agent"], "want_name": names.get(gold[i]["agent"], gold[i]["agent"]),
               "got_name": names.get(str(full_g[i]), str(full_g[i])), "lang": router.language(gold[i]["text"]), "set": "golden"} for i in broken_g if int(i) not in explained_g]
    yield {"type": "done", "culprits": [{k: v for k, v in c.items() if k not in ("back_idx", "back_g_idx", "type")} for c in culprits], "combined": combined,
           "unexplained": loose,
           "advice": advice, "ms": int((time.time() - t0) * 1000),
           "note": "lessons interact: one left out at a time, then all culprits together"}


def same_need(text: str, want_agent: str, n: int = 6) -> dict:
    """For a GOLDEN sentence no single lesson explains: fresh sentences with the same need, worded differently (never for blind ones)."""
    import doctor
    import gates
    if " ".join(text.lower().split()) not in {" ".join(g["text"].lower().split()) for g in gates.golden()}:
        raise ValueError("only golden sentences can seed examples — the blind exam must stay unseen")
    return doctor.generate(want_agent, n, hint=f"customer messages in the same language with the same need as \"{text}\" — worded differently, "
                                                f"never copying it")


def contrast(lesson: str, want_agent: str, n: int = 6) -> dict:
    """Counter-examples drafted from the LESSON (never from the exam): look like it, belong to `want_agent`."""
    import doctor
    return doctor.generate(want_agent, n, hint=f"messages that look very much like \"{lesson}\" — same language, same kind of wording — "
                                               f"but are about OpenWallet's OWN wallet and features, so they belong to this agent")
