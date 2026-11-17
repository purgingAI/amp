#!/usr/bin/env python3
"""One training job, run as its own process:  python train_job.py <job dir>

job.json says what to do; every step is appended to events.jsonl and summarised in state.json, which
the studio streams to the dashboard. Two kinds:

  rag_tune     fine-tune the document retriever on a collection, round after round, until it reaches
               the target score (or the round limit). Starts from the ACTIVE version, so earlier
               versions stay intact; the result is a new version waiting to be promoted.
  router_tune  retrain Laya's reflex (the fast classifier in front of Laya) from every agent's
               examples, scored on held-out examples against the active head.
"""
from __future__ import annotations

import collections
import json
import math
import shutil
import os
import pickle
import random
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JOB = os.path.abspath(sys.argv[1])
CFG = json.load(open(os.path.join(JOB, "job.json"), encoding="utf-8"))
os.environ.setdefault("STUDIO_THREADS", str(CFG.get("threads", 4)))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import encoder                                                # noqa: E402
import llm                                                    # noqa: E402
import rag                                                    # noqa: E402
import store                                                  # noqa: E402

STATE = {"id": os.path.basename(JOB), "kind": CFG["kind"], "status": "running", "phase": "starting", "progress": 0.0,
         "started": time.time(), "pid": os.getpid(), "config": CFG, "rounds": [], "result": None}
_stop = False


def _sig(*_):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _sig)


def emit(kind: str, **kw):
    ev = {"t": round(time.time(), 2), "type": kind, **kw}
    with open(os.path.join(JOB, "events.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    if kind == "phase":
        STATE["phase"] = kw.get("name")
    for k in ("progress", "eta_s"):
        if k in kw:
            STATE[k] = kw[k]
    STATE["updated"] = time.time()
    tmp = os.path.join(JOB, "state.json.tmp")
    json.dump(STATE, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, os.path.join(JOB, "state.json"))


def check_stop():
    if _stop:
        raise KeyboardInterrupt("cancelled")


# ════════════════════════════════════════════════════════════════════════════════════
# RAG: questions → exam → train rounds → version
# ════════════════════════════════════════════════════════════════════════════════════
QSYS = ("You write training data for a bank-wallet search engine used by UAE customers. For each passage, write questions a "
        "real OpenWallet customer would type that THIS passage answers. Mix languages the way customers do: about half in English, "
        "at least one in Hinglish (romanised Hindi, e.g. 'refund aane mein kitna time lagta hai'), and one in Arabic or Hindi "
        "(Devanagari), alternating between passages. Vary wording and length; keep one short and casual. Never copy a "
        "sentence from the passage. Output JSON only.")
_NON_EN = re.compile(r"[\u0900-\u097F\u0600-\u06FF]|\b(kya|kaise|kitna|kitne|hai|mein|mai|kab|lagta|chahiye|paisa|wapas|mujhe|mera)\b", re.I)


def gen_questions(chunks: list[dict], per: int, note: str = "") -> dict[str, list[str]]:
    """{chunk id: [questions]} — Claude when there is a key, else simple templates."""
    out: dict[str, list[str]] = {}
    if not llm.api_key():
        for c in chunks:
            head = c.get("heading") or c["text"].split(".")[0][:80]
            words = [w for w in re.findall(r"[A-Za-z]{5,}", c["text"])][:3]
            out[c["id"]] = [f"what does OpenWallet say about {head.lower()}?", f"tell me about {' '.join(words).lower()}"][:per]
        return out
    for s in range(0, len(chunks), 6):
        check_stop()
        batch = chunks[s:s + 6]
        prompt = (f"Write {per} questions for each passage{note}. Output ONLY lines of the form\n"
                  "<passage id> ||| <question>\none question per line, nothing else.\n\n" +
                  "\n\n".join(f"PASSAGE id={c['id']}\n{(c.get('heading') or '')}\n{c['text'][:1400]}" for c in batch))
        ids = {c["id"] for c in batch}
        try:
            raw = llm.complete(QSYS.replace("Output JSON only.", "Follow the output format exactly."), prompt, max_tokens=1800)
            for line in raw.splitlines():
                if "|||" not in line:
                    continue
                cid, q = (x.strip().strip("-*•").strip() for x in line.split("|||", 1))
                cid = cid.replace("id=", "").strip()
                if cid in ids and 6 < len(q) < 220 and len(out.setdefault(cid, [])) < per:
                    out[cid].append(q)
        except Exception as exc:                            # noqa: BLE001 — one bad batch must not end the run
            emit("log", level="warn", text=f"question batch {s // 6 + 1} skipped: {llm.error_text(exc)}")
        emit("step", phase="questions", done=min(s + 6, len(chunks)), total=len(chunks),
             progress=0.05 + 0.20 * min(s + 6, len(chunks)) / len(chunks))
    return out


TIMELAPSE = {"stages": [], "passages": [], "questions": []}


def _timelapse(label, chunks, exam, P, Q, ranks, golds_cos):
    """One frame of the training time-lapse: where every exam question ranks its right passage now.
    Passages keep the positions of the FIRST frame (PCA of their vectors), so motion = real change in rank."""
    if not TIMELAPSE["passages"]:
        X = P - P.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        xy = X @ vt[:2].T
        xy = xy / (np.abs(xy).max(0) + 1e-9)
        TIMELAPSE["passages"] = [{"id": c["id"], "x": round(float(xy[k, 0]), 4), "y": round(float(xy[k, 1]), 4),
                                  "doc": c.get("doc"), "heading": (c.get("heading") or "")[:60]} for k, c in enumerate(chunks)]
        TIMELAPSE["questions"] = [{"q": q, "gold": gold, "ranks": [], "cos": []} for q, gold in exam]
    TIMELAPSE["stages"].append(label)
    for i, qrec in enumerate(TIMELAPSE["questions"]):
        qrec["ranks"].append(int(ranks[i]))
        qrec["cos"].append(round(float(golds_cos[i]), 3))
    json.dump(TIMELAPSE, open(os.path.join(JOB, "timelapse.json"), "w", encoding="utf-8"), ensure_ascii=False)
    emit("timelapse", stage=label, frames=len(TIMELAPSE["stages"]))


def evaluate(model_path, chunks, exam, label=""):
    """recall@1, recall@5, MRR of `exam` questions against all passages; plus the misses."""
    enc = encoder.Encoder(model_path) if model_path is not None else encoder.get(None)
    P = enc.encode([rag._passage_text(c) for c in chunks], max_len=256)
    Q = enc.encode([q for q, _ in exam], max_len=64)
    S = Q @ P.T
    ids = [c["id"] for c in chunks]
    r1 = r5 = mrr = 0.0
    misses, hardneg = [], {}
    ranks, gcos = [], []
    for i, (q, gold) in enumerate(exam):
        order = np.argsort(-S[i])
        rank = next((k for k, j in enumerate(order) if ids[j] == gold), len(order))
        ranks.append(rank)
        gcos.append(S[i][ids.index(gold)] if gold in ids else 0.0)
        r1 += rank == 0
        r5 += rank < 5
        mrr += 1 / (rank + 1)
        if rank >= 5:
            misses.append({"q": q, "gold": gold, "got": ids[order[0]]})
    n = max(1, len(exam))
    m = {"recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4), "mrr": round(mrr / n, 4), "n": len(exam)}
    m["score"] = round(100 * m["recall@5"], 1)
    if label:
        _timelapse(label, chunks, exam, P, Q, ranks, gcos)
    return m, misses


def mine_negatives(model_path, chunks, pairs) -> dict[str, str]:
    """For each training question, the most similar WRONG passage under the current model."""
    enc = encoder.Encoder(model_path) if model_path is not None else encoder.get(None)
    P = enc.encode([rag._passage_text(c) for c in chunks], max_len=256)
    Q = enc.encode([q for q, _ in pairs], max_len=64)
    S = Q @ P.T
    ids = [c["id"] for c in chunks]
    out = {}
    for i, (q, gold) in enumerate(pairs):
        for j in np.argsort(-S[i])[:3]:
            if ids[j] != gold:
                out[q] = ids[j]
                break
    return out


def train_round(init_path, chunks, pairs, negs, rnd, out_dir, epochs, lr, batch=16, tau=0.05):
    from transformers import AutoModel, AutoTokenizer
    src = init_path or encoder.BASE
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src)
    model.embeddings.requires_grad_(False)                     # 82% of the weights; frozen = 3× faster, no forgetting
    model.train()
    text = {c["id"]: rag._passage_text(c) for c in chunks}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(pairs) / batch)
    total = steps_per_epoch * epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    emit("phase", name=f"training round {rnd}", round=rnd, steps=total, trainable=sum(p.numel() for p in params),
         frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))

    def enc(texts, n):
        e = tok(texts, padding=True, truncation=True, max_length=n, return_tensors="pt")
        return torch.nn.functional.normalize(encoder.mean_pool(model(**e).last_hidden_state, e["attention_mask"]), dim=-1)

    step, t0, losses = 0, time.time(), []
    rng = random.Random(rnd)
    for ep in range(epochs):
        order = pairs[:]
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            check_stop()
            b = order[s:s + batch]
            q = enc([x[0] for x in b], 64)
            pos = enc([text[x[1]] for x in b], 256)
            neg_ids = [negs.get(x[0]) for x in b]
            neg = enc([text[n] for n in neg_ids if n], 256) if any(neg_ids) else None
            cand = torch.cat([pos, neg]) if neg is not None else pos
            logits = q @ cand.T / tau                          # in-batch + mined hard negatives (InfoNCE)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss))
            el = time.time() - t0
            eta = el / step * (total - step)
            if step == 1 or step % 2 == 0 or step == total:
                emit("step", phase="train", round=rnd, epoch=ep + 1, step=step, total=total, loss=round(float(loss), 4),
                     loss_avg=round(sum(losses[-10:]) / len(losses[-10:]), 4), lr=float(sched.get_last_lr()[0]),
                     sec_per_step=round(el / step, 2), eta_s=int(eta),
                     progress=0.3 + 0.55 * ((rnd - 1) + step / total) / CFG.get("max_rounds", 3))
    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"steps": total, "seconds": int(time.time() - t0), "final_loss": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 4)}


def rag_tune():
    cid = CFG["collection"]
    chunks = rag.all_chunks(cid)
    if len(chunks) < 8:
        raise ValueError("this collection needs at least 8 passages to fine-tune on — upload more documents")
    target, max_rounds = float(CFG.get("target", 85)), int(CFG.get("max_rounds", 3))
    emit("phase", name="reading documents", passages=len(chunks), docs=len({c["doc"] for c in chunks}), progress=0.02)

    # 1. questions (cached per passage, so a re-run only writes questions for new passages)
    qfile = store.path(f"collections/{cid}/questions.jsonl")
    have = {}
    if os.path.exists(qfile):
        for line in open(qfile, encoding="utf-8"):
            o = json.loads(line)
            have.setdefault(o["chunk"], []).extend(o["questions"])
    todo = [c for c in chunks if c["id"] not in have]
    emit("phase", name="writing training questions", new=len(todo), cached=len(chunks) - len(todo),
         writer=("Claude " + llm.QUESTION_MODEL) if llm.api_key() else "templates (no API key)", progress=0.05)
    fresh = gen_questions(todo, int(CFG.get("questions_per_passage", 4))) if todo else {}
    with open(qfile, "a", encoding="utf-8") as fh:
        for k, qs in fresh.items():
            fh.write(json.dumps({"chunk": k, "questions": qs}, ensure_ascii=False) + "\n")
            have[k] = qs
    # passages whose questions are all English get a Hinglish + Arabic/Hindi top-up (older runs wrote mostly English)
    if llm.api_key():
        mono = [c for c in chunks if c["id"] in have and not any(_NON_EN.search(q) for q in have[c["id"]])]
        if mono:
            emit("phase", name="writing Hinglish, Hindi and Arabic questions", passages=len(mono), progress=0.22)
            extra = gen_questions(mono, 2, note=" — write them ONLY in Hinglish (romanised Hindi) and in Arabic or Hindi, never English")
            with open(qfile, "a", encoding="utf-8") as fh:
                for k, qs in extra.items():
                    fh.write(json.dumps({"chunk": k, "questions": qs}, ensure_ascii=False) + "\n")
                    have[k] = have.get(k, []) + qs
            emit("log", text=f"added {sum(len(q) for q in extra.values())} multilingual questions for {len(extra)} passages")
    live = {c["id"] for c in chunks}
    qa = [(q, k) for k, qs in have.items() if k in live for q in qs]
    langs = collections.Counter("hi" if re.search(r"[\u0900-\u097F]", q) else "ar" if re.search(r"[\u0600-\u06FF]", q)
                                else "hinglish" if _NON_EN.search(q) else "en" for q, _ in qa)
    emit("log", text="questions by language: " + ", ".join(f"{k} {v}" for k, v in langs.most_common()))
    emit("log", text=f"{len(qa)} questions for {len(live)} passages", sample=[q for q, _ in random.Random(1).sample(qa, min(6, len(qa)))])

    # 2. a fixed exam: 20% of questions, never trained on (stable across rounds and runs)
    exam = [x for x in qa if hash_bucket(x[0]) < 20]
    pairs = [x for x in qa if hash_bucket(x[0]) >= 20]
    emit("phase", name="setting the exam", exam=len(exam), train=len(pairs), progress=0.27)

    # 3. baseline: the active version (what customers get today)
    base_ver = store.kb_active(cid)                    # this knowledge base's live version — never another's
    base_m, base_miss = evaluate(store.retriever_path(base_ver), chunks, exam, label=f"live {base_ver}")
    emit("metric", name="baseline", version=base_ver, **base_m, progress=0.3)
    scratch = bool(CFG.get("from_scratch"))
    if scratch and base_ver != "base":
        # start over from the untrained model: the live version is only the yardstick, not the starting point
        s_m, s_miss = evaluate(None, chunks, exam, label="untrained")
        emit("log", text=f"starting from scratch — untrained model scores {s_m['score']} (live {base_ver}: {base_m['score']})")
        best = {"path": None, "metrics": s_m, "ver": "base", "misses": s_miss, "yardstick": base_m}
    else:
        best = {"path": store.retriever_path(base_ver), "metrics": base_m, "ver": base_ver, "misses": base_miss}
    start_ver = best["ver"]
    work = store.path(f"retriever/_work_{STATE['id']}/x")
    work = os.path.dirname(work)
    for rnd in range(1, max_rounds + 1):
        check_stop()
        if rnd > 1 and llm.api_key() and best["misses"]:
            missed = {m["gold"] for m in best["misses"]}
            extra = gen_questions([c for c in chunks if c["id"] in missed], 3,
                                  note=" — the search engine got these wrong, so write harder, differently-worded questions")
            add = [(q, k) for k, qs in extra.items() for q in qs if hash_bucket(q) >= 20]
            pairs += add
            emit("log", text=f"round {rnd}: {len(add)} harder questions for {len(missed)} passages it missed")
        emit("phase", name=f"mining hard negatives (round {rnd})", progress=0.3 + 0.55 * (rnd - 1) / max_rounds)
        negs = mine_negatives(best["path"], chunks, pairs)
        out = os.path.join(work, f"round{rnd}")
        info = train_round(best["path"], chunks, pairs, negs, rnd, out, int(CFG.get("epochs", 2)), float(CFG.get("lr", 2e-5)))
        if info["steps"]:
            store.write("speed.json", {"sec_per_step": round(info["seconds"] / info["steps"], 2), "at": time.time()})   # honest estimates next time
        emit("phase", name=f"scoring round {rnd}")
        m, miss = evaluate(out, chunks, exam, label=f"round {rnd}")
        better = m["score"] > best["metrics"]["score"] or (m["score"] == best["metrics"]["score"] and m["mrr"] > best["metrics"]["mrr"])
        STATE["rounds"].append({"round": rnd, **info, **m, "better": better})
        emit("metric", name=f"round {rnd}", round=rnd, better=better, **m, **info)
        if better:
            best = {"path": out, "metrics": m, "ver": None, "misses": miss, "yardstick": best.get("yardstick")}
        if best["metrics"]["score"] >= target:
            emit("log", text=f"target {target:g} reached ({best['metrics']['score']:g}) — stopping after round {rnd}")
            break
        if rnd < max_rounds:
            emit("log", text=f"score {best['metrics']['score']:g} is below the target {target:g} — another round")

    # 4. a new version (only if training beat what is active)
    if best["ver"] is None:
        ver = store.next_retriever_id()
        final = os.path.join(store.DATA, "retriever", ver)
        os.replace(best["path"], final)
        emit("phase", name="indexing with the new version", version=ver, progress=0.92)
        rag.index(cid, ver)
        v = store.kb_versions(cid)
        v["list"].append({"id": ver, "created": time.time(), "parent": start_ver, "collection": cid, "job": STATE["id"],
                          "from_scratch": scratch, "scores": best["metrics"], "baseline": base_m, "rounds": len(STATE["rounds"]),
                          "by": CFG.get("by")})
        better_than_live = best["metrics"]["score"] > base_m["score"] or base_ver == "base"
        if CFG.get("auto_promote") and better_than_live:
            v["active"] = ver
        store.save_kb_versions(cid, v)
        prune_retrievers(cid, keep=int(CFG.get("keep_versions", 3)))
        result = {"version": ver, "collection": cid, "promoted": v["active"] == ver, "from_scratch": scratch, "baseline": base_m, "best": best["metrics"],
                  "reached_target": best["metrics"]["score"] >= target}
    else:
        result = {"version": None, "baseline": base_m, "best": best["metrics"], "reached_target": base_m["score"] >= target,
                  "note": "no round beat the active version — nothing new to promote"}
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    if CFG.get("then_router", True):
        result["router"] = router_tune(inner=True)
    return result


def prune_retrievers(cid: str, keep: int = 3) -> None:
    """Disk is finite (~470 MB a version): per knowledge base keep its live version and the `keep`
    newest; the record of every version (scores, parent, job) stays in its versions.json."""
    import shutil
    v = store.kb_versions(cid)
    on_disk = [x for x in v["list"] if x["id"] != "base" and os.path.isdir(os.path.join(store.DATA, "retriever", x["id"]))]
    on_disk.sort(key=lambda x: x["created"], reverse=True)
    for x in on_disk[keep:]:
        if x["id"] == v["active"]:
            continue
        shutil.rmtree(os.path.join(store.DATA, "retriever", x["id"]), ignore_errors=True)
        for ext in ("npy", "json"):
            try:
                os.remove(os.path.join(store.DATA, "collections", cid, f"index-{x['id']}.{ext}"))
            except OSError:
                pass
        x["pruned"] = time.time()
        emit("log", text=f"removed retriever {x['id']} from disk to save space (its scores stay on record)")
    store.save_kb_versions(cid, v)


def hash_bucket(s: str) -> int:
    import hashlib
    return int(hashlib.md5(s.strip().lower().encode()).hexdigest()[:6], 16) % 100


# ════════════════════════════════════════════════════════════════════════════════════
# Router reflex
# ════════════════════════════════════════════════════════════════════════════════════
# Seeds from our intent corpus. Knowledge agents get NONE: each one learns only from its own documents' questions
# (and examples typed on the canvas), so two knowledge agents never fight over generic "how do I…" sentences —
# those stay with the action agent, whose engine answers the app's journeys.
SEED_MAP = {"human": {"human_handoff"}, "decline": {"out_of_scope"}}


def router_tune(inner=False):
    from sklearn.linear_model import LogisticRegression
    import router
    g = router.graph()
    agents = [n for n in g["nodes"] if n["type"] == "agent"]
    kinds = {n["id"]: n.get("kind") for n in agents}
    # THE DATA. Delta by default: the LIVE head's pinned recipe (its frozen corpus seeds + document questions) plus
    # today's agent examples and labels — so a challenger differs from live by exactly the lessons taught since.
    # A head without a recipe, or {"recipe": "fresh"}, builds from today's data (and pins itself for next time).
    import inbox
    import recipes
    live_ver = store.active("router")
    base = recipes.load(live_ver) if CFG.get("recipe", "delta") == "delta" else None
    labels = [(l["text"], l["agent"]) for l in inbox.training_rows("orchestrator")]
    xf = CFG.get("extra_labels")
    if xf and os.path.exists(xf):                           # an Autopilot mission's STAGED examples (adopted only on promotion)
        extra = [json.loads(l) for l in open(xf, encoding="utf-8") if l.strip()]
        labels += [(e["text"], e["agent"]) for e in extra if e.get("agent")]
        emit("log", text=f"{len(extra)} staged examples from the Autopilot mission joined the labels")
    # a strict delta: the label policy and regularisation are the LIVE recipe's, so only the lessons differ
    pol = dict(CFG)
    if base:
        for k in ("label_boost", "balance_labels", "class_weight", "kb_questions", "seed_corpus"):
            pol.pop(k, None)
            if (base.get("cfg") or {}).get(k) is not None:
                pol[k] = base["cfg"][k]
        asked = {k: CFG[k] for k in ("label_boost", "balance_labels", "class_weight") if k in CFG and CFG[k] != (base.get("cfg") or {}).get(k)}
        if asked:
            emit("log", text=f"kept the live recipe's label policy (asked for {asked}) — a policy change is a fresh build, not a lesson")
    asm = recipes.assemble(g, pol, labels, base, log=lambda t: emit("log", text=t))
    tagged = asm["rows"]
    labelled, lab_exam = asm["labelled"], asm["lab_exam"]
    if asm["mode"] == "delta":
        L = asm["lessons"]
        emit("lessons", base=live_ver, **{k: L[k] for k in ("n", "labels_added", "labels_removed", "labels_moved", "examples_added", "examples_removed", "agents_added", "agents_removed")})
        emit("log", text=f"challenger = live {live_ver} + {L['n']} lesson(s): {len(L['labels_added'])} label(s) added, {len(L['labels_removed'])} removed, "
                         f"{len(L['labels_moved'])} re-labelled, {len(L['examples_added'])} example(s) added, {len(L['examples_removed'])} removed"
                         + (f", agents +{len(L['agents_added'])}/−{len(L['agents_removed'])}" if L['agents_added'] or L['agents_removed'] else ""))
    else:
        emit("log", text=("built FRESH from today's data (asked for)" if CFG.get("recipe") == "fresh" else
                          f"the live head {live_ver} has no pinned recipe — built fresh from today's data; this head pins its own recipe"))
    rows = [(t, a) for t, a, _ in tagged]
    # topic ownership: an ACTION example that is nearly the same question as a knowledge agent's own
    # document questions belongs to that knowledge agent now ("refund kitne din mein aata hai" vs a refund-policy document)
    enc_path = router.encoder_path()
    enc = encoder.get(enc_path)
    kinds_of = {n["id"]: n.get("kind") for n in agents}
    owned = [(t, a) for t, a in rows if kinds_of.get(a) == "knowledge" and t not in {e for n in agents for e in n.get("examples", [])}]
    owner_vecs = None
    # OFF by default: measured 2026-10-19 — the intent encoder groups by topic, so similarity cannot tell
    # "how long do refunds take?" (policy) from "where is my refund?" (personal); even at 0.95 it moved
    # personal Arabic/Hinglish requests. Opt in per job with {"ownership": 0.97}.
    if owned and CFG.get("ownership"):
        owner_vecs = enc.encode([t for t, _ in owned], max_len=64)
        act_idx = [i for i, (_, a) in enumerate(rows) if kinds_of.get(a) == "action"]
        if act_idx:
            av = enc.encode([rows[i][0] for i in act_idx], max_len=64)
            sim = av @ owner_vecs.T
            best = sim.argmax(1)
            moved = []
            for j, i in enumerate(act_idx):
                if sim[j, best[j]] >= float(CFG.get("ownership", 0.85)):
                    rows[i] = (rows[i][0], owned[best[j]][1])
                    moved.append(rows[i][0])
            if moved:
                emit("log", text=f"{len(moved)} action examples are the same questions a knowledge agent's documents answer — they now train that agent",
                     sample=moved[:6])
    tagged = [[t, a, src] for (t, a), (_, _, src) in zip(rows, tagged)]          # ownership may have moved an agent
    counts = {}
    for _, a in rows:
        counts[a] = counts.get(a, 0) + 1
    trained = [a for a, c in counts.items() if c >= 5]
    rows = [r for r in rows if r[1] in trained]
    tagged = [r for r in tagged if r[1] in trained]
    emit("phase", name="router: embedding examples", examples=len(rows), per_agent=counts, progress=0.88 if inner else 0.1)
    if len(trained) < 2:
        raise ValueError("the router needs at least two agents with 5+ example questions each")
    exam = [r for r in rows if hash_bucket(r[0]) < 20]
    train = [r for r in rows if hash_bucket(r[0]) >= 20]
    emit("log", text=f"router encoder: {os.path.basename(enc_path) if enc_path else 'multilingual MiniLM (base)'}")
    X = enc.encode([t for t, _ in train], max_len=64)
    Xe = enc.encode([t for t, _ in exam], max_len=64)
    ye = [a for _, a in exam]
    emit("phase", name="router: training the reflex", progress=0.93 if inner else 0.6)
    best = None
    cw = pol.get("class_weight", "balanced")
    cw = None if cw in (None, "none", "None") else cw
    for C in ((float(base["C"]),) if base and base.get("C") else (2.0, 8.0, 32.0)):          # delta: the live head's C
        clf = LogisticRegression(C=C, max_iter=2000, class_weight=cw)
        clf.fit(X, [a for _, a in train])
        acc = float((clf.predict(Xe) == np.array(ye)).mean()) if ye else 0.0
        emit("log", text=f"router C={C:g}: held-out accuracy {acc:.3f}")
        if not best or acc > best[0]:
            best = (acc, C, clf)
    acc, C, clf = best
    per = {a: round(float((clf.predict(Xe[np.array(ye) == a]) == a).mean()), 3) for a in set(ye)} if ye else {}
    prev = router.head()
    prev_acc = None
    if prev and set(prev["clf"].classes_) >= set(ye) and prev.get("encoder") == enc_path:
        prev_acc = round(float((prev["clf"].predict(Xe) == np.array(ye)).mean()), 4)
    # the honest exam: a blind set no model trained on (the held-out split above shares sentences
    # with the intent model the encoder came from, so it flatters)
    blind_acc, blind_per = None, {}
    bf = os.path.join(os.path.dirname(HERE), "corpus", CFG.get("blind_corpus", "blind_v5.jsonl"))
    if os.path.exists(bf):
        def to_agent(it):
            for aid, kind in kinds.items():
                if kind in SEED_MAP and it in SEED_MAP[kind]:
                    return aid
            return next((aid for aid, k in kinds.items() if k == "action"), None)
        brows = [json.loads(l) for l in open(bf, encoding="utf-8")]
        brows = [(r["text"], to_agent(r.get("intent") or r.get("label"))) for r in brows]
        brows = [r for r in brows if r[1] in set(clf.classes_)]
        if brows and owner_vecs is not None:
            # blind sentences a knowledge agent now owns are not "action" any more — leave them out, and say how many
            bv = enc.encode([t for t, _ in brows], max_len=64)
            keep = (bv @ owner_vecs.T).max(1) < float(CFG.get("ownership", 0.85))
            dropped = int((~keep).sum())
            brows = [r for r, k in zip(brows, keep) if k]
            if dropped:
                emit("log", text=f"blind exam: {dropped} sentences now belong to a knowledge agent's topic and were left out")
        if brows:
            bp = clf.predict(enc.encode([t for t, _ in brows], max_len=64))
            by = np.array([a for _, a in brows])
            blind_acc = round(float((bp == by).mean()), 4)
            blind_per = {a: f"{int(((bp == a) & (by == a)).sum())}/{int((by == a).sum())}" for a in set(by)}
            emit("log", text=f"blind exam ({os.path.basename(bf)}, {len(brows)} unseen sentences): {100 * blind_acc:.1f}%")
    lab_acc = None
    if lab_exam:
        le = clf.predict(enc.encode([t for t, _ in lab_exam], max_len=64))
        lab_acc = round(float(np.mean([p == a for p, (_, a) in zip(le, lab_exam)])), 4)
        emit("log", text=f"human-label exam: {100 * lab_acc:.1f}% of {len(lab_exam)} held-back labels right")
    ver = store.next_version("router")
    with open(store.path(f"router/{ver}.pkl"), "wb") as fh:
        pickle.dump({"clf": clf, "version": ver, "encoder": enc_path, "labels": list(clf.classes_)}, fh)
    confusion = {}
    if ye:
        for t, p in zip(ye, clf.predict(Xe)):
            if t != p:
                confusion[f"{t}→{p}"] = confusion.get(f"{t}→{p}", 0) + 1
    v = store.versions()
    headline = blind_acc if blind_acc is not None else acc
    rec = {"id": ver, "created": time.time(), "challenger": bool(CFG.get("challenger")), "scores": {"accuracy": round(acc, 4), "score": round(100 * headline, 1), "per_agent": per,
           "blind": blind_acc, "blind_per_agent": blind_per, "labels_exam": lab_acc, "human_labels": len(labelled),
           "n": len(ye), "previous": prev_acc}, "agents": list(clf.classes_), "C": C, "examples": len(rows), "job": STATE["id"],
           "encoder": os.path.basename(enc_path) if enc_path else "base",
           "confusion": dict(sorted(confusion.items(), key=lambda x: -x[1])[:8])}
    rec.update(parent=live_ver if asm["mode"] == "delta" else None, recipe=asm["mode"],
               lessons=({k: (len(v_) if isinstance(v_, list) else v_) for k, v_ in asm["lessons"].items() if k != "pinned"} if asm["lessons"] else None))
    recipes.save(ver, {"version": ver, "parent": rec["parent"], "created": rec["created"], "mode": asm["mode"],
                       "cfg": {k: pol.get(k) for k in ("kb_questions", "label_boost", "balance_labels", "class_weight", "seed_corpus", "ownership") if pol.get(k) is not None},
                       "rows": tagged, "labels": [list(x) for x in labelled], "agents": asm["agents"], "examples": asm["examples"],
                       "C": C, "encoder": rec["encoder"], "lessons": asm["lessons"]})
    emit("log", text=f"recipe pinned: {len(tagged)} rows — the next challenger will be {ver} + lessons if {ver} goes live")
    v["router"]["list"].append(rec)
    if CFG.get("auto_promote") or v["router"]["active"] is None:
        v["router"]["active"] = ver
    store.save_versions(v)
    emit("metric", name="router", version=ver, accuracy=round(acc, 4), score=round(100 * headline, 1), blind=blind_acc,
         blind_per_agent=blind_per, per_agent=per, previous=prev_acc,
         confusion=rec["confusion"])
    return {"version": ver, "accuracy": round(acc, 4), "previous": prev_acc, "promoted": v["router"]["active"] == ver}


# ════════════════════════════════════════════════════════════════════════════════════
# Intent model (the action agent's brain): human labels → warm-started challenger → exams → gates
# ════════════════════════════════════════════════════════════════════════════════════
CX = os.path.dirname(HERE)
MODELS = os.path.join(CX, "model")
CORPUS = os.path.join(CX, "corpus")


def _eval_model(path: str, rows: list[dict], labels: list[str]) -> list[str]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            check_stop()
            b = rows[s:s + 64]
            enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=48, return_tensors="pt")
            ids = m(**enc).logits.argmax(-1).tolist()
            out += [labels[i] if i < len(labels) else "?" for i in ids]
    return out


def _metrics(rows, pred) -> dict:
    n = len(rows)
    ok = sum(p == r["intent"] for p, r in zip(pred, rows))
    leaks = sum(r["intent"] == "out_of_scope" and p != "out_of_scope" for p, r in zip(pred, rows))
    refused = sum(r["intent"] != "out_of_scope" and p == "out_of_scope" for p, r in zip(pred, rows))
    by_lang = {}
    for p, r in zip(pred, rows):
        lg = r.get("lang") or "en"
        d = by_lang.setdefault(lg, {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += p == r["intent"]
    by_intent = {}
    for p, r in zip(pred, rows):
        d = by_intent.setdefault(r["intent"], {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += p == r["intent"]
    return {"n": n, "ok": ok, "acc": round(100 * ok / max(1, n), 1), "leaks": leaks, "refused": refused, "by_lang": by_lang, "by_intent": by_intent}


def _confusion(rows, pred, top: int = 12) -> list[dict]:
    """The most frequent (wanted → got) mistakes, with an example sentence each."""
    c = collections.Counter((r["intent"], p) for p, r in zip(pred, rows) if p != r["intent"])
    ex = {}
    for p, r in zip(pred, rows):
        if p != r["intent"]:
            ex.setdefault((r["intent"], p), r["text"][:80])
    return [{"want": a, "got": b, "n": n, "example": ex.get((a, b))} for (a, b), n in c.most_common(top)]


def _probe(path: str, texts: list[str], labels: list[str]) -> list[dict]:
    """What the model attends to on a probe sentence: attention averaged over heads, per layer (small),
    plus its top intents — enough to show how the challenger's reading differs from the live model's."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="eager").eval()
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok([t], truncation=True, max_length=16, return_tensors="pt")
            o = m(**enc, output_attentions=True)
            probs = o.logits[0].softmax(-1)
            top = probs.argsort(descending=True)[:3].tolist()
            out.append({"text": t, "tokens": tok.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
                        "layers": [[[round(float(x), 3) for x in row] for row in layer[0].mean(0)] for layer in o.attentions],
                        "top": [{"intent": labels[i] if i < len(labels) else "?", "p": round(float(probs[i]), 3)} for i in top]})
    return out


PROBES = ["where is my refund", "mera balance kitna hai", "send 500 to Alex"]


def _read(name: str) -> list[dict]:
    p = name if os.path.isabs(name) else os.path.join(CORPUS, name)
    if not os.path.exists(p):
        return []
    rows = []
    for l in open(p, encoding="utf-8"):
        try:
            o = json.loads(l)
        except ValueError:
            continue
        it = o.get("intent") or o.get("label")
        if o.get("text") and it:
            rows.append({"text": o["text"], "intent": it, "lang": o.get("lang")})
    return rows


def _temperature(d: str):
    try:
        return json.load(open(os.path.join(d, "calibration.json"), encoding="utf-8")).get("temperature")
    except (OSError, ValueError):
        return None


def intent_tune():
    import actions
    import inbox
    labels = inbox.intent_labels()
    live = actions.live_run()
    live_dir = os.path.join(MODELS, actions.PREFIX + live) if live else None
    if not live_dir or not os.path.isdir(live_dir):
        raise ValueError("cannot find the live intent model on this machine")
    if shutil.disk_usage(MODELS).free < 3 * 2**30:
        raise ValueError("less than 3 GB of disk free — a new intent model needs about 1 GB; free space first")
    human = inbox.training_rows("intent")
    exam = [{"text": l["text"], "intent": l["intent"], "lang": l.get("lang")} for l in human if hash_bucket(l["text"]) < 20]
    train = [{"text": l["text"], "intent": l["intent"]} for l in human if hash_bucket(l["text"]) >= 20]
    if len(train) < int(CFG.get("min_labels", 5)):
        raise ValueError(f"only {len(train)} human intent labels to learn from — label more in the Teach-me inbox")
    emit("phase", name="building the training set", human=len(human), progress=0.04)
    # replay: a stratified sample of the corpus the live model learned from, so it does not forget
    base = _read(CFG.get("replay_corpus", "train_v24.jsonl"))
    rng = random.Random(11)
    by = collections.defaultdict(list)
    for r in base:
        by[r["intent"]].append(r)
    per = int(CFG.get("replay_per_intent", 120))
    replay = [x for it, xs in by.items() for x in (rng.sample(xs, min(per, len(xs))) if per else xs)]
    extra = []
    for k in CFG.get("extra_sets") or []:
        rows_k = _read({"abuse_train": "abuse_train.jsonl", "brd_train": "brd_train.jsonl"}.get(k, ""))
        extra += rows_k
        emit("log", text=f"extra corpus {k}: {len(rows_k)} sentences")
    share = float(CFG.get("label_share", 0.2))
    cap = max(1, int(share * (len(replay) + len(extra)) / max(0.01, 1 - share)))   # human labels ≤ share of what this run sees
    boosted = (train * 4)[:cap] if train else []
    data = replay + extra + boosted
    rng.shuffle(data)
    langs_d = collections.Counter((r.get("lang") or "en*") for r in replay + extra)
    emit("dataset", rows=len(data), replay=len(replay), extra=len(extra), labels=len(train), boosted=len(boosted), exam=len(exam),
         per_intent=dict(collections.Counter(r["intent"] for r in data).most_common()), langs=dict(langs_d.most_common(6)),
         sources=[CFG.get("replay_corpus", "train_v24.jsonl")] + list(CFG.get("extra_sets") or []) + (["teach-inbox labels"] if train else []),
         recipe={k: CFG.get(k) for k in ("epochs", "lr", "replay_per_intent", "label_share", "supcon", "freeze_embeddings", "extra_sets")})
    tf = os.path.join(JOB, "train.jsonl")
    with open(tf, "w", encoding="utf-8") as fh:
        for r in data:
            fh.write(json.dumps({"text": r["text"], "intent": r["intent"]}, ensure_ascii=False) + "\n")
    emit("log", text=f"training set: {len(replay)} replayed corpus sentences + {len(extra)} extra + {len(boosted)} from {len(train)} human labels "
                     f"(capped at {int(share * 100)}%); {len(exam)} labels held back as an exam")
    existing = [int(re.sub(r"\D", "", d[len(actions.PREFIX):]) or 0) for d in os.listdir(MODELS) if d.startswith(actions.PREFIX)]
    jn = os.path.join(CX, "docs", "model-brief", "learn", "models")
    existing += [int(re.sub(r"\D", "", f) or 0) for f in os.listdir(jn) if f.startswith("v")] if os.path.isdir(jn) else []
    run = f"v{max(existing) + 1}"
    out_name = actions.PREFIX + run
    emit("phase", name=f"training {run} (warm start from {live})", version=run, progress=0.08)
    env = dict(os.environ, CX_TRAIN_FILE=tf, CX_DEV_FILE=os.path.join(CORPUS, "dev_multi.jsonl"),
               CX_EVAL_FILE=os.path.join(CORPUS, "hard_eval_all.jsonl"), CX_INIT_FROM=live_dir, CX_OUT_NAME=out_name,
               CX_CKPT_EVERY="100000", CX_BACKBONE_PATH=encoder.BASE, CX_LOG_EVERY=str(int(CFG.get("log_every", 5))))
    import subprocess
    t0 = time.time()
    epochs = max(1, min(3, int(CFG.get("epochs", 1))))
    args = [sys.executable, os.path.join(MODELS, "train_multi.py"), "--epochs", str(epochs), "--lr", str(CFG.get("lr", 2e-5)),
            "--threads", str(CFG.get("threads", 4))]
    if float(CFG.get("supcon", 0) or 0) > 0:
        args += ["--supcon", str(float(CFG["supcon"]))]
    if CFG.get("freeze_embeddings"):
        args += ["--freeze-embeddings"]
    p = subprocess.Popen(args, cwd=MODELS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_step = (0, 0)
    for line in p.stdout:
        if _stop:
            p.terminate()
            raise KeyboardInterrupt("cancelled")
        m = re.search(r"epoch (\d+)/(\d+)\s+step (\d+)/(\d+)\s+loss ([\d.]+)\s+(\d+)s", line)
        if m:
            ep, step, total, loss, el = int(m[1]), int(m[3]), int(m[4]), float(m[5]), int(m[6])
            last_step = (step, el)
            emit("step", phase="train", round=1, epoch=ep, epochs=epochs, step=step, total=total, loss=loss, loss_avg=loss, lr=0,
                 sec_per_step=round(el / max(1, step), 2), eta_s=int(el / max(1, step) * (total - step)),
                 progress=0.1 + 0.55 * step / max(1, total))
            continue
        m = re.search(r"epoch (\d+) done\s+dev ([\d.]+)%", line)
        if m:
            emit("epoch", epoch=int(m[1]), epochs=epochs, dev_acc=float(m[2]))
            continue
        elif line.strip() and any(k in line for k in ("train ", "init ", "params", "epoch", "exported", "held-out", "LEAKED", "Error", "error")):
            emit("log", text=line.strip()[:200])
    if p.wait() != 0:
        raise RuntimeError("intent training failed — see the log")
    if last_step[0] >= 20:                                   # remember this machine's speed for the next estimate
        store.write("forge_speed.json", {"sec_per_step": round(last_step[1] / last_step[0], 2), "at": time.time(), "steps": last_step[0]})
    new_dir = os.path.join(MODELS, out_name)
    emit("phase", name="exams: the live model and the challenger on the same sentences", progress=0.68)
    sets = {"blind": _read("blind_v5.jsonl"), "brd": _read("brd_eval.jsonl"), "heldout": _read("hard_eval_all.jsonl"), "labels": exam}
    res = {}
    for i, (name, rows) in enumerate(sets.items()):
        if not rows:
            continue
        emit("phase", name=f"exam {i + 1}/{len(sets)}: {name} ({len(rows)} sentences) — live model, then the challenger", progress=0.68 + 0.04 * i)
        pl, pc = _eval_model(live_dir, rows, labels), _eval_model(new_dir, rows, labels)
        res[name] = {"live": _metrics(rows, pl), "cand": _metrics(rows, pc)}
        emit("log", text=f"{name}: {live} {res[name]['live']['acc']}% → {run} {res[name]['cand']['acc']}% ({len(rows)} sentences)")
        emit("exam", set=name, n=len(rows), live=res[name]["live"], cand=res[name]["cand"], live_version=live, version=run)
        if name == "heldout":
            emit("confusion", set=name, live=_confusion(rows, pl), cand=_confusion(rows, pc), live_version=live, version=run)
    try:
        emit("probe", live={"version": live, "probes": _probe(live_dir, PROBES, labels)}, cand={"version": run, "probes": _probe(new_dir, PROBES, labels)})
    except Exception as exc:                                # noqa: BLE001 — the probe is a picture, never worth a failed job
        emit("log", level="warn", text=f"attention probe skipped ({type(exc).__name__})")
    L = lambda k, side, f="acc": (res.get(k) or {}).get(side, {}).get(f)  # noqa: E731
    gates = []
    def gate(name, ok, detail):
        gates.append({"name": name, "pass": bool(ok), "detail": detail})
    if "brd" in res:
        gate("BRD prompts", L("brd", "cand") >= L("brd", "live"), f"{L('brd', 'cand')}% vs live {L('brd', 'live')}%")
    if "blind" in res:
        gate("Blind exam", L("blind", "cand") >= L("blind", "live") - 0.5, f"{L('blind', 'cand')}% vs live {L('blind', 'live')}% (model alone, unseen)")
    if "heldout" in res:
        gate("Held-out", L("heldout", "cand") >= L("heldout", "live") - 0.5, f"{L('heldout', 'cand')}% vs live {L('heldout', 'live')}%")
        gate("No more leaks", L("heldout", "cand", "leaks") <= L("heldout", "live", "leaks"), f"{L('heldout', 'cand', 'leaks')} vs live {L('heldout', 'live', 'leaks')}")
        gate("No more wrong refusals", L("heldout", "cand", "refused") <= L("heldout", "live", "refused"), f"{L('heldout', 'cand', 'refused')} vs live {L('heldout', 'live', 'refused')}")
        worse = [lg for lg, d in res["heldout"]["cand"]["by_lang"].items()
                 if d["ok"] / d["n"] < res["heldout"]["live"]["by_lang"].get(lg, {"ok": 0, "n": 1})["ok"] / max(1, res["heldout"]["live"]["by_lang"].get(lg, {"n": 1})["n"]) - 0.02]
        gate("Every language", not worse, "no language more than 2 points worse" if not worse else "worse in: " + ", ".join(worse))
    if "labels" in res:
        gate("Human-label exam", L("labels", "cand") >= L("labels", "live"), f"{L('labels', 'cand')}% vs live {L('labels', 'live')}% on {len(exam)} held-back labels")
    passed = all(g_["pass"] for g_ in gates)
    if passed:
        emit("phase", name="exporting the challenger for fast serving (ONNX)", progress=0.85)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(new_dir)
            mdl = AutoModelForSequenceClassification.from_pretrained(new_dir).eval()
            od = os.path.join(new_dir, "onnx")
            os.makedirs(od, exist_ok=True)
            smp = tok(["send 50 dirhams to alex"], return_tensors="pt")
            torch.onnx.export(mdl, (smp["input_ids"], smp["attention_mask"]), os.path.join(od, "model.onnx"),
                              input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                              opset_version=17, dynamo=False)
            tok.save_pretrained(od)
        except Exception as exc:                            # noqa: BLE001 — it still serves with PyTorch
            emit("log", level="warn", text=f"ONNX export failed ({type(exc).__name__}) — the model will serve with PyTorch")
    else:
        emit("log", text="gates failed — no ONNX export (saves ~0.5 GB); it can still be inspected, and promoted only with a warning")
    try:
        shutil.copy(os.path.join(live_dir, "calibration.json"), os.path.join(new_dir, "calibration.json"))
    except OSError:
        pass
    # a journal page, so it appears in Versions next to every other model
    H = res.get("heldout", {}).get("cand")
    rc = {k: CFG.get(k) for k in ("epochs", "lr", "replay_per_intent", "label_share", "supcon", "freeze_embeddings", "extra_sets")} if CFG.get("forge") else None
    page = {"run": run, "title": (f"Forged: {CFG.get('note')}" if CFG.get("note") else f"Forged challenger ({epochs} epoch{'s' if epochs > 1 else ''}, {len(data):,} sentences)") if CFG.get("forge") else f"Studio challenger from {len(train)} human labels", "date": time.strftime("%Y-%m-%d"),
            "story": {"changed": f"Warm start from {live}; {epochs} epoch(s) on {len(replay)} replayed corpus sentences"
                                 + (f" + {len(extra)} from {', '.join(CFG.get('extra_sets'))}" if extra else "")
                                 + f" + {len(train)} human labels from the Teach-me inbox (capped at {int(share * 100)}%)."
                                 + (f" Contrastive term {CFG.get('supcon')}." if float(CFG.get("supcon", 0) or 0) > 0 else "") + (" Embeddings frozen." if CFG.get("freeze_embeddings") else ""),
                      "lesson": "Trained in the Studio's Forge — promote only after reading the gates." if CFG.get("forge") else "Trained by Agent Studio — promote only after reading the gates.",
                      "hyper": {"epochs": epochs, "lr": CFG.get("lr", 2e-5), "init": f"warm start from {live}", "recipe": rc}},
            "card": {"trained_on": {"train": len(data)}},
            "heldout": {"n": H["n"], "ok": H["ok"], "leaks": H["leaks"], "refused": H["refused"], "by_lang": H["by_lang"]} if H else {},
            "brd": {"n": res["brd"]["cand"]["n"], "ok": res["brd"]["cand"]["ok"], "misses": []} if "brd" in res else {},
            "blind_model": {"set": "blind_v5 · model alone", "score": L("blind", "cand"), "n": (res.get("blind") or {}).get("cand", {}).get("n")},
            "calibration": {"temperature": _temperature(new_dir)},
            "decision": {"summary": ("passed every gate" if passed else "failed: " + ", ".join(g_["name"] for g_ in gates if not g_["pass"]))},
            "studio": {"job": STATE["id"], "challenger": True, "forge": bool(CFG.get("forge")), "recipe": rc, "gates": gates,
                       "exams": {k: {s: {kk: vv for kk, vv in v[s].items() if kk != "by_intent"} for s in v} for k, v in res.items()},
                       "human_labels": len(human), "seconds": int(time.time() - t0)}}
    os.makedirs(jn, exist_ok=True)
    json.dump(page, open(os.path.join(jn, f"{run}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # disk: keep at most two Studio-trained challengers that are not live (hand-trained versions are never touched)
    studio_runs = []
    for f in os.listdir(jn):
        try:
            d = json.load(open(os.path.join(jn, f), encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict):                         # index.json is a list of runs, not a model page
            continue
        if (d.get("studio") or {}).get("challenger") and d.get("run") != live:
            studio_runs.append(d["run"])
    for old in sorted(studio_runs, key=lambda r: int(re.sub(r"\D", "", r) or 0))[:-2]:
        shutil.rmtree(os.path.join(MODELS, actions.PREFIX + old), ignore_errors=True)
        emit("log", text=f"removed older Studio challenger {old} from disk (its journal page stays)")
    emit("metric", name="intent", version=run, passed=passed, gates=gates, exams={k: {s: {kk: v[s][kk] for kk in ("acc", "leaks", "refused", "n")} for s in v} for k, v in res.items()})
    return {"version": run, "passed": passed, "gates": gates, "live": live, "human_labels": len(human)}


def main():
    emit("phase", name="starting", progress=0.0)
    try:
        result = rag_tune() if CFG["kind"] == "rag_tune" else intent_tune() if CFG["kind"] == "intent_tune" else router_tune()
        STATE.update(status="done", result=result, finished=time.time(), progress=1.0, eta_s=0)
        emit("done", result=result, progress=1.0)
    except KeyboardInterrupt:
        STATE.update(status="cancelled", finished=time.time())
        emit("error", text="cancelled")
    except Exception as exc:                                # noqa: BLE001
        import traceback
        STATE.update(status="failed", finished=time.time(), error=f"{type(exc).__name__}: {exc}")
        emit("error", text=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    main()
