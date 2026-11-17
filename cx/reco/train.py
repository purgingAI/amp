"""Train the recommender, and narrate every step so a human can watch it learn.

The question this model answers is **"will this customer actually take this offer?"** — not
"is this offer big". A recommender that just sorts by discount size will happily push an
offer nobody acts on, which costs a notification and buys nothing.

## How it is told it is wrong

Binary cross-entropy, weighted. For one example with truth `y` and prediction `p`:

    loss = -w * [ y*log(p) + (1-y)*log(1-p) ]

Two properties matter and both are visible in the UI:

1. **Confidence is punished.** Saying 0.95 when the truth is 0 costs ~3.0; saying 0.55 costs
   ~0.8. The model is penalised for being *confidently* wrong far harder than for being
   unsure, which is exactly the behaviour we want from something that will send messages to
   customers.

2. **The two mistakes do not cost the same.** A **false positive** — we recommend, the
   customer ignores it — spends a notification and a little of their patience, and
   notification fatigue is how a channel dies. A **false negative** is a saving they did not
   hear about. We weight false positives higher (`FP_WEIGHT`), so the model has to be
   genuinely confident before it recommends anything. That weight is a product decision
   written in one place, not an emergent property.

Three models are trained and compared on the SAME chronological test window, because a
number without a baseline is decoration:

  * **base rate** — predict the historical take-up for everyone. Beats more models than
    people expect.
  * **logistic regression** (12 features) — transparent; its learned weights are published
    to the UI, so you can read what it decided mattered.
  * **MiniLM fine-tune** — the pretrained model, fine-tuned on a text rendering of each
    offer. On 144 training rows this is expected to struggle, and if it does we say so.

Everything is emitted through `on_event`, which the training server turns into SSE.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CX)

DATA = os.path.join(HERE, "data", "reco.jsonl")
MODEL_DIR = os.path.join(HERE, "model")

#: A false positive costs more than a false negative. See the docstring.
FP_WEIGHT = 2.0
FN_WEIGHT = 1.0

Emit = Callable[[str, dict], None]


def _noop(kind: str, payload: dict) -> None:
    pass


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------

def load():
    from dataset import FEATURES, split
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    rows.sort(key=lambda r: r["created_at"])
    cut = int(len(rows) * 0.75)
    return rows[:cut], rows[cut:], FEATURES


def standardise(train, test, features):
    """Zero mean, unit variance from the TRAIN window only. Fitting the scaler on everything
    leaks the test distribution into training, which quietly inflates every number below."""
    n = len(train)
    mean = {f: sum(r["features"][f] for r in train) / n for f in features}
    var = {f: sum((r["features"][f] - mean[f]) ** 2 for r in train) / n for f in features}
    std = {f: math.sqrt(v) if v > 1e-9 else 1.0 for f, v in var.items()}

    def vec(r):
        return [(r["features"][f] - mean[f]) / std[f] for f in features]

    return [vec(r) for r in train], [vec(r) for r in test], (mean, std)


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------

def auc(y, p) -> float:
    """Ranking quality, which is what a recommender is judged on: the chance a taken offer
    scores above an ignored one. Accuracy is misleading at a 28% base rate — predicting
    'nobody takes anything' scores 72%."""
    pos = [s for s, t in zip(p, y) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def metrics(y, p, threshold=0.5) -> dict:
    tp = sum(1 for t, s in zip(y, p) if t == 1 and s >= threshold)
    fp = sum(1 for t, s in zip(y, p) if t == 0 and s >= threshold)
    fn = sum(1 for t, s in zip(y, p) if t == 1 and s < threshold)
    tn = sum(1 for t, s in zip(y, p) if t == 0 and s < threshold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(len(y), 1),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "auc": round(auc(y, p), 4),
            # what it would actually cost us, in the units the weights describe
            "cost": round(fp * FP_WEIGHT + fn * FN_WEIGHT, 2)}


def weighted_bce(p: float, y: int) -> tuple[float, float]:
    """Returns (loss, weight). The weight is the penalty multiplier applied to this example."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    w = FP_WEIGHT if y == 0 else FN_WEIGHT
    return -w * (y * math.log(p) + (1 - y) * math.log(1 - p)), w


# ---------------------------------------------------------------------------------------
# 1. the bar
# ---------------------------------------------------------------------------------------

def train_base_rate(train, test, emit: Emit) -> dict:
    rate = sum(r["label"] for r in train) / len(train)
    y = [r["label"] for r in test]
    p = [rate] * len(test)
    m = metrics(y, p)
    emit("model.done", {"model": "base rate", "note": f"predicts {rate:.0%} for everyone",
                        "metrics": m})
    return m


# ---------------------------------------------------------------------------------------
# 2. logistic regression -- transparent
# ---------------------------------------------------------------------------------------

def train_logreg(train, test, features, emit: Emit, epochs=60, lr=0.12) -> dict:
    import torch

    Xtr, Xte, _ = standardise(train, test, features)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor([float(r["label"]) for r in train]).unsqueeze(1)
    xv = torch.tensor(Xte, dtype=torch.float32)
    yv = [r["label"] for r in test]

    torch.manual_seed(11)
    model = torch.nn.Linear(len(features), 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # The class weight IS the punishment described in the docstring, handed to the loss.
    w = torch.tensor([[FP_WEIGHT if r["label"] == 0 else FN_WEIGHT] for r in train])
    lossfn = torch.nn.BCEWithLogitsLoss(weight=w)

    emit("model.start", {
        "model": "logistic regression", "params": len(features) + 1,
        "loss": "weighted binary cross-entropy",
        "fp_weight": FP_WEIGHT, "fn_weight": FN_WEIGHT,
        "explain": "A wrong-and-confident answer costs far more than a wrong-and-unsure one, "
                   "and a false recommendation costs twice a missed one."})

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(xt)
        loss = lossfn(out, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(xv)).squeeze(1).tolist()
        m = metrics(yv, p)
        emit("epoch", {"model": "logistic regression", "epoch": ep + 1, "epochs": epochs,
                       "train_loss": round(loss.item(), 4), "metrics": m})
        time.sleep(0.015)          # the UI is the point; let a human see it move

    with torch.no_grad():
        weights = model.weight.squeeze(0).tolist()
        p = torch.sigmoid(model(xv)).squeeze(1).tolist()

    # What the model decided mattered. This is the bit you can argue with.
    emit("weights", {"model": "logistic regression",
                     "weights": sorted(
                         [{"feature": f, "weight": round(w_, 4)}
                          for f, w_ in zip(features, weights)],
                         key=lambda d: -abs(d["weight"]))})

    # Walk the test window one example at a time, showing what it cost.
    for r, score in zip(test, p):
        loss, penalty = weighted_bce(score, r["label"])
        verdict = ("right" if (score >= 0.5) == (r["label"] == 1) else
                   ("false recommendation" if score >= 0.5 else "missed it"))
        emit("example", {"model": "logistic regression", "text": r["human"],
                         "prediction": round(score, 4), "label": r["label"],
                         "loss": round(loss, 4), "penalty_weight": penalty,
                         "verdict": verdict})
        time.sleep(0.03)

    m = metrics(yv, p)
    emit("model.done", {"model": "logistic regression", "metrics": m})
    os.makedirs(MODEL_DIR, exist_ok=True)
    _, _, scaler = standardise(train, test, features)
    json.dump({"features": list(features), "weights": weights,
               "bias": float(model.bias.item()),
               "mean": scaler[0], "std": scaler[1],
               "fp_weight": FP_WEIGHT, "fn_weight": FN_WEIGHT, "metrics": m},
              open(os.path.join(MODEL_DIR, "reco-logreg-v1.json"), "w"), indent=2)
    return m


# ---------------------------------------------------------------------------------------
# 3. the pretrained model, fine-tuned
# ---------------------------------------------------------------------------------------

def train_minilm(train, test, emit: Emit, epochs=6, bs=16, lr=3e-5) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    emit("model.start", {
        "model": "MiniLM fine-tune", "params": 22_700_000,
        "loss": "weighted binary cross-entropy",
        "fp_weight": FP_WEIGHT, "fn_weight": FN_WEIGHT,
        "explain": "The same penalty, applied to a 22.7M-parameter pretrained encoder reading "
                   "each offer as a sentence. 144 training rows is very little for this, and "
                   "the comparison below is the honest test of whether it helps."})

    torch.manual_seed(11)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)

    def enc(rows):
        return tok([r["human"] for r in rows], truncation=True, padding=True,
                   max_length=64, return_tensors="pt")

    tr_enc, te_enc = enc(train), enc(test)
    ytr = torch.tensor([[float(r["label"])] for r in train])
    wtr = torch.tensor([[FP_WEIGHT if r["label"] == 0 else FN_WEIGHT] for r in train])
    yv = [r["label"] for r in test]

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idx = list(range(len(train)))
    for ep in range(epochs):
        model.train()
        random.Random(ep).shuffle(idx)
        total = 0.0
        for i in range(0, len(idx), bs):
            batch = idx[i:i + bs]
            opt.zero_grad()
            out = model(input_ids=tr_enc["input_ids"][batch],
                        attention_mask=tr_enc["attention_mask"][batch]).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, ytr[batch], weight=wtr[batch])
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(**te_enc).logits).squeeze(1).tolist()
        m = metrics(yv, p)
        emit("epoch", {"model": "MiniLM fine-tune", "epoch": ep + 1, "epochs": epochs,
                       "train_loss": round(total / max(1, len(idx) / bs), 4), "metrics": m})

    for r, score in list(zip(test, p))[:24]:
        loss, penalty = weighted_bce(score, r["label"])
        verdict = ("right" if (score >= 0.5) == (r["label"] == 1) else
                   ("false recommendation" if score >= 0.5 else "missed it"))
        emit("example", {"model": "MiniLM fine-tune", "text": r["human"],
                         "prediction": round(score, 4), "label": r["label"],
                         "loss": round(loss, 4), "penalty_weight": penalty,
                         "verdict": verdict})
        time.sleep(0.02)

    m = metrics(yv, p)
    emit("model.done", {"model": "MiniLM fine-tune", "metrics": m})
    return m


# ---------------------------------------------------------------------------------------

def run(emit: Emit = _noop, with_minilm: bool = True) -> dict:
    train, test, features = load()
    emit("run.start", {
        "train": len(train), "test": len(test),
        "positives_train": sum(r["label"] for r in train),
        "positives_test": sum(r["label"] for r in test),
        "features": list(features),
        "split": "chronological — train on the past, test on the future",
        "source": "campaign_redemption_ledger (real dev reserve→outcome events)",
        "fp_weight": FP_WEIGHT, "fn_weight": FN_WEIGHT,
    })
    results = {}
    results["base rate"] = train_base_rate(train, test, emit)
    results["logistic regression"] = train_logreg(train, test, features, emit)
    if with_minilm:
        try:
            results["MiniLM fine-tune"] = train_minilm(train, test, emit)
        except Exception as e:
            emit("model.failed", {"model": "MiniLM fine-tune", "error": f"{type(e).__name__}: {e}"})

    best = max(results, key=lambda k: results[k]["auc"])
    emit("run.done", {"results": results, "best": best,
                      "verdict": f"{best} ranks best by AUC on the held-out future window"})
    return results


if __name__ == "__main__":
    def p(kind, payload):
        if kind in ("run.start", "model.start", "model.done", "run.done", "weights"):
            print(f"[{kind}] {json.dumps(payload)[:400]}")
        elif kind == "epoch" and payload["epoch"] % 10 == 0:
            print(f"  {payload['model']} ep{payload['epoch']} loss {payload['train_loss']} "
                  f"auc {payload['metrics']['auc']}")
    run(p, with_minilm="--fast" not in sys.argv)
