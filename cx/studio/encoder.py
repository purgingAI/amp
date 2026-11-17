"""Sentence vectors for the studio: the router's reflex and the document retriever share one encoder.

The base model is the multilingual MiniLM our intent model was fine-tuned from (English, Hinglish,
Hindi, Arabic). A fine-tuned retriever version is just another directory with the same layout, so
`Encoder(path)` loads either. Vectors are mean-pooled over real tokens and L2-normalised, so a dot
product IS the cosine similarity.
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")              # the base model is in the local cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base-model")   # a self-contained copy (server)
BASE = os.environ.get("STUDIO_BASE_MODEL") or (_LOCAL_BASE if os.path.isfile(os.path.join(_LOCAL_BASE, "config.json"))
                                                 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
torch.set_num_threads(int(os.environ.get("STUDIO_THREADS", "3")))   # leave cores for the live engine


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


class Encoder:
    def __init__(self, path: str = BASE):
        self.path = path
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], max_len: int = 128, batch: int = 32) -> np.ndarray:
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))          # length-sorted: little padding
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            v = mean_pool(self.model(**enc).last_hidden_state, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, dim=-1).numpy().astype(np.float32)
            out.extend(zip(idx, v))
        out.sort(key=lambda x: x[0])
        return np.stack([v for _, v in out]) if out else np.zeros((0, 384), np.float32)


_cache: dict[str, Encoder] = {}
_lock = threading.Lock()


def get(path: str | None) -> Encoder:
    """One loaded copy per model directory (they are ~470 MB each)."""
    path = path or BASE
    with _lock:
        if path not in _cache:
            if len(_cache) >= 2:                              # keep at most two resident
                _cache.pop(next(iter(_cache)))
            _cache[path] = Encoder(path)
        return _cache[path]
