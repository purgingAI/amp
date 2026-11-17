"""Everything the studio remembers, as plain JSON files under studio/data/ (atomic writes).

  graph.json                       the agent graph: nodes, edges, router settings
  collections/<cid>/meta.json      a document collection and its documents' chunk lists
  collections/<cid>/files/         the uploaded originals
  collections/<cid>/questions.jsonl  generated training questions (reused across runs)
  collections/<cid>/index-<ver>.npy  chunk vectors for one retriever version
  retriever/<ver>/                 fine-tuned retriever versions (base = the shared backbone)
  router/<ver>.pkl                 trained router reflex heads
  versions.json                    every version, its scores and which one is active
  jobs/<id>/                       one folder per training job: state.json + events.jsonl
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("STUDIO_DATA", os.path.join(HERE, "data"))
_lock = threading.RLock()


def path(*parts: str) -> str:
    p = os.path.join(DATA, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def read(rel: str, default=None):
    try:
        with open(os.path.join(DATA, rel), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write(rel: str, obj) -> None:
    with _lock:
        p = path(rel)
        tmp = f"{p}.{uuid.uuid4().hex[:6]}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, p)


def new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"


# ── versions ─────────────────────────────────────────────────────────────────────────
def versions() -> dict:
    return read("versions.json", {"retriever": {"active": "base", "list": [
        {"id": "base", "created": 0, "note": "multilingual MiniLM-L12, not tuned", "scores": {}}]},
        "router": {"active": None, "list": []}})


def save_versions(v: dict) -> None:
    write("versions.json", v)


def active(kind: str) -> str | None:
    return versions()[kind]["active"]


def retriever_path(ver: str | None) -> str | None:
    """None means the base model."""
    if not ver or ver == "base":
        return None
    return os.path.join(DATA, "retriever", ver)


def next_version(kind: str) -> str:
    ids = [v["id"] for v in versions()[kind]["list"]]
    n = 1 + max([int(i[1:]) for i in ids if i[:1] == ("r" if kind == "retriever" else "h") and i[1:].isdigit()] or [0])
    return f"{'r' if kind == 'retriever' else 'h'}{n}"


# ── knowledge bases: each collection has its OWN retriever lineage ────────────────────
#   collections/<cid>/versions.json  {"active": "base" | "rN", "list": [{id, created, parent, scores, …}]}
# A new knowledge base starts from the untrained base model; training one never changes another.
BASE_ENTRY = {"id": "base", "created": 0, "note": "untrained multilingual MiniLM-L12", "scores": {}}


def kb_versions(cid: str) -> dict:
    v = read(f"collections/{cid}/versions.json")
    if v is None:
        # migration from the old single, global retriever list: take the versions trained on this collection
        old = (read("versions.json") or {}).get("retriever", {}).get("list", [])
        mine = [x for x in old if x.get("collection") == cid]
        v = {"active": "base", "list": [dict(BASE_ENTRY)] + mine}
        write(f"collections/{cid}/versions.json", v)
    return v


def save_kb_versions(cid: str, v: dict) -> None:
    write(f"collections/{cid}/versions.json", v)


def kb_active(cid: str) -> str:
    return kb_versions(cid).get("active") or "base"


def next_retriever_id() -> str:
    """Retriever ids are unique across ALL knowledge bases (they share one folder)."""
    ids = set()
    root = os.path.join(DATA, "retriever")
    if os.path.isdir(root):
        ids |= {d for d in os.listdir(root)}
    croot = os.path.join(DATA, "collections")
    for cid in os.listdir(croot) if os.path.isdir(croot) else []:
        ids |= {x["id"] for x in (read(f"collections/{cid}/versions.json") or {}).get("list", [])}
    ids |= {x["id"] for x in (read("versions.json") or {}).get("retriever", {}).get("list", [])}
    n = 1 + max([int(i[1:]) for i in ids if i[:1] == "r" and i[1:].isdigit()] or [0])
    return f"r{n}"
