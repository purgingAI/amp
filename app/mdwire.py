"""AMP md-wire: Markdown is the *cognition* form (what an LLM natively speaks); the *wire*
form is a compact binary segment with a TCP-like header carrying zlib-compressed,
codebook-substituted Markdown bytes.

    cognition:   # Plan\n- add /health to `HBLController` ...
      │  pack(md, codebook)
      ▼
    payload:     zlib( codebook_substitute(md) )           (bytes)
      │  segment(flags, sid, seq, ack, payload)
      ▼
    wire:        b'AMP1' | ver | flags | type | sid(8) | seq(4) | ack(4) | plen(4) | payload

The codebook (negotiated in the handshake) maps shared domain concepts -> tiny codes like `§7`,
so two agents that "speak the same dialect" send far fewer bytes/tokens.
"""
from __future__ import annotations
import struct, zlib

MAGIC = b"AMP1"
VER = 1
# flags
F_SYN, F_ACK, F_FIN, F_COMP, F_CODE = 1, 2, 4, 8, 16
# segment types
T_DATA, T_HELLO, T_HELLOACK, T_ACK, T_TRANSLATE, T_FIN = 0, 1, 2, 3, 4, 5

_HDR = ">4sBBB8sIII"      # magic, ver, flags, type, sid, seq, ack, plen
HDR_LEN = struct.calcsize(_HDR)   # 27


def _sub(md: str, codebook: dict) -> str:
    # replace longest concepts first so substrings don't clobber
    for concept, code in sorted(codebook.items(), key=lambda kv: -len(kv[0])):
        md = md.replace(concept, f"§{code}")
    return md


def _unsub(md: str, codebook: dict) -> str:
    for concept, code in sorted(codebook.items(), key=lambda kv: -len(str(kv[1]))):
        md = md.replace(f"§{code}", concept)
    return md


def pack(md: str, codebook: dict | None = None, compress: bool = True) -> tuple[bytes, int]:
    """Markdown -> payload bytes. Returns (payload, flags)."""
    flags = 0
    if codebook:
        md = _sub(md, codebook); flags |= F_CODE
    raw = md.encode("utf-8")
    if compress:
        raw = zlib.compress(raw, 9); flags |= F_COMP
    return raw, flags


def unpack(payload: bytes, flags: int, codebook: dict | None = None) -> str:
    if flags & F_COMP:
        payload = zlib.decompress(payload)
    md = payload.decode("utf-8")
    if flags & F_CODE and codebook:
        md = _unsub(md, codebook)
    return md


def segment(payload: bytes, flags: int = 0, typ: int = T_DATA,
            sid: bytes = b"\x00" * 8, seq: int = 0, ack: int = 0) -> bytes:
    sid = (sid + b"\x00" * 8)[:8]
    return struct.pack(_HDR, MAGIC, VER, flags, typ, sid, seq, ack, len(payload)) + payload


def parse(buf: bytes) -> dict:
    magic, ver, flags, typ, sid, seq, ack, plen = struct.unpack(_HDR, buf[:HDR_LEN])
    if magic != MAGIC:
        raise ValueError("not an AMP segment")
    return {"ver": ver, "flags": flags, "type": typ, "sid": sid, "seq": seq, "ack": ack,
            "payload": buf[HDR_LEN:HDR_LEN + plen]}


def benchmark(md: str, codebook: dict | None = None) -> dict:
    import json
    payload, flags = pack(md, codebook)
    seg = segment(payload, flags)
    raw = len(md.encode())
    as_json = len(json.dumps({"type": "RESPONSE", "intent": "plan.feature",
                              "body": {"text": md}}).encode())
    return {"markdown_bytes": raw, "json_env_bytes": as_json, "amp_segment_bytes": len(seg),
            "vs_markdown_pct": round((1 - len(seg) / max(1, raw)) * 100),
            "vs_json_pct": round((1 - len(seg) / max(1, as_json)) * 100),
            "approx_tokens_markdown": round(raw / 4), "approx_tokens_wire": round(len(seg) / 4)}
