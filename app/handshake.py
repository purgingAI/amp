"""AMP 3-way handshake — the part the world hasn't seen.

Like TCP's SYN / SYN-ACK / ACK, but what gets negotiated is *understanding*:
- each agent has a stable **claude_id** and a Markdown **agent card** (who I am, what I know,
  the vocabulary/dialect I speak);
- during the handshake they exchange cards (as md→bytes) and negotiate a **session codebook**
  (shared concepts → tiny codes) so every later message is small and mutually legible;
- a session id is derived from both ids + nonces (like TCP's initial sequence numbers).

    A ──HELLO(SYN, cardA, vocabA, isnA)──▶ B
    A ◀─HELLO-ACK(SYN|ACK, cardB, codebook, isnB, ack=isnA+1)── B
    A ──ACK(ack=isnB+1)──▶ B         ⇒ session ESTABLISHED, both hold the same codebook
"""
from __future__ import annotations
import json, hashlib, secrets, re
from dataclasses import dataclass, field
from . import mdwire

_CTRL = re.compile(r"```amp-control\n(.*?)\n```", re.S)


@dataclass
class AgentCard:
    agent_id: str
    claude_id: str
    role: str = ""
    capabilities: list = field(default_factory=list)
    vocab: list = field(default_factory=list)          # domain phrases this agent speaks

    def to_markdown(self, control: dict | None = None) -> str:
        md = (f"# Agent Card · {self.agent_id}\n"
              f"- **claude_id:** `{self.claude_id}`\n"
              f"- **role:** {self.role or '—'}\n"
              f"- **capabilities:** {', '.join(self.capabilities) or '—'}\n"
              f"- **dialect (vocab):** {', '.join(self.vocab) or '—'}\n")
        if control is not None:
            md += "\n```amp-control\n" + json.dumps(control, separators=(',', ':')) + "\n```"
        return md


def _extract_control(md: str) -> dict:
    m = _CTRL.search(md)
    return json.loads(m.group(1)) if m else {}


def _isn() -> int:
    return secrets.randbelow(2**32)


def _sid(a_claude, b_claude, na, nb) -> bytes:
    return hashlib.sha256(f"{a_claude}|{b_claude}|{na}|{nb}".encode()).digest()[:8]


def build_codebook(*vocabs) -> dict:
    """Assign compact codes to the union of concepts worth compressing (base36 codes)."""
    concepts, seen = [], set()
    for v in vocabs:
        for c in v:
            c = c.strip()
            if len(c) > 3 and c.lower() not in seen:
                seen.add(c.lower()); concepts.append(c)
    concepts.sort(key=lambda c: -len(c))     # bigger wins first
    book = {}
    for i, c in enumerate(concepts):
        code = _b36(i)
        book[c] = code
    return book


def _b36(n: int) -> str:
    d = "0123456789abcdefghijklmnopqrstuvwxyz"
    s = d[n % 36]; n //= 36
    while n:
        s = d[n % 36] + s; n //= 36
    return s


# ---------- state machine ----------
def hello(me: AgentCard) -> tuple[bytes, dict]:
    na, isn = secrets.token_hex(6), _isn()
    ctrl = {"step": "hello", "claude_id": me.claude_id, "vocab": me.vocab, "nonce": na, "isn": isn}
    payload, flags = mdwire.pack(me.to_markdown(ctrl))
    seg = mdwire.segment(payload, flags | mdwire.F_SYN, mdwire.T_HELLO, seq=isn)
    state = {"role": "initiator", "me": me, "na": na, "isn": isn}
    return seg, state


def on_hello(seg_bytes: bytes, me: AgentCard) -> tuple[bytes, dict]:
    seg = mdwire.parse(seg_bytes)
    peer_md = mdwire.unpack(seg["payload"], seg["flags"])
    ctrl = _extract_control(peer_md)
    nb, isn_r = secrets.token_hex(6), _isn()
    sid = _sid(ctrl["claude_id"], me.claude_id, ctrl["nonce"], nb)
    codebook = build_codebook(ctrl.get("vocab", []), me.vocab)
    ctrl_out = {"step": "hello-ack", "claude_id": me.claude_id, "vocab": me.vocab, "nonce": nb,
                "isn": isn_r, "ack": seg["seq"] + 1, "sid": sid.hex(), "codebook": codebook}
    payload, flags = mdwire.pack(me.to_markdown(ctrl_out))
    seg2 = mdwire.segment(payload, flags | mdwire.F_SYN | mdwire.F_ACK, mdwire.T_HELLOACK,
                          sid=sid, seq=isn_r, ack=seg["seq"] + 1)
    session = {"sid": sid.hex(), "role": "responder", "peer": ctrl["claude_id"],
               "codebook": codebook, "established": False, "peer_card_md": peer_md,
               "isn": isn_r, "nb": nb}
    return seg2, session


def on_hello_ack(seg_bytes: bytes, state: dict) -> tuple[bytes, dict]:
    seg = mdwire.parse(seg_bytes)
    peer_md = mdwire.unpack(seg["payload"], seg["flags"])
    ctrl = _extract_control(peer_md)
    if seg["ack"] != state["isn"] + 1:
        raise ValueError("handshake ack mismatch (initiator)")
    ack3 = mdwire.pack(json.dumps({"step": "ack", "sid": ctrl["sid"], "ack": seg["seq"] + 1}))
    seg3 = mdwire.segment(ack3[0], ack3[1] | mdwire.F_ACK, mdwire.T_ACK,
                          sid=bytes.fromhex(ctrl["sid"]), seq=state["isn"] + 1, ack=seg["seq"] + 1)
    session = {"sid": ctrl["sid"], "role": "initiator", "peer": ctrl["claude_id"],
               "codebook": ctrl["codebook"], "established": True, "peer_card_md": peer_md}
    return seg3, session


def on_ack(seg_bytes: bytes, session: dict) -> dict:
    seg = mdwire.parse(seg_bytes)
    if seg["ack"] != session["isn"] + 1:
        raise ValueError("handshake ack mismatch (responder)")
    session["established"] = True
    return session
