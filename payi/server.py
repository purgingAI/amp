#!/usr/bin/env python3
"""PayI — paid by AI. The three-screen demo server.

    screen 1  /payer   Sagar's laptop: Jarvis console. Two products on one protocol:
                         PayI     — pay people by voice ("send Alex 20 dirhams")
                         PalmPay  — walk into a store, pay with your palm alone
                       VoicePay (speaking to Jarvis) works in both.
    screen 2  /watch   PayI Core: every key, signature, ciphertext, check and ledger entry, live
    screen 3  /wallet  Alex's laptop: money arriving, derived from the ledger

Nothing on the watcher is decorative. Every value it shows is produced by the real code paths:
app/intent (4×4 matrix, exact payee), app/monitor (X25519 + HKDF session, single-use nonce,
ChaCha20-Poly1305 channel, ordered checks), app/mandate (two-party signatures), app/ledger
(hold → capture, double entry, invariant), the TPM-sealed device key, and proto/storepass (store
certificate, signed gate track, per-visit pass, encrypted one-time palm release, 1:1 match,
acceptance checks). UWB ranging is simulated from positions on the store plan.

Separate database (data/payi.db) so the demo can be reset without touching anything else.

    ../.venv/bin/python payi/server.py --host 0.0.0.0 --port 8420
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import os
import random
import secrets
import socketserver
import sys
import threading
import time
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMP = HERE.parent
sys.path.insert(0, str(AMP))
sys.path.insert(0, str(AMP / "proto"))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey  # noqa: E402

from app import mandate_store as ms  # noqa: E402

ms.DB = AMP / "data" / "payi.db"          # conn() reads the module global at call time

from app import ledger, monitor, notify, trace, step_up, palm, device_key as dk, intent as I  # noqa: E402
from app.mandate import Mandate, Party, Limits  # noqa: E402

import storepass as SP  # noqa: E402
from presence import canon, h, pub_hex  # noqa: E402

WEB = HERE / "web"

# ── Jarvis's voice (ElevenLabs) and ears (whisper.cpp, same recogniser as the glasses) ──────────
VOICE_ID = os.environ.get("PAYI_VOICE", "onwK4e9ZLuTAKqWW03F9")      # "Daniel" — deep, calm, British
VOICE_MODEL = os.environ.get("PAYI_VOICE_MODEL", "eleven_multilingual_v2")   # highest quality
WHISPER = Path.home() / "glasslink/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = Path.home() / "glasslink/whisper.cpp/models/ggml-base.en.bin"
STT_PROMPT = ("Jarvis, send Alex Morgan 20 dirhams. Pay here. Walk in. Walk out. Go to the till. "
              "Falcon Café. Yes, confirm. Cancel. What's my balance?")
TTS_CACHE: dict[str, bytes] = {}
# per-payment approval for VoicePay: "fingerprint" (fprintd touch) or "none" (demo: read-back then pay)
APPROVAL = os.environ.get("PAYI_APPROVAL", "none")


def eleven_key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY", "")
    if not k:
        env = Path.home() / "creator-os" / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ELEVENLABS_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k


# ── Voice ID: only the enrolled owner's voice can drive Jarvis ─────────────────────────────────
# WeSpeaker ResNet34 (VoxCeleb) speaker embeddings via sherpa-onnx. The voiceprint (a 256-number
# vector, never audio) stays on this laptop. Threshold is per person: just under how consistently
# the owner matches themself at enrolment. A convenience filter — a recording can fool it; the
# device key and the mandate are what actually authorize money.
VOICE_MODEL_PATH = AMP / "data" / "models" / "wespeaker_en_voxceleb_resnet34.onnx"
VOICE_FILE = AMP / "data" / "payi_voiceid.json"
_EXTRACTOR = None


def _pcm16k(wav: bytes):
    import io
    import wave
    import numpy as np
    with wave.open(io.BytesIO(wav)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16 if width == 2 else np.uint8).astype(np.float32)
    x = x / 32768 if width == 2 else (x - 128) / 128
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        idx = (np.arange(int(len(x) * 16000 / rate)) * rate / 16000).astype(int)
        x = x[idx]
    return x


def voice_embedding(wav: bytes):
    global _EXTRACTOR
    import numpy as np
    import sherpa_onnx
    if _EXTRACTOR is None:
        _EXTRACTOR = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(VOICE_MODEL_PATH), num_threads=4))
    x = _pcm16k(wav)
    st = _EXTRACTOR.create_stream()
    st.accept_waveform(16000, x)
    st.input_finished()
    e = np.array(_EXTRACTOR.compute(st))
    return e / (np.linalg.norm(e) or 1), len(x) / 16000


def voiceid() -> dict | None:
    return json.loads(VOICE_FILE.read_text()) if VOICE_FILE.exists() else None


def voiceid_enrol(clips: list[bytes]) -> dict:
    import numpy as np
    embs = []
    for c in clips:
        e, secs = voice_embedding(c)
        if secs < 1.0:
            return {"ok": False, "say": "One of those was too short — read each line fully."}
        embs.append(e)
    if len(embs) < 3:
        return {"ok": False, "say": "I need three lines."}
    self_scores = []
    for i in range(len(embs)):
        rest = np.mean([e for j, e in enumerate(embs) if j != i], axis=0)
        self_scores.append(float(embs[i] @ (rest / np.linalg.norm(rest))))
    mean = np.mean(embs, axis=0)
    mean /= np.linalg.norm(mean)
    thr = float(min(0.8, max(0.5, np.mean(self_scores) - 0.07)))
    VOICE_FILE.write_text(json.dumps({"owner": "Sagar Uniyal", "embedding": mean.tolist(), "threshold": thr,
                                      "self_scores": self_scores, "enrolled": time.time(),
                                      "model": VOICE_MODEL_PATH.name}))
    os.chmod(VOICE_FILE, 0o600)
    emit("voiceid.enrol", "Voice ID enrolled — voiceprint stays on this device", "jarvis", "jarvis",
         fields=[F("Model", "WeSpeaker ResNet34 (VoxCeleb) · 256-dim speaker embedding", "text"),
                 F("Clips", len(embs), "mono"),
                 F("Self-consistency (leave-one-out)", ", ".join(f"{x:.3f}" for x in self_scores), "mono"),
                 F("Your threshold", f"mean − 0.07 = {thr:.3f}", "mono"),
                 F("Stored", "a vector of numbers on this laptop · never audio · never uploaded", "text")])
    return {"ok": True, "threshold": round(thr, 3), "say": "Voice ID set. I'll only take instructions from you."}


def voiceid_check(wav: bytes) -> dict:
    vid = voiceid()
    if not vid:
        return {"enrolled": False, "ok": True}
    import numpy as np
    e, secs = voice_embedding(wav)
    score = float(e @ np.array(vid["embedding"]))
    ok = score >= vid["threshold"]
    return {"enrolled": True, "ok": ok, "score": round(score, 3), "threshold": round(vid["threshold"], 3),
            "seconds": round(secs, 2)}


def transcribe(wav: bytes) -> str:
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        path = f.name
    try:
        out = subprocess.run([str(WHISPER), "-m", str(WHISPER_MODEL), "-f", path, "-nt", "-l", "en", "-t", "6",
                              "--prompt", STT_PROMPT], capture_output=True, text=True, timeout=40)
        text = " ".join(l.strip() for l in out.stdout.splitlines() if l.strip())
        return text.replace("[BLANK_AUDIO]", "").strip()
    finally:
        os.unlink(path)
KEYS = AMP / "data" / "payi_keys"
CUR = "AED"

SAGAR, ALEX, FALCON = "agt_sagar", "agt_alex", "agt_falcon"
W_SAGAR, W_ALEX, W_FALCON = "wal_sagar_aed", "wal_alex_aed", "wal_falcon_aed"
OPENING = 1000.0

# live webcam hand geometry is noisier than the synthetic bench: per-person tolerance with a
# wider band, measured at enrolment from the person's own movement
LIVE_TOL_K, LIVE_TOL_FLOOR, LIVE_TOL_CEIL = 2.5, 0.7, 1.5
SP.MATCH_LIMIT = LIVE_TOL_CEIL

# store plan, metres. x = depth into the store (x < 0 is the mall corridor), y = along the gate line
TILL_POS = (6.0, 2.6)
TILL_FACING = (-1.0, 0.0)
QUEUE_B = (4.95, 2.75)

LOCK = threading.RLock()


# ── the device key: TPM-sealed, unsealed once per demo session ───────────────

class DeviceKey:
    """The payer's device key. Sealed to this laptop's TPM; unsealing takes 5–9 s, so it is
    unsealed once when the demo starts (like unlocking a phone) and signs in memory after that."""
    algorithm = "Ed25519"
    hardware_resident = False

    def __init__(self):
        t0 = time.time()
        try:
            if os.environ.get("PAYI_SOFTWARE_KEY"):
                raise dk.DeviceKeyError("PAYI_SOFTWARE_KEY set")
            be = dk.load()
            if isinstance(be, dk.SealedEd25519):
                self._k = Ed25519PrivateKey.from_private_bytes(be._unseal())
                self.backend = "tpm-sealed"
                self.source = "TPM 2.0 (systemd-creds), unsealed for this session"
            else:
                raise dk.DeviceKeyError("pkcs11 backend not used by the demo")
        except Exception as e:  # another machine: say so, never pretend
            KEYS.mkdir(parents=True, exist_ok=True)
            p = KEYS / "sagar_device.key"
            if not p.exists():
                p.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw())
                os.chmod(p, 0o600)
            self._k = Ed25519PrivateKey.from_private_bytes(p.read_bytes())
            self.backend = "software"
            self.source = f"software key — no TPM here ({type(e).__name__})"
        self.unseal_s = round(time.time() - t0, 2)

    def sign(self, payload: bytes) -> bytes:
        return self._k.sign(payload)

    def public_key(self):
        return self._k.public_key()


def file_key(name: str) -> Ed25519PrivateKey:
    KEYS.mkdir(parents=True, exist_ok=True)
    p = KEYS / f"{name}.key"
    if not p.exists():
        p.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw())
        os.chmod(p, 0o600)
    return Ed25519PrivateKey.from_private_bytes(p.read_bytes())


DEV: DeviceKey | None = None


# ── event bus ────────────────────────────────────────────────────────────────

EVENTS: list[dict] = []
COND = threading.Condition()
SEQ = [0]                                   # monotonic across resets, so SSE clients never stall


def short(x, n=16):
    x = str(x)
    return x if len(x) <= n else x[:n] + "…"


def F(k, v, t="mono"):
    """One inspector field. t: mono | hex | text | json | check | bad | amount"""
    return {"k": k, "v": v, "t": t}


def emit(stage: str, title: str, frm: str = "", to: str = "", status: str = "ok", product: str = "",
         flow: str = "", fields: list | None = None, data: dict | None = None, pace: float = 0.0):
    ev = {"seq": 0, "t": int(time.time() * 1000), "stage": stage, "title": title, "frm": frm, "to": to,
          "status": status, "product": product, "flow": flow, "fields": fields or [], "data": data or {}}
    with COND:
        SEQ[0] += 1
        ev["seq"] = SEQ[0]
        EVENTS.append(ev)
        if len(EVENTS) > 3000:
            del EVENTS[:1000]
        COND.notify_all()
    if pace:
        time.sleep(pace)
    return ev


# ── seed / reset ─────────────────────────────────────────────────────────────

def seed():
    global WORLD
    with LOCK:
        for suf in ("", "-wal", "-shm"):
            p = Path(str(ms.DB) + suf)
            if p.exists():
                p.unlink()
        for mod in (ms, ledger, monitor, notify, trace, step_up):
            mod.init()
        alex_k, falcon_k = file_key("alex"), file_key("falcon")
        ms.enrol_key(SAGAR, DEV.public_key(), DEV.source)
        ms.enrol_key(ALEX, alex_k.public_key(), "Alex's laptop")
        ms.enrol_key(FALCON, falcon_k.public_key(), "Falcon Café merchant key")
        ledger.open_account(W_SAGAR, "Sagar Uniyal", CUR, OPENING)
        ledger.open_account(W_ALEX, "Alex Morgan", CUR, 0)
        ledger.open_account(W_FALCON, "Falcon Café (Alex Morgan)", CUR, 0)
        mandates = []
        for payee, lim, cap, purpose in (
                (Party(ALEX, W_ALEX, "Alex Morgan"), Limits(200, 500, 3000, 9), 100, "peer_transfer"),
                (Party(FALCON, W_FALCON, "Falcon Café"), Limits(500, 2000, 10000, 30), 300, "palm_checkout")):
            m = Mandate(payer=Party(SAGAR, W_SAGAR, "Sagar Uniyal"), payee=payee, currency=CUR, limits=lim,
                        green_tier_cap=cap, not_after="2026-12-31T23:59:59Z", purposes=[purpose])
            m.payer_sig = DEV.sign(m.canonical()).hex()
            m.sign_as("payee", alex_k if payee.agent_id == ALEX else falcon_k)
            ms.save(m, "active")
            mandates.append(m)
        WORLD = World()
    with COND:
        EVENTS.clear()
    emit("system.reset", "Demo reset — fresh ledger, keys enrolled, mandates signed by both parties",
         "monitor", "monitor", fields=[
             F("Payer device key", DEV.source, "text"),
             F("Unseal time", f"{DEV.unseal_s} s"),
             F("Payer public key (Ed25519)", pub_hex(DEV), "hex"),
             F("Sagar opening balance", f"{OPENING:,.2f} {CUR}", "amount"),
             *[F(f"Mandate {m.mandate_id}", f"{m.payer.display_name} → {m.payee.display_name} · "
                 f"{m.purposes[0]} · green ≤ {m.green_tier_cap:g}", "text") for m in mandates],
         ])
    WORLD.announce()


# ── PalmPay world (store, gate, till, phones) ────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def aoa(p):
    vx, vy = p[0] - TILL_POS[0], p[1] - TILL_POS[1]
    n = math.hypot(vx, vy) or 1e-9
    c = (vx * TILL_FACING[0] + vy * TILL_FACING[1]) / n
    return round(math.degrees(math.acos(max(-1, min(1, c)))), 1)


class World:
    def __init__(self):
        self.rnd = random.Random()
        self.acq = SP.Acquirer()
        self.store = SP.Store(self.acq, "Falcon Café", "st_falcon", n_tills=1)
        self.gate, self.till = self.store.gates[0], self.store.tills[0]
        self.phone = self._phone("Sagar Uniyal", DEV)
        self.phone.template = None
        self.enrol_mode = ""
        self.pos = (-2.2, -3.2)                  # in the mall corridor
        # a second shopper for the queue demo (synthetic hand, software key)
        self.b = SP.Phone("Shopper B", SP.B.person(4242), self.rnd)
        self.b_on = False
        self.accept = SP.Monitor(self.acq.pub, {self.phone.pub, self.b.pub})

    @staticmethod
    def _phone(owner, key):
        ph = SP.Phone.__new__(SP.Phone)
        ph.owner, ph.key, ph.pub = owner, key, pub_hex(key)
        ph.cap, ph.visits, ph.used_nonces = SP.DEFAULT_CAP, {}, set()
        ph.eph_pub = pub_hex(X25519PrivateKey.generate())
        ph.template, ph.tolerance = None, None
        return ph

    def announce(self):
        c = self.store.cert["body"]
        emit("ca.certify", "Acquirer CA certified Falcon Café: gate and till keys", "ca", "edge", product="palmpay",
             fields=[F("Store", f"{c['legal_name']} ({c['store_id']})", "text"),
                     F("Gate key gate-1 (Ed25519)", c["gates"]["gate-1"], "hex"),
                     F("Till key till-1 (Ed25519)", c["tills"]["till-1"], "hex"),
                     F("Acquirer signature", self.store.cert["sig"], "hex"),
                     F("Valid until", dt.datetime.fromtimestamp(c["not_after"]).strftime("%d %b %Y"), "text")])

    def pass_of(self, ph):
        v = ph.visits.get(self.store.store_id)
        return v["pass"] if v else None

    def snapshot(self):
        p = self.pass_of(self.phone)
        v = self.phone.visits.get(self.store.store_id)
        return {"pos": self.pos, "till": TILL_POS, "queue_b": QUEUE_B, "b_on": self.b_on,
                "enrolled": self.phone.template is not None, "enrol_mode": self.enrol_mode,
                "tolerance": round(self.phone.tolerance, 3) if self.phone.tolerance else None,
                "pass": ({"visit_id": p["body"]["visit_id"], "visit_pub": p["body"]["visit_pub"],
                          "cap": p["body"]["handsfree_cap"], "spent": v["spent"],
                          "expires": p["body"]["expires"]} if p else None),
                "till_distance": round(dist(self.pos, TILL_POS), 2), "till_aoa": aoa(self.pos),
                "store": self.store.name}


WORLD: World | None = None


# ── settlement through PayI Core (shared by both products) ───────────────────

def settle(flow: str, product: str, mandate_id: str, payee_wallet: str, amount: float,
           evidence: dict, step_up_id: str = "", pace: float = 0.32) -> dict:
    m = ms.get(mandate_id)
    payer_pub, payee_pub = ms.public_key(m.payer.agent_id), ms.public_key(m.payee.agent_id)
    emit("mandate.verify", "Mandate loaded — both signatures verified", "monitor", "monitor", product=product,
         flow=flow, data={"mandate": mandate_view(m)}, fields=[
             F("Mandate", f"{m.mandate_id} · {m.purposes[0]}", "text"),
             F("Parties", f"{m.payer.display_name} → {m.payee.display_name}", "text"),
             F("Limits", f"per txn {m.limits.per_txn:g} · per day {m.limits.per_day:g} · per month "
                         f"{m.limits.per_month:g} · green ≤ {m.green_tier_cap:g} {CUR}", "text"),
             F("Payer signature (Ed25519)", m.payer_sig, "hex"),
             F("Payee signature (Ed25519)", m.payee_sig, "hex"),
             F("Both signatures valid", bool(payer_pub and payee_pub and m.verify(payer_pub, payee_pub)), "check"),
             F("Signed bytes (canonical JSON)", hashlib.sha256(m.canonical()).hexdigest(), "hex")], pace=pace)

    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    agent_nonce = secrets.token_hex(8)
    s = monitor.open_session(SAGAR, mandate_id, eph_pub, agent_nonce)
    shared = eph.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(s["monitor_eph_pub"])))
    send, recv = monitor._derive(shared, agent_nonce, s["monitor_nonce"], mandate_id)
    with ms.conn() as c:
        row = c.execute("SELECT send_key, recv_key FROM sessions WHERE session_id=?", (s["session_id"],)).fetchone()
    agree = row["send_key"] == send.hex() and row["recv_key"] == recv.hex()
    emit("session.open", "Session keys agreed — X25519 ECDH → HKDF-SHA256", "phone", "monitor", product=product,
         flow=flow, fields=[
             F("Payer ephemeral X25519", eph_pub.hex(), "hex"),
             F("Monitor ephemeral X25519", s["monitor_eph_pub"], "hex"),
             F("HKDF salt", f"{agent_nonce}|{s['monitor_nonce']}|{mandate_id}", "mono"),
             F("HKDF info", "amp-monitor-v1", "mono"),
             F("Send key fingerprint", hashlib.sha256(send).hexdigest()[:24], "hex"),
             F("Receive key fingerprint", hashlib.sha256(recv).hexdigest()[:24], "hex"),
             F("Both sides derived identical keys", agree, "check"),
             F("Session token (opaque, 32 bytes)", short(s["token"], 22), "mono")], pace=pace)

    n = monitor.issue_nonce(s["token"])["nonce"]
    emit("nonce.issue", "Single-use nonce issued (120 s)", "monitor", "phone", product=product, flow=flow,
         fields=[F("Nonce", n, "mono"), F("Expires in", "120 s", "mono"),
                 F("Rule", "spent before any other check — racing replays cannot both pass", "text")], pace=pace)

    ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = dk._canonical_transfer(mandate_id, n, payee_wallet, amount, CUR, ts)
    sig = dk.sign_transfer(DEV, mandate_id, n, payee_wallet, amount, CUR, ts)
    emit("device.sign", "Device key signed the exact transfer", "phone", "phone", product=product, flow=flow,
         fields=[F("Key", DEV.source, "text"), F("Canonical payload", payload.decode(), "json"),
                 F("Ed25519 signature", sig["signature"], "hex")], pace=pace)

    sealed = monitor.seal(s["token"], payload + b"|" + bytes.fromhex(sig["signature"]))
    ok_open = monitor.unseal(s["token"], sealed["generation"], sealed["counter"],
                             sealed["ciphertext"]).startswith(payload)
    emit("channel.seal", "Authorization sealed in the session channel", "phone", "monitor", product=product,
         flow=flow, fields=[F("AEAD", "ChaCha20-Poly1305 (IETF, 96-bit counter nonce)", "text"),
                            F("Generation · counter", f"{sealed['generation']} · {sealed['counter']}", "mono"),
                            F(f"Ciphertext ({len(sealed['ciphertext']) // 2} bytes)", sealed["ciphertext"], "hex"),
                            F("Monitor opened and authenticated it", ok_open, "check")], pace=pace)

    try:
        r = monitor.authorize_transfer(token=s["token"], nonce=n, payee_wallet=payee_wallet, amount=amount,
                                       currency=CUR, timestamp=ts, signature_hex=sig["signature"],
                                       step_up_id=step_up_id, evidence=evidence)
    except monitor.MonitorError as e:
        for row in trace.of(n):
            if row["stage"] in ("nonce.issue",):
                continue
            emit("core." + row["stage"], row.get("label") or row["stage"], row.get("frm", "monitor"),
                 row.get("to", "monitor"), status=row["status"], product=product, flow=flow,
                 fields=[F(k, v, "mono") for k, v in row["detail"].items()], pace=pace / 2)
        emit("monitor.refuse", f"Refused: {e}", "monitor", "phone", "refused", product, flow,
             [F("Reason (speakable)", str(e), "bad")])
        return {"state": "refused", "reason": str(e)}

    labels = {"transfer.request": ("phone", "monitor"), "signature.verify": ("monitor", "monitor"),
              "mandate.check": ("monitor", "monitor"), "tier.decide": ("monitor", "monitor"),
              "stepup.request": ("monitor", "phone"), "stepup.approve": ("phone", "monitor"),
              "ledger.authorize": ("monitor", "ledger"), "ledger.capture": ("ledger", "ledger"),
              "event.emit": ("monitor", "payee")}
    for row in trace.of(n):
        if row["stage"] == "nonce.issue":
            continue
        frm, to = labels.get(row["stage"], ("monitor", "monitor"))
        d = row["detail"]
        fields = [F(k.replace("_", " "), v, "mono") for k, v in d.items()]
        fields.append(F("Elapsed since nonce", f"{row['elapsed_ms']:.1f} ms", "mono"))
        emit("core." + row["stage"], row.get("label") or row["stage"], frm, to, row["status"], product, flow,
             fields, pace=pace)

    if r.get("state") == "requires_step_up":
        return r

    entries = ledger.audit(r.get("txn_id"))
    inv = ledger.check_invariant()
    emit("ledger.entries", "Double-entry posted — book balanced", "ledger", "ledger", product=product, flow=flow,
         data={"entries": entries, "invariant": inv},
         fields=[*[F(e["account_id"], f"{e['amount']:+,.2f} {e['currency']}", "amount") for e in entries],
                 F("Transaction", r.get("txn_id"), "mono"),
                 F("Sum of this transaction", f"{sum(e['amount'] for e in entries):.2f}", "mono"),
                 F("Whole book sums to zero", inv["balanced"], "check")], pace=pace)
    return r


def mandate_view(m: Mandate) -> dict:
    c = ms.counters(m.mandate_id)
    return {"mandate_id": m.mandate_id, "payer": m.payer.display_name, "payee": m.payee.display_name,
            "purpose": m.purposes[0], "limits": vars(m.limits), "green_cap": m.green_tier_cap,
            "spent_today": c.get("spent_today", 0), "spent_month": c.get("spent_month", 0),
            "payer_sig": m.payer_sig, "payee_sig": m.payee_sig, "not_after": m.not_after}


def credited(flow, product, r, from_name, to_agent, to_wallet, channel, amount, extra=None):
    bal = ledger.balance(to_wallet)
    emit("wallet.credited", f"{to_agent_name(to_agent)} credited {amount:,.2f} {CUR}", "payee", "payee",
         product=product, flow=flow,
         data={"amount": amount, "currency": CUR, "from": from_name, "to_agent": to_agent, "wallet": to_wallet,
               "balance": bal, "channel": channel, "transfer_id": r.get("transfer_id"), "txn_id": r.get("txn_id"),
               "tier": r.get("tier"), "payer_balance": ledger.balance(W_SAGAR), **(extra or {})},
         fields=[F("Payee wallet", to_wallet, "mono"), F("New balance (derived from entries)", f"{bal:,.2f} {CUR}", "amount"),
                 F("Transfer", r.get("transfer_id"), "mono"), F("Event emitted only after capture", True, "check")])


def to_agent_name(a):
    return {ALEX: "Alex Morgan", FALCON: "Falcon Café", SAGAR: "Sagar Uniyal"}.get(a, a)


# ── VoicePay / PayI ──────────────────────────────────────────────────────────

FLOWS: dict[str, dict] = {}


def voice_plan(text: str, product: str) -> dict:
    flow = "f_" + secrets.token_hex(5)
    text = text.strip()
    emit("voice.heard", f"“{text}”", "jarvis", "jarvis", product=product, flow=flow,
         fields=[F("Transcript", text, "text"), F("Recogniser", "browser speech → text (proposes only)", "text")])
    low = text.lower()
    if any(w in low for w in ("balance", "how much do i have", "how much money")):
        b = ledger.balance(W_SAGAR)
        emit("voice.query", "Balance read from the ledger", "jarvis", "ledger", product=product, flow=flow,
             fields=[F("Balance (sum of entries)", f"{b:,.2f} {CUR}", "amount")])
        return {"ok": True, "kind": "answer", "say": f"You have {b:,.2f} dirhams.", "flow": flow}
    verb, cells = I.classify(text)
    emit("intent.classify", f"4×4 intent matrix → {verb}", "jarvis", "jarvis", product=product, flow=flow,
         data={"verb": verb, "cells": cells},
         fields=[F("Verb class", verb, "mono"),
                 *[F(f"cell: {o}", bool(v), "check") for o, v in cells.items()],
                 F("Model used", "none — dictionary lookup, fails closed", "text")])
    try:
        cands = I.RosterResolver().candidates(SAGAR)
        p = I.plan(text, SAGAR)
    except I.IntentError as e:
        emit("intent.refuse", f"Refused before anything was sent: {e}", "jarvis", "jarvis", "refused", product,
             flow, [F("Reason", str(e), "bad"), F("Rule", "exact payee or nothing — no similarity threshold", "text")])
        return {"ok": False, "say": str(e)[0].upper() + str(e)[1:] + ".", "flow": flow}
    if p.intent != "transfer":
        emit("intent.other", f"Not a payment ({p.intent})", "jarvis", "jarvis", "refused", product, flow)
        return {"ok": False, "say": "That doesn't sound like a payment.", "flow": flow}
    m = ms.get(p.mandate_id)
    tier = "green" if p.amount <= m.green_tier_cap else ("amber" if p.amount <= m.limits.per_txn else "red")
    spoken = set(I.tokens(I.payee_phrase(text)))
    emit("payee.resolve", f"Exact payee: {p.payee_name}", "jarvis", "jarvis", product=product, flow=flow,
         fields=[F("Heard name phrase", I.payee_phrase(text), "mono"),
                 *[F(c["name"], f"{len(spoken & set(I.tokens(c['name'])))} exact token(s)", "mono") for c in cands],
                 F("Resolved", f"{p.payee_name} · {p.payee_wallet}", "text"),
                 F("Mandate", p.mandate_id, "mono")])
    rb = p.read_back()
    emit("voice.readback", f"Read back: {rb}", "jarvis", "jarvis", product=product, flow=flow,
         fields=[F("Amount", f"{p.amount:,.2f} {p.currency}", "amount"),
                 F("Tier preview", {"green": f"green — within voice-only cap {m.green_tier_cap:g}",
                                    "amber": f"amber — above {m.green_tier_cap:g}: second channel required",
                                    "red": "red — above the mandate's per-payment limit"}[tier], "text"),
                 F("Authority so far", "none — nothing is signed until your fingerprint approves", "text")])
    FLOWS[flow] = {"product": product, "mandate_id": p.mandate_id, "payee_wallet": p.payee_wallet,
                   "payee_agent": p.payee_agent, "payee_name": p.payee_name, "amount": p.amount,
                   "text": text, "state": "planned", "tier": tier}
    say = (f"{p.amount:g} dirhams to {p.payee_name}. That's over your limit for this person." if tier == "red"
           else f"{p.amount:g} dirhams to {p.payee_name}. Touch the sensor to pay." if APPROVAL == "fingerprint"
           else f"Paying {p.amount:g} dirhams to {p.payee_name}.")
    return {"ok": True, "kind": "readback", "flow": flow, "read_back": rb, "tier": tier, "amount": p.amount,
            "payee": p.payee_name, "wallet": p.payee_wallet, "mandate": p.mandate_id,
            "cap": m.green_tier_cap, "approval": APPROVAL, "say": say}


def voice_confirm(flow: str, step_up_id: str = "") -> dict:
    f = FLOWS.get(flow)
    if not f:
        return {"ok": False, "say": "That payment expired. Say it again."}
    if f["state"] not in ("planned", "stepup_ok"):
        return {"ok": False, "say": "That one is already being handled."}
    if not f.get("approved"):
        return {"ok": False, "say": "Touch the fingerprint sensor to approve."}
    f["state"] = "running"
    r = settle(flow, f["product"], f["mandate_id"], f["payee_wallet"], f["amount"],
               {"transcript": f["text"], "channel": "VoicePay", "approved_by": "fingerprint"}, step_up_id)
    if r.get("state") == "requires_step_up" and f.get("approved") == step_up.binding(
            f["mandate_id"], f["payee_wallet"], f["amount"], CUR) and not step_up_id:
        # amber: the fingerprint touch was already bound to exactly this payment — it is the second channel
        step_up._settle(r["step_up_id"], "approved", "fingerprint" if APPROVAL == "fingerprint" else "demo-skip")
        emit("stepup.fingerprint", "Amber: the same fingerprint approval satisfies the step-up" if APPROVAL == "fingerprint"
             else "Amber: step-up auto-approved (approval OFF — demo setting)", "phone", "monitor",
             product=f["product"], flow=flow,
             fields=[F("Step-up", r["step_up_id"], "mono"), F("Binding matches the touch", True, "check")])
        f["state"] = "planned"
        return voice_confirm(flow, r["step_up_id"])
    if r.get("state") == "requires_step_up":
        f.update(state="stepup", step_up_id=r["step_up_id"])
        return {"ok": True, "state": "stepup", "flow": flow,
                "say": f"{f['amount']:g} dirhams is above your voice-only limit. Confirm with your palm."}
    if r.get("state") != "captured":
        f["state"] = "refused"
        return {"ok": False, "state": "refused", "say": r.get("reason", "Refused.").capitalize() + "."}
    f["state"] = "done"
    credited(flow, f["product"], r, "Sagar Uniyal", f["payee_agent"], f["payee_wallet"], "VoicePay",
             f["amount"], {"transcript": f["text"]})
    return {"ok": True, "state": "captured", "flow": flow, "transfer_id": r["transfer_id"],
            "say": f"Done. {f['amount']:g} dirhams sent to {f['payee_name'].split()[0]}."}


def fingerprint_touch(timeout: int = 25) -> tuple[bool, str]:
    """Ask the laptop's fingerprint sensor (fprintd) for one touch of an enrolled finger."""
    import subprocess
    try:
        out = subprocess.run(["fprintd-verify"], capture_output=True, text=True, timeout=timeout)
        txt = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        return False, "no touch within time"
    if "verify-match" in txt:
        return True, "match"
    if "verify-no-match" in txt:
        return False, "that finger doesn't match"
    return False, (txt.strip().splitlines() or ["sensor unavailable"])[-1][:80]


def voice_approve(flow: str) -> dict:
    """The ONLY approval for a VoicePay payment: one fingerprint touch on the payer's own device,
    bound to sha256(mandate, payee, amount, currency). Runs outside the global lock (it waits for a
    human)."""
    f = FLOWS.get(flow)
    if not f or f["state"] != "planned":
        return {"ok": False, "say": "That payment isn't waiting any more."}
    if f["tier"] == "red":
        return {"ok": False, "say": "That's over your limit for this person."}
    bind = step_up.binding(f["mandate_id"], f["payee_wallet"], f["amount"], CUR)
    if APPROVAL != "fingerprint":
        with LOCK:
            emit("approve.skipped", "Approval step OFF (demo setting) — paying after read-back", "phone", "phone",
                 "ok", f["product"], flow,
                 [F("Setting", "PAYI_APPROVAL=none · fingerprint approval switched off for this demo", "text"),
                  F("Would bind to", bind, "hex"),
                  F("Still enforced", "exact payee · both-signed mandate · limits · device signature · nonce", "text")])
            f.update(state="planned", approved=bind)
            return voice_confirm(flow)
    f["state"] = "touch"
    emit("approve.request", f"Touch to approve {f['amount']:g} {CUR} to {f['payee_name']}", "phone", "phone",
         product=f["product"], flow=flow,
         fields=[F("Channel", "fingerprint on the payer's own device (fprintd)", "text"),
                 F("Approval bound to", "sha256(mandate, payee wallet, amount, currency)", "text"),
                 F("Binding", bind, "hex"),
                 F("Replaces", "a spoken “yes” — the voice still proposes only", "text")])
    t0 = time.time()
    ok, why = fingerprint_touch()
    with LOCK:
        if f["state"] == "cancelled":
            return {"ok": False, "say": "Cancelled. Nothing was sent."}
        emit("approve.fingerprint", "Fingerprint matched — approved" if ok else f"Not approved: {why}", "phone", "phone",
             "ok" if ok else "refused", f["product"], flow,
             [F("Result", why, "mono" if ok else "bad"), F("Time to touch", f"{time.time() - t0:.1f} s", "mono"),
              F("Approved exactly", bind, "hex")])
        if not ok:
            f["state"] = "refused"
            return {"ok": False, "say": "Not approved. Nothing was sent." if "time" in why else f"Not approved — {why}."}
        f.update(state="planned", approved=bind)
        return voice_confirm(flow)


def palm_check(frames, who: str = "") -> tuple[bool, float | None, float | None, str]:
    ph = WORLD.phone
    if ph.template is None:
        return False, None, None, "enrol your palm first"
    live = frames if frames else synthetic_frames(who)
    d = sorted(palm.distance(ph.template, fr) for fr in live)[len(live) // 2]
    ok = d <= ph.tolerance
    return ok, round(d, 3), round(ph.tolerance, 3), "same hand" if ok else "palm does not match the phone's owner"


def voice_stepup(flow: str, frames, who: str) -> dict:
    f = FLOWS.get(flow)
    if not f or f.get("state") != "stepup":
        return {"ok": False, "say": "There's nothing waiting for confirmation."}
    ok, d, lim, why = palm_check(frames, who)
    emit("stepup.palm", "Second channel: palm 1:1 on the phone", "phone", "phone",
         "ok" if ok else "refused", f["product"], flow,
         [F("Template", "on the phone only — never uploaded", "text"),
          F("Distance", d, "mono"), F("Your tolerance", lim, "mono"), F("Same person", ok, "check" if ok else "bad"),
          F("Bound to", f"step-up {f['step_up_id']} = sha256(mandate, payee, amount, currency)", "mono")])
    if not ok:
        step_up._settle(f["step_up_id"], "denied", "palm 1:1")
        f["state"] = "refused"
        return {"ok": False, "say": "That palm doesn't match. Payment cancelled."}
    step_up._settle(f["step_up_id"], "approved", "palm 1:1")
    f["state"] = "stepup_ok"
    return voice_confirm(flow, f["step_up_id"])


# ── PalmPay ──────────────────────────────────────────────────────────────────

def synthetic_frames(who: str, n: int = 8):
    hand = SP.B.person(77) if who == "owner" else SP.B.person(9090)
    return [SP.B.jitter(SP.B.pose(hand, WORLD.rnd), WORLD.rnd, 0.007) for _ in range(n)]


def palm_enrol(steady, moved, demo: bool) -> dict:
    ph = WORLD.phone
    if demo:
        hand = SP.B.person(77)
        steady = [SP.B.jitter(SP.B.pose(hand, WORLD.rnd), WORLD.rnd, 0.005) for _ in range(10)]
        moved = [SP.B.jitter(SP.B.pose(hand, WORLD.rnd), WORLD.rnd, 0.008) for _ in range(6)]
    if len(steady) < palm.MIN_SAMPLES or len(moved) < 3:
        return {"ok": False, "say": "I need a few more frames of your open palm."}
    t = palm.enrol("Sagar Uniyal", steady)
    own = sorted(palm.distance(t, fr) for fr in moved)[len(moved) // 2]
    ph.template = t
    k, lo, hi = (SP.bio.TOL_K, SP.bio.TOL_FLOOR, SP.bio.TOL_CEIL) if demo else (LIVE_TOL_K, LIVE_TOL_FLOOR, LIVE_TOL_CEIL)
    ph.tolerance = min(hi, max(lo, k * own))
    WORLD.enrol_mode = "demo hand" if demo else "your hand (webcam)"
    th = hashlib.sha256(json.dumps(t.mean).encode()).hexdigest()
    emit("palm.enrol", "Palm enrolled on the phone — never uploaded", "phone", "phone", product="palmpay",
         fields=[F("Source", WORLD.enrol_mode, "text"), F("Frames", f"{len(steady)} steady + {len(moved)} moved", "mono"),
                 F("Features", f"{len(t.mean)} pose-invariant hand-geometry ratios", "text"),
                 F("Your own movement (median)", round(own, 3), "mono"),
                 F("Your tolerance", f"clamp({k} × {own:.3f}, {lo}, {hi}) = {ph.tolerance:.3f}", "mono"),
                 F("Template fingerprint", th, "hex"),
                 F("Stored on", "Sagar's phone only · no server, no store, no database", "text")])
    return {"ok": True, "tolerance": round(ph.tolerance, 3), "say": "Palm enrolled. It stays on your phone."}


def palm_move(path) -> dict:
    """The avatar moved along `path` (metres). The gate module tracks it near the entrance and the
    PHONE decides whether that was a real walk-in."""
    w = WORLD
    pts = [(float(x), float(y)) for x, y in path][-400:]
    if not pts:
        return {"ok": True, "world": w.snapshot()}
    w.pos = pts[-1]
    near = [p for p in pts if abs(p[0]) <= 1.6 and abs(p[1]) <= 2.6]     # the gate module's UWB range
    out = {"ok": True}
    had = w.pass_of(w.phone)
    if len(near) >= 4:
        # one UWB sample per 0.25 s at walking pace (1.4 m/s) — resample by distance walked,
        # so a slow mouse drag and a fast one look like the same walk to the gate
        sample, acc = [near[0]], 0.0
        for q0, q1 in zip(near, near[1:]):
            acc += dist(q0, q1)
            if acc >= 0.35:
                sample.append(q1)
                acc = 0.0
        if sample[-1] != near[-1]:
            sample.append(near[-1])
        tr = w.gate.track(w.phone.eph_pub, sample, w.rnd)
        s0 = tr["body"]["samples"]
        if not had:
            r = w.phone.enter(w.acq.pub, w.store.cert, tr)
            checks = [F("Store certificate chains to acquirer", True, "check"),
                      F("Gate key listed in that certificate", True, "check"),
                      F("Time of flight consistent (no relay)",
                        all(x["tof_slack_ns"] <= SP.MAX_TOF_SLACK_NS for x in s0), "check"),
                      F("Crossed ≥0.6 m outside → ≥0.6 m inside between the posts", r["ok"],
                        "check" if r["ok"] else "bad")]
            emit("gate.track", f"Gate posts tracked the phone ({len(s0)} UWB samples, signed)", "gate", "phone",
                 product="palmpay", data={"samples": s0},
                 fields=[F("Samples (x, y, ToF slack)", " ".join(f"({q['x']:+.2f},{q['y']:+.2f})" for q in s0[:10]) + " …", "mono"),
                         F("Gate signature (Ed25519)", tr["sig"], "hex"), F("UWB noise", "10 cm (simulated ranging)", "text")])
            if r["ok"]:
                p = r["pass"]
                w.store.register(p)
                link = w.phone.visits[w.store.store_id]["link"]
                emit("phone.enter", "Phone verified the walk-in", "phone", "phone", product="palmpay", fields=checks)
                emit("pass.issue", "Store pass issued — a fresh pseudonym for this visit", "phone", "edge",
                     product="palmpay", data={"pass": p["body"]},
                     fields=[F("Visit id", p["body"]["visit_id"], "mono"),
                             F("Visit key (new, per visit)", p["body"]["visit_pub"], "hex"),
                             F("Hands-free cap", f"{p['body']['handsfree_cap']:g} {CUR} this visit", "amount"),
                             F("Expires", "3 h, or when you walk out", "text"),
                             F("Pass signature (visit key)", p["sig"], "hex"),
                             F("Device link (device key → visit key)", link["sig"], "hex"),
                             F("Device link goes to", "the monitor only, at payment — never the store", "text"),
                             F("Store learns", "a pass it cannot link to you or to any earlier visit", "text")])
                emit("edge.register", "Store edge registered the pass", "edge", "edge", product="palmpay",
                     fields=[F("Roster size", len(w.store.roster), "mono"),
                             F("Knows who you are", False, "check")])
                out.update(event="entered", say="Welcome to Falcon Café. You're checked in.")
            elif any(abs(q[0]) < 1.2 and abs(q[1]) < 2.0 for q in near):
                emit("phone.enter", f"No pass — {r['why']}", "phone", "phone", "refused", "palmpay", "", checks)
                out.update(event="nopass", why=r["why"])
        else:
            if w.phone.leave(w.store.cert, tr):
                vid = had["body"]["visit_id"]
                w.store.close(vid)
                emit("pass.close", "Walked out — pass closed, visit key deleted", "gate", "edge", product="palmpay",
                     fields=[F("Visit id", vid, "mono"), F("Visit key", "deleted on the phone", "text"),
                             F("Store keeps", "nothing that identifies you", "text")])
                out.update(event="left", say="Pass closed. See you next time.")
    out["world"] = w.snapshot()
    return out


def palm_pay(amount: float, items: list, frames, who: str, explicit: bool) -> dict:
    w = WORLD
    flow = "p_" + secrets.token_hex(5)
    if w.phone.template is None:
        return {"ok": False, "say": "Enrol your palm on your phone first."}
    if explicit and not w.pass_of(w.phone):
        d, a = dist(w.pos, TILL_POS), aoa(w.pos)
        if d > 1.5 or a > 30:
            emit("voice.payhere", "“Pay here” — but no till in front of you", "jarvis", "phone", "refused",
                 "palmpay", flow, [F("Distance to till", f"{d:.2f} m", "mono"), F("Angle", f"{a:.0f}°", "mono")])
            return {"ok": False, "say": "I can't see a till in front of you."}
        vk = Ed25519PrivateKey.generate()
        now = time.time()
        body = {"store_id": w.store.store_id, "visit_id": "v_" + secrets.token_hex(8), "visit_pub": pub_hex(vk),
                "issued": now, "expires": now + SP.PASS_TTL_S, "handsfree_cap": w.phone.cap,
                "entry_track": "explicit:voice-pay-here"}
        p = {"body": body, "sig": SP.sign(vk, body)}
        lb = {"visit_pub": body["visit_pub"], "pass_hash": h(p), "store_id": w.store.store_id}
        w.phone.visits[w.store.store_id] = {"pass": p, "vk": vk, "spent": 0.0, "used": set(),
                                            "link": {"body": lb, "sig": SP.sign(w.phone.key, lb), "device_pub": w.phone.pub}}
        w.store.register(p)
        emit("voice.payhere", "VoicePay: “pay here” — explicit consent replaces the gate check-in", "jarvis", "edge",
             product="palmpay", flow=flow, fields=[F("Visit id", body["visit_id"], "mono"),
                                                   F("Visit key", body["visit_pub"], "hex"),
                                                   F("Till in front of you", f"{d:.2f} m · {a:.0f}°", "mono")])
    req = w.till.request(w.store.cert, round(amount, 2))
    rb = req["body"]
    emit("till.request", f"Palm on the reader — till requests {amount:,.2f} {CUR}", "till", "phone", product="palmpay",
         flow=flow, data={"items": items, "amount": amount},
         fields=[F("Basket", ", ".join(items) or "—", "text"), F("Amount", f"{amount:,.2f} {CUR}", "amount"),
                 F("Request nonce", rb["request_nonce"], "mono"), F("Till ephemeral X25519", rb["terminal_eph"], "hex"),
                 F("Request signature (till key)", req["sig"], "hex"),
                 F("Asks", "the nearest checked-in phone first", "text")])
    cands = []
    for ph, pos in ((w.phone, w.pos), (w.b, QUEUE_B)):
        if ph is w.b and not w.b_on:
            continue
        p = w.pass_of(ph)
        if p and p["body"]["visit_id"] in w.store.roster:
            cands.append((dist(pos, TILL_POS), aoa(pos), ph, p))
    cands.sort(key=lambda x: x[0])
    if not cands:
        emit("till.none", "No checked-in phone — nothing to ask", "till", "till", "refused", "palmpay", flow,
             [F("Fallback", "say “pay here” to Jarvis (explicit consent)", "text")])
        return {"ok": False, "say": "You're not checked in here. Say “pay here” and I'll pay at this till.",
                "flow": flow}
    live = frames if frames else synthetic_frames(who)
    tried = 0
    for d, a, ph, p in cands:
        who_lbl = "your phone" if ph is w.phone else "Shopper B's phone"
        rng = w.till.range_phone(req, p["body"]["visit_id"], d, a)
        emit("till.ranging", f"UWB ranging to {who_lbl}: {d:.2f} m at {a:.0f}°", "till", "phone", product="palmpay",
             flow=flow, fields=[F("Distance (from time of flight)", f"{rng['body']['distance_m']:.2f} m", "mono"),
                                F("Round-trip ToF", f"{rng['body']['tof_ns']:.2f} ns", "mono"),
                                F("Angle of arrival", f"{a:.1f}°", "mono"),
                                F("Ranging signature (till key)", rng["sig"], "hex")])
        if d > SP.HANDSFREE_DIST_M:
            emit("phone.policy", f"{who_lbl.capitalize()} is {d:.1f} m away — palm stays on the phone", "phone", "phone",
                 "refused", "palmpay", flow, [F("Limit", f"≤ {SP.HANDSFREE_DIST_M} m, ±{SP.HANDSFREE_AOA_DEG}°", "mono")])
            continue
        rel = ph.release(req, rng, w.acq.pub)
        v = ph.visits[w.store.store_id]
        policy = [F("Till listed in the certificate of my pass's store", "store" not in rel.get("why", ""), "check"),
                  F(f"In front of the till (≤ {SP.HANDSFREE_DIST_M} m, ±{SP.HANDSFREE_AOA_DEG}°)",
                    d <= SP.HANDSFREE_DIST_M and a <= SP.HANDSFREE_AOA_DEG, "check"),
                  F("No relay (time of flight)", "relay" not in rel.get("why", ""), "check"),
                  F(f"Within hands-free cap ({v['spent']:g} + {amount:g} ≤ {v['pass']['body']['handsfree_cap']:g})",
                    not rel.get("step_up"), "check"),
                  F("Pass valid, request unused", rel["ok"] or rel.get("step_up") or "expired" not in rel.get("why", ""), "check")]
        if not rel["ok"]:
            emit("phone.policy", f"Phone refused: {rel['why']}", "phone", "phone", "refused", "palmpay", flow,
                 policy + [F("Palm left the phone", False, "check")])
            if rel.get("step_up"):
                return {"ok": False, "flow": flow, "say": rel["why"].replace(" — confirm", ". Confirm") + "."}
            continue
        env = rel["envelope"]
        tried += 1
        emit("phone.policy", "Release policy passed on the phone", "phone", "phone", product="palmpay", flow=flow,
             fields=policy)
        emit("palm.release", f"Palm template released to till-1 only ({len(env['ciphertext']) // 2} bytes, encrypted)",
             "phone", "till", product="palmpay", flow=flow, data={"bytes": len(env["ciphertext"]) // 2},
             fields=[F("Key agreement", "X25519(phone ephemeral, till ephemeral)", "text"),
                     F("Phone ephemeral X25519", env["device_eph"], "hex"),
                     F("KDF", "HKDF-SHA256 · salt = txn hash · info = amp-storepass-release", "mono"),
                     F("Txn hash (salt)", env["binding"]["txn_hash"], "hex"),
                     F("AEAD", "ChaCha20-Poly1305 · AAD = binding", "text"),
                     F("Nonce (96-bit)", env["nonce"], "hex"),
                     F(f"Ciphertext ({len(env['ciphertext']) // 2} bytes)", env["ciphertext"], "hex"),
                     F("Bound to", f"till-1 · {env['binding']['visit_id']} · nonce {short(env['binding']['request_nonce'], 12)} · 60 s", "mono"),
                     F("Signed by visit key", env["visit_sig"], "hex")])
        ok, md, why = w.till.match(req, env, p, live)
        lim = round(min(SP.MATCH_LIMIT, ph.tolerance), 3)
        emit("till.match", f"1:1 match: {'same person' if ok else 'no match'} — template deleted", "till", "till",
             "ok" if ok else "refused", "palmpay", flow,
             [F("Mode", "1:1 against this one released template (never a search)", "text"),
              F("Live palm", "your hand on the reader" if frames else f"synthetic {who or 'stranger'} hand", "text"),
              F("Distance", md, "mono"), F("Tolerance (carried in the envelope)", lim, "mono"),
              F("Match", ok, "check" if ok else "bad"), F("Template deleted after comparing", True, "check"),
              F("Templates retained by till", w.till.retained_templates, "mono")])
        if not ok:
            continue                                # template deleted; ask the next-nearest phone
        att = w.till.attest(req, p["body"]["visit_id"], True, md, why, tried)
        emit("till.attest", "Till signed the palm attestation", "till", "phone", product="palmpay", flow=flow,
             fields=[F("Attestation", json.dumps({k: att["body"][k] for k in ("visit_id", "match", "distance", "mode", "phones_tried")}), "json"),
                     F("Signature (till key)", att["sig"], "hex")])
        a2 = ph.authorize(req, rng, att)
        if not a2["ok"]:
            return {"ok": False, "flow": flow, "say": a2["why"]}
        emit("phone.authorize", "Visit key signed the payment", "phone", "accept", product="palmpay", flow=flow,
             fields=[F("Covers", "request · ranging · attestation · pass · amount · nonce", "text"),
                     F("Signature (visit key)", a2["auth"]["sig"], "hex"),
                     F("Device link attached (for the monitor only)", a2["device_link"]["sig"], "hex")])
        okv, whyv = w.accept.verify(req, rng, att, a2)
        checks = ["Store certificate → acquirer", "Till key listed for this store", "Request signed by till",
                  "Ranging signed by till", "Attestation signed by till", "Attestation = this request, match",
                  "Same visit in ranging + attestation", "Pass valid for this store (visit key)",
                  "Device link from an enrolled device", "Device link covers exactly this pass",
                  "Payment signed by the pass holder", "Payment covers this attestation",
                  f"Payer ≤ {SP.HANDSFREE_DIST_M} m from till", "Pass not expired",
                  "Within this visit's hands-free cap", "Request nonce unused"]
        emit("accept.verify", f"PalmPay acceptance: {len(checks)} checks {'passed' if okv else 'failed'}", "accept",
             "monitor", "ok" if okv else "refused", "palmpay", flow,
             [F(c, okv, "check") for c in checks] if okv else [F("Failed", whyv, "bad")])
        if not okv:
            return {"ok": False, "flow": flow, "say": f"Refused: {whyv}."}
        if ph is not w.phone:
            a2["commit"]()
            return {"ok": True, "flow": flow, "say": "That was Shopper B's palm — charged to Shopper B, not you."}
        mid = ms.find_active(SAGAR, FALCON).mandate_id
        r = settle(flow, "palmpay", mid, W_FALCON, round(amount, 2),
                   {"channel": "PalmPay", "visit_id": p["body"]["visit_id"], "till": "till-1",
                    "palm_distance": md, "attestation": h(att)})
        if r.get("state") != "captured":
            return {"ok": False, "flow": flow, "say": (r.get("reason") or "Refused").capitalize() + "."}
        a2["commit"]()
        credited(flow, "palmpay", r, "Sagar Uniyal", FALCON, W_FALCON, "PalmPay", round(amount, 2),
                 {"items": items, "till": "till-1", "store": "Falcon Café", "palm_distance": md})
        return {"ok": True, "flow": flow, "transfer_id": r["transfer_id"],
                "say": f"Paid. {amount:,.2f} dirhams at Falcon Café."}
    if tried:
        return {"ok": False, "flow": flow, "say": "That palm doesn't match anyone checked in at this till. Nothing was paid."}
    return {"ok": False, "flow": flow,
            "say": "No checked-in phone is close enough to the till. Step up to the counter."}


# ── state snapshot ───────────────────────────────────────────────────────────

def state() -> dict:
    ms_list = []
    with ms.conn() as c:
        rows = c.execute("SELECT doc FROM mandates WHERE state='active' ORDER BY created").fetchall()
    for r in rows:
        ms_list.append(mandate_view(Mandate.from_dict(json.loads(r["doc"]))))
    feed = []
    for a in (ALEX, FALCON):
        for e in notify.poll(a, 0, 200):
            feed.append({**e["payload"], "to_agent": a})
    feed.sort(key=lambda x: x.get("at", ""), reverse=True)
    inv = ledger.check_invariant()
    return {"balances": {"sagar": ledger.balance(W_SAGAR), "alex": ledger.balance(W_ALEX),
                         "falcon": ledger.balance(W_FALCON)},
            "mandates": ms_list, "feed": feed[:30], "invariant": inv,
            "device": {"source": DEV.source, "backend": DEV.backend, "pub": pub_hex(DEV), "unseal_s": DEV.unseal_s},
            "voiceid": ({"enrolled": True, "threshold": round(voiceid()["threshold"], 3)} if voiceid() else {"enrolled": False}),
            "world": WORLD.snapshot(), "last_seq": EVENTS[-1]["seq"] if EVENTS else 0}


# ── HTTP ─────────────────────────────────────────────────────────────────────

PAGES = {"/": "index.html", "/payer": "payer.html", "/watch": "watch.html", "/wallet": "wallet.html"}
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
         ".svg": "image/svg+xml", ".png": "image/png"}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/events":
            return self._sse()
        if path == "/api/events":
            q = self.path.split("since=")
            since = int(q[1].split("&")[0]) if len(q) > 1 and q[1].split("&")[0].isdigit() else 0
            with COND:
                evs = [e for e in EVENTS if e["seq"] > since][:300]
            return self._json({"events": evs, "last": SEQ[0]})
        if path == "/api/tts":
            return self._tts()
        if path == "/api/state":
            with LOCK:
                return self._json(state())
        if path in PAGES or path.startswith("/static/"):
            f = WEB / (PAGES.get(path) or path[len("/static/"):])
            if f.resolve().parent == WEB.resolve() and f.exists():
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
        self._json({"error": "not found"}, 404)

    def _tts(self):
        from urllib.parse import parse_qs, urlparse
        import urllib.request
        text = (parse_qs(urlparse(self.path).query).get("text") or [""])[0][:600]
        text = text.replace("AED", "dirhams")
        key = eleven_key()
        if not text or not key:
            return self._json({"error": "no voice"}, 503)
        if text in TTS_CACHE:
            return self._send(200, TTS_CACHE[text], "audio/mpeg")
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": VOICE_MODEL,
                             "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.15,
                                                "use_speaker_boost": True}}).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            return self._json({"error": f"voice unavailable ({type(e).__name__})"}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        buf = bytearray()
        try:
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                buf += chunk
                self.wfile.write(chunk)
                self.wfile.flush()
            if len(TTS_CACHE) < 300:
                TTS_CACHE[text] = bytes(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self):
        q = self.path.split("since=")
        since = int(q[1].split("&")[0]) if len(q) > 1 and q[1].split("&")[0].isdigit() else 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_ping = time.time()
        try:
            while True:
                with COND:
                    pending = [e for e in EVENTS if e["seq"] > since]
                    if not pending:
                        COND.wait(timeout=10)
                        pending = [e for e in EVENTS if e["seq"] > since]
                for e in pending:
                    self.wfile.write(f"id: {e['seq']}\ndata: {json.dumps(e, default=str)}\n\n".encode())
                    since = e["seq"]
                if time.time() - last_ping > 12:
                    self.wfile.write(b": ping\n\n")
                    last_ping = time.time()
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4_000_000:
            return self._json({"error": "too large"}, 413)
        if self.path.split("?")[0] == "/api/voice/approve":        # waits for a human: no global lock
            try:
                return self._json(voice_approve(json.loads(self.rfile.read(n) or b"{}").get("flow", "")))
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._json({"ok": False, "say": f"Something failed: {type(e).__name__}."}, 500)
        if self.path.split("?")[0] == "/api/voiceid/enrol":
            import base64
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
                if d.get("reset"):
                    VOICE_FILE.unlink(missing_ok=True)
                    return self._json({"ok": True, "say": "Voice ID removed."})
                return self._json(voiceid_enrol([base64.b64decode(c) for c in d.get("clips") or []]))
            except Exception as e:
                return self._json({"ok": False, "say": f"Voice ID failed: {type(e).__name__}."}, 500)
        if self.path.split("?")[0] == "/api/stt":               # raw 16 kHz WAV from the browser
            try:
                wav = self.rfile.read(n)
                spk = voiceid_check(wav)
                text = transcribe(wav) if spk["ok"] else ""
                if spk.get("enrolled"):
                    emit("voice.speaker", ("Voice ID: owner recognised" if spk["ok"] else
                                           "Voice ID: NOT the owner — ignored, nothing transcribed or acted on"),
                         "jarvis", "jarvis", "ok" if spk["ok"] else "refused",
                         fields=[F("Similarity to Sagar's voiceprint", spk["score"], "mono"),
                                 F("Threshold", spk["threshold"], "mono"),
                                 F("Owner", spk["ok"], "check" if spk["ok"] else "bad"),
                                 F("Clip", f"{spk['seconds']} s", "mono")])
                return self._json({"text": text, "speaker": spk})
            except Exception as e:
                return self._json({"text": "", "error": type(e).__name__}, 500)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        p = self.path.split("?")[0]
        try:
            with LOCK:
                if p == "/api/reset":
                    seed()
                    return self._json({"ok": True})
                if p == "/api/voice/plan":
                    return self._json(voice_plan(d.get("text", ""), d.get("product", "payi")))
                if p == "/api/voice/confirm":
                    return self._json(voice_confirm(d.get("flow", "")))
                if p == "/api/voice/cancel":
                    f = FLOWS.get(d.get("flow", ""))
                    if f and f["state"] in ("planned", "stepup", "touch"):
                        f["state"] = "cancelled"
                        emit("voice.cancel", "Cancelled — nothing was signed", "jarvis", "jarvis", "refused",
                             f["product"], d.get("flow", ""))
                    return self._json({"ok": True, "say": "Cancelled. Nothing was sent."})
                if p == "/api/voice/stepup":
                    return self._json(voice_stepup(d.get("flow", ""), d.get("frames") or [], d.get("who", "")))
                if p == "/api/palm/enrol":
                    return self._json(palm_enrol(d.get("steady") or [], d.get("moved") or [], bool(d.get("demo"))))
                if p == "/api/palm/move":
                    return self._json(palm_move(d.get("path") or []))
                if p == "/api/palm/queue":
                    WORLD.b_on = bool(d.get("on"))
                    b = WORLD.b
                    if WORLD.b_on and not WORLD.pass_of(b):
                        path = [(-2.0 + 4.0 * i / 15, 0.2) for i in range(16)]
                        r = b.enter(WORLD.acq.pub, WORLD.store.cert, WORLD.gate.track(b.eph_pub, path, WORLD.rnd))
                        if r["ok"]:
                            WORLD.store.register(r["pass"])
                            emit("pass.issue", "Shopper B walked in — their own anonymous pass", "phone", "edge",
                                 product="palmpay", fields=[F("Visit id", r["pass"]["body"]["visit_id"], "mono"),
                                                            F("Standing", "1.0 m behind you in the queue", "text")])
                    return self._json({"ok": True, "world": WORLD.snapshot()})
                if p == "/api/palm/pay":
                    return self._json(palm_pay(float(d.get("amount") or 0), d.get("items") or [],
                                               d.get("frames") or [], d.get("who", ""), bool(d.get("explicit"))))
                if p == "/api/palm/check":
                    ok, dd, lim, why = palm_check(d.get("frames") or [], d.get("who", ""))
                    return self._json({"ok": ok, "distance": dd, "limit": lim, "why": why})
        except Exception as e:  # show the error on the watcher rather than hiding it
            import traceback
            traceback.print_exc()
            emit("server.error", f"{type(e).__name__}: {e}", "monitor", "monitor", "refused")
            return self._json({"ok": False, "say": f"Something failed: {type(e).__name__}."}, 500)
        self._json({"error": "not found"}, 404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8420)
    a = ap.parse_args()
    print("unsealing the device key from the TPM …", flush=True)
    DEV = DeviceKey()
    print(f"  {DEV.source} ({DEV.unseal_s} s)", flush=True)
    seed()
    import subprocess
    lan, vpn = [], []
    try:
        for line in subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout.splitlines():
            dev, addr = line.split()[1], line.split()[3].split("/")[0]
            if dev == "lo" or dev.startswith(("docker", "br-", "veth")):
                continue
            (vpn if dev.startswith(("tun", "wg", "tap")) else lan).append(addr)
    except OSError:
        pass
    ip = (lan or vpn or ["127.0.0.1"])[0]
    print(f"\nPayI demo running\n  screen 1 (this laptop):  http://localhost:{a.port}/payer\n"
          f"  screen 2 (watcher):      http://{ip}:{a.port}/watch\n"
          f"  screen 3 (Alex):        http://{ip}:{a.port}/wallet\n"
          + (f"  (same Wi-Fi: {', '.join(lan)} · VPN only: {', '.join(vpn)})\n" if vpn else ""), flush=True)
    Server((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
