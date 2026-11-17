"""PayI -> OpenWallet: two screens, one real transaction.

  /send    speak or type "send 20 dirhams to alex"  (the payer screen)
  /watch   OpenWallet-branded module-by-module trace of what actually happened

The authority half is the real PayI flow — the roster match, the both-signed mandate
from data/pay.db, and a real TPM device signature. Only once that passes does the
money half run, and that is a real call to the dev OpenWallet API; nothing here is mocked.

Stages are tagged so the watch screen can be honest about provenance:
  live    observed here (we did it, or the API told us)
  db      verified by reading dev MySQL after the fact
  path    the module the request travels through inside OpenWallet, from the code —
          shown for orientation, not claimed as an observation
"""
import argparse, json, os, queue, re, socketserver, subprocess, sys, tempfile, threading, time
import http.server
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.dirname(HERE)
sys.path.insert(0, AMP)
sys.path.insert(0, HERE)

import openwallet_api
from app import intent  # the one amount parser, shared with the PayI flow

# The customer-experience engine lives in its own package so that the decision policy, the
# scope wall and the model client can be reviewed (and tested) without reading the payment
# demo. It never imports anything that moves money -- it PROPOSES, and the proposal comes
# back through the same mandate + device-signature path as the voice flow.
sys.path.insert(0, os.path.join(AMP, "cx"))
from engine import engine as cx_engine   # noqa: E402
from engine import prefilter as cx_prefilter   # noqa: E402
from engine import client as cx_client   # noqa: E402
from engine.matrix import explain_matrix  # noqa: E402
import data as cx_data                   # noqa: E402
sys.path.insert(0, os.path.join(AMP, "cx", "reco"))

WEB = os.path.join(HERE, "web")
SUBS: list[queue.Queue] = []
SUBS_LOCK = threading.Lock()
TRACE: list[dict] = []


UTTERANCES = os.path.join(AMP, "data", "utterances.jsonl")


def record_utterance(source, text, **fields):
    """Append one spoken or typed utterance, and what the parser made of it.

    This exists because we could not measure anything: the demo kept no record of what was
    said, so "how often does the amount parse correctly" had no answer and the 1-for-1-million
    defect went unnoticed until it happened in front of people. Text only -- no audio is
    written, and nothing here is read back into the payment path.
    """
    try:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "text": text}
        row.update(fields)
        os.makedirs(os.path.dirname(UTTERANCES), exist_ok=True)
        with open(UTTERANCES, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass            # a logging failure must never stop or alter a payment


def emit(stage, kind, detail, tag="live"):
    ev = {"ts": time.strftime("%H:%M:%S"), "t": time.time(),
          "stage": stage, "kind": kind, "detail": detail, "tag": tag}
    try:   # whose event this is — the app only shows a customer their own
        import userctx
        _u = userctx._CUR.get()
        if _u and _u.get("user_id"):
            ev["uid"] = _u["user_id"]
    except Exception:                                              # noqa: BLE001
        pass
    TRACE.append(ev)
    with SUBS_LOCK:
        for q in list(SUBS):
            q.put(ev)
    return ev


# ---------- the authority half (real PayI) ----------

def payi_authorise(sentence, amount):
    """Roster match, both-signed mandate, TPM signature. Raises on refusal."""
    # The magnitude read-back. It is on screen before the mandate is checked and before
    # anything is signed, so "1 million, that is 1,000,000" is visible while it can still
    # be stopped -- the failure mode here was a magnitude nobody ever saw.
    emit("PayI · Intent", "intent.parse",
         f'heard "{sentence}" → transfer / person / amount'
         f' · reading it as {intent.say_amount(float(amount))} AED')

    from app import mandate_store as ms
    from app.intent import RosterResolver
    payee = RosterResolver().resolve(sentence, "agt_sagar")
    emit("PayI · Intent", "payee.resolve",
         f'exact token match → {payee["name"]} ({payee["mandate_id"]})')

    m = ms.get(payee["mandate_id"])
    payer_pub, payee_pub = ms.public_key(m.payer.agent_id), ms.public_key(m.payee.agent_id)
    if not (payer_pub and payee_pub and m.verify(payer_pub, payee_pub)):
        raise RuntimeError("that mandate does not carry two valid signatures")
    emit("PayI · Mandate", "mandate.verify",
         f"both signatures valid · per_txn {m.limits.per_txn:g} · per_day {m.limits.per_day:g}"
         f" · green cap {m.green_tier_cap:g} {m.currency}")

    tier = m.check(float(amount), "AED", 0, 0, 0)
    emit("PayI · Mandate", "tier.decide", f"{tier.upper()} — " + (
        "within the no-touch cap" if tier == "green" else "over the no-touch cap, step-up required"))

    blob = json.dumps({"payee": payee["agent_id"], "amount": float(amount), "cur": "AED"},
                      sort_keys=True, separators=(",", ":")).encode()
    sig, backend, hardware = sign_payload(blob)
    emit("PayI · Device", "device.sign",
         f"{backend} · Ed25519 · {sig.hex()[:24]}…"
         + ("" if hardware else "   ⚠ SOFTWARE KEY — not hardware-held"),
         tag="live" if hardware else "path")
    return payee, tier


# device_key.load() refuses to fall back to a key on disk, by design: "a payment key's
# protection must never be a surprise". That is right for the real thing and wrong for a
# rehearsal on a laptop whose TPM has gone away, so the downgrade is opt-in and loud —
# OPENWALLET_SOFTWARE_KEY=1 — and every screen is told it is software, never silently.
_SOFT_KEY = None


def sign_payload(blob: bytes):
    """Sign with the hardware key, or say plainly that we could not.

    The fallback is NOT silent. If the TPM is unreachable — on this laptop the user is not in
    the 'tss' group, so /dev/tpmrm0 cannot be opened — we sign with a software key and the
    reason travels with the signature to the watch screen, which renders it as
    "⚠ SOFTWARE KEY — not hardware-held". A demo that quietly downgraded its own security
    guarantee would be worse than one that stops.
    """
    global _SOFT_KEY
    reason = "forced by OPENWALLET_SOFTWARE_KEY" if os.environ.get("OPENWALLET_SOFTWARE_KEY") else None
    if reason is None:
        from app import device_key
        try:
            k = device_key.load()
            return k.sign(blob), type(k).__name__, True
        except Exception as e:
            # Fail fast and loudly rather than hanging the payment on a device we cannot open.
            reason = str(e) or type(e).__name__
            emit("PayI · Device", "device.key.unavailable",
                 f"hardware key unusable — {reason}", tag="guard")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if _SOFT_KEY is None:
        _SOFT_KEY = Ed25519PrivateKey.generate()
    return _SOFT_KEY.sign(blob), f"software ({reason})", False


# ---------- the money half (real OpenWallet) ----------

OPENWALLET_PATH = [
    ("JWTFilterBean", "verifies HS512, builds RequestContext from the claims"),
    ("EncryptionAdvice", "unwraps encKey with the static key, then encData"),
    ("TransactionHandlerFactory", "product SEND_MONEY → WalletToWalletHandler"),
    ("discountValidator", "no campaign on this txn"),
    ("productAndCountryValidator", "product 4 active for AE, slab + fees resolved"),
    ("senderValidator", "sender wallet ACTIVE, KYC Successful"),
    ("receiverValidator", "receiver resolved BY MOBILE via globalWalletKeyMapper"),
    ("VelocityService", "per-txn / daily / receiver max-balance caps"),
    ("Ledger", "debit sender leg + credit receiver leg"),
    ("Kafka → Notification", "transaction event published to the notification engine"),
]


def run_payment(sentence, amount, receiver_mobile=None, discount=None, session=None,
                payee_hint: str = ""):
    """`receiver_mobile`/`discount` are set for a merchant payment: the authority half is
    identical (the merchant's own both-signed mandate and limits), and the money half settles
    to the merchant's wallet with the campaign the incentive engine quoted."""
    if cx_data.logged_in():
        # A customer logged in through the app. The PayI mandates and the enrolled device key
        # belong to the configured demo customer only, so the authority here is what the real
        # app relies on: the payee is someone on THIS customer's own roster (resolved on the
        # server, by name — a mobile from the browser is never used), the wallet has verified
        # their PIN, and the wallet's own velocity limits apply at /transact.
        if not receiver_mobile:
            who = cx_data.roster_entry(payee_hint) if payee_hint else None
            if not who:
                msg = "I can only send to people you've paid before — I couldn't match that name."
                emit("OpenWallet · Roster", "payee.unknown", msg, tag="guard")
                emit("", "done", json.dumps({"ok": False, "message": msg}))
                return
            receiver_mobile, send_product = who["mobile"], "SEND_MONEY"
            emit("OpenWallet · Roster", "payee.resolve",
                 f"{who['name']} — on this customer's own payment history (wallet •••{who['mobile'][-3:]})",
                 tag="db")
        emit("CX · PIN", "pin.verified", "the wallet verified this customer's PIN before release",
             tag="guard")
    else:
        try:
            payee, tier = payi_authorise(sentence, amount)
        except Exception as e:
            emit("PayI", "refused", str(e), tag="live")
            emit("", "done", json.dumps({"ok": False, "message": str(e)}))
            return

    emit("OpenWallet · Treasury", "tms.skip",
         "treasuryEnabled=false in treasury_configuration — TMS/FX not consulted", tag="db")

    def on_stage(name, detail):
        pretty = ", ".join(f"{k}={v}" for k, v in detail.items())
        emit("OpenWallet · Client", name, pretty)

    _product = ("SEND_MONEY" if locals().get("send_product") else
                "MERCHANT_PAYMENT" if receiver_mobile else None)
    out = openwallet_api.transact(float(amount), on_stage=on_stage,
                             receiver_mobile=receiver_mobile, discount=discount,
                             product=_product)

    for mod, what in OPENWALLET_PATH:
        emit("OpenWallet · Server", mod, what, tag="path")

    data = (out or {}).get("data") or {}
    ok = str(out.get("status")) == "200" and (data.get("status") == "SUCCESS")
    if ok and session:
        # Tell the conversation what it just did, so "I want a refund" a moment later is about
        # THIS payment rather than an older one that merely looks more broken.
        try:
            from engine import conversation
            conversation.remember_payment(session, str(data.get("txnId")),
                                          payee_name_of(data), float(data.get("amount") or 0))
            # A merchant bill is the thing people actually split, and the moment right after
            # paying it is the only moment they are all still standing there. Offering then
            # is the difference between a feature people use and one they have to go looking
            # for. It is an OFFER — the split is created only if they say yes, and only for
            # people already on their own list.
            if receiver_mobile and _product == "MERCHANT_PAYMENT":
                conversation.offer_split(session, str(data.get("txnId")),
                                         payee_name_of(data),
                                         float(data.get("amount") or 0))
        except Exception:
            pass
    if ok:
        emit("OpenWallet · Result", "transaction.success",
             f'txnId {data.get("txnId")} · {data.get("amount")} {data.get("currency")} '
             f'· fee {data.get("serviceCharges")} · total {data.get("totalAmount")}')
        emit("OpenWallet · Result", "parties",
             f'{data.get("sender")} → {data.get("receiver")}')
        try:
            from engine import brd_handlers as _brd
            _b, _ = _brd.wallet_balance()
            emit("OpenWallet · Ledger", "wallet.balance", f"your balance is now {_b:,.2f} AED", tag="db")
        except Exception as e:
            emit("OpenWallet · Ledger", "wallet.balances", f"not read ({type(e).__name__})", tag="db")
    else:
        emit("OpenWallet · Result", "transaction.failed",
             out.get("message") or data.get("responseMessage") or json.dumps(out)[:200])

    emit("", "done", json.dumps({"ok": ok, "data": data, "message": out.get("message")}))


def stack_info() -> dict:
    """Which models are serving right now — for the backend console."""
    out = {"stt": {"model": os.path.basename(WHISPER_MODEL), "server": WHISPER_SERVER,
                   "multilingual": WHISPER_MULTILINGUAL},
           "intent": None, "second_opinion": None}
    try:
        import urllib.request
        h = json.loads(urllib.request.urlopen("http://127.0.0.1:8450/health", timeout=3).read())
        card = h.get("card") or {}
        out["intent"] = {"name": card.get("name"), "backbone": card.get("backbone"),
                         "params_millions": card.get("params_millions"),
                         "labels": len(card.get("labels") or []), "latency": h.get("latency"),
                         "eval": card.get("hand_written_eval")}
        out["second_opinion"] = h.get("laya") or h.get("second_opinion")
    except Exception:                                          # noqa: BLE001
        pass
    return out


def cx_home() -> dict:
    """Everything the customer app's home screen shows, from real data.

    Balance from wallet_master (what the wallet itself says), recipients from who this
    customer has actually paid (people first, merchants after), and the last few payments.
    Read-only: nothing here can move money.
    """
    import userctx
    _nm = (userctx.name() or cx_data.CUSTOMER["name"]).split()
    out = {"name": _nm[0].title() if _nm else "", "currency": "AED",
           "balance": None, "recipients": [], "recent": [], "live": False}
    try:
        from engine import brd_handlers as _brd
        bal, _src = _brd.wallet_balance()          # wallet_master for THIS customer
        if bal is not None:
            out["balance"], out["live"] = float(bal), True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        txns = cx_data.transactions_for(limit=40)
        seen = []
        for t in txns:
            nm = (t.payee or "").strip()
            if nm and nm not in seen and not getattr(t, "is_merchant", False):
                seen.append(nm)
        for t in txns:
            nm = (t.payee or "").strip()
            if nm and nm not in seen:
                seen.append(nm)
        for nm in seen[:8]:
            words = [w for w in nm.split() if w]
            out["recipients"].append({"name": nm, "short": words[0].title() if words else nm,
                                      "initials": "".join(w[0] for w in words[:2]).upper()})
        for t in txns[:5]:
            pub = t.public()
            out["recent"].append({"payee": t.payee, "amount": t.amount, "currency": t.currency,
                                  "status": pub.get("status"), "when": pub.get("when"),
                                  "ref": t.ref})
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}"
    return out


def read_balances():
    """Read the two wallets straight from dev MySQL, so the screen shows ground truth."""
    from db import db_conn  # local copy of the Vault->MySQL helper
    cfgj = openwallet_api.cfg()
    conn = db_conn(); cur = conn.cursor()
    out = {}
    for uid, who in ((cfgj["sender"], "Sagar"), (cfgj["receiver"], "Alex")):
        cur.execute("select balance from wallet_master where user_id=%s", (uid,))
        r = cur.fetchone()
        out[who] = str(r["balance"]) if r else "?"
    return out



# ---------- speech to text ----------
# Browser SpeechRecognition only exists in Chromium. Rather than tell people to switch
# browsers, transcribe server-side with the whisper.cpp build that GlassLink already uses:
# MediaRecorder exists everywhere, and whisper-cli reads ogg/opus (Firefox's default) and
# webm directly, so no ffmpeg hop is needed.
WHISPER = os.path.expanduser("~/glasslink/whisper.cpp/build/bin/whisper-cli")
# small.en over base.en: base mis-hears the two words that matter most here — it gave
# "Send 35 dams to Vika's datage" for "send 35 dirhams to Alex Morgan". Roughly 3x the
# compute on a 3-second clip, which is nothing next to the TPM unseal already in the flow.
#
# 2026-10-14: the customer-app BRD requires English AND Arabic voice at launch, and small.en
# cannot transcribe Arabic at all (it was also run with "-l en", which forced English on
# everything). The default is now the MULTILINGUAL small model, 5-bit quantised (190 MB):
# it auto-detects the language and handles English, Arabic and Hindi. small.en remains as a
# fallback if the multilingual file is missing.
_MULTI = os.path.expanduser("~/glasslink/whisper.cpp/models/ggml-small-q5_1.bin")
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL", _MULTI if os.path.isfile(_MULTI)
    else os.path.expanduser("~/glasslink/whisper.cpp/models/ggml-small.en.bin"))
WHISPER_MULTILINGUAL = not WHISPER_MODEL.endswith(".en.bin")

# Priming whisper with the vocabulary it is about to hear. Names and "dirhams" are
# exactly what a general model guesses wrong, and the payee is matched by EXACT token,
# so a mangled name is a refusal rather than a wrong payment.
WHISPER_PROMPT = os.environ.get(
    "WHISPER_PROMPT",
    "Send 20 dirhams to Alex Morgan. Pay Sagar Uniyal. AED dirhams wallet transfer.")


def stt_available():
    """Speech works if whisper-server answers (the normal path) OR the CLI fallback exists.
    Checking only for the CLI binary made a server-only deploy refuse every utterance with
    "whisper.cpp is not built" — shown in the app as "I didn't catch any words"."""
    if os.path.isfile(WHISPER) and os.path.isfile(WHISPER_MODEL):
        return True
    try:
        import urllib.request
        urllib.request.urlopen(WHISPER_SERVER + "/", timeout=2).read(64)
        return True
    except Exception:                                              # noqa: BLE001
        return False


#: The last language whisper detected, for the caller to report (and to answer in).
LAST_STT_LANG: dict = {"lang": None}

#: whisper-server keeps the model in memory. whisper-cli reloads 190 MB from disk on every
#: call, which cost several seconds per utterance; the server answers the same request in
#: about two. Started by ~/amp/stt/run.sh; when it is not running, whisper-cli is used.
WHISPER_SERVER = os.environ.get("WHISPER_SERVER", "http://127.0.0.1:8178")

#: Priming per language. Whisper spells what it has seen in the prompt: without it "DEWA"
#: came out as "Deva", "Daliya" as "Delhi", and Arabic numbers lost letters. The prompt is
#: in the language being spoken — an English prompt drags Arabic speech towards English.
STT_PROMPTS = {
    # NO English prompt. Every prompt tried made whisper copy from it somewhere: example
    # sentences were echoed back ("Send 200 AED to Alex. Send 200 AED to Alex."), a word
    # list was CONTINUED ("150 AED, 50 AED, VVVV…" for "send 50 dirhams to Alex"), and one
    # turned "Pay" into "I paid". For a payment flow a faithful transcript matters more than
    # spelling; known words are corrected afterwards by normalise_vocab(), which can only map
    # onto vocabulary we already have and can never invent an amount or a payee.
    #
    # …except ONE short phrase in the customer's own words, which is the exception that
    # held up on real voice. With nothing, the user's "send 200 dirhams to Alex" came out
    # "same 200 the rounds to the cause"; with "Send dirhams to <their top payee>." it was
    # right on all eight test clips and was never echoed. Longer prompts were.
    "en": "Send dirhams to {top}.",
    "ar": "أرسل ثلاثمائة درهم إلى محمد. ادفع فاتورة ديوا. كم رصيدي الآن. اشحن رقمي. ذكرني بالدفع.",
    "hi": "मोबाइल रिचार्ज कर दो। पचास दिरहम भेजो। बिल भरना है। बैलेंस कितना है।",
}
_WHISPER_LANG = {"english": "en", "arabic": "ar", "hindi": "hi", "urdu": "hi"}

#: The language each session speaks, detected on its FIRST utterance and reused after.
#: Detection is a second full pass over the audio (~1-3 s); a customer does not switch
#: language between "send 50 to Alex" and "yes".
STT_SESSION_LANG: dict[str, str] = {}

#: Decoding settings for every request. temperature_inc=0 turns off whisper's fallback
#: re-decoding: when it is unsure it otherwise re-decodes at rising temperatures, and on
#: Arabic forced through English that loop ran for 41 seconds and produced "camera see I
#: Now". One pass, no timestamps; greedy for live partials, beam for the final text.
_DECODE = {"temperature_inc": "0", "no_timestamps": "true"}


def _wav_seconds(audio: bytes) -> float | None:
    """Length of a PCM WAV clip from its header, or None if it is not one."""
    try:
        if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            return None
        import struct
        i = 12
        rate = ch = bits = None
        while i + 8 <= len(audio):
            cid, size = audio[i:i + 4], struct.unpack("<I", audio[i + 4:i + 8])[0]
            if cid == b"fmt ":
                ch, rate = struct.unpack("<HI", audio[i + 10:i + 16])
                bits = struct.unpack("<H", audio[i + 22:i + 24])[0]
            if cid == b"data" and rate:
                return size / (rate * ch * (bits // 8))
            i += 8 + size + (size & 1)
    except Exception:                                          # noqa: BLE001
        return None
    return None


#: One utterance at a time through whisper-server, finals first. On a CPU server a live
#: partial costs about as much as the gap between partials, so they queued up behind each
#: other and the FINAL transcript waited behind the queue: "send money to alex" took 53 s.
#: A partial now runs only if whisper is idle; a final always gets its turn.
_STT_LOCK = threading.Lock()


def _stt_server(audio: bytes, fields: dict, partial: bool = False) -> dict | None:
    """One multipart request to whisper-server; None if it is not reachable. A partial
    returns {"text": "", "skipped": True} at once if whisper is busy."""
    if partial:
        # CX_STT_PARTIALS=0 (CPU servers): no live partials at all, so a final never waits
        if os.environ.get("CX_STT_PARTIALS", "1") == "0" or not _STT_LOCK.acquire(blocking=False):
            return {"text": "", "skipped": True}
    elif not _STT_LOCK.acquire(timeout=60):
        return None
    try:
        return _stt_server_locked(audio, fields)
    finally:
        _STT_LOCK.release()


def _stt_server_locked(audio: bytes, fields: dict) -> dict | None:
    import urllib.request
    import uuid
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f'Content-Type: application/octet-stream\r\n\r\n'.encode() + audio + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(WHISPER_SERVER + "/inference", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())
    except Exception:                                          # noqa: BLE001
        return None


def stt_prompt(lang: str, names: list[str] | None = None) -> str:
    p = STT_PROMPTS.get(lang) or ""
    if "{top}" in p:
        from collections import Counter
        firsts = [n.split()[0].title() for n in (names or []) if n and n.split()]
        top = Counter(firsts).most_common(1)[0][0] if firsts else "Alex"
        return p.replace("{top}", top)
    nm = [n.split()[0].title() for n in (names or []) if n][:6]
    return p.replace("{names}", ", ".join(dict.fromkeys(nm)) or "Alex, Sonu")


#: Whisper's annotations and its well-known phrases invented from silence or noise. None of
#: them is something a customer said, and none may reach the screen or the engine.
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪+|♫+")
_SILENCE_PHRASES = {"thank you", "thanks for watching", "thank you for watching", "you", "bye",
                    "so", "okay", "music", "applause", "silence", "blank audio", "شكرا", "شكراً",
                    "اشتركوا في القناة", "ترجمة نانسي قنقر", "धन्यवाद"}


#: Spellings whisper produces for words we know, mapped to the word. Each was observed.
_VOCAB_FIX = [
    (re.compile(r"\b(?:deva|devil|dev a|diva|dewa|dewaa|duwa)\b(?=\s+bill|\s*$|\s+account)", re.I), "DEWA"),
    (re.compile(r"\b(?:dyrums?|durhams?|dirhems?|dirhames?|dirham)\b", re.I), "dirhams"),
    (re.compile(r"\b(?:etisalath|etisalaat|itisalat|ethisalat)\b", re.I), "Etisalat"),
    (re.compile(r"\bA\.?\s?E\.?\s?D\b\.?"), "AED"),
    # number + a mangled "dirhams": "200 the rams", "200 the rounds", "50 drums"
    (re.compile(r"(?<=\d)\s+(?:the\s+)?(?:rams|rums|rounds|runs|drums|durhams?|dirhams?|dhams?)\b", re.I), " dirhams"),
    # a mangled verb at the very START, right before an amount: "same 200…", "sen 200…"
    (re.compile(r"^(?:it'?s\s+)?(?:same|sen|sand|sent|send|son|cent|scent)\b(?=\s+\d)", re.I), "Send"),
    # the "S" of a clipped "Send money" is lost: "And money", "End money", "Sand money"
    (re.compile(r"^(?:and|end|sand|sen|cent)\s+(?=money\b)", re.I), "Send "),
]


def _sound(w: str) -> str:
    """A rough sound key: spellings that are pronounced alike compare alike."""
    w = w.lower()
    for x, y in (("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "s"),
                 ("oo", "u"), ("ee", "i"), ("y", "i"), ("h", "")):
        w = w.replace(x, y)
    return re.sub(r"(.)\1+", r"\1", w)


def _close(a: str, b: str) -> float:
    import difflib
    return max(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(),
               difflib.SequenceMatcher(None, _sound(a), _sound(b)).ratio())


def normalise_vocab(text: str, names: list[str] | None = None) -> str:
    """Correct near-misses onto vocabulary we already have: billers, currency words, and the
    customer's OWN payees. A word is only replaced by a payee name when it is a close match
    (≥ 0.7) and sits where a payee goes ("to X", "pay X") — nothing is ever invented."""
    t = text or ""
    for rx, word in _VOCAB_FIX:
        t = rx.sub(word, t)
    first = list(dict.fromkeys(n.split()[0].title() for n in (names or []) if n))
    def fix(m):
        w = m.group(2)
        best = max(first, key=lambda n: _close(w, n), default=None)
        if best and best.lower() != w.lower() and _close(w, best) >= 0.7:
            return m.group(1) + best
        return m.group(0)
    if first:
        t = re.sub(r"(\b(?:to|pay|send|for|with)\s+)([A-Za-z]{3,})", fix, t, flags=re.I)
    return t


def _degenerate(text: str, secs: float | None) -> bool:
    """A whisper decoding loop rather than words: "पचास पचास पचास…", "ट्ट्ट्ट्…", or more words
    than anyone can say in the clip (people manage ~3 words a second)."""
    if not text:
        return False
    words = re.findall(r"[\w\u0600-\u06FF\u0900-\u097F]+", text)
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        best = max(best, run)
    if best >= 4:
        return True
    if re.search(r"(.{1,3})\1{6,}", text):          # a syllable repeated over and over
        return True
    if secs and len(words) > max(8, secs * 5):
        return True
    return False


def clean_transcript(text: str) -> str:
    """The words the customer spoke, or "" if whisper heard only noise or silence."""
    t = _ANNOTATION.sub(" ", text or "")
    t = re.sub(r"\s+", " ", t).strip(" .,!?-—")
    if t.lower().strip(" .!") in _SILENCE_PHRASES or not re.search(r"[\w\u0600-\u06FF\u0900-\u097F]", t):
        return ""
    return t


def _collapse_repeats(text: str) -> str:
    """Whisper sometimes says the same sentence twice. "Send 200 AED to Alex. Send 200 AED to
    Alex." is one instruction, not two — and a doubled instruction is the last thing a
    payment flow should be handed."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?؟।])\s+", text or "") if p.strip()]
    out = []
    for p in parts:
        if not out or p.lower().rstrip(".!?؟।") != out[-1].lower().rstrip(".!?؟।"):
            out.append(p)
    return " ".join(out)


def transcribe_fast(audio: bytes, lang: str = "auto", names: list[str] | None = None,
                    session: str | None = None, partial: bool = False) -> dict | None:
    """Detect (if asked) then transcribe with a language-specific prompt, on the warm server.

    The audio window is sized to the clip: whisper otherwise processes a full 30 seconds
    for a 3-second sentence. Only for WAV, whose length is known exactly — a window shorter
    than the speech would silently cut the end off.
    """
    secs = _wav_seconds(audio)
    ctx = 0 if secs is None else min(1500, int(secs * 50) + 64)
    t0 = time.time()
    detected, prob = None, None
    sticky = False
    # Only an EXPLICIT language choice sticks (STT_SESSION_LANG is set when the app sends
    # en/ar). A detected language used to stick too, and one noisy first clip detected as
    # Hindi forced every later utterance into Hindi — "send money" came back "मनि" and the
    # forced language sent whisper into "पचास पचास पचास…" loops. Detection is ~0.2 s on the
    # GPU; it runs on every utterance.
    if lang == "auto" and session and STT_SESSION_LANG.get(session, "").startswith("explicit:"):
        lang, sticky = STT_SESSION_LANG[session].split(":", 1)[1], True
    if lang == "auto" and os.environ.get("CX_STT_DETECT", "1") == "0":
        lang = "en"              # no GPU: see below — detection is the expensive part on CPU
    if lang == "auto" and partial:
        # a live partial does not pay for detection: one pass, language left to whisper
        r = _stt_server(audio, dict(_DECODE, language="auto", audio_ctx=ctx, response_format="json",
                                    beam_size="1", best_of="1"), partial=True)
        if r is None:
            return None
        return {"text": clean_transcript(re.sub(r"\s+", " ", (r.get("text") or "")).strip()), "lang": None,
                "partial": True, "seconds": secs, "audio_ctx": ctx,
                "ms": int((time.time() - t0) * 1000), "engine": "whisper-server"}
    detect_ms = 0
    detected = prob = None
    r = None
    if lang == "auto" and os.environ.get("CX_STT_DETECT", "1") == "0":
        # No GPU: language detection costs 1-3 s of CPU per utterance against 0.6 s for the
        # whole English transcript. Auto means English here; Arabic speakers pick ع once in
        # the app and it sticks for the session.
        lang = "en"
    if lang == "auto":
        # ONE pass: transcribe with language "auto" and read the detected language from the
        # same result. A separate detect-only call encoded the full 30 s window whatever the
        # clip length — invisible on the laptop GPU, 1-10 s per utterance on a server CPU.
        # Primed with the English prompt, since English is what most turns are.
        r = _stt_server(audio, dict(_DECODE, language="auto", audio_ctx=ctx,
                                    response_format="verbose_json", prompt=stt_prompt("en", names)))
        if r is None:
            return None
        detected = _WHISPER_LANG.get(str(r.get("detected_language") or r.get("language") or "").lower())
        prob = r.get("detected_language_probability") or r.get("language_probability")
        # Auto chooses between the LAUNCH languages only: Arabic when it is confidently
        # Arabic, otherwise English. Hindi is never auto-selected — Indian-accented English
        # scores "hi" at 0.84 on the laptop GPU and 0.86 on the server CPU for the SAME clip,
        # and forcing Hindi turns "How are you" into nothing. Hinglish is transcribed fine in
        # Latin script, which the engine reads. (The app's toggle is Auto / EN / ع.)
        lang = "ar" if detected == "ar" and (prob or 0) >= 0.6 else "en"
        detect_ms = int((time.time() - t0) * 1000)
        if detected != lang:
            r = None                     # decoded in the wrong language: once more, in `lang`
    if r is None:
        params = dict(_DECODE, language=lang, audio_ctx=ctx, response_format="json",
                      prompt=stt_prompt(lang, names))
        if partial:
            params.update(beam_size="1", best_of="1")
        r = _stt_server(audio, params, partial=partial)
        if r is None:
            return None
    LAST_STT_LANG["lang"] = lang
    _txt = normalise_vocab(clean_transcript(_collapse_repeats(re.sub(r"\s+", " ", (r.get("text") or "")).strip())), names)
    if _degenerate(_txt, secs):
        _txt = ""                        # a decoding loop, not speech: never sent on
    if _txt:
        _txt = cx_prefilter.correct(_txt)   # "Sen money" -> "send money", shown to the customer as such
    return {"text": _txt, "no_speech": not _txt, "lang": lang,
            "detected": detected, "detected_p": prob, "sticky_lang": sticky, "partial": partial,
            "seconds": secs, "audio_ctx": ctx, "detect_ms": detect_ms,
            "decode": "greedy" if partial else "beam 5",
            "model": os.path.basename(WHISPER_MODEL),
            "ms": int((time.time() - t0) * 1000), "engine": "whisper-server"}


def transcribe(audio: bytes, suffix=".ogg", lang: str = "auto") -> str:
    """`lang` is "auto" (detect), or a code such as "en" / "ar" to force one. Forcing is
    faster and avoids a mis-detection on very short clips; auto is the default because a
    customer does not announce which language they are about to speak."""
    if not WHISPER_MULTILINGUAL:
        lang = "en"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        cmd = [WHISPER, "-m", WHISPER_MODEL, "-f", path, "-nt", "-np", "-l", lang or "auto",
               "-t", str(min(8, os.cpu_count() or 4))]
        # The English priming prompt biases detection towards English, so it is only used
        # when English has been asked for.
        if (lang or "auto") == "en":
            cmd += ["--prompt", WHISPER_PROMPT]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        m = re.search(r"auto-detected language: (\w+)", r.stderr or "")
        LAST_STT_LANG["lang"] = m.group(1) if m else (None if lang == "auto" else lang)
        text = " ".join(line.strip() for line in r.stdout.splitlines()
                        if line.strip() and not line.startswith("["))
        return re.sub(r"\s+", " ", text).strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------- http ----------

def payee_name_of(data: dict) -> str:
    return (data.get("receiverName") or data.get("receiverFullName")
            or data.get("receiverMobile") or "")


REFUNDS: list[dict] = []        # refund proposals raised in this session

#: The recommender is cached for a short while: every call quotes the live Incentive Engine
#: once per merchant, and a UI that re-ranks on every poll would hammer it for no new answer.
_RECO_CACHE: dict = {"at": 0.0, "value": None}
_RECO_TTL = 25.0


def reco_for_demo(force: bool = False) -> dict:
    import agent as reco_agent
    now = time.time()
    if not force and _RECO_CACHE["value"] and now - _RECO_CACHE["at"] < _RECO_TTL:
        return _RECO_CACHE["value"]
    user = next(iter(reco_agent.ALLOWED_USERS))
    try:
        recs = [r.public() for r in reco_agent.recommend(user, limit=8)]
        err = None
    except Exception as e:
        recs, err = [], f"{type(e).__name__}: {e}"
    out = {"user": user, "recommendations": recs, "error": err,
           "merchants": cx_data.nearby(len(cx_data.MERCHANTS)),
           "restricted_to": sorted(reco_agent.ALLOWED_USERS),
           "generated_at": time.strftime("%H:%M:%S")}
    _RECO_CACHE.update(at=now, value=out)
    return out


def reco_draft(merchant: str | None) -> dict:
    """Draft a notification for one recommendation. The matrix decides whether it is allowed,
    and the agent refuses anything that is not."""
    import agent as reco_agent
    user = next(iter(reco_agent.ALLOWED_USERS))
    try:
        recs = reco_agent.recommend(user, limit=8)
        rec = next((r for r in recs if r.merchant == merchant), None)
        if not rec:
            return {"ok": False, "error": "I no longer have that recommendation."}
        out = reco_agent.draft_notification(user, rec)
        emit("Reco · Draft", "notification.drafted",
             f"#{out['campaign_id']} \"{out['headline']}\" → DRAFT, audience "
             f"{out['audience']}, awaiting human approval", tag="act")
        return {"ok": True, **out}
    except reco_agent.Refused as e:
        emit("Reco · Draft", "refused", str(e), tag="guard")
        return {"ok": False, "refused": True, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def req_session(handler) -> str | None:
    """The session on a GET, so /api/cx/context can give prompts for THIS conversation."""
    from urllib.parse import parse_qs
    return (parse_qs(urlparse(handler.path).query).get("session") or [None])[0]


def _studio_take(text: str, session: str) -> dict | None:
    """Ask Agent Studio (127.0.0.1:8467) whether the Orchestrator answers this message. None = the engine does."""
    if os.environ.get("CX_STUDIO_ROLLOUT", "1") == "0" or not (text or "").strip():
        return None
    import urllib.request
    from engine import conversation as _conv
    # The engine's own conversation state travels with the message, so the Orchestrator never takes the
    # answer to a question the engine is waiting on ("How much?" → "100"), or a yes/no to a payment.
    try:
        f = _conv.get(session)
        eng = {"frame": {"intent": f.intent, "asked_for": f.asked_for, "age": int(time.time() - f.at)} if f else None,
               "topic": _conv.topic(session), "proposal": bool(_conv.proposal(session)), "lang": _conv.last_lang(session)}
    except Exception:                                          # noqa: BLE001 — no state is still a valid state
        eng = {}
    try:
        req = urllib.request.Request(os.environ.get("STUDIO_APP_URL", "http://127.0.0.1:8467/app/route"),
                                     data=json.dumps({"text": text, "session": session, "engine": eng}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=float(os.environ.get("STUDIO_APP_TIMEOUT", "20"))).read())
    except Exception:                                          # noqa: BLE001 — the engine always has an answer
        return None
    if not r.get("handled") or not (r.get("say") or "").strip():
        return None
    # The conversation moved to the Orchestrator's agent: the engine's old topic must not pull the next
    # "show me more" back into a thread the customer has left.
    try:
        _conv._TOPIC.pop(session, None)
    except Exception:                                          # noqa: BLE001
        pass
    say = re.sub(r"\s*\[\d{1,2}\]", "", r["say"]).strip()
    return {"say": say, "intent": "knowledge_query", "confidence": None, "model": "studio-orchestrator",
            "outcome": "ANSWERED", "options": [], "sources": r.get("sources") or [], "agent": r.get("agent_name"),
            "lane_trace": [{"lane": "orchestrator", "what": f"answered by {r.get('agent_name')}",
                            "detail": f"Agent Studio decision #{r.get('ledger')}", "ms": r.get("ms")}]}


def cx_ask(text: str, session: str = "default", pick: dict | None = None) -> dict:
    """Answer a customer question and narrate the decision onto the live event stream.

    Every lane step becomes an event, so the watch screen shows the reasoning as it happens
    rather than a verdict that appeared from nowhere. The decision itself is made in
    cx/engine -- this function only translates it into something the UI can watch.
    """
    TRACE.clear()
    from engine import nudge as cx_nudge
    cx_nudge.reset(session)              # they spoke; the idle clock starts over
    emit("CX · Question", "cx.start", text, tag="live")
    answer = cx_engine.ask(text, session=session, pick=pick)
    for st in answer.lane_trace:
        emit(f"CX · {st['lane'].title()}", st["what"], st["detail"], tag=_cx_tag(st["tag"]))
    emit("CX · Answer", "cx.answer", json.dumps({
        "intent": answer.intent, "outcome": answer.outcome, "say": answer.say,
        "confidence": answer.confidence, "model": answer.model, "top": answer.top,
        "degraded": answer.degraded, "txn": answer.txn, "slots": answer.slots,
        "options": answer.options, "proposal": answer.proposal,
        "advisory": answer.advisory, "ms": answer.ms}), tag="live")
    stash_plan(session, answer, text)
    out = cx_engine.to_dict(answer)
    # Fresh prompts ride back with the answer. The page used to fetch these separately and,
    # in practice, never did — so the strip above the input was fixed from the greeting
    # onwards. Sending them here means they cannot drift from the turn they belong to, and
    # costs no extra round trip.
    try:
        out["suggestions"] = cx_data.suggestions(session)
    except Exception:
        out["suggestions"] = []
    return out


def cx_image(image: bytes, session: str = "default", caption: str = "") -> dict:
    """A screenshot instead of a sentence. Narrated onto the same event stream as a typed
    question, so the watch screen shows the reading and the lookup rather than an answer
    that appeared from a picture."""
    TRACE.clear()
    from engine import nudge as cx_nudge
    cx_nudge.reset(session)
    emit("CX · Question", "cx.start", f"[screenshot, {len(image):,} bytes]", tag="live")
    answer = cx_engine.ask_image(image, session=session, caption=caption)
    for st in answer.lane_trace:
        emit(f"CX · {st['lane'].title()}", st["what"], st["detail"], tag=_cx_tag(st["tag"]))
    emit("CX · Answer", "cx.answer", json.dumps({
        "intent": answer.intent, "outcome": answer.outcome, "say": answer.say,
        "txn": answer.txn, "options": answer.options, "ms": answer.ms}), tag="live")
    stash_plan(session, answer, "[screenshot]")
    out = cx_engine.to_dict(answer)
    try:
        out["suggestions"] = cx_data.suggestions(session)
    except Exception:                                         # noqa: BLE001
        out["suggestions"] = []
    return out


def _cx_tag(tag: str) -> str:
    """Map engine tags onto the watch screen's existing tag vocabulary, so the new lanes
    look like they belong to the same screen rather than bolted on."""
    return {"model": "model", "laya": "laya", "guard": "guard", "act": "act",
            "db": "db", "input": "live"}.get(tag, "path")


def request_share(rec, participant, emit):
    """Ask one person for their share, with REQUEST_TO_PAY.

    Product 57 is a real transaction with the parties swapped — the counterparty is debited,
    we are credited. `ReceiverValidator` treats it specially for exactly that reason, so this
    is the platform's own mechanism for asking rather than something invented here.

    Returns (txnId, reason). A None txnId is an honest failure: nobody was asked, and the
    caller must not pretend otherwise.
    """
    import openwallet_api as _api
    try:
        mobile = None
        if cx_data.logged_in():
            mobile = next((p["mobile"] for p in cx_data.roster()
                           if p["userId"] == participant["userId"]), None)
        for uid, v in (_api.cfg().get("people") or {}).items():
            if not mobile and uid == participant["userId"] and not cx_data.logged_in():
                mobile = v.get("mobile")
        if not mobile:
            return None, "no wallet mobile configured for them in this environment"
        out = _api.transact(float(participant["amount"]), product="REQUEST_TO_PAY",
                            receiver_mobile=mobile)
        data = (out or {}).get("data") or {}
        txn = data.get("txnId")
        if not txn:
            return None, (out or {}).get("message") or "the request was not accepted"
        return txn, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def pin_prompt(session: str, amount=None, who: str = "") -> dict:
    """The wording on the PIN sheet: what is being confirmed, in the customer's language.

    A prompt for a credential with nothing on it asks somebody to authorise they-know-not-
    what, and the chat message that used to carry the amount is now behind an overlay.
    """
    confirming = ""
    if amount is not None:
        try:
            confirming = (f"<b>AED {float(amount):,.2f}</b>"
                          + (f" to {who}" if who else ""))
        except (TypeError, ValueError):
            confirming = f"to {who}" if who else ""
    try:
        from engine import conversation as _conv, lang as _lang
        say = _lang.t("enter_pin", _conv.last_lang(session)) or "Enter your PIN to confirm."
    except Exception:
        say = "Enter your PIN to confirm."
    return {"ok": True, "needPin": True, "confirming": confirming, "say": say}


def start_payment(sentence, amount, receiver_mobile=None, discount=None, session=None,
                  pin_ok: bool = False, who: str = ""):
    """Start a real payment, or hold it for a PIN. THE ONLY WAY run_payment is started.

    Every screen funnels through here — the chat, the voice send screen, the merchant till
    — because the PIN gate is only a gate if nothing can go round it. It used to live in
    `cx_act` alone, which covered the chat and left the voice send screen paying out with
    no PIN at all: same wallet, same money, different page.

    `pin_ok` is set only by `release_pin` after the wallet itself has verified the PIN. The
    caller cannot pass it in from a request body.
    """
    from engine import pin as cx_pin
    session = session or "send-screen"
    if cx_pin.required() and not pin_ok:
        cx_pin.challenge(session, {"kind": "_payment", "sentence": sentence,
                                   "amount": amount, "receiver_mobile": receiver_mobile,
                                   "discount": discount, "session": session, "who": who})
        emit("CX · PIN", "pin.required",
             "CX_REQUIRE_PIN is on — holding this payment until the wallet verifies the "
             "PIN", tag="guard")
        return pin_prompt(session, amount, who)

    TRACE.clear()
    import userctx
    userctx.spawn(run_payment, sentence, amount, receiver_mobile, discount, session, who or "")
    return {"ok": True, "started": True, "sentence": sentence, "amount": amount,
            "receiver": "merchant" if receiver_mobile else "configured peer"}


def release_pin(session: str, pin_text: str) -> dict:
    """Verify a PIN with the wallet and resume what was held. One release path, every screen."""
    from engine import pin as cx_pin
    held = cx_pin.pending(session)
    if not held:
        return {"ok": False, "error": "There's nothing waiting for a PIN."}
    ok, msg = cx_pin.validate(session, str(pin_text or ""))
    if not ok:
        emit("CX · PIN", "pin.rejected", msg, tag="guard")
        return {"ok": False, "error": msg,
                "retry": cx_pin.pending(session) is not None}
    action = dict(held["action"])
    cx_pin.clear(session)
    emit("CX · PIN", "pin.ok", "PIN verified by the wallet — releasing the payment",
         tag="act")
    # READ the kind; do not pop it. Popping removed it from the dict handed to cx_act,
    # so a chat payment came back from a correct PIN as "I do not know how to do that" —
    # the money was authorised and then the instruction describing it had been deleted.
    if action.get("kind") == "_payment":
        return start_payment(pin_ok=True,
                             **{k: v for k, v in action.items() if k != "kind"})
    return cx_act(action)


def cx_act(req: dict) -> dict:
    """Carry out what the customer picked. Money-moving choices go through run_payment --
    the same signed path as the voice flow -- and a refund is only ever RAISED, never paid."""
    kind = req.get("kind")
    session = req.get("session") or "default"

    # ── the PIN gate ────────────────────────────────────────────────────────────────────
    # One flag, every money-moving action. Checked HERE rather than inside each handler so
    # there is exactly one place a payment can slip past it. A gate some flows honour and
    # others skip is not a gate.
    from engine import pin as cx_pin

    if kind == "pin_cancel":
        cx_pin.clear(session)
        emit("CX · PIN", "pin.cancelled", "the customer backed out of the PIN prompt",
             tag="guard")
        return {"ok": True, "say": "No problem — I've left it. Nothing has moved."}

    if kind == "pin":
        return release_pin(session, req.get("pin"))

    if kind == "cancel_scheduled":
        import scheduled_pay
        import userctx
        ok = scheduled_pay.cancel(userctx.user_id(), str(req.get("id") or ""))
        emit("CX · Act", "payment.unscheduled", f"{req.get('id')}: {'cancelled' if ok else 'not found'}",
             tag="act")
        return {"ok": ok, "done": ok,
                "say": "Cancelled — that scheduled payment won't go. Nothing was sent." if ok
                       else "I couldn't find that scheduled payment — it may have already gone."}

    # A statement moves no money, but it sends the customer's history out of the app — so it
    # ALWAYS needs the PIN, whatever CX_REQUIRE_PIN says. The address is resolved here from
    # the customer's profile; nothing in the request decides where it goes.
    if kind == "email_statement":
        import statement as cx_statement
        count = max(1, min(int(req.get("count") or 5), cx_statement.MAX_ROWS))
        if not req.get("_pin_ok"):
            cx_pin.challenge(session, {"kind": "email_statement", "count": count,
                                       "session": session, "_pin_ok": True})
            to, _src = cx_statement.recipient()
            emit("CX · PIN", "pin.required",
                 f"emailing {count} transactions — holding it until the PIN is verified", tag="guard")
            try:
                from engine import conversation as _conv, lang as _lang
                _say = _lang.t("enter_pin", _conv.last_lang(session)) or "Enter your PIN to confirm."
            except Exception:
                _say = "Enter your PIN to confirm."
            return {"ok": True, "needPin": True, "say": _say,
                    "confirming": f"your last <b>{count}</b> transactions to <b>{cx_statement.mask(to)}</b>"}
        try:
            r = cx_statement.send(count)
        except Exception as exc:                                    # noqa: BLE001
            emit("CX · Email", "email.failed", f"{type(exc).__name__}", tag="guard")
            return {"ok": False, "say": "I couldn't send the email just now — the mail service "
                                        "didn't answer. Nothing else was affected; try again in a bit."}
        if not r.get("ok"):
            return {"ok": False, "say": r.get("error")}
        emit("CX · Email", "email.sent",
             f"{r['count']} transactions to {r['to']} ({r['source']})", tag="act")
        return {"ok": True, "done": True,
                "say": f"Sent — your last {r['count']} transactions are on their way to {r['to']}."}

    if kind in ("send", "pay_merchant", "split", "bill_pay", "bill_schedule") \
            and cx_pin.required() \
            and not req.get("_pin_ok"):
        cx_pin.challenge(session, dict(req, _pin_ok=True))
        emit("CX · PIN", "pin.required",
             "CX_REQUIRE_PIN is on — holding this payment until the PIN is verified by the "
             "wallet", tag="guard")
        # Say WHAT is being confirmed on the PIN sheet itself. A prompt for a credential
        # with no amount on it asks someone to authorise they-know-not-what, and the chat
        # message it replaced is now behind an overlay.
        amt = req.get("grandTotal")
        if amt is None:
            amt = req.get("payable") if req.get("payable") is not None else req.get("amount")
        who = (req.get("merchant") or req.get("payee")
               or (req.get("bills") or [{}])[0].get("nickname") or "")
        confirming = (f"<b>AED {float(amt):,.2f}</b>" + (f" to {who}" if who else "")
                      if amt is not None else "")
        # The sheet's own wording. The server has no Answer object here, so the language
        # comes from the sentence that opened the flow, kept on the conversation.
        try:
            from engine import conversation as _conv, lang as _lang
            _l = _conv.last_lang(session)
            _say = _lang.t("enter_pin", _l) or "Enter your PIN to confirm."
        except Exception:
            _say = "Enter your PIN to confirm."
        return {"ok": True, "needPin": True, "confirming": confirming, "say": _say}

    if kind == "raise_refund":
        ref = req.get("ref")
        txn = cx_data.find_txn(ref)
        if not txn:
            return {"ok": False, "error": "I could not find that payment."}
        if txn.refund_raised:
            return {"ok": True, "already": True,
                    "say": "A refund is already raised for that payment."}
        txn.refund_raised = True
        rec = {"ref": txn.ref, "amount": txn.amount, "currency": txn.currency,
               "raised_at": time.strftime("%H:%M:%S"), "state": "AWAITING_APPROVAL"}
        REFUNDS.append(rec)
        emit("CX · Refund", "refund.raised",
             f"{txn.amount:g} {txn.currency} on {txn.ref} — queued for approval; "
             f"no money moves until a person approves it", tag="act")
        return {"ok": True, "refund": rec,
                "say": f"Refund raised for {txn.amount:g} {txn.currency}. It needs approval "
                       f"before the money moves, and you will get a notification."}

    if kind == "split":
        # Creating a split moves no money on its own: it records what each person owes and
        # notifies them. The shares are sent exactly as the customer was shown them — the
        # backend recomputes and rejects the request if its own arithmetic disagrees, which
        # is what stops the assistant quietly rounding anyone's share.
        from split import api as split_api
        parts = [{"userId": p["userId"], "displayValue": float(p["displayValue"])}
                 for p in (req.get("participants") or []) if p.get("userId")]
        if not parts:
            return {"ok": False, "error": "I need at least one person to split with."}
        ref = req.get("referenceTxnId")
        if not ref:
            return {"ok": False, "error": "I need the payment to split."}
        emit("CX · Split", "split.create",
             f"{req.get('splitType')} of {req.get('currency','AED')} "
             f"{float(req.get('total') or 0):g} at {req.get('merchant')} between "
             + ", ".join(f"{p['name'].split()[0]} {p['displayValue']:g}"
                         for p in req["participants"])
             + f", {float(req.get('initiatorDisplayValue') or 0):g} mine", tag="act")
        split_id = split_api.create_post_txn(
            reference_txn_id=ref, participants=parts,
            initiator_display=float(req.get("initiatorDisplayValue") or 0),
            split_type=req.get("splitType") or "SHARE",
            bill_description=req.get("billDescription") or "",
            currency=req.get("currency") or "AED")
        if not split_id:
            why = split_api.last_error() or "unknown"
            emit("CX · Split", "split.failed", why, tag="guard")
            return {"ok": False,
                    "error": f"I couldn't create that split — {why}. Nothing has changed."}
        names = ", ".join(p["name"].split()[0] for p in req["participants"])
        emit("CX · Split", "split.created",
             f"{split_id} created; {names} notified", tag="act")
        # Follow it, so the moment somebody settles their share the chat says so instead of
        # the customer having to ask.
        try:
            from split import watcher as split_watcher
            split_watcher.watch(split_id)
            split_watcher.start(emit)
        except Exception:
            pass
        return {"ok": True, "splitId": split_id,
                "say": f"Done — {names} {'have' if len(parts) > 1 else 'has'} been asked "
                       f"for their share. Reference {split_id}."}

    if kind == "split_nudge":
        from split import api as split_api
        sid, uid = req.get("splitId"), req.get("userId")
        if not sid:
            return {"ok": False, "error": "I need to know which split to chase."}
        ok = split_api.nudge(sid, [uid] if uid else None)
        emit("CX · Split", "split.nudge", f"{sid} reminder sent" if ok else
             (split_api.last_error() or "nudge failed"), tag="act" if ok else "guard")
        if ok:
            return {"ok": True, "say": "I've sent them a reminder."}
        why = split_api.last_error() or "unknown"
        if "permission" in why.lower():
            # Not the customer's problem and not a transient failure: this account is not
            # permitted to send reminders. Say what IS possible instead of repeating a
            # permissions error nobody outside the team can act on.
            return {"ok": False,
                    "error": "I can't send reminders from here yet. They can see what they "
                             "owe in their own OpenWallet app."}
        return {"ok": False, "error": f"I couldn't send that reminder — {why}."}

    if kind == "bill_schedule":
        # A promise to pay later. Nothing moves now; the amount SHOWN is the amount stored,
        # so a bill that grows before the due date does not quietly take more than the
        # customer agreed to.
        from bills import schedule as bill_schedule
        import openwallet_api as _api
        bills = req.get("bills") or []
        when_at = req.get("whenAt")
        if not bills or not when_at:
            return {"ok": False, "error": "I need a bill and a time to schedule it."}
        import userctx
        me = userctx.user_id()
        ids = []
        for b in bills:
            ids.append(bill_schedule.add(
                me, {"planId": b["planId"], "productId": b.get("productId"),
                     "identifier": b["identifier"], "nickname": b.get("nickname"),
                     "biller": b.get("biller")},
                float(b.get("amount") or 0), when_at,
                pay_mode=req.get("payMode") or "WALLET"))
        emit("CX · Bill", "bill.scheduled",
             f"{len(ids)} payment(s) for {req.get('whenAt')} via "
             f"{req.get('payMode')} — {', '.join(ids)}", tag="act")
        when = req.get("when") or when_at[:10]
        return {"ok": True, "scheduleIds": ids,
                "say": f"Scheduled for {when}. I'll pay AED "
                       f"{float(req.get('total') or 0):.2f} then — say the word any time "
                       f"before and I'll cancel it."}

    if kind == "bill_pay":
        from bills import pay as bill_pay
        out = bill_pay.pay_now(req, emit)
        return out

    if kind == "bill_split":
        # Open the collection, then ASK each person. Recorded before asking, so a failure to
        # reach one of them cannot leave a half-made split with no record of who was asked.
        from bills import billsplit
        import openwallet_api as _api
        parts = [p for p in (req.get("participants") or []) if p.get("userId")]
        if not parts or not req.get("planId"):
            return {"ok": False, "error": "I need a bill and someone to split it with."}
        import userctx
        me = userctx.user_id()

        # Is one already running for THIS bill? Asking again is how the customer ended up
        # requesting the same 253 from Alex four times: each failed attempt at the biller
        # started a brand-new collection while his money sat in the old one.
        existing = billsplit.open_for_bill(me, req["planId"], req["identifier"])
        if existing:
            prog = billsplit.progress(existing)
            if prog["ready"]:
                # Everything is already collected — the only thing left is the biller.
                from bills import billsplit_watch
                billsplit_watch.watch(existing["id"])
                billsplit_watch.start(emit)
                emit("CX · Bill split", "billsplit.reuse",
                     f"{existing['id']} already has every share — retrying the biller "
                     f"instead of asking again", tag="guard")
                return {"ok": True, "splitId": existing["id"], "reused": True,
                        "say": f"{', '.join(prog['paid'])} already paid their share of "
                               f"{prog['bill']} — I'm not asking again. Trying the biller "
                               f"now with the AED {existing['total']:.2f} collected."}
            waiting = ", ".join(prog["waiting"]) or "them"
            emit("CX · Bill split", "billsplit.reuse",
                 f"{existing['id']} is still collecting — not starting a second one",
                 tag="guard")
            return {"ok": True, "splitId": existing["id"], "reused": True,
                    "say": (f"That one's already going — "
                            + (f"{', '.join(prog['paid'])} paid, " if prog["paid"] else "")
                            + f"still waiting on {waiting} for AED "
                              f"{prog['outstanding']:.2f}. I won't ask twice.")}

        rec = billsplit.create(
            me, {"planId": req["planId"], "productId": req.get("productId"),
                 "identifier": req["identifier"], "nickname": req.get("bill"),
                 "biller": req.get("biller")},
            float(req.get("total") or 0),
            [{"userId": p["userId"], "name": p["name"]} for p in parts])
        emit("CX · Bill split", "billsplit.created",
             f"{rec['id']} · {rec['nickname']} AED {rec['total']:g} · "
             + ", ".join(f"{p['name'].split()[0]} {p['amount']:g}"
                         for p in rec["participants"])
             + f" · mine {rec['my_share']:g}", tag="act")

        asked, failed = [], []
        for p in rec["participants"]:
            txn, why = request_share(rec, p, emit)
            billsplit.mark_requested(rec["id"], p["userId"], txn)
            (asked if txn else failed).append(p["name"].split()[0])
            if not txn:
                emit("CX · Bill split", "billsplit.ask_failed",
                     f"{p['name'].split()[0]}: {why}", tag="guard")
        try:
            from bills import billsplit_watch
            billsplit_watch.watch(rec["id"])
            billsplit_watch.start(emit)
        except Exception:
            pass

        if not asked:
            return {"ok": False,
                    "error": f"I couldn't ask {', '.join(failed)} for their share, so I've "
                             f"not started this. Your {rec['nickname']} bill is still due."}
        say = (f"Asked {', '.join(asked)} for AED "
               f"{rec['participants'][0]['amount']:.2f}. The moment it arrives I'll pay "
               f"{rec['biller']} — nothing leaves your wallet until then.")
        if failed:
            say += f" I couldn't reach {', '.join(failed)}."
        return {"ok": True, "splitId": rec["id"], "say": say}

    if kind in ("send", "pay_merchant"):
        # The payable is what the customer owes AFTER the offer. Taking req["amount"] first
        # would charge the pre-discount price and quietly pocket the discount.
        amount = float(req.get("payable") if req.get("payable") is not None
                       else req.get("amount") or 0)
        if amount <= 0:
            return {"ok": False, "error": "I need an amount."}
        # The configured demo keeps its old default; an app login never gets one — a send
        # with no payee must not quietly become a send to somebody.
        who = req.get("payee") or req.get("merchant") or ("" if cx_data.logged_in() else "alex")
        if kind == "send" and not who:
            return {"ok": False, "error": "Who should I send it to?"}
        sentence = req.get("sentence") or f"send {amount:g} dirhams to {who}"
        if kind == "pay_merchant":
            if req.get("discount"):
                emit("CX · Act", "offer.applied",
                     f"AED {float(req['discount']):g} off {req.get('campaign') or 'campaign'} — "
                     f"paying {amount:g} instead of {float(req.get('amount') or amount):g}",
                     tag="act")
        receiver_mobile = req.get("receiverMobile") if kind == "pay_merchant" else None
        discount = None
        if kind == "pay_merchant" and req.get("campaignId"):
            discount = {"campaignId": req["campaignId"], "promoCode": req.get("campaign"),
                        "payeeName": req.get("merchant"),
                        "discountAmount": req.get("discount")}
        if kind == "pay_merchant" and not receiver_mobile:
            # REFUSE, do not fall back. Settling to the configured peer would look like a
            # successful merchant payment on every screen and be nothing of the sort — the
            # money would reach the wrong wallet. A payment that cannot reach its payee is a
            # failure, and saying so is the only honest outcome.
            emit("CX · Act", "merchant.unconfigured",
                 f"no wallet mobile configured for {who} — refusing rather than settling to "
                 f"the configured peer, which would pay the wrong wallet", tag="guard")
            emit("", "done", json.dumps(
                {"ok": False, "message": f"I cannot pay {who} yet — their wallet is not "
                                         f"wired up in this environment."}))
            return {"ok": False,
                    "error": f"I cannot pay {who} yet — their wallet is not wired up in "
                             f"this environment. Nothing was sent."}
        # A payment AT a time ("pay alex 100 in 2 minutes"). We only get here after the PIN
        # gate above — the wallet has verified the PIN — so what is held is an authorised
        # payment, not a credential. It is released through this same branch when due.
        if req.get("scheduleAt"):
            import scheduled_pay
            from engine.engine import SCHEDULE_MAX_MIN, _dubai
            got = scheduled_pay.schedule(dict(req, sentence=sentence), SCHEDULE_MAX_MIN)
            if not got.get("ok"):
                return got
            emit("CX · Act", "payment.scheduled",
                 f"AED {amount:g} to {who} held until {got['due']} — PIN verified now, "
                 f"released by the scheduler (the PIN is not kept)", tag="act")
            return {"ok": True, "done": True, "scheduled": got["id"],
                    "say": f"Scheduled — AED {amount:g} to {who} will go at {_dubai(got['due'])}. "
                           f"Say \"cancel my scheduled payment\" to stop it before then.",
                    "options": [{"kind": "cancel_scheduled", "id": got["id"],
                                 "label": "Cancel this payment"}]}
        # Through the chokepoint, like every other screen. The gate above has already
        # cleared this one; `_pin_ok` carries that through rather than asking twice.
        return start_payment(sentence, amount, receiver_mobile, discount,
                             req.get("session"), pin_ok=True, who=who or "")

    # ── from the customer-app BRD ─────────────────────────────────────────────────────
    if kind == "recharge":
        # No prepaid top-up rail is connected in this environment. Saying so is the only
        # honest answer — a "done" here would be a receipt for money that never moved.
        return {"ok": False, "say": (f"Mobile recharge isn't connected to a live operator in this "
                                     f"environment yet, so nothing was charged. The request was "
                                     f"{req.get('operator', '')} {req.get('phone', '')}, "
                                     f"AED {float(req.get('amount') or 0):,.2f}.")}
    if kind == "navigate":
        screens = {"add_money": "Add Money", "withdraw_money": "Withdraw Money",
                   "link_bank": "Connect Bank"}
        return {"ok": True, "navigate": req.get("screen"),
                "say": f"In the app this opens the {screens.get(req.get('screen'), 'requested')} screen."}
    return {"ok": False, "error": "I do not know how to do that."}


PAGES = {"/": "send.html", "/send": "send.html", "/watch": "watch.html",
         "/wallet": "wallet.html",
         # New surfaces. The two existing customer screens above are untouched.
         "/assist": "assist.html", "/console": "console.html", "/merchant": "merchant.html",
         "/reco": "reco.html",
         #: The seven-web view of one answer. Built to show how a question is actually
         #: processed, and reusable for any intent — the webs come from the engine's own
         #: trace, not from anything spend-specific.
         "/webs": "webs.html"}


#: The most recent crawl plan per session, so the /webs screen can follow a conversation
#: happening on another surface. Built from the SAME answer the chat produced — never by
#: asking the engine again, which for a payment would mean running it twice.
LAST_PLAN: dict = {}


def stash_plan(session: str, answer, question: str) -> None:
    try:
        import webs as webslib
        LAST_PLAN[session] = webslib.build(answer, question)
        emit("CX · Webs", "cx.webs", session, tag="live")
    except Exception as e:                                   # noqa: BLE001
        log(f"webs plan not built: {type(e).__name__}: {e}")


def webs_plan(question: str, session: str = "webs") -> dict:
    """One question, answered, then rendered as the seven webs the crawler walked.

    The answer is produced by exactly the same `engine.ask` the chat surface calls — this is
    a view of a real turn, not a replay or a simulation. If it were a simulation the whole
    point of showing it would be gone.
    """
    try:
        import webs as webslib
        from engine import engine as E
        a = E.ask(question, session=session)
        return webslib.build(a, question)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


class _Pending:
    """Logins part-way through (OTP sent, PIN not yet entered). Ten minutes, in memory."""
    def __init__(self):
        self._d, self._lock = {}, threading.Lock()

    def start(self, st: dict) -> str:
        import secrets as _s
        tx = "lgn_" + _s.token_urlsafe(18)
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._d.items() if now - v["at"] > 600]:
                self._d.pop(k, None)
            self._d[tx] = dict(st, at=now)
        return tx

    def get(self, tx):
        with self._lock:
            st = self._d.get(tx or "")
            return st if st and time.time() - st["at"] < 600 else None

    def drop(self, tx):
        with self._lock:
            self._d.pop(tx or "", None)


_PENDING_LOGINS = _Pending()


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ── who is asking ─────────────────────────────────────────────────────────────────
    # Every request runs AS the customer its app token names (userctx). Customer-data calls
    # without a login are refused when CX_REQUIRE_LOGIN=1; locally, with the flag off, the
    # configured sender stays the default so tools and tests keep working.
    _CUSTOMER_PATHS = ("/api/cx/home", "/api/cx/ask", "/api/cx/act", "/api/cx/login",
                       "/api/cx/session", "/api/stt", "/api/cx/image", "/api/pin")

    def _as_customer(self, inner):
        import userctx
        path = urlparse(self.path).path
        if path.startswith("/api/auth/"):
            return self._auth(path)
        user = userctx.lookup(self.headers.get("X-Cx-Auth"))
        if not user and userctx.require_login() and path.startswith(self._CUSTOMER_PATHS):
            return self._send(401, json.dumps({"ok": False, "error": "Please log in."}))
        tok = userctx.set_current(user)
        try:
            return inner()
        except userctx.NotLoggedIn:
            return self._send(401, json.dumps({"ok": False, "error": "Please log in."}))
        finally:
            userctx.reset(tok)

    def do_GET(self):
        return self._as_customer(self._do_GET)

    def do_POST(self):
        return self._as_customer(self._do_POST)

    def _auth(self, path):
        """mobile -> OTP -> PIN against the wallet's own login. The wallet tokens stay here;
        the browser gets an opaque app token naming the login."""
        import userctx
        import wallet_login as wl
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            req = {}
        bad = lambda msg, code=400: self._send(code, json.dumps({"ok": False, "error": msg}))  # noqa: E731
        try:
            if path == "/api/auth/otp":
                mobile = wl.normalise_mobile(req.get("mobile"))
                if not mobile:
                    return bad("Enter a UAE mobile number — 9 digits starting with 5.")
                tok = wl.send_otp(mobile)
                tx = _PENDING_LOGINS.start({"mobile": mobile, "otp_token": tok})
                emit("CX · Login", "otp.sent", f"OTP requested for •••{mobile[-4:]}", tag="act")
                return self._send(200, json.dumps({"ok": True, "tx": tx}))
            if path == "/api/auth/verify":
                st = _PENDING_LOGINS.get(req.get("tx"))
                otp = str(req.get("otp") or "")
                if not st:
                    return bad("That login has expired — start again with your mobile number.")
                if not (otp.isdigit() and len(otp) == 6):
                    return bad("Enter the 6-digit OTP.")
                got = wl.verify_otp(st["mobile"], otp, st["otp_token"])
                st.update(got, otp_ok=True)
                emit("CX · Login", "otp.ok", f"OTP verified for •••{st['mobile'][-4:]} · userId "
                     f"{'present' if got.get('userId') else 'MISSING'} · session key stored by wallet: "
                     f"{ {True: 'yes', False: 'NO — different key', None: 'unknown'}[wl.key_registered(got.get('userId') or '', got['encKey'])] }",
                     tag="act")
                return self._send(200, json.dumps({"ok": True, "tx": req.get("tx")}))
            if path == "/api/auth/pin":
                st = _PENDING_LOGINS.get(req.get("tx"))
                pin = str(req.get("pin") or "")
                if not st or not st.get("otp_ok"):
                    return bad("That login has expired — start again with your mobile number.")
                if not (pin.isdigit() and len(pin) == 4):
                    return bad("Enter your 4-digit PIN.")
                tokens = wl.validate_pin(st["mobile"], st["encKey"], st["accessToken"], pin,
                                         st.get("userId"))
                uid = st.get("userId")
                if not uid:
                    return bad("Logged in, but the wallet did not say which customer this is.", 502)
                prof = wl.profile(uid)
                user = {"user_id": uid, "mobile": st["mobile"], "name": prof.get("name") or "",
                        "access_token": tokens.get("accessToken") or st["accessToken"],
                        "refresh_token": tokens.get("refreshToken") or st.get("refreshToken"),
                        "enc_key": st["encKey"]}
                _PENDING_LOGINS.drop(req.get("tx"))
                auth = userctx.remember(user)
                emit("CX · Login", "login.ok", f"{(user['name'] or 'customer').split()[0]} logged in "
                     f"(•••{st['mobile'][-4:]})", tag="act")
                return self._send(200, json.dumps({"ok": True, "auth": auth, "user": userctx.public(user)}))
            if path == "/api/auth/logout":
                userctx.forget(self.headers.get("X-Cx-Auth"))
                return self._send(200, json.dumps({"ok": True}))
            return bad("not found", 404)
        except wl.LoginError as e:
            w = getattr(e, "wallet", None)
            emit("CX · Login", "login.failed", f"{path.rsplit('/', 1)[-1]}: wallet said "
                 + (f"HTTP {w['http']} · status {w['status']} · {w['message'] or '(no message)'}" if w
                    else f"HTTP {e.status} · {e}"), tag="guard")
            return bad(str(e), 401 if path == "/api/auth/pin" else 400)
        except Exception as e:                                       # noqa: BLE001
            emit("CX · Login", "login.error", type(e).__name__, tag="guard")
            return bad("Login didn't complete — please try again.", 502)

    def _do_GET(self):
        path = urlparse(self.path).path
        if path == "/events":
            return self.events()
        if path == "/api/webs/last":
            from urllib.parse import parse_qs
            q = parse_qs(urlparse(self.path).query)
            sess = (q.get("session") or ["default"])[0]
            return self._send(200, json.dumps(LAST_PLAN.get(sess) or {}))
        if path == "/api/trace":
            return self._send(200, json.dumps(TRACE))
        if path == "/api/cx/home":
            return self._send(200, json.dumps(cx_home()))
        if path == "/api/cx/console/overview":
            import sessions as cx_sessions
            return self._send(200, json.dumps(dict(cx_sessions.overview(), stack=stack_info())))
        if path == "/api/cx/console/turns":
            import sessions as cx_sessions
            from urllib.parse import parse_qs as _pq     # do_GET imports parse_qs locally further down
            sid = (_pq(urlparse(self.path).query).get("session") or [""])[0]
            return self._send(200, json.dumps(cx_sessions.turns(sid), default=str))
        if path == "/api/balance":
            try:
                b = read_balances()
                return self._send(200, json.dumps({"sagar": b.get("Sagar"), "alex": b.get("Alex")}))
            except Exception as e:
                return self._send(200, json.dumps({"error": type(e).__name__}))
        if path == "/api/reco":
            return self._send(200, json.dumps(reco_for_demo()))
        if path == "/api/reco/matrix":
            import matrix as rmatrix
            return self._send(200, json.dumps({"cells": rmatrix.explain_matrix()}))
        if path == "/api/reco/drafts":
            import agent as reco_agent
            try:
                return self._send(200, json.dumps({"drafts": reco_agent.list_drafts()}))
            except Exception as e:
                return self._send(200, json.dumps({"drafts": [], "error": str(e)}))
        if path == "/api/cx/matrix":
            return self._send(200, json.dumps({"cells": explain_matrix()}))
        if path == "/api/cx/context":
            return self._send(200, json.dumps({
                "customer": cx_data.CUSTOMER,
                "transactions": [t.public() for t in cx_data.transactions_for()],
                "beneficiaries": cx_data.sendable(),
                "suggestions": cx_data.suggestions(req_session(self)),
                "merchants": cx_data.nearby(len(cx_data.MERCHANTS)),
                "model": cx_client.health()}))
        if path == "/api/cx/nudge":
            # The page owns the clock, the server owns the policy. Keeping the decision here
            # means the rules about what may be said, and how often, live next to everything
            # else that decides what this assistant says — not in a timer in a browser.
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            sess = (qs.get("session") or ["default"])[0]
            try:
                idle = float((qs.get("idle") or ["0"])[0])
            except ValueError:
                idle = 0.0
            from engine import nudge as cx_nudge
            out = cx_nudge.due(sess, idle)
            if out:
                out["suggestions"] = cx_data.suggestions(sess)
                emit("CX · Nudge", "cx.nudge", out["say"], tag="live")
            return self._send(200, json.dumps(out or {}))
        if path == "/api/stt/available":
            return self._send(200, json.dumps({"available": stt_available()}))
        if path in PAGES:
            try:
                body = open(os.path.join(WEB, PAGES[path]), "rb").read()
            except OSError:
                return self._send(404, "page missing")
            return self._send(200, body, "text/html; charset=utf-8")

        if path == "/dev-cert.pem":
            # So the phone can fetch the certificate to trust. The PUBLIC half only —
            # dev-key.pem is never served, and the check below is a literal path, not
            # anything derived from the request.
            try:
                pem = open(os.path.join(HERE, "dev-cert.pem"), "rb").read()
            except OSError:
                return self._send(404, b"no certificate - run openwallet/make-cert.sh",
                                  "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-x509-ca-cert")
            self.send_header("Content-Disposition", "attachment; filename=openwallet-dev.pem")
            self.send_header("Content-Length", str(len(pem)))
            self.end_headers()
            return self.wfile.write(pem)

        if path == "/faceid.js":
            try:
                js = open(os.path.join(WEB, "faceid.js"), "rb").read()
            except OSError:
                return self._send(404, b"", "application/javascript")
            return self._send(200, js, "application/javascript; charset=utf-8")
        self._send(404, json.dumps({"error": "not found"}))

    def _do_POST(self):
        if urlparse(self.path).path.startswith("/api/stt") and urlparse(self.path).path != "/api/stt/available":
            n = int(self.headers.get("Content-Length") or 0)
            audio = self.rfile.read(n)
            if not stt_available():
                return self._send(200, json.dumps(
                    {"ok": False, "error": "whisper.cpp is not built at " + WHISPER}))
            ctype = self.headers.get("Content-Type") or ""
            suffix = ".webm" if "webm" in ctype else ".ogg" if "ogg" in ctype else ".wav"
            _q = parse_qs(urlparse(self.path).query)
            want = (_q.get("lang") or ["auto"])[0]
            stt_session = (_q.get("session") or [None])[0]
            partial = (_q.get("partial") or ["0"])[0] == "1"
            if want not in ("auto", "en", "ar", "hi"):
                want = "auto"
            if want != "auto" and stt_session:
                STT_SESSION_LANG[stt_session] = "explicit:" + want   # the customer CHOSE this
            try:
                t0 = time.time()
                # names on this customer's roster prime the English prompt, so the payee is
                # spelled the way the exact-match payee check expects
                try:
                    if cx_data.logged_in():
                        names = [p["name"] for p in cx_data.roster()]
                    else:
                        names = [v.get("name") for v in (json.load(open(
                            os.path.join(AMP, "openwallet", ".secrets.json"))).get("people") or {}).values()
                            if v.get("name")]
                except Exception:                              # noqa: BLE001
                    names = []
                try:   # and everyone this customer has actually paid
                    names += [t.payee for t in cx_data.transactions_for(limit=25) if t.payee]
                except Exception:                                  # noqa: BLE001
                    pass
                # Dev only: keep the last few FINAL clips so a mis-heard phrase can be replayed
                # against other models and prompts. Local disk, newest 20, never partials.
                if not partial and os.environ.get("CX_KEEP_STT_CLIPS", "1") == "1":
                    try:
                        _cd = os.path.join(AMP, "data", "stt_clips")
                        os.makedirs(_cd, exist_ok=True)
                        _fn = os.path.join(_cd, time.strftime("%H%M%S") + f"-{len(audio)}.wav")
                        open(_fn, "wb").write(audio)
                        for _old in sorted(os.listdir(_cd))[:-20]:
                            os.unlink(os.path.join(_cd, _old))
                    except Exception:                              # noqa: BLE001
                        pass
                fast = transcribe_fast(audio, want, names, session=stt_session, partial=partial)
                if fast is not None:
                    if not partial:
                        record_utterance("stt", fast["text"], bytes=len(audio),
                                         parsed_amount=intent.parse_amount(fast["text"]))
                    return self._send(200, json.dumps(dict(fast, ok=True,
                                                           model=os.path.basename(WHISPER_MODEL))))
                text = transcribe(audio, suffix, lang=want)
                # The ASR output as whisper produced it, before anything interprets it --
                # the only honest input for measuring the parser or any model we add later.
                record_utterance("stt", text, bytes=len(audio),
                                 parsed_amount=intent.parse_amount(text))
                return self._send(200, json.dumps({"ok": True, "text": text,
                                                   "lang": LAST_STT_LANG.get("lang"),
                                                   "ms": int((time.time() - t0) * 1000),
                                                   "model": os.path.basename(WHISPER_MODEL)}))
            except Exception as e:
                return self._send(200, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        if urlparse(self.path).path == "/api/cx/login":
            # Identity for the demo is the configured dev wallet (there is no customer login in
            # front of this server); the bot_id it returns is stable per customer and agent.
            import sessions as cx_sessions
            import userctx
            return self._send(200, json.dumps(cx_sessions.login(
                userctx.user_id(), userctx.name() or cx_data.CUSTOMER["name"])))
        if urlparse(self.path).path == "/api/cx/session":
            import sessions as cx_sessions
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            got = cx_sessions.start(str(req.get("bot_id") or ""), str(req.get("channel") or "app"))
            return self._send(200 if got else 404, json.dumps(got or {"error": "unknown bot_id — log in first"}))
        if urlparse(self.path).path == "/api/cx/ask":
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            session = req.get("session") or "default"
            t_ask = time.time()
            # Agent Studio's Orchestrator may answer — only a covered knowledge-agent question, only for
            # sessions inside its rollout percentage (0% = never). Option picks are engine flows. Any
            # error or timeout falls through to the engine exactly as before.
            out = None if req.get("pick") else _studio_take(req.get("text") or "", session)
            if out is None:
                out = cx_ask(req.get("text") or "", session, req.get("pick"))
            try:
                import sessions as cx_sessions
                out["turn_id"] = cx_sessions.record(session, req.get("text") or "", req.get("stt"),
                                                    out, int((time.time() - t_ask) * 1000))
            except Exception:                                  # noqa: BLE001
                pass
            return self._send(200, json.dumps(out, default=str))

        if urlparse(self.path).path == "/api/cx/image":
            # Raw bytes with the session in the query string, rather than multipart or a
            # base64 JSON field: the image is the whole body, so there is nothing to parse
            # and nothing to get wrong, and a 6MB screenshot does not become 8MB of text.
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 12 * 1024 * 1024:
                return self._send(200, json.dumps(
                    {"say": "That image is too large for me to read — anything up to about "
                            "12MB is fine.", "outcome": "NEED_INFO", "options": []}))
            body = self.rfile.read(n)
            q = parse_qs(urlparse(self.path).query)
            return self._send(200, json.dumps(cx_image(
                body, (q.get("session") or ["default"])[0],
                (q.get("caption") or [""])[0])))

        if urlparse(self.path).path == "/api/webs":
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            return self._send(200, json.dumps(webs_plan(req.get("q") or "",
                                                        req.get("session") or "webs")))

        if urlparse(self.path).path == "/api/reco/draft":
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            return self._send(200, json.dumps(reco_draft(req.get("merchant"))))

        if urlparse(self.path).path == "/api/cx/act":
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            # Underscore fields are the SERVER's own markers — `_pin_ok` above all, which
            # release_pin sets after the wallet has verified the PIN. Taken from a client they
            # skipped the PIN: {"kind": "send", "_pin_ok": true, ...} went straight through.
            req = {k: v for k, v in (req if isinstance(req, dict) else {}).items()
                   if not str(k).startswith("_")}
            return self._send(200, json.dumps(cx_act(req)))

        if urlparse(self.path).path == "/api/pin":
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length") or 0)) or b"{}")
            sess = body.get("session") or "send-screen"
            if body.get("cancel"):
                from engine import pin as _p
                _p.clear(sess)
                emit("CX · PIN", "pin.cancelled",
                     "the customer backed out of the PIN prompt", tag="guard")
                return self._send(200, json.dumps(
                    {"ok": True, "cancelled": True,
                     "say": "No problem — I've left it. Nothing has moved."}))
            return self._send(200, json.dumps(release_pin(sess, body.get("pin"))))

        if urlparse(self.path).path != "/api/send":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        sentence = (req.get("sentence") or "send 20 dirhams to alex morgan").strip()
        # The amount comes from the SAME parser as every other entry point. This used to be a
        # local regex taking the first digit run, which read "send 1 million dirhams" as 1 --
        # the guardrail existed, the demo just was not behind it.
        # And the SAME intent gate. /api/send went straight to the payee match and the
        # device signature, so "did my 500 dirhams to alex go through" -- a question --
        # reached the mandate and was stopped only by the 200 limit. Under it, it would
        # have paid. A question is not an instruction.
        verb, _ = intent.classify(sentence)
        if verb != "transfer":
            return self._send(200, json.dumps(
                {"accepted": False, "sentence": sentence,
                 "error": "That sounded like a question, not an instruction to send money."}))
        amount = req.get("amount")
        amount = float(amount) if amount else intent.parse_amount(sentence)
        record_utterance("send", sentence, parsed_amount=amount,
                         typed_amount=bool(req.get("amount")))
        if amount is None:
            return self._send(200, json.dumps(
                {"accepted": False, "sentence": sentence,
                 "error": "I could not read one clear amount in that. Say it again, "
                          "for example: send one million dirhams to Alex."}))
        # Name the payee on the PIN sheet. The roster match is the same one the payment
        # itself will use a moment later, and it is read-only — asking for a credential
        # while showing only an amount tells somebody what they are spending but not who
        # is receiving it, which is the half that a wrong payment gets wrong.
        who = ""
        try:
            from app.intent import RosterResolver      # imported where payi_authorise does
            who = (RosterResolver().resolve(sentence, "agt_sagar") or {}).get("name", "")
        except Exception:
            who = ""        # unresolved here is fine: the payment path refuses it properly
        out = start_payment(sentence, amount, session=req.get("session") or "send-screen",
                            who=who)
        if out.get("needPin"):
            return self._send(200, json.dumps(
                {"accepted": False, "needPin": True, "confirming": out["confirming"],
                 "say": out["say"], "sentence": sentence, "amount": amount}))
        self._send(200, json.dumps({"accepted": True, "sentence": sentence,
                                    "amount": amount}))

    def events(self):
        # An app login sees only its own events (?auth=, since EventSource cannot set
        # headers). The watch screens and console, with no login, see everything as before.
        import userctx
        _viewer = userctx.lookup((parse_qs(urlparse(self.path).query).get("auth") or [None])[0])
        _mine = (lambda ev: ev.get("uid") == _viewer["user_id"]) if _viewer else (lambda ev: True)
        q = queue.Queue()
        with SUBS_LOCK:
            SUBS.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            # History is replayed so a screen opened mid-run catches up. Flag it: the
            # wallet must not play the "money just arrived" animation for a payment that
            # landed before the page was even open.
            for ev in [e for e in list(TRACE) if _mine(e)]:
                self.wfile.write(f"data: {json.dumps(dict(ev, replay=True))}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                    if not _mine(ev):
                        continue
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with SUBS_LOCK:
                if q in SUBS:
                    SUBS.remove(q)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def warm_caches() -> None:
    """Fill the slow caches in the background at start-up.

    The first /api/cx/context took 7.6s and every one after it 21ms — Mongo connect, the
    ledger read, store locations, and a wallet lookup per merchant, all on the customer's
    first page load. Doing it before anyone asks costs nothing and removes the one slow
    moment they would actually notice.
    """
    def go():
        try:
            from split import watcher as split_watcher
            split_watcher.start(emit)
            split_watcher.adopt_open_splits()
            from bills import billsplit_watch
            billsplit_watch.adopt()
            billsplit_watch.start(emit)
        except Exception:
            pass
        try:
            # Reminders that fall due are pushed to the customer's app (Redis -> Kafka).
            import reminders_store
            reminders_store.start_scheduler(
                10.0, log=lambda *a: emit("CX · Reminder", "reminder.fire", " ".join(str(x) for x in a),
                                          tag="act"))
            print("  reminder scheduler running (Redis → app push)", flush=True)
            import scheduled_pay
            scheduled_pay.start(cx_act, 5.0, log=lambda *a: emit(
                "CX · Act", "payment.scheduled.fire", " ".join(str(x) for x in a), tag="act"))
            print("  scheduled-payment runner on (held, PIN-verified payments)", flush=True)
        except Exception as e:
            print(f"  reminder scheduler not started: {type(e).__name__}: {e}", flush=True)
        try:
            cx_data.transactions_for()
            cx_data.nearby(len(cx_data.MERCHANTS))
            cx_data.suggestions()
            print("  caches warm (ledger, stores, merchant wallets)", flush=True)
        except Exception as e:
            print(f"  cache warm-up skipped: {type(e).__name__}: {e}", flush=True)
    threading.Thread(target=go, daemon=True).start()


def lan_address() -> str:
    """The address this machine is actually reachable on.

    Not `hostname -I`[0]: that returns every address in no guaranteed order, and this machine
    carries docker bridges (172.17-19.x) and a VPN tunnel alongside the real one. Opening a
    UDP socket toward a public address makes the kernel pick the interface it would really
    route over — no packet is sent, and it keeps working when the network changes underneath.
    """
    import socket
    import subprocess

    # Probe toward the DEFAULT GATEWAY, not a public address. Probing 8.8.8.8 returns the
    # VPN tunnel's address whenever a VPN holds a more specific route — technically the
    # interface traffic leaves by, but useless for "open this on your phone", which needs the
    # address on the local network.
    gateway = None
    try:
        out = subprocess.run(["ip", "route", "show", "default"],
                             capture_output=True, text=True, timeout=3).stdout.split()
        if "via" in out:
            gateway = out[out.index("via") + 1]
    except Exception:
        pass

    for target in (gateway, "127.0.0.1", "8.8.8.8"):
        if not target:
            continue
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sk.connect((target, 80))
                return sk.getsockname()[0]
            finally:
                sk.close()
        except OSError:
            continue
    return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8440)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print("PayI → OpenWallet demo")
    ip = lan_address()
    for label, path in (("assist  (phone)", "assist"), ("watch   (big screen)", "watch"),
                        ("reco", "reco"), ("merchant till", "merchant"),
                        ("send", "send"), ("wallet", "wallet")):
        print(f"  {label:22}: http://{ip}:{a.port}/{path}")
    warm_caches()
    srv = Server((a.host, a.port), H)

    # TLS is OPTIONAL and off by default — the demo has always run over http on the LAN
    # and nothing about that changed. It exists because WebAuthn (Face ID) is only
    # exposed to a SECURE CONTEXT: over http://192.168.x.x the browser hides the API
    # entirely and no amount of page code can reach it.
    #
    # A self-signed certificate is enough for the browser to call the page secure, but
    # iOS will not trust it until the profile is installed AND switched on under
    # Settings → General → About → Certificate Trust Settings. Until then Safari shows a
    # warning and still withholds Face ID.
    if os.environ.get("OPENWALLET_TLS") == "1":
        import ssl
        cert = os.environ.get("OPENWALLET_CERT", os.path.join(HERE, "dev-cert.pem"))
        key = os.environ.get("OPENWALLET_KEY", os.path.join(HERE, "dev-key.pem"))
        if not (os.path.exists(cert) and os.path.exists(key)):
            print(f"  TLS asked for but {cert} / {key} are missing — "
                  f"see openwallet/make-cert.sh. Serving plain http instead.")
        else:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
            print(f"  TLS on — https://{ip}:{a.port}/  (Face ID needs this)")

    srv.serve_forever()


if __name__ == "__main__":
    main()
