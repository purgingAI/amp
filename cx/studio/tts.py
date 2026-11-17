"""High-quality narration for the 🎤 Pitch: ElevenLabs text-to-speech, cached per sentence.

The key never reaches a browser: it is read from STUDIO_ELEVEN_KEY or data/tts_key (owner-only file). Every
rendered sentence is cached as data/tts/<sha1(voice|model|settings|text)>.mp3, so a pitch is paid for once and a
rename only re-renders the sentences that contain the names. Only the pitch script is ever sent — never customer data.

    status()             configured?, voices, the account's character quota, cache size
    get(text, voice, …)  (mp3 bytes, cached?) — renders only if allowed (admins), else cached audio only
    plan(texts, voice)   how many characters a pre-render would spend (what is already cached is free)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

import store

API = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE = os.environ.get("STUDIO_ELEVEN_VOICE", "JBFqnCBsd6RMkjVDRZzb")     # George — warm British storyteller
MODEL = os.environ.get("STUDIO_ELEVEN_MODEL", "eleven_multilingual_v2")           # the highest-quality model
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True}
MAX_TEXT = 700
_lock = threading.Lock()
_voices: dict = {"at": 0, "list": []}


def _key() -> str:
    k = os.environ.get("STUDIO_ELEVEN_KEY", "").strip()
    if k:
        return k
    p = os.path.join(store.DATA, "tts_key")
    return open(p, encoding="utf-8").read().strip() if os.path.exists(p) else ""


def _dir() -> str:
    d = os.path.join(store.DATA, "tts")
    os.makedirs(d, exist_ok=True)
    return d


def _id(text: str, voice: str) -> str:
    return hashlib.sha1(json.dumps([voice, MODEL, SETTINGS, text], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _req(path: str, body: dict | None = None) -> urllib.request.Request:
    return urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                  headers={"xi-api-key": _key(), "content-type": "application/json", "accept": "*/*"},
                                  method="POST" if body is not None else "GET")


def voices() -> list[dict]:
    if not _key():
        return []
    if time.time() - _voices["at"] < 3600 and _voices["list"]:
        return _voices["list"]
    try:
        with urllib.request.urlopen(_req("/voices"), timeout=20) as r:
            d = json.loads(r.read())
        _voices["list"] = [{"id": v["voice_id"], "name": v["name"].split(" - ")[0], "desc": (v["name"].split(" - ") + [""])[1],
                            "accent": (v.get("labels") or {}).get("accent"), "gender": (v.get("labels") or {}).get("gender")} for v in d.get("voices", [])]
        _voices["at"] = time.time()
    except Exception:                                       # noqa: BLE001
        pass
    return _voices["list"]


def quota() -> dict | None:
    if not _key():
        return None
    try:
        with urllib.request.urlopen(_req("/user/subscription"), timeout=20) as r:
            s = json.loads(r.read())
        return {"used": s.get("character_count"), "limit": s.get("character_limit"), "resets": s.get("next_character_count_reset_unix"), "tier": s.get("tier")}
    except Exception:                                       # noqa: BLE001
        return None


def status() -> dict:
    d = _dir()
    files = [f for f in os.listdir(d) if f.endswith(".mp3")]
    return {"configured": bool(_key()), "provider": "elevenlabs", "model": MODEL, "default_voice": DEFAULT_VOICE,
            "voices": voices(), "quota": quota(), "cached": len(files),
            "cache_mb": round(sum(os.path.getsize(os.path.join(d, f)) for f in files) / 2**20, 1)}


def cached(text: str, voice: str) -> bool:
    return os.path.exists(os.path.join(_dir(), _id(text, voice) + ".mp3"))


def plan(texts: list[str], voice: str) -> dict:
    todo = [t for t in dict.fromkeys(t.strip() for t in texts if t and t.strip()) if not cached(t, voice)]
    return {"sentences": len(set(texts)), "to_render": len(todo), "characters": sum(len(t) for t in todo)}


def get(text: str, voice: str | None = None, may_render: bool = False) -> tuple[bytes | None, str]:
    """→ (mp3, how): how ∈ cached | rendered | not-cached | not-configured | error: …"""
    text = " ".join((text or "").split())[:MAX_TEXT]
    voice = (voice or DEFAULT_VOICE).strip()
    if not text:
        return None, "error: empty text"
    p = os.path.join(_dir(), _id(text, voice) + ".mp3")
    if os.path.exists(p):
        return open(p, "rb").read(), "cached"
    if not _key():
        return None, "not-configured"
    if not may_render:
        return None, "not-cached"
    with _lock:                                             # one render at a time; a parallel request for the same line waits
        if os.path.exists(p):
            return open(p, "rb").read(), "cached"
        body = {"text": text, "model_id": MODEL, "voice_settings": SETTINGS}
        try:
            with urllib.request.urlopen(_req(f"/text-to-speech/{voice}?output_format=mp3_44100_128", body), timeout=90) as r:
                audio = r.read()
        except urllib.error.HTTPError as e:
            return None, f"error: elevenlabs {e.code} {e.read()[:160].decode('utf-8', 'replace')}"
        except Exception as exc:                            # noqa: BLE001
            return None, f"error: {type(exc).__name__}"
        tmp = p + ".tmp"
        open(tmp, "wb").write(audio)
        os.replace(tmp, p)
        return audio, "rendered"
