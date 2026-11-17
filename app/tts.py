"""Server-side TTS proxy — rich 3rd-party British voices; provider keys stay off the client.

Provider order (first available wins): ElevenLabs (if AMP_ELEVEN_KEY) → AWS Polly (if AWS creds
present via key or instance role). Returns (audio_bytes, mime). synth() degrades to (None, None,
reason) when nothing is configured so the client can fall back to the browser voice.

Config (env, all optional):
  AMP_TTS_PROVIDER   force 'elevenlabs' | 'polly' | 'auto' (default auto)
  AMP_ELEVEN_KEY     ElevenLabs API key
  AMP_ELEVEN_VOICE   voice id (default 'George' — a warm British male)
  AMP_ELEVEN_MODEL   default 'eleven_turbo_v2_5' (low latency)
  AMP_POLLY_VOICE    default 'Amy' (British neural female; 'Arthur' male)
  AMP_POLLY_REGION   default from AWS_REGION or 'eu-north-1'
"""
from __future__ import annotations
import os, json, urllib.request, urllib.error

FORCE = os.environ.get("AMP_TTS_PROVIDER", "auto").lower()
ELEVEN_KEY = os.environ.get("AMP_ELEVEN_KEY", "").strip()
ELEVEN_VOICE = os.environ.get("AMP_ELEVEN_VOICE", "JBFqnCBsd6RMkjVDRZzb")   # "George" — British
ELEVEN_MODEL = os.environ.get("AMP_ELEVEN_MODEL", "eleven_turbo_v2_5")
POLLY_VOICE = os.environ.get("AMP_POLLY_VOICE", "Amy")                       # British neural
POLLY_REGION = os.environ.get("AMP_POLLY_REGION") or os.environ.get("AWS_REGION") or "eu-north-1"


def _polly_ready() -> bool:
    try:
        import boto3
        return boto3.session.Session().get_credentials() is not None
    except Exception:
        return False


def provider() -> str | None:
    if FORCE == "elevenlabs":
        return "elevenlabs" if ELEVEN_KEY else None
    if FORCE == "polly":
        return "polly" if _polly_ready() else None
    if ELEVEN_KEY:
        return "elevenlabs"
    if _polly_ready():
        return "polly"
    return None


def status() -> dict:
    p = provider()
    return {"enabled": bool(p), "provider": p,
            "voice": ELEVEN_VOICE if p == "elevenlabs" else (POLLY_VOICE if p == "polly" else None),
            "accent": "British"}


def synth(text: str):
    """Return (audio_bytes, mime, None) or (None, None, reason)."""
    text = (text or "").strip()[:5000]
    if not text:
        return None, None, "empty text"
    p = provider()
    try:
        if p == "elevenlabs":
            return _eleven(text)
        if p == "polly":
            return _polly(text)
        return None, None, "no TTS provider configured (set AMP_ELEVEN_KEY or AWS creds for Polly)"
    except urllib.error.HTTPError as e:
        return None, None, f"{p} error {e.code}: {e.read()[:160].decode('utf-8','replace')}"
    except Exception as e:
        return None, None, f"{p} error: {str(e)[:180]}"


def synth_timed(text: str):
    """ElevenLabs with-timestamps: returns {audio_b64, chars[], starts[]} for exact caption/anim
    sync, or (None, reason). Only ElevenLabs supports this; callers fall back to plain synth()."""
    text = (text or "").strip()[:5000]
    if not text:
        return None, "empty text"
    if provider() != "elevenlabs":
        return None, "timestamps require the elevenlabs provider"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/with-timestamps?output_format=mp3_44100_128"
    body = json.dumps({"text": text, "model_id": ELEVEN_MODEL,
                       "voice_settings": {"stability": 0.45, "similarity_boost": 0.75, "style": 0.15}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": ELEVEN_KEY, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        al = d.get("alignment") or d.get("normalized_alignment") or {}
        return {"audio_b64": d.get("audio_base64"), "mime": "audio/mpeg",
                "chars": al.get("characters", []),
                "starts": al.get("character_start_times_seconds", [])}, None
    except urllib.error.HTTPError as e:
        return None, f"elevenlabs {e.code}: {e.read()[:140].decode('utf-8', 'replace')}"
    except Exception as e:
        return None, str(e)[:160]


def _eleven(text: str):
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
           "?output_format=mp3_44100_128")
    body = json.dumps({"text": text, "model_id": ELEVEN_MODEL,
                       "voice_settings": {"stability": 0.45, "similarity_boost": 0.75, "style": 0.15}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": ELEVEN_KEY, "content-type": "application/json", "accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), "audio/mpeg", None


def _polly(text: str):
    import boto3
    c = boto3.client("polly", region_name=POLLY_REGION)
    r = c.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=POLLY_VOICE, Engine="neural")
    return r["AudioStream"].read(), "audio/mpeg", None
