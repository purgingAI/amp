"""The Studio's language-model helper. Two paths:

  internal work (drafting examples, exam questions, red-team attacks, Autopilot, Genesis, fact-checks)
      → a Claude Code session on this server (`claude -p`, no tools, no plugins, nothing persisted), authenticated
        by data/claude_token or the user's own login; falls back to the API if no session is available.
  the customer path (fast=True: follow-up rewrites, query translation) and streamed answers
      → the API only, behind a circuit breaker: after a credit/auth error it stops calling for a while so a
        customer never waits on a request that cannot succeed. Every customer-path caller has its own fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

ANSWER_MODEL = os.environ.get("STUDIO_ANSWER_MODEL", "claude-sonnet-5-5")
QUESTION_MODEL = os.environ.get("STUDIO_QUESTION_MODEL", "claude-haiku-5-5")


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    for p in filter(None, [os.environ.get("TEN_ENV_FILE"), os.path.expanduser("~/amp/.env.ten"), os.path.expanduser("~/creator-os/.env")]):
        try:
            for line in open(p, encoding="utf-8"):
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            continue
    return ""


_down = {"until": 0.0, "why": ""}
_cli_slots = threading.Semaphore(int(os.environ.get("STUDIO_CLI_CONCURRENCY", "2")))
CLI = os.environ.get("STUDIO_CLAUDE_CLI") or shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "claude_token")


def _api_open() -> None:
    if time.time() < _down["until"]:
        raise RuntimeError(f"model API paused ({_down['why']})")


def _api_failed(exc: Exception) -> None:
    """Credit or auth errors cannot fix themselves in seconds: pause the API so customers don't wait on it."""
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 429):
        try:
            msg = exc.read().decode("utf-8", "replace")
        except Exception:                                    # noqa: BLE001
            msg = ""
        if exc.code == 429:
            _down.update(until=time.time() + 60, why="rate limited")
        elif exc.code in (401, 403) or "credit balance" in msg:
            _down.update(until=time.time() + 600, why="credit balance too low" if "credit" in msg else "not authorised")


def cli_status() -> dict:
    tok = os.path.exists(TOKEN_FILE) and os.path.getsize(TOKEN_FILE) > 20
    login = os.path.exists(os.path.expanduser("~/.claude/.credentials.json"))
    return {"installed": bool(CLI and os.path.exists(CLI)), "auth": "token" if tok else "login" if login else None, "api_paused": _down["why"] if time.time() < _down["until"] else None}


def _cli(system: str, prompt: str, model: str, timeout: int = 180) -> str:
    env = dict(os.environ)
    if os.path.exists(TOKEN_FILE):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = open(TOKEN_FILE, encoding="utf-8").read().strip()
    env.pop("ANTHROPIC_API_KEY", None)                       # the session's own auth, never the API key
    args = [CLI, "-p", "--output-format", "json", "--model", model, "--system-prompt", system, "--tools", "",
            "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"]
    with _cli_slots:
        p = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=timeout, env=env, cwd="/tmp")
    try:
        d = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError(f"claude session: {(p.stderr or p.stdout)[:200]}")
    if d.get("is_error"):
        raise RuntimeError(f"claude session: {str(d.get('result'))[:200]}")
    return str(d.get("result") or "")


def _req(body: dict) -> urllib.request.Request:
    return urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), headers={
        "x-api-key": api_key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})


def stream(system: str, messages: list[dict], max_tokens: int = 1000, model: str = ANSWER_MODEL):
    """Yields text pieces as they are written (API only — the customer path)."""
    _api_open()
    try:
        r = urllib.request.urlopen(_req({"model": model, "max_tokens": max_tokens, "system": system,
                                         "messages": messages, "stream": True}), timeout=90)
    except Exception as exc:                                # noqa: BLE001
        _api_failed(exc)
        raise
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if ev.get("type") == "content_block_delta" and (ev.get("delta") or {}).get("type") == "text_delta":
                yield ev["delta"]["text"]
            elif ev.get("type") == "error":
                raise RuntimeError((ev.get("error") or {}).get("message", "stream error"))
            elif ev.get("type") == "message_stop":
                return


def complete(system: str, prompt: str, max_tokens: int = 1500, model: str = QUESTION_MODEL, fast: bool = False) -> str:
    """fast=True: the customer path — API only, never the (slower) session. Otherwise the session first."""
    if not fast and os.environ.get("STUDIO_INTERNAL_LLM", "session") == "session":
        st = cli_status()
        if st["installed"] and st["auth"]:
            try:
                return _cli(system, prompt, model)
            except Exception:                               # noqa: BLE001 — fall through to the API
                if not api_key():
                    raise
    _api_open()
    try:
        with urllib.request.urlopen(_req({"model": model, "max_tokens": max_tokens, "system": system,
                                          "messages": [{"role": "user", "content": prompt}]}), timeout=120) as r:
            data = json.loads(r.read())
    except Exception as exc:                                # noqa: BLE001
        _api_failed(exc)
        raise
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def error_text(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            return f"HTTP {exc.code}: " + json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:                                    # noqa: BLE001
            return f"HTTP {exc.code}"
    return f"{type(exc).__name__}: {exc}"[:200]
