# AMP — Agent Message Protocol

A **prompt-native protocol** that lets autonomous agents talk to each other and to a central
Orchestrator. Every message is human-readable text an LLM can write and parse directly, carried in a
JSON envelope that is **HMAC-signed per agent**, so even a compromised hub can't forge messages.

**What's in this repo**
| Path | What it is |
|---|---|
| `AMP-SPEC.md` | The protocol spec: identities, envelope, message types, threading |
| `app/` | Hub + Orchestrator (FastAPI): agent registry, routing, code-graph-aware answers |
| `web/` | Live mesh observer, client and monitor UIs |
| `payi/`, `openwallet/` | Demos: **voice-to-payment** (speak → intent → exact payee → signed mandate → device signature → ledger) |
| `cx/` | A customer-support assistant engine: own intent model, guardrails, phrase bank, eval suites |
| `stt/` | Local speech-to-text (whisper.cpp) |
| `demo_two_agents.py`, `demo_handshake.py` | Smallest runnable demos of two agents talking over AMP |

```bash
python -m venv .venv && .venv/bin/pip install fastapi uvicorn anthropic httpx
export ANTHROPIC_API_KEY=...
./run.sh                 # hub + UI
./run-demo.sh            # two agents negotiating over AMP
```

All company-specific names have been replaced with a fictional wallet ("OpenWallet"), and integrations
read hosts and credentials from environment variables.
