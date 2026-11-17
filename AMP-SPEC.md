# AMP — Agent Message Protocol (v1, draft)

AMP is a **prompt-native** protocol for autonomous agents to talk to each other and to a central
**Orchestrator** that holds live access to all company code. Instead of each engineer's Claude
working in isolation, their agent speaks AMP to the Orchestrator, which implements + tests changes
with full-repo context and awareness of everyone else's in-flight work (so merges stay clean).

"Prompt-native" means: **every message is human-readable text that an LLM can author and parse
directly.** The wire format is JSON for transport, but each message has a canonical *prompt form*
(below) that agents embed straight into their context.

---

## 1. Identities & trust
- Each agent registers once and receives: `agent_id` (e.g. `sagar.laptop`), a **bearer `token`**
  (auth to the hub), and a **shared `secret`** (HMAC signing of message bodies).
- Every message carries `sig = HMAC_SHA256(secret, canonical(envelope_without_sig))`.
  The hub verifies the bearer token; the *recipient* verifies `sig` — so even a compromised hub
  can't forge an agent's messages. Tokens are unique per agent and revocable.

## 2. Envelope (JSON on the wire)
```json
{
  "amp": "1",
  "id": "msg_<uuid>",
  "thread": "thr_<uuid>",
  "ts": "2026-09-08T12:00:00Z",
  "from": "sagar.laptop",
  "to": "orchestrator",            // agent_id | "orchestrator" | "broadcast"
  "token": "<bearer>",             // stripped by hub before delivery
  "type": "REQUEST",               // REQUEST | RESPONSE | EVENT | ACK | ERROR
  "intent": "implement.feature",   // dotted verb (see §4)
  "body": { "text": "…", "...": "structured fields" },
  "sig": "<hex>"
}
```

## 3. Canonical *prompt form* (what agents read/write)
```
⟦AMP/1 REQUEST implement.feature⟧
id: msg_ab12   thread: thr_9f   from: sagar.laptop → orchestrator   ts: 2026-09-08T12:00:00Z
token: •••   sig: 3af1…
---
Add a /health endpoint to the HBL service returning build sha + upstream reachability.
Repos: core-api. Constraints: no secrets in response. Tests: unit + a curl smoke.
⟦/AMP⟧
```
Parsers key off the `⟦AMP/1 <TYPE> <intent>⟧ … ⟦/AMP⟧` fences. Anything between `---` and the
closing fence is the free-text body; structured hints use `key: value` lines above `---`.

## 4. Core intents (extensible)
| intent | from → to | meaning |
|---|---|---|
| `chat` | any → any | free-form message (used by the test app) |
| `query.knowledge` | agent → orchestrator | ask about people/projects/code/history |
| `plan.feature` | agent → orchestrator | produce an implementation plan (no writes) |
| `implement.feature` | agent → orchestrator | implement + test in an isolated worktree (Phase 2) |
| `review.diff` | agent → orchestrator | review a diff/branch |
| `claim.files` | agent → orchestrator | reserve files/modules you're about to touch |
| `release.claim` | agent → orchestrator | free a claim |
| `collision.check` | agent → orchestrator | who else is touching these paths? |
| `presence` | agent → hub (EVENT) | heartbeat / online status |

Responses reuse the thread and set `type: RESPONSE` (or `ERROR`), `intent` mirrors the request.

## 5. Merge-safety (the "easy merge" promise)
The Orchestrator keeps a **claims ledger**: `(agent, repo, paths[], intent, status, ts)`.
- Before any `implement.feature`/`plan.feature`, it runs `collision.check` against active claims
  and recent commits, and returns overlaps ("Alex is editing `TransactionController.java`").
- Implementations happen on a **per-request git worktree/branch**, so parallel work never shares a
  tree. The ledger + branch-per-task is what keeps merges trivial.

## 6. Transport
- `POST /amp/register` → `{agent_id, token, secret}`
- `POST /amp/send` (envelope) → hub validates token, stores, routes to recipient mailbox(es)
- `GET  /amp/poll?agent_id=&token=&wait=` → pending messages (long-poll)
- `GET  /amp/directory` → agents, capabilities, presence
- `GET  /amp/thread/{thread}` → full thread
- `WS   /amp/ws` (later) → realtime push

## 7. Phasing
1. **Phase 1 (this):** hub + registry + signed messaging + threads + Orchestrator that does
   `query.knowledge`, `plan.feature`, `claim.files`, `collision.check`; 2-device web test client.
2. **Phase 2:** `implement.feature` — Orchestrator applies changes in a worktree, runs tests,
   returns a diff + test log; PR creation.
3. **Phase 3:** move to AWS (hub on ECS/API-GW, mailboxes on SQS/Dynamo, secrets in Vault/KMS),
   native iOS test app via TestFlight.

---

## 8. Identity & the Agent Card
Every agent has a stable **`claude_id`** and a Markdown **Agent Card** describing who it is, its
capabilities, and its **dialect (vocab)** — the domain phrases it speaks. Cards are exchanged
during the handshake (as md→bytes), so agents *learn who they're talking to* before talking.

## 9. The 3-way handshake (SYN / SYN-ACK / ACK)
Modeled on TCP, but it negotiates **understanding**, not just a connection:
```
A ──HELLO (SYN, cardA, vocabA, isnA)──────────────▶ B
A ◀─HELLO-ACK (SYN|ACK, cardB, CODEBOOK, isnB, ack=isnA+1)── B
A ──ACK (ack=isnB+1)──────────────────────────────▶ B      ⇒ ESTABLISHED
```
- `isn` = initial sequence number (random, like TCP). Each side +1's the other's to prove liveness.
- **session id** = `sha256(claude_A | claude_B | nonceA | nonceB)[:8]` — a shared, unguessable id.
- **CODEBOOK** = compact codes for the *union* of both agents' vocab (`provider_bank_coverage → §0`).
  Both sides hold the same codebook after ACK, so every later message is smaller and mutually legible.

## 10. md-wire — Markdown is cognition, bytes are transport
Messages are authored as **Markdown** (what an LLM natively produces/understands), then serialized:
```
markdown ──codebook-substitute──▶ §-coded md ──zlib──▶ payload
segment = b"AMP1" | ver | flags | type | sid(8) | seq(4) | ack(4) | plen(4) | payload   (27-byte header)
```
`flags`: SYN=1 ACK=2 FIN=4 COMP=8 CODEBOOK=16. Typically **~45–60% smaller than a JSON envelope**,
and fewer tokens when an agent reads a transcript of segments in-context. Receipt reverses it back
to the exact Markdown. (See `app/mdwire.py`, `app/handshake.py`; try `GET /amp/demo/handshake`.)

## 11. The Orchestrator as language broker
When an agent receives a term outside the session codebook (a foreign dialect), it asks the
Orchestrator (`POST /amp/translate`) to rewrite the message into its own terms — being an LLM, the
broker *understands* both dialects. This is how any two agents, from any company, interoperate even
without a pre-shared vocabulary: **negotiate what you can, broker the rest.**

*Security note:* AMP is for authorized internal automation. The hub never logs `token`/`secret`;
bodies may contain code but never plaintext credentials.
