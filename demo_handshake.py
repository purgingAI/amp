"""End-to-end AMP handshake OVER THE HUB relay, then a compressed md message exchange.

Two agents (Sagar's + Alex's) never talk directly — every handshake segment travels as an AMP
message through the hub. Proves: SYN/SYN-ACK/ACK across the wire, a shared session + codebook,
and a Markdown message delivered as compressed bytes and reconstructed losslessly.

Run the hub (./run.sh), then: ./.venv/bin/python demo_handshake.py [hub_url]
"""
import sys, json, base64, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from app import handshake as H, mdwire as W

HUB = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"


def post(p, o):
    return json.load(urllib.request.urlopen(urllib.request.Request(
        HUB + p, data=json.dumps(o).encode(), headers={"Content-Type": "application/json"}), timeout=40))


def get(p):
    return json.load(urllib.request.urlopen(HUB + p, timeout=40))


def reg(name):
    return post("/amp/register", {"name": name})


def send(me, to, intent, seg: bytes):
    post("/amp/send-lite", {"from": me["agent_id"], "token": me["token"], "to": to,
                            "intent": intent, "body": {"seg": base64.b64encode(seg).decode()}})


def wait(me, intent):
    for _ in range(20):
        for m in get(f"/amp/poll?agent_id={me['agent_id']}&token={me['token']}&wait=10")["messages"]:
            if m["intent"] == intent:
                return base64.b64decode(m["body"]["seg"])
        time.sleep(0.3)
    raise TimeoutError(intent)


if __name__ == "__main__":
    a = reg("Sagar Laptop"); b = reg("Alex Laptop")
    A = H.AgentCard(a["agent_id"], "claude-" + a["agent_id"], "backend",
                    ["plan.feature"], ["cross-border payout", "provider_bank_coverage",
                                       "FetchAccountTitle", "merge collision", "campaign reserve"])
    B = H.AgentCard(b["agent_id"], "claude-" + b["agent_id"], "backend",
                    ["plan.feature"], ["cross-border payout", "settlement leg",
                                       "provider_bank_coverage", "merge collision"])

    # --- 3-way handshake, relayed by the hub ---
    syn, stA = H.hello(A);                      send(a, b["agent_id"], "hs.hello", syn)
    synack, sessB = H.on_hello(wait(b, "hs.hello"), B); send(b, a["agent_id"], "hs.hello-ack", synack)
    ack, sessA = H.on_hello_ack(wait(a, "hs.hello-ack"), stA); send(a, b["agent_id"], "hs.ack", ack)
    sessB = H.on_ack(wait(b, "hs.ack"), sessB)
    print("🤝 handshake done | sid match:", sessA["sid"] == sessB["sid"],
          "| established:", sessA["established"] and sessB["established"])
    print("   codebook:", dict(list(sessA["codebook"].items())[:5]))

    # --- a real markdown message as compressed bytes over the session ---
    msg = ("# Heads up\nStarting the **cross-border payout** fix — touching "
           "`provider_bank_coverage` + FetchAccountTitle. Watch for a **merge collision** on the "
           "campaign reserve path; I'll claim those files.")
    payload, flags = W.pack(msg, sessA["codebook"])
    seg = W.segment(payload, flags, W.T_DATA, sid=bytes.fromhex(sessA["sid"]), seq=1)
    send(a, b["agent_id"], "data", seg)
    got = wait(b, "data")
    p = W.parse(got); recovered = W.unpack(p["payload"], p["flags"], sessB["codebook"])
    print("\n📦 message over the wire:", W.benchmark(msg, sessA["codebook"]))
    print("   lossless on Alex's side:", recovered == msg)
    print("\n✅ AMP handshake + md-wire demo complete.")
