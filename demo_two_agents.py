"""Prove AMP: two agents register, message each other (signed), and one asks the Orchestrator.

Run the hub first (./run.sh), then: ./.venv/bin/python demo_two_agents.py
"""
import sys, time, json, urllib.request

HUB = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from app import protocol


def post(path, obj):
    req = urllib.request.Request(HUB + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(HUB + path, timeout=60))


def register(name):
    return post("/amp/register", {"name": name, "capabilities": ["chat"]})


def send(sender, to, intent, text):
    env = protocol.make(sender["agent_id"], to, intent, {"text": text}, sender["secret"], token=sender["token"])
    return post("/amp/send", env)


def poll(agent, wait=20):
    return get(f"/amp/poll?agent_id={agent['agent_id']}&token={agent['token']}&wait={wait}")["messages"]


if __name__ == "__main__":
    alice = register("Alice Phone")
    bob = register("Bob Phone")
    print("registered:", alice["agent_id"], "&", bob["agent_id"])

    # 1) agent -> agent, signed
    send(alice, bob["agent_id"], "chat", "Hey Bob — AMP handshake test 👋")
    got = poll(bob)
    print("\n[Bob inbox]", [(m["from"], m["intent"], m["body"]["text"]) for m in got])
    assert any("handshake" in m["body"]["text"] for m in got), "agent->agent failed"

    # 2) agent -> orchestrator (LLM plan). Give it a beat to run.
    r = send(alice, "orchestrator", "plan.feature",
             "Plan a /health endpoint on the HBL service returning build sha + upstream reachability. "
             "List exact files to change and check merge safety.")
    print("\n[sent to orchestrator, thread]", r["thread"])
    for _ in range(6):
        msgs = poll(alice, wait=20)
        done = [m for m in msgs if m["from"] == "orchestrator"]
        if done:
            print("\n[ORCHESTRATOR REPLY]\n")
            print(done[0]["body"]["text"][:1600])
            break
    else:
        print("(no orchestrator reply yet)")
    print("\n✅ AMP demo complete.")
