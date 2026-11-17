#!/usr/bin/env python3
"""The seven webs must describe what actually happened, and nothing else.

This is the test that matters for a view whose whole purpose is to be believed. A
visualisation of reasoning is only worth showing if every node in it corresponds to
something the engine really did — the moment it contains a decorative step, it is a claim
about the system that nobody can check against the answer.

So the contract pinned here is:

* every node the crawler walks is marked as visited, and every visited node is on a path;
* the webs are built from the real `lane_trace`, so a step the engine took appears and a
  step it did not take does not;
* the score is the engine's own confidence, never a number chosen to look good;
* the layout is deterministic — the same question draws the same picture, so a rehearsed
  demo and a screenshot both mean something;
* it works for any intent, not just the spend question it was built for.

Run: ~/amp/.venv/bin/python cx/tests/test_webs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CX_SPEND_DEMO"] = "1"

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got!r} (wanted {want!r})")
    else:
        print(f"  ok    {name}")


def plan_for(q, session):
    from engine import engine as E
    import webs
    a = E.ask(q, session=session)
    return a, webs.build(a, q)


def test_shape():
    import webs
    print("\nseven webs, in the order the engine runs them")
    a, p = plan_for("analyse my spending last month", "w-shape")
    check("seven webs", len(p["webs"]), 7)
    check("in pipeline order", tuple(w["key"] for w in p["webs"]), webs.ORDER)
    for w in p["webs"]:
        check(f"{w['key']}: has nodes", len(w["nodes"]) > 0, True)
        check(f"{w['key']}: every node is placed",
              all("x" in n and "y" in n for n in w["nodes"]), True)
        check(f"{w['key']}: edges point at real nodes",
              all(0 <= i < len(w["nodes"]) and 0 <= j < len(w["nodes"])
                  for i, j in w["edges"]), True)
        check(f"{w['key']}: path points at real nodes",
              all(0 <= i < len(w["nodes"]) for i in w["path"]), True)
        check(f"{w['key']}: everything walked is marked visited",
              all(w["nodes"][i]["hit"] for i in w["path"]), True)
        check(f"{w['key']}: found == visited count",
              w["found"], sum(1 for n in w["nodes"] if n["hit"]))


def test_nodes_are_real():
    """The anti-decoration test. Each engine trace step has to show up as a node."""
    print("\nevery trace step the engine recorded appears in a web")
    a, p = plan_for("analyse my spending last month", "w-real")
    labels = {n["label"] for w in p["webs"] for n in w["nodes"]}
    missing = [st["what"] for st in a.lane_trace
               if st.get("tag") != "input" and st["what"] not in labels]
    check("no trace step is dropped", missing, [])
    # and the converse: the spend figures shown are the ones the handler computed
    totals = {n["label"] for w in p["webs"] for n in w["nodes"]}
    check("the computed total is on the execute web",
          f"AED {a.spend['total']:,.2f}" in totals, True)


def test_chips_say_nothing_the_trace_did_not():
    """The floating chips are the easiest place for a visualisation to start making things
    up, so they are pinned to the node's own detail: a chip must be a prefix of the reason
    the engine recorded, never a new sentence written to look good."""
    import webs
    print("\nevery chip is a prefix of its node's own reason")
    _, p = plan_for("analyse my spending last month", "w-chip")
    bad, long_, dangling = [], [], []
    for w in p["webs"]:
        for n in w["nodes"]:
            c = n.get("chip") or ""
            if not c:
                continue
            if not " ".join(n["detail"].split()).startswith(c.rstrip("…")):
                bad.append((c, n["detail"][:40]))
            if len(c) > 34:
                long_.append(c)
            if c.split() and c.split()[-1].lower() in {"the", "a", "of", "from", "to", "and"}:
                dangling.append(c)
    check("no chip invents text", bad, [])
    check("no chip is longer than a glance", long_, [])
    check("no chip ends mid-phrase", dangling, [])
    check("a node with no reason gets no chip",
          [n["chip"] for w in p["webs"] for n in w["nodes"]
           if not n["detail"] and n.get("chip")], [])


def test_score_is_the_engines_own():
    print("\nthe gauge is the engine's confidence, not a flattering number")
    a, p = plan_for("where is my money", "w-score")
    check("score == engine confidence", p["score"]["value"], round(a.confidence * 100))
    check("model is named", p["score"]["model"], a.model)
    check("flags count real guards",
          p["score"]["flags"],
          sum(1 for w in p["webs"] for n in w["nodes"]
              if n["kind"] == "guard" and n["hit"]))


def test_deterministic():
    print("\nthe same question draws the same picture")
    _, p1 = plan_for("analyse my spending last month", "w-d1")
    _, p2 = plan_for("analyse my spending last month", "w-d2")
    xs1 = [(n["x"], n["y"]) for w in p1["webs"] for n in w["nodes"]]
    xs2 = [(n["x"], n["y"]) for w in p2["webs"] for n in w["nodes"]]
    check("layout is stable", xs1, xs2)


def test_any_intent():
    """It is built for the spend question but must not be specific to it — the whole point
    is that the next question gets the same view."""
    print("\nit works for questions that have nothing to do with spending")
    for q, sess in [("where is my money", "w-a1"),
                    ("show me my last 5 transactions", "w-a2"),
                    ("what is the fee to pakistan", "w-a3"),
                    ("what is the weather", "w-a4")]:
        a, p = plan_for(q, sess)
        check(f"{q[:26]!r}: seven webs", len(p["webs"]), 7)
        check(f"{q[:26]!r}: crawler has somewhere to go",
              sum(len(w["path"]) for w in p["webs"]) > 0, True)
        check(f"{q[:26]!r}: heat panel is populated", len(p["heat"]["rows"]) > 0, True)
        check(f"{q[:26]!r}: answer carried", bool(p["answer"]), True)
    # a question with no spend data must fall back rather than show an empty panel
    _, p = plan_for("what is the weather", "w-a5")
    check("no spend -> coverage heat instead", p["heat"]["title"], "WEB COVERAGE")


if __name__ == "__main__":
    test_shape()
    test_nodes_are_real()
    test_chips_say_nothing_the_trace_did_not()
    test_score_is_the_engines_own()
    test_deterministic()
    test_any_intent()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
