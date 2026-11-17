"""Pin every cell of the recommendation matrix, and the behaviours that protect the channel."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import Action, Reach, Signals, Strength, MATRIX, decide, strength_of, reach_of


def test_every_cell_is_mapped():
    for st in Strength:
        for re in Reach:
            assert (st, re) in MATRIX, f"unmapped cell {st}x{re}"


def test_only_one_cell_may_interrupt():
    # The whole point: a notification is earned by a strong offer the customer can use NOW.
    pushy = [(s, r) for (s, r), a in MATRIX.items() if a is Action.NOTIFY]
    assert pushy == [(Strength.STRONG, Reach.AT_HAND)], pushy


def test_blocked_is_always_silent():
    for st in Strength:
        assert MATRIX[(st, Reach.BLOCKED)] is Action.STAY_QUIET


def test_small_money_never_pushes():
    for re in Reach:
        assert MATRIX[(Strength.SMALL, re)] not in (Action.NOTIFY, Action.RECOMMEND)


def test_strength_is_relative_not_absolute():
    # AED 50 off 705 is a smaller deal than AED 24 off 159, and the matrix must see that.
    assert strength_of(Signals(saving=50, basket=705)) is Strength.GOOD
    assert strength_of(Signals(saving=23.85, basket=159)) is Strength.STRONG


def test_tiny_savings_are_never_strong():
    assert strength_of(Signals(saving=4.0, basket=10)) is Strength.NONE


def test_reach_needs_a_reachable_basket():
    assert reach_of(Signals(saving=20, basket=60, min_txn=100)) is Reach.STRETCH
    assert reach_of(Signals(saving=20, basket=160, min_txn=100, metres=40)) is Reach.AT_HAND
    assert reach_of(Signals(eligible=False)) is Reach.BLOCKED


def test_urgency_promotes_but_never_rescues_a_blocked_offer():
    v = decide(Signals(saving=30, basket=200, metres=50, days_left=1))
    assert v.action is Action.NOTIFY and v.urgency
    blocked = decide(Signals(saving=30, basket=200, eligible=False, days_left=1))
    assert blocked.action is Action.STAY_QUIET


def test_a_stretch_offer_says_so_rather_than_pretending():
    v = decide(Signals(saving=25, basket=60, min_txn=100, metres=30))
    assert v.action is Action.SUGGEST_BIGGER_BASKET


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ok   {name}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if failed else 'all green'} ({failed} failing)")
    raise SystemExit(1 if failed else 0)
