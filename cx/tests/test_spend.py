#!/usr/bin/env python3
"""The spend breakdown: what it must categorise, and what it must never swallow.

Three classes of bug are pinned here, each of which this codebase has shipped before in
another form:

**1. A payment instruction answered with a report.** "pay MClothes 50" contains the word
"pay" and a merchant; a generous spend detector turns it into a pie chart and the customer
never makes their payment. The detector is deliberately narrow, and the negatives below are
the contract.

**2. Substring matching where word matching was meant.** "du" is a UAE telecom and lives
inside "dukan". Every keyword in `spend.py` is matched on a word boundary for exactly this
reason, and `dukan` is pinned as uncategorised.

**3. English-only code behind a multilingual model.** The first version of the detector had
only Latin patterns, so "मेरा खर्चा कितना है" fell through to fees_query. Python's `\b` never
fires between Devanagari characters, so those scripts are matched by containment instead —
and the answer has to come back with its period and category names translated too, or it
reads "the last 30 days me aapne ... kharch kiye".

Run: ~/amp/.venv/bin/python cx/tests/test_spend.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CX_SPEND_DEMO"] = "1"      # a known corpus, so the numbers are assertable

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(name)
        print(f"  FAIL  {name}  -> {got!r} (wanted {want!r})")
    else:
        print(f"  ok    {name}")


def test_categorisation():
    import spend
    print("\nmerchants land in the right category, or honestly in none")
    for name, want in [
        ("Carrefour", "food"), ("Talabat", "food"),
        ("Sharaf DG", "electronics"), ("Aster Clinic", "healthcare"),
        ("VOX Cinemas", "entertainment"), ("du", "services"),
        ("Etisalat LandLine", "services"),
        # keyword rules, not in the explicit table
        ("Al Noor Pharmacy", "healthcare"), ("City Supermarket", "food"),
        # the substring trap: "du" must not be found inside these
        ("dukan", "other"), ("Dubai Trading", "other"),
        # dev junk is uncategorised, never guessed into a bucket
        ("KVVIZM WRGGZ", "other"), ("aaa", "other"), ("", "other"),
    ]:
        check(f"category({name!r})", spend.categorise(name)[0], want)


def test_period_and_focus():
    import spend
    print("\nperiod and category are read from the sentence, in every script")
    for said, want in [
        ("my spending last week", "week"), ("spends for the last 7 days", "week"),
        ("analyse my spending", "month"), ("budget for 90 days", "quarter"),
        ("pichle hafte ka kharcha", "week"),
        ("पिछले हफ़्ते खाने पर कितना खर्च", "week"),
        ("مصروفي آخر أسبوع", "week"),
    ]:
        check(f"period({said[:28]!r})", spend.period_from(said), want)
    for said, want in [
        ("how much did I spend on food", "food"),
        ("my healthcare spends", "healthcare"),
        ("khane par kitna kharch kiya", "food"),
        ("दवाई पर कितना खर्च", "healthcare"),
        ("analyse my spending", None),
    ]:
        check(f"focus({said[:28]!r})", spend.focus_from(said), want)


def test_detector_negatives():
    """The important half. A spend question is a report; these are instructions."""
    from engine.engine import _is_spend_question
    print("\nwhat is and is not a spend question")
    for said, want in [
        ("analyse my spending last month", True),
        ("where did my money go", True),
        ("how much did I spend on food", True),
        ("mera kharcha kitna hai is mahine", True),
        ("मेरा खर्चा कितना है", True),
        ("أين ذهبت أموالي", True),
        # instructions — must never be answered with a breakdown
        ("send 100 to alex", False),
        ("pay MClothes 50", False),
        ("send money to my sister", False),
        ("alex ko 50 bhej do", False),
        # other questions that merely mention money
        ("where is my money", False),
        ("show me my last 5 transactions", False),
        ("what is my balance", False),
    ]:
        check(f"is_spend({said[:30]!r})", _is_spend_question(said), want)


def test_the_report():
    import spend
    print("\nthe report adds up and compares against the period before")
    r = spend.analyse("month")
    check("has categories", len(r["categories"]) >= 4, True)
    check("totals match the categories",
          round(sum(c["total"] for c in r["categories"]), 2), r["total"])
    check("previous period was read", r["previous_total"] > 0, True)
    check("demo corpus is declared as not live", r["live"], False)
    # a named category wins over the biggest mover
    check("focus wins", spend.analyse("month", focus="healthcare")["mover"], "healthcare")
    # uncategorised is reported rather than hidden
    check("uncategorised is a number, not a silent bucket",
          isinstance(r["uncategorised"], float), True)


def test_a_warm_cache_cannot_shrink_the_window():
    """The breakdown must not get smaller because something else asked for fewer rows first.

    `history.recent` caches, and the cache used to be served to ANY later caller regardless of
    how many rows it actually held. The chat path asks for 8, which fills the cache with 30;
    the spend breakdown then asks for 600, was handed those 30, and reported a total over a
    fraction of the period — silently, with no error and a plausible-looking number. Wrong
    figures about a customer's own money are the one failure this code cannot have.
    """
    import history, spend
    print("\nthe breakdown is the same whether or not a small read warmed the cache")
    os.environ.pop("CX_SPEND_DEMO", None)          # this one needs the real cache path
    try:
        history._CACHES.clear()   # per-customer caches since logins
        cold = spend.analyse("quarter")
        if not cold["live"]:
            print("  skip  (no ledger available here)")
            return
        history._CACHES.clear()   # per-customer caches since logins
        history.recent(limit=8)                    # the chat path warms it small
        warm = spend.analyse("quarter")
        check("total survives a warm cache", warm["total"], cold["total"])
        check("payment count survives", warm["payments"], cold["payments"])
    finally:
        os.environ["CX_SPEND_DEMO"] = "1"
        history._CACHES.clear()   # per-customer caches since logins


def test_answers_are_in_the_customers_language():
    from engine import engine as E
    import spend
    print("\nthe whole sentence is translated, not half of it")
    cases = [("analyse my spending", "en"), ("mera kharcha kitna hai", "hinglish"),
             ("मेरा खर्चा कितना है", "hi"), ("أين ذهبت أموالي", "ar")]
    for said, lang in cases:
        a = E.ask(said, session=f"t-spend-{lang}")
        check(f"{lang}: intent", a.intent, "spend_analysis")
        check(f"{lang}: answered in {lang}", a.lang, lang)
        if lang != "en":
            # the period label must not still be English inside a translated sentence
            check(f"{lang}: period label translated",
                  spend.period_label("month", "en") in (a.say or ""), False)


if __name__ == "__main__":
    test_categorisation()
    test_period_and_focus()
    test_detector_negatives()
    test_the_report()
    test_a_warm_cache_cannot_shrink_the_window()
    test_answers_are_in_the_customers_language()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("all good")
