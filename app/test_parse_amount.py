"""Amount parsing: the magnitude must survive, and anything unreadable must refuse.

Written after a live demo in which "send 1 million dirhams to Alex" moved AED 1. Two
parsers each dropped the word "million": the real one knew no magnitude above "thousand",
and the demo server never called it at all. These tests pin both halves.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.intent import parse_amount, say_amount, payee_phrase, classify


def test_the_regression_that_cost_a_million():
    # The literal sentence from the demo. Before the fix this returned 1.0.
    assert parse_amount("send 1 million dirhams to alex") == 1_000_000


def test_digit_plus_magnitude():
    assert parse_amount("send 5 thousand dirhams to alex") == 5_000
    assert parse_amount("send 1.5 million dirhams") == 1_500_000
    assert parse_amount("send 2 crore rupees") == 20_000_000
    assert parse_amount("send 3 lakh rupees") == 300_000
    assert parse_amount("send 1 billion dirhams") == 1_000_000_000


def test_words_plus_magnitude():
    assert parse_amount("send one million dirhams to alex") == 1_000_000
    assert parse_amount("send twenty thousand dirhams") == 20_000
    assert parse_amount("send five hundred dirhams") == 500
    assert parse_amount("send one lakh rupees") == 100_000
    assert parse_amount("send a million dirhams") == 1_000_000


def test_compound_magnitudes_descend():
    assert parse_amount("send one million five hundred thousand dirhams") == 1_500_000
    assert parse_amount("send 1 million 500 thousand dirhams") == 1_500_000
    assert parse_amount("send one thousand and fifty dirhams") == 1_050


def test_plain_amounts_still_work():
    # The green-tier demo path. Any regression here breaks the working flow.
    assert parse_amount("send 20 dirhams to alex") == 20
    assert parse_amount("send twenty dirhams to alex") == 20
    assert parse_amount("pay alex 50 dirhams") == 50
    assert parse_amount("send 19.99 dollars") == 19.99


def test_refuses_rather_than_guesses():
    assert parse_amount("send twenty twenty dirhams to alex") is None   # a year, not an amount
    assert parse_amount("send 20 dirhams and 30 dirhams") is None        # two numbers
    assert parse_amount("send 20 20 dirhams") is None
    assert parse_amount("send money to alex") is None                   # no amount at all
    assert parse_amount("send million dirhams") is None                  # magnitude, no count
    assert parse_amount("send five thousand two million dirhams") is None  # not a number


def test_a_is_only_a_number_in_front_of_a_magnitude():
    # Otherwise "a" anywhere in a sentence would silently become an amount of 1.
    assert parse_amount("send money to a friend") is None


def test_read_back_says_the_magnitude_out_loud():
    # The spoken safety net: a 1000x error has to be audible before the mandate is signed.
    assert say_amount(1_000_000) == "1 million, that is 1,000,000"
    assert say_amount(1_500_000) == "1.5 million, that is 1,500,000"
    assert say_amount(5_000) == "5 thousand, that is 5,000"
    assert say_amount(20) == "20"
    assert say_amount(None) == "no amount"


def test_magnitude_words_are_not_mistaken_for_the_payee():
    assert payee_phrase("send 1 million dirhams to alex morgan") == "alex morgan"
    assert payee_phrase("send one lakh rupees to alex") == "alex"


def test_a_question_about_money_is_not_an_instruction_to_send_it():
    # Found by measuring the gates, not by guessing: a currency word alone used to promote
    # any sentence to a transfer, so asking whether a payment had gone through proposed
    # making it again.
    assert classify("did my 500 dirhams to alex go through")[0] == "query"
    assert classify("how much did i send alex last week")[0] == "query"
    assert classify("who did i pay 20 dirhams to yesterday")[0] == "query"
    assert classify("what is my balance")[0] == "query"


def test_a_completed_transfer_is_not_a_new_one():
    assert classify("i sent 20 dirhams to alex already")[0] != "transfer"
    assert classify("i paid alex 50 dirhams yesterday")[0] != "transfer"


def test_real_instructions_still_classify_as_transfers():
    for said in ("send 20 dirhams to alex", "send 1 million dirhams to alex",
                 "pay alex 50 dirhams", "transfer 100 aed to alex now",
                 "give alex twenty dirhams"):
        assert classify(said)[0] == "transfer", said



if __name__ == "__main__":
    # pytest is not installed on either box, and an amount parser is not a good place to
    # have untested code waiting for a dependency. Runs standalone; still a pytest file.
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ok   {name}")
            except AssertionError as e:
                failed += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if failed else 'all green'} ({failed} failing)")
    raise SystemExit(1 if failed else 0)
