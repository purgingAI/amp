"""The closed world of things a OpenWallet customer may ask about.

This file is the contract. The model, the decision matrix, the UI lanes and the scope guard
all read their label space from here, so there is exactly one place where "what we will
answer" is defined — and adding an intent is a deliberate edit to this file, never an
emergent behaviour of a model.

Two rules are structural rather than advisory:

1. **The set is CLOSED.** Anything that is not one of these intents is OUT_OF_SCOPE and gets
   a refusal. A customer assistant that will have a go at any question is a liability; one
   that answers fourteen things well and declines the rest is a product.

2. **Nothing here can reach the codebase.** The knowledge graph in ~/ace-go knows commit
   authors, file ownership and internal module names. A customer must never see any of it.
   JOURNEY_HELP answers from `journeys.py`, which is hand-written customer-facing content —
   there is no retrieval path from the customer surface to the code graph at all. That is
   enforced by construction (the engine never imports it) and pinned by a test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """What the customer wants. One of these, or we decline."""

    # --- money that has already moved (or should have) ---
    TXN_STATUS = "txn_status"                    # "where is my money", "did it go through"
    TXN_FAILED = "txn_failed"                    # "my transfer failed"
    DEBITED_NOT_CREDITED = "debited_not_credited"  # taken from me, never arrived
    TXN_ON_HOLD = "txn_on_hold"                  # stopped by compliance / TMS
    #: "show me my last 5 transactions" — MANY payments, not one. Without this the question
    #: had nowhere to go: the model had to pick between TXN_STATUS (which answers about a
    #: single payment and so silently picked one) and BENEFICIARY_QUERY (which answered a
    #: different question entirely). A missing class is not something training can fix.
    TXN_HISTORY = "txn_history"
    #: "where did my money go last month", "analyse my spending". A question about PATTERN
    #: rather than about any one payment — the answer is a breakdown by category and a
    #: comparison against the period before, never a payment status.
    SPEND_ANALYSIS = "spend_analysis"
    #: "that can't be right", "how can it be the same", "you said 4,549 before". The
    #: customer is contradicting something WE said. It is not a new question and it is not
    #: out of scope — it is a claim of ours being challenged, and the only honest reply is
    #: to work the figure out again and then either correct it or show why it holds.
    DISPUTED = "disputed"
    #: "who did I pay the most", "what's my biggest payment", "how many payments did I
    #: make", "how much did I send Alex". A question whose answer is a NAME or a single
    #: NUMBER. Distinct from txn_history, which answers with a list — and a list is not an
    #: answer to "who". Distinct from spend_analysis, which ranks categories, not people.
    SPEND_RANK = "spend_rank"

    # --- getting it back ---
    REFUND_REQUEST = "refund_request"
    REFUND_STATUS = "refund_status"

    # --- what it costs and how long it takes ---
    FEES_QUERY = "fees_query"
    TIME_QUERY = "time_query"
    LIMITS_QUERY = "limits_query"

    # --- moving money now ---
    SEND_MONEY = "send_money"
    PAY_MERCHANT = "pay_merchant"
    BENEFICIARY_QUERY = "beneficiary_query"

    # --- splitting a bill with people ---
    #: "split this with Alex", "split the bill three ways". Distinct from SEND_MONEY: the
    #: customer is not paying a person, they are dividing something they paid (or are about
    #: to pay) a MERCHANT. The money flow is the opposite direction.
    SPLIT_PAYMENT = "split_payment"
    #: "who owes me", "what do I owe Alex", "did he pay his share yet". A question about
    #: balances and participant states, never an instruction to move money.
    SPLIT_STATUS = "split_status"

    # --- bills ---
    #: "how many bills are due", "what do I owe this month". A QUESTION about balances
    #: across every saved biller — distinct from TXN_HISTORY, which is about payments
    #: already made, and from LIMITS_QUERY, which is about caps.
    BILL_DUE = "bill_due"
    #: "pay my electricity bill", "settle the DEWA one". Moves money.
    BILL_PAY = "bill_pay"
    #: "pay it on Friday", "schedule the bill for the 1st". Moves money LATER, which is a
    #: different promise and needs a different confirmation.
    BILL_SCHEDULE = "bill_schedule"

    #: "alex ne paise bhej diye hai", "he paid me", "paisa aa gaya" — the customer
    #: TELLING us somebody has settled up, not asking us to send anything.
    #:
    #: It exists because the sentence is a trap for a classifier: it contains a person and
    #: a transfer verb, so "alex ne paise bhej diye hai" scored as SEND_MONEY and the
    #: engine replied "alex ko kitne bhejne hai?" — offering to send money to the person
    #: who had just paid. The giveaway is the direction: ne/sent/paid is reporting a
    #: completed inbound payment; ko/to/bhejna is requesting an outbound one.
    #:
    #: It never moves money. It checks what actually arrived and says so.
    PAYMENT_CLAIM = "payment_claim"

    # --- the product itself ---
    #: "I need help", "hi", "can you help me" — a real request, not noise. Answering it with
    #: the out-of-scope wall tells a customer who came for help that we cannot help them.
    GREETING_HELP = "greeting_help"
    OFFERS_NEARBY = "offers_nearby"
    JOURNEY_HELP = "journey_help"                # "how do I add a beneficiary"

    # --- from the customer-app BRD (v.34), 2026-10-14 ---
    #: "recharge my du with 50", "buy data for my phone". §2.3. v9 read these as
    #: refund_request at 95% — "re-charge" and "re-fund" are one prefix apart.
    MOBILE_RECHARGE = "mobile_recharge"
    #: "remind me to pay DEWA on the 18th". §2.6. A reminder NOTIFIES; it never pays. That
    #: is the whole difference from BILL_SCHEDULE, and the BRD is explicit about it.
    REMINDER_SET = "reminder_set"
    #: "show my reminders", "delete the DEWA reminder".
    REMINDER_LIST = "reminder_list"
    #: "what's my balance". Read from the customer's own ledger, never from text.
    BALANCE_QUERY = "balance_query"
    #: "try again" after a failure. §2.1 T11-12, §2.2 P8-9, §2.7. The engine decides whether a
    #: retry is SAFE: a failed payment may be retried with a new key; a payment whose status
    #: is still unknown must not be, or the customer is charged twice.
    RETRY_PAYMENT = "retry_payment"
    #: "no, cancel", "don't send it". §2.1 T10. Ends whatever is being prepared; nothing moves.
    CANCEL_ACTION = "cancel_action"
    #: "connect me to a human". §4.1: handoff on request, with the conversation's context.
    HUMAN_HANDOFF = "human_handoff"
    #: "I sent money to the wrong person". FAQ 14 / 65: a recall request plus support.
    WRONG_TRANSFER = "wrong_transfer"
    #: "I don't recognize this transaction". FAQ 66: straight to the fraud team.
    REPORT_FRAUD = "report_fraud"
    #: Product KNOWLEDGE: "how does split payment work", "can I send money abroad", "which
    #: bills can I pay". §2.8's 47 "Answer" rows. This is the RETRIEVAL lane — answered from
    #: approved documents, never from customer data and never by acting. "Pay my DEWA bill"
    #: is an action; "how does paying DEWA work" is this; "how do I pay DEWA" (steps) is
    #: JOURNEY_HELP. Both of the last two are retrieval.
    KNOWLEDGE_QUERY = "knowledge_query"
    #: Home-screen products the assistant navigates to (BRD objective 2). These open the
    #: app's own screen; the assistant does not move money for them.
    ADD_MONEY = "add_money"
    WITHDRAW_MONEY = "withdraw_money"
    LINK_BANK = "link_bank"

    # --- the wall ---
    OUT_OF_SCOPE = "out_of_scope"


#: Intents that move money. These never execute on the model's say-so — they produce a
#: proposal that goes through the existing mandate + TPM + read-back path, exactly like the
#: voice flow does. The model chooses the lane; the guardrail decides.
MONEY_MOVING = frozenset({Intent.SEND_MONEY, Intent.PAY_MERCHANT, Intent.SPLIT_PAYMENT,
                          Intent.BILL_PAY, Intent.BILL_SCHEDULE})

#: Everything about this customer's BILLS. Sticky for the same reason TXN_DOMAIN is: once
#: somebody is working through what they owe, "and the internet one" or "pay that one too"
#: are follow-ups within the subject, not new requests.
BILL_DOMAIN = frozenset({Intent.BILL_DUE, Intent.BILL_PAY, Intent.BILL_SCHEDULE})

#: Intents whose answer depends on a specific transaction. If we cannot identify which one,
#: we ask rather than guess — picking "probably the latest" is how a customer gets told about
#: somebody else's money.
NEEDS_TXN = frozenset({
    Intent.TXN_STATUS, Intent.TXN_FAILED, Intent.DEBITED_NOT_CREDITED,
    Intent.TXN_ON_HOLD, Intent.REFUND_REQUEST, Intent.REFUND_STATUS,
})

#: Everything that is ABOUT THIS CUSTOMER'S OWN MONEY that has already moved. Once a
#: conversation is in here it should stay here: "and the one before that", "what about last
#: month", "why did that one fail" are all follow-ups within this domain, and letting the
#: classifier throw them to SEND_MONEY mid-thread is the single most jarring thing the
#: assistant did. `engine.py` uses this to keep the thread sticky.
TXN_DOMAIN = frozenset({
    Intent.TXN_STATUS, Intent.TXN_HISTORY, Intent.TXN_FAILED,
    Intent.DEBITED_NOT_CREDITED, Intent.TXN_ON_HOLD,
    Intent.REFUND_REQUEST, Intent.REFUND_STATUS,
})

#: Intents answered purely from curated content — no customer data is read at all.
STATIC_CONTENT = frozenset({Intent.JOURNEY_HELP, Intent.FEES_QUERY, Intent.TIME_QUERY,
                            Intent.GREETING_HELP, Intent.KNOWLEDGE_QUERY})

#: The retrieval lane: answered from approved documents and curated content, never by acting
#: and never from customer data. "how to pay" lives here; "pay" never does.
RETRIEVAL = frozenset({Intent.JOURNEY_HELP, Intent.KNOWLEDGE_QUERY})

#: Answered by handing the customer to a person (support, recall desk, fraud team).
HANDOFF = frozenset({Intent.HUMAN_HANDOFF, Intent.WRONG_TRANSFER, Intent.REPORT_FRAUD})

#: Opens one of the app's own screens rather than running a flow in chat.
NAVIGATION = frozenset({Intent.ADD_MONEY, Intent.WITHDRAW_MONEY, Intent.LINK_BANK})


@dataclass(frozen=True)
class IntentSpec:
    """Everything the rest of the system needs to know about one intent."""

    intent: Intent
    label: str                       # what the UI calls it
    blurb: str                       # one line, shown in the decision panel
    slots: tuple[str, ...] = ()      # what must be filled before we can act
    lane: str = "support"            # which watch-screen lane this lights up


#: The UI lanes. The watch screen groups the decision trace by these, so a viewer can see
#: *where* in the pipeline a decision was made rather than just what came out.
LANES = ("understand", "resolve", "decide", "act")


SPECS: dict[Intent, IntentSpec] = {
    Intent.TXN_STATUS: IntentSpec(
        Intent.TXN_STATUS, "Transaction status",
        "Where a specific payment has got to.",
        slots=("txn_ref",), lane="resolve"),
    Intent.TXN_FAILED: IntentSpec(
        Intent.TXN_FAILED, "Failed transaction",
        "A payment that did not complete, and why.",
        slots=("txn_ref",), lane="resolve"),
    Intent.DEBITED_NOT_CREDITED: IntentSpec(
        Intent.DEBITED_NOT_CREDITED, "Debited, not received",
        "Money left the wallet but never reached the beneficiary.",
        slots=("txn_ref",), lane="decide"),
    Intent.TXN_ON_HOLD: IntentSpec(
        Intent.TXN_ON_HOLD, "On hold",
        "Stopped for a compliance or treasury check.",
        slots=("txn_ref",), lane="decide"),
    Intent.TXN_HISTORY: IntentSpec(
        Intent.TXN_HISTORY, "Transaction history",
        "A list of this customer's recent payments.",
        lane="resolve"),
    Intent.REFUND_REQUEST: IntentSpec(
        Intent.REFUND_REQUEST, "Refund requested",
        "The customer wants their money back.",
        slots=("txn_ref",), lane="decide"),
    Intent.REFUND_STATUS: IntentSpec(
        Intent.REFUND_STATUS, "Refund status",
        "Where an already-raised refund has got to.",
        slots=("txn_ref",), lane="resolve"),
    Intent.FEES_QUERY: IntentSpec(
        Intent.FEES_QUERY, "Fees",
        "What a transfer costs on a given corridor.",
        slots=("corridor",), lane="resolve"),
    Intent.TIME_QUERY: IntentSpec(
        Intent.TIME_QUERY, "Timing",
        "How long a corridor takes to settle.",
        slots=("corridor",), lane="resolve"),
    Intent.LIMITS_QUERY: IntentSpec(
        Intent.LIMITS_QUERY, "Limits",
        "Per-transaction, daily and monthly caps.",
        lane="resolve"),
    Intent.SEND_MONEY: IntentSpec(
        Intent.SEND_MONEY, "Send money",
        "Repeat or start a transfer to a known beneficiary.",
        slots=("payee", "amount"), lane="act"),
    Intent.PAY_MERCHANT: IntentSpec(
        Intent.PAY_MERCHANT, "Pay a merchant",
        "Pay a nearby merchant without scanning a QR.",
        slots=("merchant", "amount"), lane="act"),
    Intent.BENEFICIARY_QUERY: IntentSpec(
        Intent.BENEFICIARY_QUERY, "Beneficiaries",
        "Who this customer can already send to.",
        lane="resolve"),
    Intent.SPLIT_PAYMENT: IntentSpec(
        Intent.SPLIT_PAYMENT, "Split a bill",
        "Divide a payment between this customer and people they know.",
        slots=("participants",), lane="act"),
    Intent.SPLIT_STATUS: IntentSpec(
        Intent.SPLIT_STATUS, "Split balances",
        "Who owes whom, and which shares are still outstanding.",
        lane="resolve"),
    Intent.BILL_DUE: IntentSpec(
        Intent.BILL_DUE, "Bills due",
        "What this customer owes across their saved billers.",
        lane="resolve"),
    Intent.BILL_PAY: IntentSpec(
        Intent.BILL_PAY, "Pay a bill",
        "Settle a utility or telecom bill from the wallet.",
        slots=("bill",), lane="act"),
    Intent.BILL_SCHEDULE: IntentSpec(
        Intent.BILL_SCHEDULE, "Schedule a bill",
        "Pay a bill at a time the customer chooses.",
        slots=("bill", "when"), lane="act"),
    Intent.PAYMENT_CLAIM: IntentSpec(
        Intent.PAYMENT_CLAIM, "Someone has paid me",
        "The customer reporting that a person has sent them money — usually a share of "
        "a split. Verified against the ledger; never moves money.",
        lane="resolve"),
    Intent.SPEND_ANALYSIS: IntentSpec(
        Intent.SPEND_ANALYSIS, "Where my money went",
        "A breakdown of merchant spending by category for a period, against the period "
        "before it. Reads the ledger; never moves money.",
        slots=("period",), lane="resolve"),
    Intent.SPEND_RANK: IntentSpec(
        Intent.SPEND_RANK, "Most, biggest, how many",
        "A ranking or a tally over the customer's own payments — the top payee, the "
        "largest single payment, a count or a total. Reads the ledger; never moves money.",
        slots=("period", "payee"), lane="resolve"),
    Intent.DISPUTED: IntentSpec(
        Intent.DISPUTED, "That can't be right",
        "The customer contradicting a figure we gave them. Re-reads the ledger from "
        "scratch and either corrects the claim or shows why it holds. Never moves money.",
        lane="decide"),
    Intent.GREETING_HELP: IntentSpec(
        Intent.GREETING_HELP, "Hello / I need help",
        "An opening or an open-ended request for help.",
        lane="understand"),
    Intent.OFFERS_NEARBY: IntentSpec(
        Intent.OFFERS_NEARBY, "Nearby offers",
        "Merchants near the customer and their live campaigns.",
        lane="resolve"),
    Intent.JOURNEY_HELP: IntentSpec(
        Intent.JOURNEY_HELP, "How do I…",
        "A customer-facing walkthrough of a product journey.",
        slots=("journey",), lane="resolve"),
    Intent.OUT_OF_SCOPE: IntentSpec(
        Intent.OUT_OF_SCOPE, "Out of scope",
        "Not something this assistant answers.",
        lane="understand"),
    Intent.MOBILE_RECHARGE: IntentSpec(
        Intent.MOBILE_RECHARGE, "Mobile recharge",
        "Prepaid top-up or data for a phone number.",
        slots=("phone", "amount"), lane="act"),
    Intent.REMINDER_SET: IntentSpec(
        Intent.REMINDER_SET, "Set a reminder",
        "A reminder to pay a bill or a person. It notifies; it never pays.",
        slots=("what", "when"), lane="act"),
    Intent.REMINDER_LIST: IntentSpec(
        Intent.REMINDER_LIST, "Reminders",
        "The customer's active reminders, to view, edit or delete.",
        lane="resolve"),
    Intent.BALANCE_QUERY: IntentSpec(
        Intent.BALANCE_QUERY, "Balance",
        "What is in the wallet now, from the customer's own ledger.",
        lane="resolve"),
    Intent.RETRY_PAYMENT: IntentSpec(
        Intent.RETRY_PAYMENT, "Retry",
        "Try a failed payment again — only when retrying cannot charge twice.",
        lane="decide"),
    Intent.CANCEL_ACTION: IntentSpec(
        Intent.CANCEL_ACTION, "Cancel",
        "Stop what is being prepared. Nothing moves.",
        lane="decide"),
    Intent.HUMAN_HANDOFF: IntentSpec(
        Intent.HUMAN_HANDOFF, "Talk to support",
        "Hand the conversation, with its context, to a person.",
        lane="act"),
    Intent.WRONG_TRANSFER: IntentSpec(
        Intent.WRONG_TRANSFER, "Wrong recipient",
        "Money sent to the wrong person: a recall request and support.",
        slots=("txn_ref",), lane="decide"),
    Intent.REPORT_FRAUD: IntentSpec(
        Intent.REPORT_FRAUD, "Unrecognised payment",
        "A payment the customer did not make: straight to the fraud team.",
        lane="decide"),
    Intent.KNOWLEDGE_QUERY: IntentSpec(
        Intent.KNOWLEDGE_QUERY, "Product question",
        "General product knowledge, answered from approved documents.",
        lane="resolve"),
    Intent.ADD_MONEY: IntentSpec(
        Intent.ADD_MONEY, "Add money",
        "Top up the wallet — opens the Add Money screen.", lane="act"),
    Intent.WITHDRAW_MONEY: IntentSpec(
        Intent.WITHDRAW_MONEY, "Withdraw",
        "Move wallet money to the customer's bank — opens the Withdraw screen.", lane="act"),
    Intent.LINK_BANK: IntentSpec(
        Intent.LINK_BANK, "Connect a bank",
        "Link a bank account — opens the Connect Bank screen.", lane="act"),
}

#: Training and inference order. **APPEND-ONLY.**
#:
#: This used to be `tuple(i.value for i in Intent)`, which tied the model's output indices to
#: the order intents happen to be declared in. Adding TXN_HISTORY next to its siblings would
#: then have shifted every label after it by one, silently relabelling the trained checkpoint:
#: the file would still load, and every prediction from `refund_request` onward would come
#: back as the wrong intent. Declaration order is for humans; this list is the wire format.
#: New intents go on the END, and nothing is ever removed or reordered.
LABELS: tuple[str, ...] = (
    "txn_status", "txn_failed", "debited_not_credited", "txn_on_hold",
    "refund_request", "refund_status", "fees_query", "time_query", "limits_query",
    "send_money", "pay_merchant", "beneficiary_query", "greeting_help",
    "offers_nearby", "journey_help", "out_of_scope",
    # ── appended 2026-10-08 ──
    "txn_history",
    "split_payment", "split_status",
    "bill_due", "bill_pay", "bill_schedule",
    # ── appended 2026-10-09 ──
    "payment_claim",
    # ── appended 2026-10-12 ──
    #: Not yet in the trained label space: openwallet-cx-intent-ml-v7 has 23 outputs and cannot
    #: predict this one. It is reached by a deterministic rule in the engine until the next
    #: retrain, exactly as payment_claim was.
    "spend_analysis",
    #: Same footing as spend_analysis: reached by rule, not predictable by the deployed
    #: model, and the rows for it are being collected for the next training set.
    "disputed",
    "spend_rank",
    # ── appended 2026-10-14, from the customer-app BRD v.34 ──
    "mobile_recharge", "reminder_set", "reminder_list", "balance_query",
    "retry_payment", "cancel_action", "human_handoff", "wrong_transfer", "report_fraud",
    "knowledge_query", "add_money", "withdraw_money", "link_bank",
)
assert set(LABELS) == {i.value for i in Intent}, \
    "LABELS and Intent have diverged — add the new intent to the END of LABELS"
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


# ---------------------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------------------

@dataclass
class Slots:
    """What we managed to pull out of the sentence. Every field may legitimately be None —
    a missing slot is a question to ask, not a value to invent."""

    txn_ref: str | None = None
    amount: float | None = None
    currency: str | None = None
    payee: str | None = None
    merchant: str | None = None
    corridor: str | None = None          # e.g. "AE-PK"
    journey: str | None = None
    time_window: str | None = None
    #: How many rows the customer asked for ("last 5"). None means "use the default" — it is
    #: never invented, because "show me my transactions" and "show me my last 3" are different
    #: requests and guessing the number makes the answer look authoritative when it is not.
    count: int | None = None
    #: The window resolved to a day count, so the handler never re-parses English.
    window_days: int | None = None
    #: "my failed transactions", "anything still pending". Offered in the prompt chips, so it
    #: has to be a filter rather than a word the parser tolerates.
    status: str | None = None
    #: Who the bill is being split with. Names as the customer said them, resolved against
    #: the people they actually have splits with (or their roster) before anything is sent —
    #: a participant list is a list of real userIds or it is nothing.
    participants: list | None = None
    #: EQUAL / CUSTOM for a split before the merchant is paid; SHARE / AMOUNT / PERCENTAGE
    #: for one after. Named by the customer ("equally", "60-40", "he pays 20") or defaulted.
    split_type: str | None = None
    #: How many ways, when the customer says "three ways" without naming anybody.
    ways: int | None = None
    #: Which saved bill the customer means ("the electricity one", "DEWA", a nickname).
    bill: str | None = None
    #: The planIds actually resolved, once we have narrowed them. Carried through the
    #: conversation so answering "from my wallet" cannot silently widen a payment from one
    #: bill back to all of them — which is the difference between AED 506 and AED 667.
    bill_ids: list | None = None
    #: When a scheduled payment should run, as the customer said it ("friday", "the 1st").
    when: str | None = None
    #: Resolved to an absolute ISO datetime, so nothing downstream re-parses English.
    when_at: str | None = None
    #: WALLET or OPEN_BANKING. Asked for before a bill payment, never assumed.
    pay_mode: str | None = None

    def missing(self, spec: IntentSpec) -> list[str]:
        return [s for s in spec.slots if getattr(self, s, None) in (None, "")]

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


#: Corridors we actually operate, with the facts the FEES/TIME answers are built from. This
#: is deliberately a small curated table rather than a live pricing call: a customer-facing
#: quote must be something we can stand behind, and the authoritative number still comes from
#: the product slab at transaction time. These are indicative, and the copy says so.
CORRIDORS: dict[str, dict] = {
    "AE-PK": {"name": "UAE → Pakistan", "currency": "PKR",
              "fee_aed": 6.80, "sla": "within minutes", "sla_hours": 0.5,
              "providers": ("HBL", "UBL")},
    "AE-IN": {"name": "UAE → India", "currency": "INR",
              "fee_aed": 8.00, "sla": "same day", "sla_hours": 6,
              "providers": ("TerraPay",)},
    "AE-BD": {"name": "UAE → Bangladesh", "currency": "BDT",
              "fee_aed": 9.50, "sla": "within 2 hours", "sla_hours": 2,
              "providers": ("CityBank",)},
    "AE-LK": {"name": "UAE → Sri Lanka", "currency": "LKR",
              "fee_aed": 9.00, "sla": "same day", "sla_hours": 6,
              "providers": ("HNB",)},
    "AE-AE": {"name": "Within UAE (wallet to wallet)", "currency": "AED",
              "fee_aed": 6.80, "sla": "instant", "sla_hours": 0,
              "providers": ()},
}

#: Country and city names in every language the assistant understands. English-only here
#: meant a correctly classified Arabic "send money to India" lost its corridor and fell back
#: to the local roster — the model understood it, our lookup table did not.
COUNTRY_TO_CORRIDOR = {
    # ── Pakistan ──
    "pakistan": "AE-PK", "pak": "AE-PK", "pk": "AE-PK", "karachi": "AE-PK",
    "lahore": "AE-PK", "islamabad": "AE-PK",
    "पाकिस्तान": "AE-PK", "कराची": "AE-PK", "लाहौर": "AE-PK",
    "باكستان": "AE-PK", "كراتشي": "AE-PK", "لاهور": "AE-PK",
    # ── India ──
    "india": "AE-IN", "ind": "AE-IN", "mumbai": "AE-IN", "delhi": "AE-IN",
    "bombay": "AE-IN", "kerala": "AE-IN", "hindustan": "AE-IN",
    "भारत": "AE-IN", "इंडिया": "AE-IN", "मुंबई": "AE-IN", "दिल्ली": "AE-IN",
    "हिंदुस्तान": "AE-IN", "केरल": "AE-IN",
    "الهند": "AE-IN", "هند": "AE-IN", "مومباي": "AE-IN", "دلهي": "AE-IN",
    # ── Bangladesh ──
    "bangladesh": "AE-BD", "bd": "AE-BD", "dhaka": "AE-BD",
    "बांग्लादेश": "AE-BD", "ढाका": "AE-BD",
    "بنغلاديش": "AE-BD", "دكا": "AE-BD",
    # ── Sri Lanka ──
    "sri lanka": "AE-LK", "srilanka": "AE-LK", "lk": "AE-LK", "colombo": "AE-LK",
    "श्रीलंका": "AE-LK", "श्री लंका": "AE-LK", "कोलंबो": "AE-LK",
    "سريلانكا": "AE-LK", "سري لانكا": "AE-LK", "كولومبو": "AE-LK",
    # ── domestic ──
    "uae": "AE-AE", "dubai": "AE-AE", "abu dhabi": "AE-AE", "sharjah": "AE-AE",
    "दुबई": "AE-AE", "यूएई": "AE-AE", "शारजाह": "AE-AE",
    "الإمارات": "AE-AE", "دبي": "AE-AE", "أبوظبي": "AE-AE", "الشارقة": "AE-AE",
}
