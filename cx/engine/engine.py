"""One question in, one decided answer out, with the whole reasoning chain attached.

Four lanes, in order, each able to stop the pipeline:

    UNDERSTAND  rules first, then the model  -> intent
    RESOLVE     slots, and which payment     -> evidence
    DECIDE      the policy matrix            -> outcome
    ACT         a proposal, never an action  -> what we offer to do

The ordering is the design. The scope rules run BEFORE the classifier, because a classifier
trained on polite customer questions is the wrong thing to ask about an adversarial one. The
matrix runs AFTER both, because no amount of model confidence should be able to change what
a refund policy says. And ACT only ever produces a proposal — money moves through the
existing mandate + device-signature path, which this module cannot reach.

Everything a customer sees passes `guard.scrub` on the way out.
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import asdict, dataclass, field

CX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CX not in sys.path:
    sys.path.insert(0, CX)

import data                                                    # noqa: E402
import journeys                                                # noqa: E402
import voice                                                   # noqa: E402
import aggregate
import learn
import history as histmod
import ocr as ocrmod
import neighbours
import recheck as recheckmod
import spend as spendmod                                      # noqa: E402
from taxonomy import (BILL_DOMAIN, TXN_DOMAIN, CORRIDORS, Intent, MONEY_MOVING, NEEDS_TXN, SPECS)  # noqa: E402
from engine import lang as langmod
from engine import watchdog
from engine import prefilter
from engine import security
import phrasebank
from engine import (client, conversation, guard, incentive, refine,  # noqa: E402
                    slots as slotlib)
from engine.matrix import Outcome, decide as run_matrix        # noqa: E402

#: Below this, we ask instead of acting. Chosen to sit above the fallback classifier's
#: typical score so that a degraded classifier asks rather than guesses.
#: Every `kind` a PROPOSAL can carry — i.e. every thing the customer can be asked to
#: confirm. Declared here rather than left implicit because the surface has to be able to
#: render each one: a bill proposal fell through a send-shaped fallback and was offered as
#: "Send AED NaN to undefined", with a button that would have fired a transfer. A test pins
#: this list against the page's own switch, so adding a kind without a way to show it fails
#: rather than shipping.
PROPOSAL_KINDS = frozenset({"send", "pay_merchant", "split", "bill_pay", "bill_schedule",
                            "bill_split", "refund"})

CONFIDENT = 0.55

#: Money-moving intents need more than that, because the cost of being wrong is a payment.
CONFIDENT_FOR_MONEY = 0.75


@dataclass
class Answer:
    intent: str
    say: str
    lane_trace: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    slots: dict = field(default_factory=dict)
    txn: dict | None = None
    session: str = ""        # so handlers can see what this conversation already did
    inferred_txn: bool = False     # true when we chose the payment rather than being told
    outcome: str | None = None
    proposal: dict | None = None
    options: list[dict] = field(default_factory=list)
    #: A LIST answer (transaction history). Carried as structured rows rather than prose so
    #: the phone can render a statement; an assistant that reads out five amounts, five
    #: payees and five dates in one paragraph is unreadable on a small screen.
    rows: list[dict] = field(default_factory=list)
    #: True when this turn was the customer saying "yes" to an offer from a previous one.
    #: The surface executes it exactly as a button press would.
    confirmed: bool = False
    #: Which language to ANSWER in, decided from what the customer wrote. Carried here so
    #: every handler uses the same decision rather than re-deriving it.
    lang: str = "en"
    #: What the customer actually typed. Kept so the single exit point can judge the answer
    #: against the question without every return site having to pass it along.
    said: str = ""
    #: An earlier task this utterance reached back to, if any — see conversation.find_task
    recalled: object = None
    advisory: dict | None = None
    model: str = ""
    top: list = field(default_factory=list)   # the classifier's own ranking, for the UI
    degraded: bool = False
    #: Why this turn was written to the learning store, or None if it was not worth keeping.
    #: Set by `_finish`, so it reflects what was actually served.
    kept: str | None = None
    #: The ranking or tally this turn produced, when it was that kind of question.
    rank: dict | None = None
    #: The full spend breakdown, when this turn produced one. Carried structured rather than
    #: only as prose so the phone can draw it and the neural-web view can crawl it.
    spend: dict | None = None
    ms: int = 0

    def step(self, lane, what, detail, tag="rule"):
        self.lane_trace.append({"lane": lane, "what": what, "detail": detail, "tag": tag,
                                "t": time.time()})


#: "ادفع ٣٠ درهم" — Arabic-Indic (and Persian) digits are numbers too; the amount parser reads ASCII.
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_SMART_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                               "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def ask(text: str, want_second_opinion: bool = True,
        session: str = "default", pick: dict | None = None) -> Answer:
    """`pick` is a choice the customer made from options WE offered — a payee from their own
    roster, a merchant from the nearby list, a transaction from their own statement.

    It is trusted and bypasses classification, because it is our own UI state coming back, not
    free text to be interpreted. Round-tripping it through the classifier is how "Send to
    Alex Morgan" became "send money to Alex Morgan" became a 54% guess became a refusal.
    It can still only name things the customer was actually offered — it carries no amount
    authority and no new payee.
    """
    t0 = time.time()
    a = Answer(intent=Intent.OUT_OF_SCOPE.value, say=guard.REFUSAL)
    a.session = session
    said = (text or "").strip()
    # Phone keyboards type ’ and “ ”. "What’s up" read as bills at 89% while "What's up" read
    # as small talk — the model and every rule here were written against the ASCII forms.
    said = said.translate(_SMART_QUOTES).translate(_ARABIC_DIGITS)
    said = _strip_vocative(said)
    # Garbled payment words are fixed before anything reads them — "senmoney", "sen 200
    # dirhms", "pey my bil". The model then sees the words it was trained on.
    heard = said
    said = prefilter.correct(said)
    _learned = watchdog.override(said)
    # Decide the answer language from what they wrote, once, here — every handler then
    # uses the same decision instead of re-deriving it.
    a.said = said
    a.lang = langmod.detect(said)
    if a.lang == "en" and not _speaks_english(said):
        # A bare "100" or "yes" has no language in it. Switching back to English mid-
        # conversation because the customer typed a number is not a decision about them —
        # it is us forgetting. The session's established language carries on.
        #
        # But a real English SENTENCE is English. One misheard Hinglish phrase ("Dil pay
        # karna hai") switched a session to Hinglish and every English question after it —
        # "Bill payment", "Last month, spend analysis" — was answered in Hinglish.
        a.lang = conversation.last_lang(session)
    if a.lang != "en":
        a.step("understand", "language",
               f"written in {a.lang} — answering in {a.lang}", tag="rule")
    # Tagged "input" so the output sweep skips it. The sweep exists to stop OUR text leaking
    # internal vocabulary; the customer's own sentence is not a leak, and flagging it would
    # mean the assistant could be silenced simply by someone typing the word "mysql".
    _sec = security.check(heard)
    if _sec == "credential":
        # A code or card number the customer typed is never kept, logged or echoed.
        heard, said = security.scrub(heard), security.scrub(said)
        a.said = said
    a.step("understand", "received", f'"{heard[:120]}"', tag="input")
    if said != heard:
        a.step("understand", "spelling", f'read as "{said[:120]}"', tag="input")
    if security.CHANGE_PIN.search(said or "") and not _sec:
        # "I forgot my PIN, how do I change it?" in any language — the customer's own PIN, an
        # app task with a fixed answer (never a credential exchange, never a guess)
        a.step("understand", "pin.help", "asks how to change or reset their own PIN or login", tag="rule")
        a.intent, a.model, a.confidence = Intent.JOURNEY_HELP.value, "pin-help", 1.0
        _h = HANDLERS.get(Intent.JOURNEY_HELP)
        out = _h(a, slotlib.extract(said, a.intent), "change my pin") if _h else a
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)
    if _sec == "human":
        a.step("understand", "security.human", "asks for a person, not the assistant — handing over", tag="rule")
        a.intent, a.model, a.confidence = Intent.HUMAN_HANDOFF.value, "human-request", 1.0
        _h = HANDLERS.get(Intent.HUMAN_HANDOFF)
        out = _h(a, slotlib.extract(said, a.intent), said) if _h else a
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)
    if _sec == "fraud_victim":
        a.step("understand", "security.fraud_victim",
               "says their own money was stolen or the account was used without them — straight to "
               "the fraud team", tag="rule")
        a.intent, a.model, a.confidence = Intent.REPORT_FRAUD.value, "fraud-victim", 1.0
        _sx = slotlib.extract(said, a.intent)
        _h = HANDLERS.get(Intent.REPORT_FRAUD)
        out = _h(a, _sx, said) if _h else a
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)
    if _sec:
        return _finish(_security_answer(a, _sec, heard, session, t0))

    # ---- lane 1: UNDERSTAND --------------------------------------------------------
    # Abuse first, by rule. The model has no class for an insult: "gandu" scored greeting_help
    # and got a cheerful welcome. Nothing-but-abuse is declined calmly; abuse wrapped round a
    # real request has the abusive words removed and the request answered.
    # Policy, decided by the business: help once, then stop.
    #   * abuse and nothing else        -> a calm boundary, declined
    #   * the FIRST abusive message that carries a real question -> answered, with a
    #     boundary line in front (the money problem is real, and refusing it helps nobody)
    #   * any abuse after that          -> the conversation ends; support is offered, and
    #     nothing more is answered in this conversation, however politely it is asked
    rude = guard.abuse(said)
    if rude or conversation.abuse_stopped(session):
        if rude:
            words, cleaned, only = rude
            count = conversation.note_abuse(session)
        else:
            words, cleaned, only, count = [], said, False, 2
        if a.lang == "en" and words and guard.hinglish_abuse(words):
            a.lang = langmod.HINGLISH
        if count >= 2:
            a.step("understand", "abuse.stop",
                   ("abusive language again in this conversation" if rude else
                    "this conversation was ended for repeated abuse")
                   + " — not answering further; pointing to customer support", tag="guard")
            a.intent, a.model, a.confidence = Intent.OUT_OF_SCOPE.value, "abuse-stop", 1.0
            # Nothing offered earlier may still be confirmed: a stopped conversation does not
            # carry a standing "send 50 to Alex" that a button press could complete.
            conversation.drop_proposal(session)
            conversation.clear(session)
            a.outcome = "REFUSED"
            a.say = _tr(a, "abuse_stopped",
                        "I can't continue this conversation because of the abusive language. "
                        "If something is wrong with a payment, please contact OpenWallet customer "
                        "support and they will look into it.")
            a.options = []
            a.ms = int((time.time() - t0) * 1000)
            return _finish(a)
        if only:
            a.step("understand", "abuse.rule",
                   f"abusive language and no request — {len(words)} word(s) matched; declined "
                   "politely without asking the model", tag="guard")
            a.intent = Intent.OUT_OF_SCOPE.value
            a.model = "abuse-rule"
            a.confidence = 1.0
            a.outcome = "REFUSED"
            a.say = _tr(a, "abuse_boundary",
                        "I'm here to help, but please keep it respectful. If something has gone "
                        "wrong with a payment, tell me what happened and I'll look into it.")
            a.options = []
            a.ms = int((time.time() - t0) * 1000)
            return _finish(a)
        out = ask(cleaned, want_second_opinion=want_second_opinion, session=session, pick=pick)
        out.lane_trace.insert(0, {"lane": "understand", "what": "abuse.rule",
                                  "detail": f"abusive language removed ({len(words)} word(s)); "
                                            "first time in this conversation, so answering the "
                                            "request that was left — any more abuse ends it",
                                  "tag": "guard", "t": time.time()})
        prefix = _tr(out, "abuse_prefix", "Please keep it respectful.")
        if out.say and prefix:
            out.say = f"{prefix} {out.say}"
        out.ms = int((time.time() - t0) * 1000)
        return out

    if _IDENTITY.match((said or "").strip()):
        # "What are you" got a fee quote (the nearest known question was about fees) and
        # "Who are you" a refusal. Who we are is the one personal question worth answering.
        a.step("understand", "identity", "asks what the assistant is — introducing ourselves", tag="rule")
        a.intent, a.model, a.confidence = Intent.GREETING_HELP.value, "identity", 1.0
        a.say = _tr(a, "identity",
                    "I'm the OpenWallet Assistant. I can send money to people you've paid, pay a shop or a "
                    "bill, recharge a phone, split a bill, set payment reminders, and tell you about your "
                    "transactions, balance, fees and refunds. What would you like to do?")
        a.say = _ph(a, "identity", a.say)           # the bank names the OpenWallet Assistant itself
        a.options = [{"kind": "ask_text", "label": l, "text": t} for l, t in (
            ("Send money", "I want to send money"), ("Pay a bill", "pay my bill"),
            ("My last payments", "show my last 5 transactions"), ("My balance", "what's my balance"))]
        a.outcome = "ANSWERED"
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    _go = _GO_SCREEN.match((said or "").strip())
    if _go:
        # "Go to profile" was answered with the wallet balance; "Account management" with
        # "didn't catch". The assistant cannot open screens — it can say where they are.
        screen = (_go.group(1) or _go.group(2) or "").lower()
        where = next((w for k, w in _SCREENS.items() if k in screen), "in the menu at the bottom of the app")
        a.step("understand", "navigation", f"asks to go to '{screen}' — saying where it is", tag="rule")
        a.intent, a.model, a.confidence = Intent.JOURNEY_HELP.value, "navigation", 1.0
        a.say = _tr(a, "go_screen", f"I can't open that screen from here — you'll find {screen.title()} {where}. "
                    f"Anything about your payments I can help with?", screen=screen.title(), where=where)
        a.outcome = "EXPLAINED"
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    if _is_small_talk(said):
        a.step("understand", "small.talk",
               "courtesy, not a request — answered kindly and declined, nothing looked up", tag="guard")
        a.intent, a.model, a.confidence = Intent.OUT_OF_SCOPE.value, "small-talk", 1.0
        if a.lang == "en" and re.search(r"\b(kaise|kaisa|haal|badhiya|badia|theek)\b", said.lower()):
            a.lang = langmod.HINGLISH
        a.say = _ph(a, "small_talk", _tr(a, "small_talk",
                    "I'm doing well, thanks for asking! I can only help with your OpenWallet payments — "
                    "sending money, bills, your transactions and balance. What would you like to do?"))
        a.outcome = "REFUSED"
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    forced = guard.forced_out_of_scope(said)
    if forced:
        a.step("understand", "scope.rule", f"declined by rule — {forced}", tag="guard")
        a.confidence = 1.0
        a.model = "scope-rule"
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    # The options we showed LAST turn, and only last turn: popped here, set again by
    # _finish. "The first one" can never reach back to a menu from three turns ago.
    _shown = conversation.subject(session, "last_options")
    conversation.forget_subject(session, "last_options")
    # Section headings ("Someone you know") are not choices; "the first one" means the first
    # thing that can actually be picked.
    _shown = [o for o in (_shown or []) if _option_request(o)]
    if not pick and _shown:
        _o = _spoken_pick(said, _shown)
        _req = _option_request(_o) if _o else None
        if _req:
            # Exactly what tapping that button does — same text, same structured pick — so
            # saying it can never do more than tapping it. Money still stops at the
            # confirmation card and the PIN.
            _text, _pk = _req
            out = ask(_text, want_second_opinion, session=session, pick=_pk)
            out.lane_trace.insert(0, {"lane": "understand", "what": "picked",
                                      "detail": f'"{said[:40]}" picks option {_shown.index(_o) + 1} '
                                                f'of the ones we just showed: {_o.get("label", "")[:60]}',
                                      "tag": "rule", "t": time.time()})
            return out

    # "Send my last 5 transactions to my email." A rule, not a model label: the taxonomy is
    # append-only and retraining for one phrase family is not worth it. The engine only
    # OFFERS it — the address is the masked one on the customer's own profile, the count is
    # what they said — and the server sends it only after confirm + PIN (kind
    # "email_statement" in cx_act), resolving the address again itself.
    if not pick and _wants_statement_email(said):
        import statement
        n = _statement_count(said)
        to, source = statement.recipient()
        a.intent, a.model, a.confidence = Intent.TXN_HISTORY.value, "email-rule", 1.0
        a.step("understand", "email.statement",
               f"asks to email {n} transactions — a statement, not a payment", tag="rule")
        if not to:
            a.say = ("There's no verified email address on your profile yet, so I can't send "
                     "it. Add one in Settings and ask again.")
            a.outcome = "NONE"
            a.step("act", "email.none", source, tag="guard")
        else:
            a.step("resolve", "email.to", f"{statement.mask(to)} — from the {source}", tag="db")
            a.proposal = {"kind": "email_statement", "count": n, "to": statement.mask(to)}
            a.say = _tr(a, "email_statement",
                        f"I'll email your last {n} transactions to {statement.mask(to)}. "
                        f"Confirm, and I'll ask for your PIN.", n=n, to=statement.mask(to))
            a.outcome = "PROPOSED"
        _remember(session, a, slotlib.extract(said, a.intent))
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    if pick and pick.get("intent") in {i.value for i in Intent}:
        a.intent = pick["intent"]
        a.confidence = 1.0
        a.model = "chosen"
        a.step("understand", "chosen",
               f"the customer picked this from the options we offered — "
               + ", ".join(f"{k}={v}" for k, v in (pick.get("slots") or {}).items()),
               tag="rule")
        base = conversation.get(session)
        merged = dict(base.slots) if base and base.intent == a.intent else {}
        # Start from what the SENTENCE says, then let the pick override it. The pick used to
        # be the only source, so an option carrying an intent but no slots discarded
        # everything the text contained — "show me my payments to Alex" arrived with no
        # payee and answered "I don't have anyone called Alex on your list". The pick still
        # wins every field it actually sets, which is the whole point of a pick.
        merged.update(slotlib.extract(said, a.intent).as_dict())
        merged.update({k: v for k, v in (pick.get("slots") or {}).items() if v not in (None, "")})
        s_obj = slotlib.Slots(**{k: v for k, v in merged.items()
                                 if k in slotlib.Slots.__annotations__})
        a.slots = s_obj.as_dict()
        handler = HANDLERS.get(Intent(a.intent))
        out = handler(a, s_obj, said) if handler else a
        _remember(session, out, s_obj)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # ── yes / no, before anything else ──────────────────────────────────────────────────
    held = conversation.proposal(session)
    if _is_no(said):
        conversation.drop_proposal(session)
        conversation.clear(session)
        # Labelled for what it IS — a cancel — whatever was on offer. The held intent used to
        # be reused, so the record said "greeting" or "send money" for a turn that cancelled.
        a.intent = Intent.CANCEL_ACTION.value
        a.model = "answered"
        a.confidence = 1.0
        a.step("understand", "declined",
               "a bare no while something was on offer — dropping it rather than letting the "
               "classifier find a subject for a word that has none", tag="rule")
        a.say = (_tr(a, "cancelled", "Payment cancelled. No money was sent.")   # BRD 2.1 T10
                 if held and (held.get("kind") in ("send", "pay_merchant", "bill_pay", "split",
                                                   "bill_split", "recharge"))
                 else _tr(a, "cancelled_other", "All right, I've left it. Nothing has moved.") if held
                 else _tr(a, "cancelled", "No problem. What would you like to do?"))
        a.outcome = "CANCELLED"
        a.options = []
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    if held and _is_yes(said, held):
        conversation.drop_proposal(session)
        conversation.clear(session)
        a.intent = held.get("intent") or Intent.SEND_MONEY.value
        a.model = "answered"
        a.confidence = 1.0
        a.step("act", "confirmed",
               "the customer said yes to the offer still standing from the previous turn",
               tag="rule")
        a.proposal = held
        # The engine still does not execute. `confirmed` tells the surface to do exactly what
        # pressing the button would have done, so there is one execution path, not two.
        a.confirmed = True
        a.say = ""
        a.outcome = "CONFIRMED"
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    # A correction learned from this exact sentence earlier. Applied before the model,
    # because the model is the thing that got it wrong — and only ever to a safe,
    # non-money intent (watchdog._SAFE_TARGETS).
    if _learned and _learned != a.intent:
        a.step("understand", "learned",
               f"this exact sentence was mis-answered before and corrected to "
               f"{_learned.replace('_', ' ')} — applying that", tag="rule")
        a.intent, a.confidence, a.model = _learned, 1.0, "learned"

    # Is this NARROWING an open question rather than answering it? "which has offers",
    # "which is nearest" — the same question over a smaller list.
    #
    # Checked BEFORE the classifier, and regardless of what it would say: "which is nearest"
    # scores confidently OUT OF SCOPE on its own, because on its own it IS meaningless. It
    # only means anything relative to the list already on screen, which the classifier cannot
    # see. The detector is strict (a question must be open, the phrase must be short, and it
    # must carry no slot of its own), so this cannot swallow a real request.
    _open = conversation.get(session)
    if _open:
        _kind = refine.detect(said, slotlib.extract(said, "").as_dict())
        if _kind:
            a.intent = _open.intent
            a.model = "refinement"
            a.confidence = 1.0
            a.step("understand", "refine",
                   f"narrowing the open {_open.intent.replace('_', ' ')} question — "
                   f"{refine.describe(_kind)}", tag="rule")
            out = _refine_options(a, _kind, _open)
            _remember(session, out, slotlib.Slots(**{
                k: v for k, v in _open.slots.items()
                if k in slotlib.Slots.__annotations__}))
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)

    # Somebody telling us money ARRIVED. Decided before the classifier, because the
    # classifier is confidently wrong here: "alex ne paise bhej diye hai" carries a name
    # and a transfer verb, scores send_money, and the engine offered to send money to the
    # person who had just paid.
    # Asking where the money went. Decided before the classifier because the classifier
    # cannot reach this intent: spend_analysis is the 24th label and the deployed model has
    # 23 outputs. See `_is_spend_question`.
    # "I'm talking about the one I uploaded" — however many turns ago that was.
    #
    # Reported: a screenshot, then two unrelated flows, then this — answered about the most
    # recent payment instead, and when pressed, "I don't have a figure from this
    # conversation to check". The upload had left no record at all, so there was nothing to
    # come back to. It is in the task ledger now, which outlives topic changes by design.
    _up = conversation.subject(session, "uploaded_ref")
    if _up and _means_the_upload(said):
        a.step("understand", "recall.upload",
               "this points back at the screenshot sent earlier in this conversation — "
               "picking up that payment rather than the most recent one", tag="rule")
        _s = slotlib.extract(said, "")
        _s.txn_ref = _up
        # What they want DONE with it still comes from the sentence: "refund the one I
        # uploaded" is a refund, "what happened to the one I uploaded" is a status check.
        _want = client.classify(said).get("intent")
        _intent = (_want if _want in (Intent.REFUND_REQUEST.value,
                                      Intent.REFUND_STATUS.value,
                                      Intent.TXN_FAILED.value,
                                      Intent.DEBITED_NOT_CREDITED.value)
                   else Intent.TXN_STATUS.value)
        a.intent, a.confidence, a.model = _intent, 1.0, "recall-upload"
        _h = HANDLERS.get(Intent(_intent))
        out = _h(a, _s, said) if _h else a
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # "show me the details" — about what we were just talking about.
    #
    # The reported failure was "yes show me the detials" after a refund answer, which came
    # back "I don't have anyone called Detials on your list": the typo made it a name, and
    # even spelled correctly it listed five unrelated payments. Both halves were the same
    # mistake — the sentence has no subject of its own, so it can only mean the subject we
    # already have, and the engine was reading it as a fresh request.
    _detail_ref = conversation.subject(session, "txn_ref")
    if _detail_ref and _wants_detail(said):
        a.intent, a.confidence, a.model = Intent.TXN_STATUS.value, 1.0, "in-hand"
        a.step("understand", "in.hand",
               "this asks for more about the payment we were just discussing — it names no "
               "subject of its own, so the subject is the one already in hand", tag="rule")
        _s = slotlib.extract(said, a.intent)
        _s.txn_ref = _s.txn_ref or _detail_ref
        out = h_txn(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # A ranking is a thread too. "who did I pay the most this month" -> "and last month?"
    # asks the SAME question about another window — but the follow-up holds no "who" and no
    # "most", so the rank rule cannot see it, and the model read it as transaction history
    # (92%) and listed five payments. Mid-ranking, a bare change of window re-asks the
    # ranking: the previous question with its window swapped for the new one. Same residue
    # test as the spend thread, so anything naming a payee, an amount or a list is untouched.
    _prev_rank = conversation.subject(session, "rank_q")
    if (_prev_rank and conversation.topic(session) == Intent.SPEND_RANK.value
            and _rank_follow_up(said)):
        _asked = _swap_period(_prev_rank, said)
        _win = spendmod.follow_up_period(_prev_rank, said)
        a.intent, a.confidence, a.model = Intent.SPEND_RANK.value, 1.0, "rank-thread"
        a.step("understand", "stay.rank",
               f"a new window and nothing else, straight after a ranking — asking the same "
               f"ranking again: “{_asked}”"
               + (f", for {spendmod.period_label(_win)}, because “{said.strip()}” straight "
                  f"after “{_prev_rank.strip()}” means the window before it" if _win else ""),
               tag="rule")
        _s = slotlib.extract(_asked, "")
        out = h_spend_rank(a, _s, _asked, period=_win)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # A ranking or a tally. Checked BEFORE the breakdown rule, because "how much did I
    # spend on food" and "who did I pay the most" both look like spending questions and only
    # one of them is answered with a pie chart.
    if _is_rank_question(said):
        a.intent, a.confidence, a.model = Intent.SPEND_RANK.value, 1.0, "rank-rule"
        a.step("understand", "ranking",
               "the answer to this is a name or a single number, not a list of payments",
               tag="rule")
        _s = slotlib.extract(said, "")
        out = h_spend_rank(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # "that can't be right, I didn't spend 4,549" pushes back on a FIGURE — the model decides
    # (it reads disputed); the spending rule would answer with the same breakdown again.
    if (_is_spend_question(said) and not _DISPUTE.search(_flat(said))
            and not any(w in _flat(said) for w in _DISPUTE_INTL)
            and not re.search(r"\b(?:allowed|allow|limit|limits|max(?:imum)?|kitne\s+tak)\b", said or "", re.I)):
        a.intent, a.confidence, a.model = Intent.SPEND_ANALYSIS.value, 1.0, "spend-rule"
        a.step("understand", "spending",
               "this asks about the pattern of spending, not about one payment — "
               "building a category breakdown", tag="rule")
        _s = slotlib.extract(said, "")
        out = h_spend_analysis(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # Contradicting a figure we gave. Decided before the classifier for the same reason as
    # the two above — `disputed` is the 25th label and the deployed model has 23 outputs —
    # and before the scope guards, because "that can't be right" on its own looks like
    # nothing the assistant handles and was being refused.
    # "is there any offer" while a merchant payment is on the table. Checked before the
    # classifier, which reads it as a request for the nearby-offers list — correct for the
    # sentence alone, and wrong when they are standing in the shop looking at a total.
    _held_prop = conversation.proposal(session) or {}
    if (_held_prop.get("kind") == "pay_merchant" and _held_prop.get("merchant")
            and _asks_about_offer(said)
            and not slotlib.extract(said, "").merchant):
        a.intent = Intent.OFFERS_NEARBY.value
        _s = slotlib.extract(said, a.intent)
        out = h_offer_on_this(a, _s, said)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # A question ABOUT the last figure, checked before the dispute rule — "is it merchant
    # related" is asking what the number covered, not claiming it is wrong, and the two
    # need different answers.
    if conversation.last_claim(session) and _asks_about_last(said) \
            and not _is_dispute(said, session):
        a.intent, a.confidence, a.model = Intent.DISPUTED.value, 1.0, "about-rule"
        a.step("understand", "about.last",
               "this asks what the figure I just gave covered, rather than challenging it",
               tag="rule")
        _s = slotlib.extract(said, "")
        out = h_about_last(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    if _is_dispute(said, session):
        a.intent, a.confidence, a.model = Intent.DISPUTED.value, 1.0, "dispute-rule"
        a.step("understand", "disputed",
               "this contradicts a figure I gave earlier in this conversation — working it "
               "out again from the ledger rather than repeating it", tag="rule")
        _s = slotlib.extract(said, "")
        out = h_disputed(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # The direction rule the other way round. `_is_payment_claim` declining to fire only
    # keeps the RULE quiet — the classifier can still reach payment_claim on its own, and it
    # does: "i paid alex 100" scores 98%. First person plus a payment verb is money going
    # out, and answering it with "Yes, Alex's 253 came in" tells the customer their own
    # payment was somebody else's. The rule knows the direction; it should be allowed to say
    # so, because the cost here is the same asymmetry that justified the rule to begin with.
    if (_NOT_ARRIVED.search(_flat(said))
            and client.classify(said).get("intent") == Intent.PAYMENT_CLAIM.value):
        # The model reads "alex ne paise nahi bheje" as money arriving, at full
        # confidence — the sentence is almost exactly the one that means the opposite, and
        # "nahi" is one short token. Declining to fire the rule only keeps the RULE quiet;
        # the classifier still reaches payment_claim on its own, and a customer saying the
        # money never came gets told it did. The negation is unambiguous, so it wins.
        a.intent = Intent.DEBITED_NOT_CREDITED.value
        a.confidence, a.model = 1.0, "direction-rule"
        a.step("understand", "direction",
               "this says the money did NOT arrive — the opposite of money coming in, "
               "however close the two sentences look", tag="rule")
        _s = slotlib.extract(said, a.intent)
        _h = HANDLERS.get(Intent.DEBITED_NOT_CREDITED)
        out = _h(a, _s, said) if _h else a
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    if _I_PAID.search(" ".join(said.lower().split())) and not _WANTS_TO_SEND.search(said.lower()):
        _pred = client.classify(said)
        if _pred.get("intent") == Intent.PAYMENT_CLAIM.value:
            a.intent = Intent.TXN_STATUS.value
            a.confidence, a.model = 1.0, "direction-rule"
            a.step("understand", "direction",
                   "this describes a payment the customer MADE, not one they received — "
                   "looking it up rather than reporting it as money in", tag="rule")
            _s = slotlib.extract(said, a.intent)
            out = h_txn(a, _s, said)
            _remember(session, out, _s)
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)

    if _is_payment_claim(said) and _claim_rule_may_override(said):
        a.intent, a.confidence, a.model = Intent.PAYMENT_CLAIM.value, 1.0, "direction-rule"
        a.step("understand", "direction",
               "this reports money that has already arrived, not a request to send any — "
               "checking the ledger instead of offering a payment", tag="rule")
        _s = slotlib.extract(said, "")
        out = h_payment_claim(a, _s, said)
        _remember(session, out, _s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    pred = client.classify(said)
    a.intent = pred["intent"]
    a.confidence = float(pred.get("confidence") or 0)
    a.model = pred.get("model", "?")
    a.degraded = bool(pred.get("degraded"))
    a.top = pred.get("top") or []
    tops = ", ".join(f"{t['intent']} {t['p']:.2f}" for t in pred.get("top", [])[:3])
    a.step("understand", "classify",
           f"{a.model} → {a.intent} ({a.confidence:.0%}){'  [' + tops + ']' if tops else ''}",
           tag="model")
    if a.degraded:
        a.step("understand", "degraded",
               "the trained model is unreachable — running on keyword rules, so I will ask "
               "rather than assume", tag="guard")

    # Nobody trained on this exact sentence. Look for the nearest one we DO know.
    #
    # Consulted only when the classifier is unsure or has landed on out-of-scope, so a
    # confident reading is never second-guessed by a similarity. Two things it catches:
    #
    #   * a typo — "show me my transations" shares almost every character triple with
    #     "show me my transactions", while to a word-based classifier it is a word it has
    #     never seen contributing nothing at all;
    #   * a phrasing nobody wrote down — "who got the most of my cash this month" is not in
    #     the bank, and sits next to "who gets most of my money", which is.
    #
    # It can never reach a money-moving intent; `neighbours.FORBIDDEN` enforces that at the
    # source rather than trusting each caller. A similarity is not an understanding, and
    # "send 100 to alex" being the nearest known sentence must not turn an ambiguous one
    # into a payment.
    # "pay <a merchant we know>" is a merchant payment, whatever the score. v16 read "pay Asad
    # and Sons FZ LLC" as pay_merchant at only 52% (send_money 35%), the neighbour lookup
    # below then answered a who-did-I-pay-most ranking, and the "100 dirhams" that followed
    # had no payment to complete. The verb is PAY, the object resolves to exactly one of our
    # merchants, nobody on the roster is named, and it is not a question: that is a complete
    # reading. It still stops at the amount, the confirmation card and the PIN.
    if a.intent != Intent.PAY_MERCHANT.value or a.confidence < CONFIDENT:
        if (re.search(r"\b(?:pay|payment)\b|ادفع|भुगतान", said or "", re.I)
                and not _is_question(said)):
            _sm = slotlib.extract(said, Intent.PAY_MERCHANT.value)
            if _sm.merchant and not _sm.payee:
                a.step("understand", "merchant.shape",
                       f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it says "
                       f"pay and names one of our merchants ({_sm.merchant}) — a merchant payment",
                       tag="rule")
                a.intent, a.model = Intent.PAY_MERCHANT.value, "merchant-shape"
                a.confidence = max(a.confidence, CONFIDENT)

    # "Can you tell me how your transactions work" fell to a weak reading and was answered
    # "I don't have anyone called Work on your list". How something WORKS is product knowledge.
    # "How do I split?" — v23 read it as split STATUS at 72% and answered with the customer's
    # balances. "How do I <do an action>" read as a status/data lookup asks for the steps.
    if (_HOW_DO_I_ACT.match(said or "") and a.intent in _STATUS_INTENTS
            and a.confidence < 0.9):
        a.step("understand", "how.do.i",
               f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it asks HOW to do "
               f"something — the steps", tag="rule")
        a.intent, a.model, a.confidence = Intent.JOURNEY_HELP.value, "how-do-i", max(a.confidence, CONFIDENT)
    # "Can I pay a bill partially?" — v22 read it as a split at 44% and the split handler,
    # finding nobody named, refused it. "Can I <do an action>?" read WEAKLY as that action is
    # asking whether the product allows it.
    _can_i_action = (_CAN_I.match(said or "") and _is_question(said)
                     and a.intent in _ACTION_INTENTS)
    if (a.confidence < CONFIDENT_FOR_MONEY and (_ABOUT_THE_FEATURE.search(_flat(said)) or _WHICH_CAN_I.match(said or "")
                                                or _can_i_action)
            and a.intent not in (Intent.KNOWLEDGE_QUERY.value, Intent.JOURNEY_HELP.value)):
        a.step("understand", "about.feature",
               f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it asks how "
               f"something works — product knowledge", tag="rule")
        a.intent, a.model = Intent.KNOWLEDGE_QUERY.value, "about-feature"
        a.confidence = max(a.confidence, CONFIDENT)
    elif (a.intent == Intent.KNOWLEDGE_QUERY.value and a.confidence < CONFIDENT
            and (_ABOUT_THE_FEATURE.search(_flat(said)) or _WHICH_CAN_I.match(said or ""))):
        # "Which bills can I pay?" — the model already read a product question, weakly; the
        # nearest-question lookup must not turn it into "your bills due".
        a.step("understand", "about.feature", "a weak product-question reading, kept: it asks what "
               "the product supports", tag="rule")
        a.confidence = CONFIDENT

    # "I need to send money to someone" (44%) and "Can you send money to someone" were sent
    # to the nearest-question lookup and answered with a beneficiary LIST. "send/transfer
    # money" with nobody named is a send request — the send handler asks who, and offers the
    # customer's own people. "Who/which/how … send money" questions are left alone.
    if (a.intent == Intent.REMINDER_SET.value and a.confidence < CONFIDENT_FOR_MONEY
            and not _REMIND_WORD.search(said or "")):
        a.step("understand", "not.a.reminder",
               f"read as a reminder at only {a.confidence:.0%}, but nothing in it asks to be reminded",
               tag="rule")
        a.confidence = min(a.confidence, CONFIDENT - 0.01)
    if (_HELP_DO.search(said or "") and not re.match(r"^\W*how\b", said or "", re.I)
            and (a.intent in (Intent.JOURNEY_HELP.value, Intent.REMINDER_SET.value, Intent.GREETING_HELP.value)
                 # "help me initiating a transfer" read weakly as a support handoff (38%)
                 or (a.intent == Intent.OUT_OF_SCOPE.value and a.confidence < CONFIDENT_FOR_MONEY)
                 # a handoff needs the customer to ask for a person; "help me initiate a
                 # transfer" never does, however sure the model is (v23: confident handoff)
                 or (a.intent == Intent.HUMAN_HANDOFF.value and not _SUPPORT_WORD.search(said or "")))):
        a.step("understand", "help.do", "asks us to help DO a payment, not how to — starting it", tag="rule")
        a.intent, a.model = (Intent.SEND_MONEY.value if re.search(r"send|transfer", said, re.I)
                             else Intent.BILL_PAY.value if _has_bill_word(said)
                             else Intent.PAY_MERCHANT.value), "help-do"
        a.confidence = max(a.confidence, CONFIDENT)
    if (a.confidence < CONFIDENT_FOR_MONEY
            and _SEND_SHAPE.search(said or "")
            and not _REMIND_WORD.search(said or "")                 # "remind me to send money to Sara"
            and not _CAN_I.match(said or "")                         # "Can I send money abroad?" — a product question
            and not _ALREADY_SENT.search(said or "")                 # "kal jo paise bheje the… pahuche?" — a status question
            and not re.match(r"^\W*(?:who|whom|which|how|where|when|why)\b", said or "", re.I)):
        a.step("understand", "send.shape",
               f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it asks to send "
               f"money — a transfer; the send flow will ask who", tag="rule")
        a.intent, a.model = Intent.SEND_MONEY.value, "send-shape"
        a.confidence = max(a.confidence, CONFIDENT)

    # "is there any offer on sending money to india" read as send_money / fees / out_of_scope.
    # An OFFER question about another country or abroad is an offers question.
    if (_OFFER_WORD.search(said or "") and (_ABROAD.search(said or "") or _COUNTRY_WORD.search(said or ""))
            and a.intent != Intent.OFFERS_NEARBY.value):
        a.step("understand", "xb.offer", f"asks about OFFERS on a transfer abroad (read as "
               f"{a.intent.replace('_', ' ')} {a.confidence:.0%})", tag="rule")
        a.intent, a.model, a.confidence = Intent.OFFERS_NEARBY.value, "xb-offer", 1.0
    # "Did I pay Carrefour yesterday?" ASKS whether a payment happened — a history lookup. v21
    # read it as fraud at high confidence (its new "a merchant charged me, I never bought
    # anything" rows). Fraud needs the customer to deny it: never, not me, unauthorised…
    if (_DID_I_PAY.match(said or "") and not _DENIES.search(said or "")
            and a.intent == Intent.REPORT_FRAUD.value):
        a.step("understand", "did.i.pay",
               "asks WHETHER they paid — a history lookup; nothing in it says the payment "
               "wasn't theirs", tag="rule")
        a.intent, a.model, a.confidence = Intent.TXN_HISTORY.value, "did-i-pay", max(a.confidence, CONFIDENT)
    # Strictly not ours. "Help me buy a flight" was read as app how-to at 94%; "how do i lose
    # belly fat" and recipes likewise. A sentence built around one of these words is declined
    # outright — unless it is clearly about one of the customer's own payments ("my flight
    # booking payment failed", a biller or merchant of theirs, an amount).
    _deny = _HARD_OOS.search(_flat(said))
    if _deny and not _PAYMENT_CONTEXT.search(said or "") \
            and not (_has_transfer_verb(said) and re.search(r"\d", re.sub(r"(?i)openwallet", "", said or ""))):
        # "OpenWallet" itself fuzzy-matches a dev merchant ("OpenWallet Trading …"); our own name is not
        # the customer naming a shop, so it is left out of this lookup
        _sx = slotlib.extract(re.sub(r"(?i)\bpay\s*10\b", " ", said or ""), "")
        if not (_sx.payee or _sx.merchant or _sx.bill or _sx.txn_ref):
            a.step("understand", "deny.word",
                   f"built around “{_deny.group(0)}”, which is not something a payments assistant "
                   f"does — declining whatever the model read ({a.intent.replace('_', ' ')} "
                   f"{a.confidence:.0%})", tag="guard")
            a.intent, a.model, a.confidence = Intent.OUT_OF_SCOPE.value, "deny-word", 1.0
    # Today's exchange rate for a transfer is ours (answered honestly: the live rate is on the
    # send screen); a FORECAST is caught by the deny list above.
    if a.model != "deny-word" and _EXCHANGE_RATE.search(said or "") and _FX_FORECAST.search(said or ""):
        a.step("understand", "fx.forecast", "asks where a rate is GOING — a forecast, not ours", tag="guard")
        a.intent, a.model, a.confidence = Intent.OUT_OF_SCOPE.value, "deny-word", 1.0
    if a.model != "deny-word" and _EXCHANGE_RATE.search(said or ""):
        a.step("understand", "exchange.rate",
               "asks for the exchange rate on a transfer — a fees and rates question", tag="rule")
        a.intent, a.model, a.confidence = Intent.FEES_QUERY.value, "exchange-rate", 1.0

    # The nearest-question lookup may rescue a WEAK reading, but it must not overturn a fairly
    # confident refusal: "Can you dance" (out_of_scope 70%) was turned into a greeting with
    # the customer's last payment in it.
    if a.intent == Intent.OUT_OF_SCOPE.value and a.confidence >= 0.6:
        pass
    elif os.environ.get("CX_NEAREST", "0") == "0":
        # OFF by default since v24 (2026-10-16): the nearest-question fallback was worse or no
        # better in four separate blind measurements (it overturned right readings more often
        # than it rescued wrong ones). CX_NEAREST=1 turns it back on.
        pass
    elif a.confidence < CONFIDENT or a.intent == Intent.OUT_OF_SCOPE.value:
        _near = neighbours.suggest(said)
        if _near and _near["intent"] != a.intent:
            _ex_near = " · ".join(f"\u201c{n['text']}\u201d" for n in _near["neighbours"][:2])
            a.step("understand", "nearest",
                   f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, which is "
                   f"not confident — the closest questions we know are {_ex_near}, and "
                   f"{_near['share']:.0%} of that neighbourhood is "
                   f"{_near['intent'].replace('_', ' ')}", tag="model")
            a.intent = _near["intent"]
            a.confidence = max(a.confidence, _near["confidence"])
            a.model = "nearest"

    # Is this an answer to a question we just asked?
    #
    # Checked AFTER classifying and only when the classifier is WEAK. "100 dirhams" scores
    # a feeble send_money and is obviously the answer to "how much?"; "refund 250" scores a
    # confident refund_request and must NOT be swallowed into an open payment just because it
    # contains a number. Continuation fills gaps; it never overrides a clear new request.
    # A PURE SLOT answer continues regardless of the classifier's confidence. Gating this on
    # low confidence assumed the model would be unsure about a bare "100 dirhams"; the
    # multilingual model is confidently wrong about it instead.
    # Did we just ask what KIND of payment this is? "a bill", "someone", "a shop" carry no
    # slot and classify as nothing on their own, so the open question has to be consulted
    # before the classifier rather than after it.
    _pk = conversation.get(session)
    if (_pk and _pk.asked_for == "pay_kind") or conversation.subject(session,
                                                                    "offered_pay_kind"):
        # A sentence with its own verb ("send money to a shop", "pay my dewa bill") is a new
        # request and is classified as one; only a bare choice ("a shop", "a friend") answers.
        _kind = None if _is_instruction(said) else _pay_kind(said)
        if _kind:
            a.step("understand", "answered",
                   f"we asked what kind of payment this is and this says {_kind}",
                   tag="rule")
            a.intent = {"bill": Intent.BILL_PAY.value,
                        "person": Intent.SEND_MONEY.value,
                        "merchant": Intent.PAY_MERCHANT.value}[_kind]
            a.confidence, a.model = 1.0, "answered"
            _s = slotlib.extract(said, a.intent)
            handler = HANDLERS.get(Intent(a.intent))
            out = handler(a, _s, said) if handler else a
            if _kind == "person" and not _s.payee and out.outcome == "NEED_INFO":
                # They chose "a person" — asking "someone, or a merchant?" again would be
                # ignoring the answer. Narrow the menu to their people.
                people = [o for o in (out.options or []) if o.get("kind") == "send_to"]
                if people:
                    out.options = people
                    out.say = _tr(out, "which_person", "Which one of your people?")
            _remember(session, out, _s)
            # The menu is spent only when the choice actually led somewhere — a proposal to
            # confirm, or a payment made. Anything else and the customer is still choosing.
            #
            # The first version forgot it on any outcome that was not a question, which
            # included NONE — the outcome of a biller we could not reach. So "pay" → "bill"
            # → (biller down) → "a shop" was refused, which is the exact sequence that was
            # reported. A failed attempt is the moment they MOST need the other options.
            if out.proposal or out.outcome in ("PROPOSED", "CONFIRMED", "PAID"):
                conversation.forget_subject(session, "offered_pay_kind")
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)

    _ex = slotlib.extract(said, "").as_dict()

    # A short reply to an open recharge or reminder question ("0552223344", "50", "on the
    # 25th") answers it. Those flows hold details the shared slot set has no field for, so
    # the generic path below cannot see that the reply is an answer; v9 read a bare "50"
    # after "How much?" as a limits question.
    _bframe = conversation.get(session)
    if _bframe and _brd.answers_open(_bframe.intent, _bframe.asked_for, said):
        a.intent, a.confidence, a.model = _bframe.intent, 1.0, "answered"
        a.step("understand", "answered",
               f"we asked for {_bframe.asked_for or 'a detail'} of the open "
               f"{_bframe.intent.replace('_', ' ')} and this supplies it", tag="rule")
        _bs = slotlib.extract(said, a.intent)
        out = HANDLERS[Intent(a.intent)](a, _bs, said)
        _remember(session, out, _bs)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # An answer to the slot we LITERALLY JUST ASKED FOR wins, whatever the classifier scored.
    # This sits outside the confidence gate below on purpose: the gate assumes a reply to an
    # open question will look uncertain, and it does not. "Pay from my wallet" — offered as a
    # button right after we asked where the money should come from — scored as a merchant
    # payment at full confidence and opened a different flow. Narrow by construction: there
    # must be an open question, it must name the slot it asked for, and this utterance must
    # supply exactly that.
    #: Slots that ONLY an open question asks for. A sentence supplying one of these is
    #: answering us and nothing else — nobody opens a conversation with "from my wallet".
    #: `amount`, `payee`, `merchant` and `txn_ref` are deliberately NOT here: half the
    #: taxonomy uses them, so "refund 250" supplies an amount while plainly being a refund.
    #: Those go through the continuation path below, which already weighs a clear new request
    #: against the open one — this fast path skipped that and swallowed them.
    _ANSWER_ONLY = {"pay_mode", "when", "when_at", "bill", "split_type", "participants",
                    "ways"}

    _frame = conversation.get(session)
    # A CORRECTION of the payee kind: "Send to Alex" -> "No, to a merchant". Read on its own
    # the reply is vague, so it was held open and "How much to Alex?" was asked three times
    # while the customer kept saying merchant. "no / sorry / actually" + the other kind
    # replaces the open payment with the one they asked for.
    if (_frame and _frame.intent in (Intent.SEND_MONEY.value, Intent.PAY_MERCHANT.value)
            and _CORRECTION.match(said or "")):
        _want = (Intent.PAY_MERCHANT.value if _CORR_MERCHANT.search(said)
                 else Intent.SEND_MONEY.value if _CORR_PERSON.search(said) else None)
        if _want and _want != _frame.intent:
            a.step("understand", "correction",
                   f"corrects the open {_frame.intent.replace('_', ' ')} — the customer wants "
                   f"{_want.replace('_', ' ')} instead; dropping the old one", tag="rule")
            conversation.clear(session)
            conversation.forget_subject(session, "payee")
            conversation.forget_subject(session, "merchant")
            a.intent, a.model, a.confidence = _want, "correction", 1.0
            _frame = None
    # A saved person's name on its own, with no question open: "Alex". Read cold it scored
    # split_payment 66% and asked "Which payment should I split?" — but with nothing asked,
    # a name from your own list means you want to pay them.
    elif (not _frame and _ex.get("payee") and len(said.split()) <= 3
          and not _is_question(said) and not _is_instruction(said)
          and not _has_transfer_verb(said) and not _has_split_verb(said)
          and _only_names(said, _ex.get("payee"))):
        a.step("understand", "bare.name",
               f"just a name from the customer's own list ({_ex.get('payee')}) and nothing was "
               f"asked — starting a payment to them", tag="rule")
        a.intent, a.model, a.confidence = Intent.SEND_MONEY.value, "bare-name", 1.0
    if _frame:
        # The slot we named wins; failing that, ANY of these slots counts, because none of
        # them is a way to open a conversation. The exact-match rule was too narrow and it
        # broke the same flow in Hindi that worked in English: "बिल भरना है" asked which
        # payment method, the customer answered "Home electricity" — getting ahead of the
        # question rather than answering it — and because `bill` was not `pay_mode` the
        # sentence went to the classifier, scored out_of_scope at 73%, and was refused. The
        # customer had named a biller on their own list while a bill payment was open.
        # The widened half stands down when the sentence carries a verb of its own. Without
        # that it swallows real requests: "split my internet bill", said while a bill
        # payment was waiting on a payment method, supplies `bill` and was read as answering
        # that question — so the split was lost and the customer was asked again where the
        # money for the OTHER bill should come from. A sentence naming an action is a new
        # request however many of our slots it happens to fill.
        _own_verb = _has_transfer_verb(said) or _has_split_verb(said)
        _slot = (_frame.asked_for if _frame.asked_for in _ANSWER_ONLY
                 and _ex.get(_frame.asked_for) not in (None, "", [])
                 else None if _own_verb
                 else next((k for k in _ANSWER_ONLY
                            if _ex.get(k) not in (None, "", [])
                            and k in _SLOTS_OF.get(_frame.intent, _ANSWER_ONLY)), None))
        _answer = _ex.get(_slot) if _slot else None
        # A QUESTION is not an answer. "what's my balance", asked while "who are you paying?"
        # was open, supplied pay_mode=WALLET ("balance") and was swallowed as the answer.
        if _answer not in (None, "", []) and _is_question(said):
            a.step("understand", "new.question",
                   "this is a question of its own, not an answer to the one we asked", tag="rule")
            _answer = None
        if _answer not in (None, "", []):
            merged = dict(_frame.slots)
            # Naming ONE bill narrows a selection of several. The frame was holding both of
            # the customer's bills and waiting on a payment method; "Home electricity"
            # supplies a bill, and merging it while keeping `bill_ids` left the handler
            # still paying both — "2 bills for AED 667.00. Where from?" in answer to a
            # sentence that named one of them. A reply that names a member of a set the
            # question offered is choosing from it, not adding to it.
            if _ex.get("bill") and _ex.get("bill") != _frame.slots.get("bill"):
                merged.pop("bill_ids", None)
            # Take EVERYTHING the reply carried, not just the slot we named. "On Friday"
            # yields both `when` and the absolute `when_at` derived from it; copying only
            # `when` left the handler with a date it could not act on, so it asked the same
            # question again — which is exactly what a customer reads as not being listened to.
            merged.update({k: v for k, v in _ex.items() if v not in (None, "", [])})
            a.step("understand", "answered",
                   (f"we asked for {_frame.asked_for} and this supplies it"
                    if _slot == _frame.asked_for else
                    f"we asked for {_frame.asked_for} and this supplies {_slot} instead, "
                    f"which is still an answer to the open "
                    f"{_frame.intent.replace('_', ' ')}")
                   + f" — reading it as the answer rather than as a new "
                     f"{a.intent.replace('_', ' ')} ({a.confidence:.0%})", tag="rule")
            a.intent, a.confidence, a.model = _frame.intent, 1.0, "answered"
            s_obj = slotlib.Slots(**{k: v for k, v in merged.items()
                                     if k in slotlib.Slots.__annotations__})
            a.slots = s_obj.as_dict()
            handler = HANDLERS.get(Intent(a.intent))
            out = handler(a, s_obj, said) if handler else a
            _remember(session, out, s_obj)
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)

    # A bare NAME in reply to "which merchant?" / "who to?" is the answer, however sure the
    # model is of something else: "mcloths" after "Which merchant?" scored offers_nearby at
    # 80% and listed the shop's offer instead of paying it. Narrow: the question must be for
    # that slot, the reply must fill it, and be a few words with no verb and no question —
    # "refund mclothes" or "what offers does mclothes have" still go to the classifier.
    _named = bool(_frame and _frame.asked_for in ("merchant", "payee")
                  and _ex.get(_frame.asked_for) and len(said.split()) <= 3
                  and not _is_question(said) and not _is_instruction(said))
    if a.confidence < CONFIDENT or _is_pure_slot(said, _ex) or _named:
        carried = conversation.merge(session, _ex)
        if not carried:
            open_frame = conversation.get(session)

            # Weak signal, nothing usable in it, and a question already open. Restating the
            # open question beats starting a new half-understood task: in the live transcript
            # "I want to send money to merchant" (43%) wiped an in-progress merchant payment,
            # and the customer's next answer went to the wrong frame.
            # ...but NOT a real question or a request to go somewhere: "Can you tell me how your
            # transactions work" and "Go to profile" each got "How much?" again in the
            # regression sessions. And never more than twice in a row — a third vague reply
            # means the customer has moved on, so the question is dropped.
            _held = conversation.subject(session, "held_count") or 0
            if (open_frame and open_frame.intent != a.intent
                    and (_is_question(said) or _NAV.search(said or ""))):
                a.step("understand", "not.held",
                       f"a question about {open_frame.intent.replace('_', ' ')} is open, but this is "
                       f"a question or a request of its own — not re-asking", tag="rule")
                open_frame = None
            elif open_frame and open_frame.intent != a.intent and _held >= 2:
                conversation.clear(session)
                conversation.forget_subject(session, "held_count")
                a.step("understand", "dropped",
                       f"third unclear reply to the open {open_frame.intent.replace('_', ' ')} "
                       f"question — dropping it rather than asking a fourth time", tag="rule")
                a.intent, a.model, a.confidence = Intent.GREETING_HELP.value, "start-over", 1.0
                a.say = _ph(a, "start_over", _tr(a, "start_over", "Let's start again — what would you like to do?"))
                a.outcome = "NEED_INFO"
                a.options = []
                a.ms = int((time.time() - t0) * 1000)
                return _finish(a)
            if open_frame and open_frame.intent != a.intent:
                conversation.note_subject(session, "held_count", _held + 1)
                a.intent = open_frame.intent
                a.model = "held-open"
                a.step("understand", "hold",
                       f"too vague to act on, and a question about "
                       f"{open_frame.intent.replace('_', ' ')} is still open — asking it again "
                       f"rather than starting over", tag="guard")
                s_obj = slotlib.Slots(**{k: v for k, v in open_frame.slots.items()
                                         if k in slotlib.Slots.__annotations__})
                a.slots = s_obj.as_dict()
                handler = HANDLERS.get(Intent(a.intent))
                out = handler(a, s_obj, said) if handler else a
                _remember(session, out, s_obj)
                out.ms = int((time.time() - t0) * 1000)
                return _finish(out)
        if carried:
            intent, merged = carried
            a.intent, a.confidence, a.model = intent, 1.0, "continuation"
            a.step("understand", "continue",
                   f"weak on its own ({pred.get('intent')} {pred.get('confidence', 0):.0%}) but "
                   f"it answers the open question about {intent.replace('_', ' ')} — "
                   + ", ".join(f"{k}={v}" for k, v in merged.items()), tag="rule")
            s_obj = slotlib.Slots(**{k: v for k, v in merged.items()
                                     if k in slotlib.Slots.__annotations__})
            a.slots = s_obj.as_dict()
            handler = HANDLERS.get(Intent(a.intent))
            out = handler(a, s_obj, said) if handler else a
            _remember(session, out, s_obj)
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)

    if a.intent == Intent.OUT_OF_SCOPE.value:
        # A mishearing or a mash while a question is open is re-asked, not refused — the
        # garbled voice reply to "how much?" was refused before this check ever ran.
        _r = _reask_if_open(a, session, said, slotlib.extract(said, "").as_dict(), t0)
        if _r is not None:
            return _r
        # "and what about food?" is out of scope read on its own, and in scope read after a
        # spend breakdown. The refusal returns immediately, so the thread has to be given its
        # chance here — the LAST place before the wall goes up.
        if conversation.topic(session) == Intent.SPEND_ANALYSIS.value and _sticks_to_spend(
                said, a.intent, a.confidence, slotlib.extract(said, "")):
            a.step("understand", "stay.spend",
                   "out of scope on its own, but we are mid-thread about spending and this "
                   "names a category — answering about the breakdown rather than refusing",
                   tag="guard")
            a.intent, a.model = Intent.SPEND_ANALYSIS.value, "sticky"
            a.confidence = max(a.confidence, CONFIDENT)
            _s2 = slotlib.extract(said, a.intent)
            out = h_spend_analysis(a, _s2, said)
            _remember(session, out, _s2)
            out.ms = int((time.time() - t0) * 1000)
            return _finish(out)
        a.say = _ph(a, "refuse_scope", _tr(a, "refusal", guard.REFUSAL))
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    # Laya's second opinion. Advisory only: it can lower our confidence and make us ask, and
    # that is the entire extent of its authority.
    # Only worth its ~1.5 s when the first model is unsure. It is advisory — it can lower our
    # confidence and make us ask, nothing more — so on a 97% read it was adding a second and
    # a half to every turn for a vote that could not change the outcome.
    _so_below = float(os.environ.get("CX_SECOND_OPINION_BELOW", "0.90"))
    if want_second_opinion and a.intent == Intent.OUT_OF_SCOPE.value:
        # Declining needs no second vote: whatever Laya said, we would not act. Seen in the
        # regression sessions: "what is 2+2" spent 1.9 s of CPU to reach the same refusal.
        want_second_opinion = False
        a.step("understand", "laya.skipped",
               "the model says this is outside what we do — declining without a second "
               "opinion (it could not make us act)", tag="laya")
    if want_second_opinion and a.confidence >= _so_below:
        want_second_opinion = False
        a.step("understand", "laya.skipped",
               f"first model is {a.confidence:.0%} sure (≥ {_so_below:.0%}) — second opinion "
               f"not needed", tag="laya")
    if want_second_opinion:
        op = client.second_opinion(said, a.intent)
        a.advisory = op
        if op.get("available"):
            if op.get("agrees"):
                a.step("understand", "laya.agrees",
                       f"an independently-trained model also reads this as "
                       f"{op.get('topic')} ({op.get('confidence', 0):.0%}, {op.get('ms', 0)}ms)",
                       tag="laya")
            elif op.get("actionable"):
                # Only a CONFIDENT disagreement lowers our confidence. An unsure second
                # opinion disagreeing says more about the model than about the sentence.
                a.confidence = min(a.confidence, CONFIDENT - 0.01)
                a.step("understand", "laya.disagrees",
                       f"the second opinion reads this as {op.get('topic')} "
                       f"({op.get('confidence', 0):.0%}) where we read {op.get('expected')} — "
                       f"two models disagreeing means it is ambiguous, so I will ask",
                       tag="laya")
            else:
                a.step("understand", "laya.unsure",
                       f"second opinion leans {op.get('topic')} but only at "
                       f"{op.get('confidence', 0):.0%} — not confident enough to act on",
                       tag="laya")
        else:
            a.step("understand", "laya.absent",
                   f"second opinion unavailable ({op.get('reason', 'unknown')}) — proceeding "
                   f"on the policy table alone", tag="laya")

    # ACTION GATE. Some intents DO something the moment they are read: raise a recall,
    # hand the customer to support, open a recharge. Read weakly by the model, they ask
    # first. From the regression sessions: "Lang changed language to Haryanvi" (voice) at 43%
    # raised a recall on the customer's last payment and opened a support ticket; "I need to
    # ask money from someone" at 69% handed off to support; "I want to buy" at 31% opened a
    # phone recharge. A rule, a pick or an answer to our own question is not gated — only the
    # model's own weak reading.
    # A handoff needs the customer to ask for a person. "I need to ask money from someone" was
    # handed to support at 82% (v21) — nothing in it asks for help from us. Without a support
    # word, a handoff reading is treated as weak, so the gate below asks first.
    if (a.intent == Intent.HUMAN_HANDOFF.value and str(a.model).startswith("openwallet-cx-intent")
            and not _SUPPORT_WORD.search(said or "")):
        a.confidence = min(a.confidence, CONFIDENT_FOR_MONEY - 0.01)
    if a.intent in _ASK_FIRST and a.confidence < CONFIDENT_FOR_MONEY:
        # garbage typed while we wait for an amount is a re-ask, not "connect you to support?"
        _r = _reask_if_open(a, session, said, slotlib.extract(said, "").as_dict(), t0)
        if _r is not None:
            return _r
    if (a.intent in _ASK_FIRST and a.confidence < CONFIDENT_FOR_MONEY
            and (str(a.model).startswith("openwallet-cx-intent") or a.model == "nearest")):
        q, yes = _ASK_FIRST[a.intent]
        a.step("decide", "ask.first",
               f"read as {a.intent.replace('_', ' ')} at only {a.confidence:.0%} — that would act "
               f"straight away, so asking first", tag="guard")
        a.say = _tr(a, "ask_first_" + a.intent, q)
        a.outcome = "NEED_CONFIRMATION"
        a.options = [{"kind": "ask_text", "label": "Yes", "text": yes},
                     {"kind": "ask_text", "label": "No, something else", "text": "something else"}]
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    # ---- lane 2: RESOLVE -----------------------------------------------------------
    s = slotlib.extract(said, a.intent)

    # Carry the SUBJECT forward. Only gaps are filled — anything named in this utterance
    # wins, always. This is what makes "split kardo" mean "split the bill we were just
    # talking about" instead of starting from nothing, and it is the difference between a
    # conversation and a sequence of unrelated requests.
    _carried = []
    # Is this sentence reaching back past the current topic to something started earlier?
    # The single-subject memory below only remembers the LAST bill, payee and merchant, so
    # after "pay electricity" -> "split it" -> "my offers" -> "send Asad 50" it has been
    # overwritten twice and "electricity wala kya hua" has nothing to attach to. The task
    # ledger keeps each of those separately and resolves which one is meant by name.
    # ...unless the live topic is a better referent than any older task. "is that more than
    # before?" points back with "that", and the ledger's anaphora fallback happily attached
    # it to the one still-open payment — carrying `payee=Alex` forward, which then made the
    # spend-stickiness rule below stand down, because a sentence naming a payee is a new
    # task. The result: ask for a breakdown, ask whether it went up, get a list of payments
    # to Alex. Mid-thread, "that" means the thing we were just talking about.
    _task = None
    if not (conversation.topic(session) == Intent.SPEND_ANALYSIS.value
            and _sticks_to_spend(said, a.intent, a.confidence, s)):
        _task = conversation.find_task(session, said)
    _other_person = _named_unknown_payee(said, s)
    if _other_person:
        a.step("understand", "payee.named",
               f"names {_other_person}, who is not on the customer's list — nothing about an "
               f"earlier payee is carried into this", tag="guard")
        _task = None
    if _task:
        for _k in ("bill", "payee", "merchant", "ref"):
            _tv = _task.key.get(_k)
            _slot = "txn_ref" if _k == "ref" else _k
            if _tv and getattr(s, _slot, None) in (None, "", []):
                setattr(s, _slot, _tv)
                _carried.append(f"{_slot}={_tv}")
        a.recalled = _task
        a.step("understand", "recall",
               f"this is about “{_task.label}” from earlier in the conversation "
               f"({_task.status})", tag="rule")

    for _k in ("bill", "txn_ref", "payee", "merchant"):
        if _other_person and _k in ("payee", "merchant"):
            continue
        if getattr(s, _k, None) in (None, "", []):
            _v = conversation.subject(session, _k)
            if _v not in (None, "", []):
                setattr(s, _k, _v)
                _carried.append(f"{_k}={_v}")
    if _carried:
        a.step("understand", "remembered",
               "carried over from earlier in this conversation: " + ", ".join(_carried),
               tag="rule")
    a.slots = s.as_dict()
    spec = SPECS[Intent(a.intent)]
    a.step("resolve", "slots", ", ".join(f"{k}={v}" for k, v in a.slots.items()) or "none")

    # A breakdown is a thread like any other. Asked "was it more than the previous one?" or
    # "and food?" straight after one, the classifier reads them as something else entirely —
    # and the scope guards below turn that into a refusal and return, so this has to be
    # decided BEFORE them or the thread is lost to "I can only help with your payments".
    # A follow-up with nothing to follow. "is that more than before?" and "and healthcare?"
    # only mean something against a thread, and with no thread open they mean nothing — so
    # the honest answer is to ask, not to find a subject for them.
    #
    # Left alone, the classifier obliges: it read "is that more than before?" as transaction
    # history and the handler went looking for a payee in it, answering "I don't have anyone
    # called Than on your list". "and healthcare?" listed every unpaid bill. Both sound like
    # an assistant that heard a different question, which is worse than one that asks.
    #
    # The test is the same residue test the stickiness rules use, so a sentence carrying any
    # substance of its own — an amount, a payee, a merchant, a transfer verb — is untouched.
    if (not conversation.topic(session) and _dangling_follow_up(said, a.intent, s)
            and not _DISPUTE.search(_flat(said)) and not any(w in _flat(said) for w in _DISPUTE_INTL)):
        a.step("understand", "no.thread",
               "this only means something as a follow-up, and there is no earlier answer in "
               "this conversation for it to follow — asking rather than guessing a subject",
               tag="guard")
        a.intent, a.model = Intent.GREETING_HELP.value, "no-thread"
        a.confidence = 1.0
        a.say = _tr(a, "no_thread",
                    "I'm not sure what to compare that with yet — we haven't looked at "
                    "anything together in this conversation. Ask me about your spending or "
                    "your payments and I'll take it from there.")
        a.outcome = "NEED_INFO"
        a.options = []
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    if conversation.topic(session) == Intent.SPEND_ANALYSIS.value and _sticks_to_spend(
            said, a.intent, a.confidence, s):
        a.step("understand", "stay.spend",
               f"read as {a.intent.replace('_', ' ')} ({a.confidence:.0%}) but it names no "
               f"amount, payee or merchant and we are mid-thread about spending — staying "
               f"with the breakdown rather than answering about a payment", tag="guard")
        a.intent, a.model = Intent.SPEND_ANALYSIS.value, "sticky"
        a.confidence = max(a.confidence, CONFIDENT)
        _s2 = slotlib.extract(said, a.intent)
        _win = spendmod.follow_up_period(conversation.subject(session, "spend_q") or "", said)
        if _win:
            a.step("understand", "window.before",
                   f"“{said.strip()}” straight after a breakdown of the same span means the "
                   f"window before it — {spendmod.period_label(_win)}", tag="rule")
        out = h_spend_analysis(a, _s2, said, period=_win)
        _remember(session, out, _s2)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    # A money-moving intent with NEITHER an amount NOR anybody to pay is not a payment
    # instruction, whatever the classifier scored. "Open a bank account for me" and "delete
    # my data" both landed in send_money; they carry no amount and name no payee, and a send
    # with neither is not a send. Cheap, principled, and it closes the only leaks that were
    # reaching a money-moving lane.
    # A split is in MONEY_MOVING — it gets the same confidence bar — but the test below is
    # about SENDING. "Split the bill" and "we were four people, divide it up" name no amount,
    # no payee and no merchant, and are still complete requests: a split's subject is the
    # customer's own last merchant payment, which we look up rather than being told. Applying
    # the send-shaped test to them refused eight perfectly good split requests across all
    # four languages, which is how this was caught.
    # A short reply we cannot read, while we are waiting on an answer, is a MISHEARING, not
    # an out-of-scope request. "Alex" heard as "because" after "who are you splitting the
    # AED 80 with?" scored split_payment, named nobody, and the split guard below refused it —
    # throwing away the payment we had already found. Ask the same question again, with the
    # same options. A real question or a new instruction still goes on as normal.
    _r = _reask_if_open(a, session, said, _ex, t0)
    if _r is not None:
        return _r

    if Intent(a.intent) in (Intent.BILL_PAY, Intent.BILL_SCHEDULE):
        # Same shape as the split test above, and for the same reason: a bill payment's
        # subject is the customer's OWN saved billers, looked up rather than stated.
        if not (s.bill or s.amount or s.when_at or s.pay_mode or _has_bill_verb(said)):
            # An incomplete INSTRUCTION is not an out-of-scope request. A bare "pay"
            # classifies as bill_pay at 89% and was refused with "I can only help with your
            # OpenWallet payments" — said to somebody who had just asked to pay something. The
            # guard is right that there is no bill here; it was wrong about what to do with
            # that. "send", "transfer" and "split" all ask on their own; "pay" was the only
            # one that turned a missing object into a refusal.
            if _has_transfer_verb(said) or _has_bill_word(said):
                a.step("understand", "incomplete",
                       "a payment instruction with nothing to pay yet — asking what, "
                       "rather than declining a request we do handle", tag="rule")
                a.intent = Intent.GREETING_HELP.value
                a.model = "incomplete"
                a.confidence = 1.0
                a.say = _tr(a, "pay_what",
                            "Happy to. Is this a bill, someone on your list, or a shop "
                            "you're standing in?")
                a.outcome = "NEED_INFO"
                a.options = [
                    {"kind": "ask_text", "label": "A bill", "text": "pay my bill"},
                    {"kind": "ask_text", "label": "Someone", "text": "send money"},
                    {"kind": "ask_text", "label": "A shop", "text": "pay a merchant"},
                ]
                # Remember that we asked. Without this the question was rhetorical: the
                # customer answered "a shop" and the next turn had no idea a question was
                # open, so it classified two words on their own and refused them. Asking
                # something and then not holding the question is worse than not asking.
                conversation.hold(session, Intent.GREETING_HELP.value, {}, "pay_kind")
                # And remember the MENU separately from the question. The frame is cleared
                # the moment the next turn resolves — so after "pay" → "bill" → (the biller
                # is down), saying "a shop" found no open question and was refused. The
                # customer was picking the third option we had just offered them. An offer
                # outlives the first answer to it.
                conversation.note_subject(session, "offered_pay_kind", 1)
                conversation.note_lang(session, a.lang)
                a.ms = int((time.time() - t0) * 1000)
                return _finish(a)
            a.step("understand", "scope.nobill",
                   "classified as a bill payment but it names no biller, no amount, no time "
                   "and no billing word — declining rather than guessing at a bill to pay",
                   tag="guard")
            a.intent = Intent.OUT_OF_SCOPE.value
            a.say = _ph(a, "refuse_scope", guard.REFUSAL)
            a.outcome = None
            a.ms = int((time.time() - t0) * 1000)
            return a
    elif Intent(a.intent) is Intent.SPLIT_PAYMENT:
        if not (s.participants or s.ways or s.split_type or s.amount
                or _has_split_verb(said)):
            a.step("understand", "scope.nosplit",
                   "classified as a split but it names nobody, no number of ways, no way of "
                   "dividing and no splitting word — declining rather than guessing at a "
                   "bill to divide", tag="guard")
            a.intent = Intent.OUT_OF_SCOPE.value
            a.say = _ph(a, "refuse_scope", guard.REFUSAL)
            a.outcome = None
            a.ms = int((time.time() - t0) * 1000)
            return a
    elif (Intent(a.intent) in MONEY_MOVING
            and Intent(a.intent) is not Intent.SPLIT_PAYMENT
            and _has_split_verb(said) and not _has_transfer_verb(said)
            and not (s.amount or s.payee or s.merchant)):
        # The word in the sentence is DIVIDE, and the classifier read SEND. The bare Arabic
        # imperative `قسّمها` scores send_money at 89% — there is very little of it to go on
        # — and the guard below then refused it outright, because a send with no amount and
        # no payee is not a send. Both readings are wrong and the sentence says which:
        # splitting is not sending, and a split's subject is the bill already on screen.
        #
        # Same asymmetry as the other direction rules. Reading a split as a send points
        # money at a person; reading it as a split at worst asks who to divide it with.
        a.step("understand", "direction",
               "the verb here is to divide, not to send — reading it as a split of what is "
               "already on screen rather than as a payment", tag="rule")
        a.intent, a.model = Intent.SPLIT_PAYMENT.value, "direction-rule"
        a.confidence = max(a.confidence, CONFIDENT)
        out = h_split(a, s, said)
        _remember(session, out, s)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)
    elif Intent(a.intent) in MONEY_MOVING and not (
            s.amount or s.payee or s.merchant or _refers_to_someone(said)
            or _has_transfer_verb(said)):
        a.step("understand", "scope.nomoney",
               "classified as a payment but it names no amount, no payee and no merchant — "
               "declining rather than opening a payment flow", tag="guard")
        a.intent = Intent.OUT_OF_SCOPE.value
        a.say = _ph(a, "refuse_scope", guard.REFUSAL)
        a.outcome = None
        a.ms = int((time.time() - t0) * 1000)
        return a

    # A "split" that says nothing about splitting. Misheard speech ("same 200 the rounds to
    # Alex") reached the model with its verb gone and scored split_payment at 59%. A split
    # sentence names a way of dividing; this one carries an amount and one of the customer's
    # own payees after "to" — the shape of a transfer. Reading it as one is safe: a transfer
    # still stops at the confirmation card and the PIN.
    if (a.intent == Intent.SPLIT_PAYMENT.value and not _SPLIT_WORDS.search(said or "")
            and s.amount and s.payee and re.search(r"\bto\s+\w+", said or "", re.I)):
        a.step("understand", "not.a.split",
               f"read as split ({a.confidence:.0%}) but it names no way of dividing — it has an "
               f"amount and a payee after 'to', which is a transfer", tag="rule")
        a.intent, a.model = Intent.SEND_MONEY.value, "transfer-shape"
        a.confidence = max(a.confidence, CONFIDENT)
        s = slotlib.extract(said, a.intent)
        a.slots = s.as_dict()

    # "agar mai pakistan 1000 aed bhejunga to kitna jaega" — how much ARRIVES — reached v23
    # as time_query at 54% and was answered with a delivery window. A sentence that names a
    # corridor and an amount and asks HOW MUCH, without asking how LONG, is a quote.
    if (a.intent != Intent.FEES_QUERY.value and a.confidence < CONFIDENT
            and s.corridor and s.corridor != "AE-AE" and s.amount
            and _HOW_MUCH.search(said or "") and not _HOW_LONG.search(said or "")):
        a.step("understand", "corridor.quote",
               f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it names "
               f"{s.corridor} and an amount and asks how much — a fee quote", tag="rule")
        a.intent, a.model = Intent.FEES_QUERY.value, "corridor-quote"
        a.confidence = max(a.confidence, CONFIDENT)
    # ...and one that names a corridor and asks how LONG is the delivery window ("pakistan
    # transfer kitni der mein jata hai" read as spend_rank at 32%).
    elif (a.intent != Intent.TIME_QUERY.value and a.confidence < CONFIDENT
            and s.corridor and s.corridor != "AE-AE" and _HOW_LONG.search(said or "")
            and not _HOW_MUCH.search(said or "")):
        a.step("understand", "corridor.time",
               f"read as {a.intent.replace('_', ' ')} at {a.confidence:.0%}, but it names "
               f"{s.corridor} and asks how long — the delivery window", tag="rule")
        a.intent, a.model = Intent.TIME_QUERY.value, "corridor-time"
        a.confidence = max(a.confidence, CONFIDENT)

    topic = conversation.topic(session)
    if topic and Intent(topic) in TXN_DOMAIN and _sticks_to_txn_domain(
            said, a.intent, a.confidence, s):
        # A bare follow-up ("show me more") after a LIST means more of the list; after a
        # single payment it means the conversation is still about that payment.
        back = (Intent.TXN_HISTORY.value
                if topic == Intent.TXN_HISTORY.value or _is_follow_up(said) else topic)
        a.step("understand", "stay",
               f"read as {a.intent.replace('_', ' ')} ({a.confidence:.0%}) but it names no "
               f"amount, payee or merchant and we are mid-thread about "
               f"{topic.replace('_', ' ')} — staying there rather than switching flow",
               tag="guard")
        a.intent, a.model = back, "sticky"
        a.confidence = max(a.confidence, CONFIDENT)
        s = slotlib.extract(said, a.intent)
        a.slots = s.as_dict()

    # A data answer needs a reason in the sentence. "and" scored txn_history at 85% and
    # "and how are you" bill_due at 90% — confident, and about nothing — and the customer was
    # shown their payments and bills. An answer built from customer data requires some sign
    # the sentence is ABOUT that data: a payment word, an amount, a payee, a biller, a period.
    # A real follow-up inside a thread ("and before that") is the exception.
    if (Intent(a.intent) in DATA_INTENTS and a.model not in ("sticky", "held-open", "answered")
            and not _has_evidence(said, slotlib.extract(said, ""))   # the SENTENCE, not memory
            and not (topic and (_is_follow_up(said) or _COMPARATIVE.search(said.lower())
                                or _points_back(said)
                                or (spendmod.period_named(said) and _only_a_period(" ".join(said.lower().split())))))):
        fragment = len(re.findall(r"[\w\u0600-\u06FF\u0900-\u097F]+", said)) <= 2
        a.step("decide", "scope.noevidence",
               f"read as {a.intent.replace('_', ' ')} ({a.confidence:.0%}) but the sentence says "
               f"nothing about payments, bills or money — not answering from your data", tag="guard")
        a.intent = Intent.GREETING_HELP.value if fragment else Intent.OUT_OF_SCOPE.value
        a.say = (_ph(a, "didnt_catch", _tr(a, "didnt_catch", "Sorry, I didn't catch that — what would you like to do?"))
                 if fragment else _ph(a, "refuse_scope", _tr(a, "refusal", guard.REFUSAL)))
        a.outcome = "NEED_INFO" if fragment else "REFUSED"
        a.options, a.rows = [], []
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    handler = HANDLERS.get(Intent(a.intent))
    out = handler(a, s, said) if handler else a
    _remember(session, out, s)
    out.ms = int((time.time() - t0) * 1000)
    return _finish(out)


#: Intents answered from the customer's own data. Each needs evidence in the sentence.
DATA_INTENTS = frozenset({
    Intent.TXN_HISTORY, Intent.TXN_STATUS, Intent.TXN_FAILED, Intent.TXN_ON_HOLD,
    Intent.DEBITED_NOT_CREDITED, Intent.REFUND_STATUS, Intent.BILL_DUE, Intent.SPEND_ANALYSIS,
    Intent.SPEND_RANK, Intent.SPLIT_STATUS, Intent.BENEFICIARY_QUERY, Intent.LIMITS_QUERY})
_EVIDENCE = re.compile(
    r"\b(?:pa(?:y|id|ying|yment|yments)|transactions?|txns?|transfers?|transferred|sent|send|sending|"
    r"received?|money|cash|paise|paisa|pese|rupees?|dirhams?|dhs|aed|bills?|due|owe[sd]?|spend|spent|"
    r"spending|kharch\w*|balance|history|statement|refund\w*|split\w*|merchants?|shops?|stores?|"
    r"limits?|fees?|charges?|status|pending|failed|stuck|hold|credited|debited|deducted|wallet|"
    r"recharge|beneficiar\w*|recipients?|bhej\w*|diya|mila|"
    r"aaya|gaya|pahunch\w*|transaction|dewa|sewa|fewa|etisalat|du|salik|internet|electricity|water|"
    r"go(?:ne)?\s+through|went\s+through|arriv\w*|reach\w*|land\w*|deliver\w*|receiv\w*|"
    r"bheja|bhejna|mile|openwallet|my\s+list|on\s+my\s+list|saved|contacts?|added|"
    # what happened to a payment, in the words people actually use
    r"review|processing|compliance|rejected|declined|bounced|error|approv\w*|reversed|"
    r"atk\w*|fail\w*|hisaa?b|lenden|"
    # limits ("can i do 5000 in one shot", "am i capped")
    r"capped|cap|maximum|max|allowance|one\s+shot|at\s+once|"
    r"utilit\w*|outstanding|top\s*up|commission|settle\w*|chipped\s+in)\b|"
    # a sum of money: a number of 3+ digits that is not a year
    r"(?<![\d.,])(?!(?:19|20)\d\d\b)\d{3,}(?![\d])|"
    r"पेंडिंग|रिफंड|क्रेडिट|लेनदेन|लेन-देन|स्टेटमेंट|हिसाब|अटका|अटकी|फेल|कमीशन|लिमिट|सीमा|लिस्ट|"
    r"استرداد|الاسترداد|استرجاع|معلق|فشل|فشلت|الحد|أموال|الأموال|المال|فلوس|كشف|دفعت|"
    r"عملي|العملية|عمليات|فاشل|القسمة|مقسوم|حوّل|حولت|حولي|حوالة|الحوالة|الخصم|خصم|انخصم|تحقق|إيقاف|موقوف|مستفيد|"
    r"مصاري|مصاريف|مصروف|صرفت|حساب|أحول|احول|بحول|تحويل|شحن|تعبئة|بنك|محفظ|انسحب|انسرق|مديون|"
    r"स्टेटस|ट्रांसफर|भुगतान|प्रोसेस|होल्ड|जाँच|जांच|राशि|कट|प्राप्तकर्ता|लाभार्थी|सूची|"
    r"len-?den|payees?|"
    r"पेमेंट|पैसे|पैसा|बिल|खर्च|ट्रांज|भेज|बैलेंस|दिया|मिला|"
    r"دفع|مدفوع|فاتور|فواتير|رصيد|تحويل|حوال|أرسل|ارسل|مبلغ|درهم|معامل|مستفيد|قائمت", re.I)


def _points_back(said: str) -> bool:
    """"is it done?", "wo pahuncha?" — a short sentence pointing at what we were discussing."""
    t = (said or "").lower()
    return len(t.split()) <= 8 and bool(re.search(r"\b(?:it|that|this|those|them|wo|woh|ye|yeh|uska|unka)\b|ذلك|هذا|वो|यह", t))


def _has_evidence(said: str, s) -> bool:
    # a period alone ("yesterday", "this week") says nothing about money: "who won the match
    # yesterday" listed yesterday's payments
    if any(getattr(s, k, None) for k in ("amount", "payee", "merchant", "txn_ref", "bill")):
        return True
    return bool(_EVIDENCE.search(said or ""))


_SMALL_TALK = re.compile(
    r"^\W*(?:and\s+|so\s+|hey\s+|hi\s+|hello\s+|aur\s+|or\s+|arre\s+)?(?:bhai\s+|yaar\s+)?"
    r"(?:how\s+(?:are|r)\s+(?:you|u)(?:\s+doing)?|how'?s\s+it\s+going|"
    r"what'?s\s+up|wassup|how\s+have\s+you\s+been|kaise\s+ho|kaise\s+h(?:ai)?|kaisa\s+h(?:ai)?|kya\s+haal(?:\s+hai)?|"
    r"sab\s+(?:badhiya|badia|badiya|theek|thik|changa)|good\s+(?:morning|evening|afternoon|night)|"
    r"كيف\s+حالك|شلونك|كيفك|आप\s+कैसे\s+हैं|कैसे\s+हो)"
    r"(?:[\s,.!?؟]+(?:sab\s+(?:badhiya|badia|badiya|theek)|bhai|yaar|sir|dear|buddy|today|"
    r"and\s+you|aur\s+aap|hai|ho)?)*[\s,.!?؟]*$", re.I)


_IDENTITY = re.compile(
    r"^\W*(?:hey\s+|hi\s+|ok\s+|so\s+|and\s+)?"
    r"(?:(?:who|what)\s+(?:are|r)\s+(?:you|u)|what\s+(?:all\s+)?(?:can|do)\s+(?:you|u)\s+do|"
    r"what'?s\s+your\s+name|(?:tum|aap)\s+(?:kaun|kon)\s+(?:ho|hai|h)|(?:tum|aap)\s+kya\s+kar\s+sakte\s+(?:ho|hai)|"
    r"من\s+أنت|ماذا\s+تستطيع\s+أن\s+تفعل|(?:आप|तुम)\s+कौन\s+(?:हो|हैं))"
    r"(?:\s+(?:bro|bhai|yaar|dude|exactly|really|for\s+me|here|again))*[\s.?!؟]*$", re.I)


def _is_small_talk(said: str) -> bool:
    return bool(_SMALL_TALK.match((said or "").strip()))


#: Phrases that only make sense as a continuation of something already on screen. They carry
#: no subject of their own, which is precisely why the classifier cannot be trusted with them:
#: "and before that" is not a sentence about money at all.
_FOLLOW_UP = re.compile(
    r"^(?:and\s+)?(?:what\s+about\s+)?(?:the\s+)?(?:ones?\s+)?"
    r"(?:before\s+that|previous(?:\s+one)?s?|older(?:\s+ones?)?|earlier(?:\s+ones?)?|"
    r"more|the\s+rest|rest|next|next\s+page|go\s+back|scroll\s+back|keep\s+going|"
    r"show\s+me\s+more|anything\s+older|any\s+older(?:\s+ones?)?|what\s+came\s+before)"
    r"[\s.?!]*$",
    re.IGNORECASE)

_FOLLOW_UP_NONLATIN = (
    "उससे पहले", "और दिखाओ", "और पुराने", "पुराने दिखा", "और पीछे", "पहले वाले",
    "usse pehle", "aur dikhao", "aur purane", "purane dikhao", "aur peeche", "pehle wale",
    "وقبل ذلك", "قبل ذلك", "أظهر المزيد", "المزيد", "أقدم", "ارجع أكثر",
)


def _is_follow_up(said: str) -> bool:
    low = " ".join((said or "").lower().split())
    if _FOLLOW_UP.match(low):
        return True
    return any(p in low for p in _FOLLOW_UP_NONLATIN)


#: Comparisons and bare category mentions — the natural way to carry on talking about a
#: breakdown. "was it more than last time" names nothing at all, so without this it was read
#: as a question about one payment and answered with the customer's last transaction.
_COMPARATIVE = re.compile(
    r"\b(more|less|higher|lower|than|compar\w*|versus|vs|increase|decrease|"
    # "up"/"down" only as a level ("is it up?", "gone down", "up or down") — never the verb
    # particle in "settle up", "top up", "divide it up", which made those sentences
    # "a comparison with nothing to compare" and refused them
    r"(?:is|was|it's|its|gone|went|going|been|go)\s+(?:up|down)|up\s+or\s+down|"
    # "before" as a deadline ("before evening", "before 5") is not a comparison
    r"zyada|kam|pehle|pichl\w*|before(?!\s+(?:evening|morning|night|noon|tonight|tomorrow|eod|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|the\s+weekend|\d))|previous|last time)\b", re.I)
#: Matched as WHOLE words (see _has_intl_comparative): "कम" is inside "कमीशन" (commission).
_COMPARATIVE_INTL = ("ज़्यादा", "ज्यादा", "कम", "पिछली", "पिछले", "أكثر", "أقل", "السابق")


def _has_intl_comparative(low: str) -> bool:
    words = set(re.split(r"[\s,.!?؟،]+", low))
    return any(w in words for w in _COMPARATIVE_INTL)


def _sticks_to_spend(said: str, intent: str, confidence: float, s) -> bool:
    """Should this turn stay with the spend breakdown we just gave?

    Same shape, and the same caution, as `_sticks_to_txn_domain`: anything carrying the
    evidence of a NEW task is let through untouched, and only the residue is held. What
    counts as residue here is a comparison or a bare category — "was it more than the
    previous one", "and food?" — which name no amount, no payee and no merchant, and which
    on their own classify as something else entirely.
    """
    if intent == Intent.SPEND_ANALYSIS.value:
        return False                                   # already where it belongs
    low0 = " ".join((said or "").lower().split())
    # Asking for a LIST is asking for a list. "pichle 5 transactions dikhao" contains
    # "pichle", which is in the comparative list for the sake of "pichle se zyada?", and
    # that was enough to hold a plain request for five payments inside the breakdown.
    if _LIST_NOUN.search(low0):
        return False
    # The number in "show me 7 days instead" is a WINDOW, not an amount — but the slot
    # extractor cannot know that, so it files 7 as money and this rule then stood down
    # because "a sentence naming an amount is a new task". A number that belongs to a named
    # period is not a payment.
    _amount_is_period = bool(spendmod.period_named(said) and _only_a_period(low0))
    if (s.amount and not _amount_is_period) or s.payee or s.merchant or s.txn_ref:
        return False                                   # names a new task; let it go
    if _has_transfer_verb(said):
        return False                                   # an explicit instruction
    low = " ".join((said or "").lower().split())
    if _COMPARATIVE.search(low) or _has_intl_comparative(low):
        return True
    if spendmod.focus_from(said):
        return True                                    # "and healthcare?" — a category, alone
    if _is_follow_up(said):
        return True
    # A bare change of window — "and last week?", "pichle hafte?" — is the commonest way to
    # carry on reading a breakdown, and it was the worst-failing family in the corpus (5 of
    # 37 clean). It classified as transaction history, which is a defensible reading of the
    # sentence alone and the wrong one mid-thread: the customer gets a list of five payments
    # where they asked for the same categories over a shorter window.
    #
    # It has to be a BARE change, though. "show me last week's payments" names a window too
    # and really is a history request, so the period words are stripped and the sentence only
    # sticks if nothing else of substance is left.
    if spendmod.period_named(said) and _only_a_period(low):
        return True
    return False


#: Period words plus the connectives a follow-up is made of. Anything outside this set is
#: substance, and substance means a new request.
_PERIOD_RESIDUE = re.compile(
    r"\b(?:and|also|then|what|whats|what's|about|how|much|ok|okay|so|is|it|that|the|"
    r"last|past|previous|this|prev|pichla|pichle|pehle|"
    r"show|see|give|make|change|try|instead|rather|just|only|now|then|"
    r"dikhao|dikha|karo|kardo|batao|"
    r"my|me|for|in|of|a|an|aur|phir|kya|kitna|kitne|mera|meri|mujhe|ka|ki|ke|me|"
    r"\u0914\u0930|\u092b\u093f\u0930|\u0915\u094d\u092f\u093e|\u0915\u093f\u0924\u0928\u093e|\u092e\u0947\u0930\u093e|"
    r"\u0648|\u0645\u0627\u0630\u0627|\u0639\u0646|\u0643\u0645|\u0641\u064a)\b", re.I)


#: Nouns that ask for a LIST of payments rather than a figure about them.
_LIST_NOUN = re.compile(
    r"\b(?:transaction|transactions|txn|txns|statement|history|receipt|receipts|"
    r"ledger|list)\b"
    r"|\u0932\u0947\u0928\u0926\u0947\u0928|\u0938\u094d\u091f\u0947\u091f\u092e\u0947\u0902\u091f|"
    r"\u0645\u0639\u0627\u0645\u0644\u0627\u062a|\u0643\u0634\u0641", re.I)


#: Subjects a question can be about, beyond the spend nouns. Between the two lists, any
#: sentence that names its own topic is excluded from the dangling-follow-up guard.
_SUBJECT_NOUN = re.compile(
    r"\b(?:refund|refunds|limit|limits|fee|fees|offer|offers|balance|statement|history|"
    r"beneficiary|beneficiaries|split|splits|list|wallet|account|"
    r"dikhao|dikha|batao|bata|"
    r"transactions?|payments?|hisaa?b|lenden|"
    r"\u0926\u093f\u0916\u093e\u0913|\u092c\u0924\u093e\u0913)\b|"
    # Devanagari and Arabic WITHOUT \b: a word ending in a vowel sign or virama ("लिस्ट")
    # has no \b after it, so these never matched
    r"लिस्ट|लेनदेन|ट्रांज़?ैक्शन|स्टेटमेंट|हिसाब|पेमेंट|कमीशन|फीस|"
    r"\u0623\u0638\u0647\u0631|\u0642\u0627\u0626\u0645\u0629|\u0631\u0635\u064a\u062f|معاملات|كشف|"
    r"صرفت|مصاريف|مصروف|مطاعم|بقالة|مواصلات|فواتير", re.I)


#: The words a continuation opens with. "and healthcare?" carries on; "my bills" does not.
_CONTINUES = re.compile(
    r"^(?:and|also|then|what about|how about|and what about|ok|okay|so|plus|"
    r"aur|phir|aur kya|to|"
    r"\u0914\u0930|\u092b\u093f\u0930|"
    r"\u0648|\u0648\u0645\u0627\u0630\u0627|\u0645\u0627\u0630\u0627 \u0639\u0646)\b", re.I)


def _only_a_period(low: str) -> bool:
    """Is a window the only thing this sentence says?"""
    rest = low
    for _, words in (spendmod._PERIOD_WORDS_INTL + spendmod._PERIOD_WORDS):
        for w in words:
            rest = rest.replace(w, " ")
    rest = spendmod._SPAN.sub(" ", rest)
    rest = _PERIOD_RESIDUE.sub(" ", rest)
    return not re.sub(r"[^\w\u0900-\u097f\u0600-\u06ff]+", "", rest)


def _dangling_follow_up(said: str, intent: str, s) -> bool:
    """Does this sentence only mean something as a follow-up to an answer we never gave?

    Narrow on purpose. It must carry no subject of its own, and it must be the kind of
    sentence that is *only* ever a continuation: a comparison, a bare category, a bare
    window, or an explicit "and before that". A greeting is not one of those, and neither is
    a short request — "my bills" names its own subject and is answerable cold.
    """
    if s.amount or s.payee or s.merchant or s.txn_ref or s.bill:
        return False
    if _has_transfer_verb(said):
        return False
    low = " ".join((said or "").lower().split())
    if not low or len(low.split()) > 8:
        return False                                   # a long sentence is making its own point
    if _is_yes(low) or _is_no(low):
        return False                                   # handled long before this
    # A sentence that names what it is ABOUT is not a dangling follow-up, whatever else it
    # contains. "pichle 5 transactions dikhao" is a plain request for the last five
    # payments, and it was being refused with "there's nothing to compare that with yet" —
    # because "pichle" (last/previous) is in the comparative list, which is right for
    # "pichle se zyada?" and wrong the moment a subject follows it. Same failure as "my
    # bills" being read as the services category: residue plus a noun is not residue.
    if _has_spend_noun(low) or _SUBJECT_NOUN.search(low):
        return False
    comparative = bool(_COMPARATIVE.search(low)) or _has_intl_comparative(low)
    if comparative or _is_follow_up(said):
        return True                                    # "more than before", "and before that"
    # A bare category or window only dangles when it is OFFERED as a continuation. Said on
    # its own it is a request in its own right, and swallowing those is how "my bills" —
    # which reads as the services category, because a bill is a service — stopped being
    # answerable as the first thing somebody types.
    if not _CONTINUES.match(low):
        return False
    return bool(spendmod.focus_from(said)
                or (spendmod.period_named(said) and _only_a_period(low)))


def _sticks_to_txn_domain(said: str, intent: str, confidence: float, s) -> bool:
    """Should this turn stay in the transaction domain rather than going where the
    classifier points?

    The complaint this answers: ask about your payments, then ask a natural follow-up, and
    the assistant threw you into "who do you want to send money to?". Mid-thread that is not
    a mild misread — it is the assistant losing the plot.

    Deliberately narrow, because the opposite failure is worse. A customer who really does
    want to send money after looking at their history must be able to, so anything carrying
    the evidence of a NEW task — an amount, a payee, a merchant, an explicit transfer verb —
    is let through untouched. What gets held is the residue: short referential phrases, and
    weak classifications that name nothing at all.
    """
    if Intent(intent) in TXN_DOMAIN:
        return False                                   # already where it belongs
    if s.amount or s.payee or s.merchant:
        return False                                   # names a new task; let it go
    if _has_transfer_verb(said) or Intent(intent) is Intent.OUT_OF_SCOPE:
        return False                                   # an explicit instruction, or the wall
    if _is_follow_up(said):
        return True                                    # "and before that" — unambiguous
    return confidence < CONFIDENT                       # weak and subjectless: keep the thread


#: Which intents are worth remembering as a THING THE SESSION IS DOING, and how to label
#: one. Not every answer is a task: "what are my limits" is a question that is over when it
#: is answered, and filling the ledger with those would bury the two or three real ones the
#: customer might come back to.
_TASK_KINDS = {
    "bill_split": "split {bill} with {who}",
    "split_payment": "split {merchant} with {who}",
    "bill_pay": "{bill} bill payment",
    "bill_schedule": "scheduled {bill} payment",
    "send_money": "send {amount} to {who}",
    "pay_merchant": "pay {merchant}",
    "refund_request": "refund for {ref}",
}

#: Outcome -> what state that leaves the task in. Anything else leaves it alone, because an
#: answer that neither advanced nor ended a task should not touch its status.
_TASK_STATUS = {
    "NEED_INFO": "open", "NEED_CONFIRMATION": "open", "OVER_LIMIT": "open",
    "CONFIRMED": "waiting", "SCHEDULED": "waiting", "COLLECTING": "waiting",
    "DONE": "done", "PAID": "done", "SENT": "done", "INFORMED": None,
    "CANCELLED": "cancelled", "FAILED": "failed", "REFUSED": None,
}


def _track_task(session: str, a: Answer, s) -> None:
    """Record what this session is doing, so it can be asked about later.

    This is the ledger behind "alex ne paise bhej diye hai" four topics after the split was
    set up. The frame cannot answer that — it holds one thing and expires in half an hour —
    so the durable record of WHAT EXISTS lives here instead, separately from what is
    currently being filled in.
    """
    tmpl = _TASK_KINDS.get(a.intent)
    if not tmpl:
        return
    status = _TASK_STATUS.get(a.outcome or "", "open")
    if status is None:
        return
    who = (s.payee or "") or ", ".join(
        p.get("name", "") if isinstance(p, dict) else str(p)
        for p in (s.participants or []))[:40]
    label = tmpl.format(bill=s.bill or "the bill", who=who.split()[0] if who else "someone",
                        merchant=s.merchant or "the merchant",
                        amount=f"AED {s.amount:g}" if s.amount else "money",
                        ref=s.txn_ref or "a payment").strip()
    key = {k: v for k, v in (
        ("bill", s.bill), ("payee", who or None), ("merchant", s.merchant),
        ("ref", s.txn_ref), ("amount", str(s.amount) if s.amount else None),
        ("split_id", (a.proposal or {}).get("splitId") or (a.rows or {}).get("splitId")
         if isinstance(a.rows, dict) else (a.proposal or {}).get("splitId")),
    ) if v}
    try:
        conversation.open_task(session, a.intent, label, key, status)
    except Exception:
        pass            # bookkeeping must never break an answer


def _remember(session: str, a: Answer, s) -> None:
    """Hold the frame open while we are waiting for something, drop it once we are not.

    Clearing on completion matters as much as holding: a stray "100" after a finished payment
    must not reopen it.
    """
    _track_task(session, a, s)
    if a.model != "held-open":
        conversation.forget_subject(session, "held_count")   # only CONSECUTIVE re-asks count
    # Which payment the answer was ABOUT. The handlers already resolve one — a refund
    # answer names the transaction it is going to refund — but nothing kept it, so the next
    # turn had no idea what "the details" referred to.
    if a.txn and a.txn.get("ref"):
        conversation.note_subject(session, "txn_ref", a.txn["ref"])

    # An offer stands while it is still the subject. Once the conversation has moved to
    # something else, it has lapsed — and must stop being confirmable.
    #
    # This is the half of the reported bug that survived making "split it" kind-specific.
    # An unanswered split offer outlived two bill-payment turns, because each of those was
    # NEED_INFO and left the offer standing:
    #
    #     split it with alex   -> offer: split that transaction   (never answered)
    #     bill pay karna hai    -> NEED_INFO, offer still standing
    #     Home electricity      -> NEED_INFO, offer still standing
    #     "split it"            -> confirmed the four-turn-old transaction split,
    #                              with an empty reply, instead of splitting the bill
    #
    # It has to run here rather than next to the other `hold_proposal` calls below, because
    # the waiting-on-a-slot branch returns early and would skip it — which is exactly how
    # the offer survived in the first place.
    if not a.confirmed and not a.proposal:
        _standing = conversation.proposal(session)
        if _standing and (_standing.get("intent") or "") != a.intent:
            conversation.hold_proposal(session, None)

    # OVER_LIMIT is still an open question — we asked for a DIFFERENT amount, not for nothing.
    # Clearing the frame here orphaned the customer's next number, so "10000" then "1000" then
    # "150" each started from scratch.
    if a.outcome in ("NEED_INFO", "NEED_CONFIRMATION", "OVER_LIMIT"):
        # If the HANDLER already said what it is waiting for, that wins. This function runs
        # after the handler and used to overwrite it from a send-shaped guess: a bill payment
        # waiting on `pay_mode` was recorded as waiting on `amount`, so the customer's answer
        # ("pay from my wallet") no longer matched the open question and went to the
        # classifier, which opened a merchant payment instead.
        existing = conversation.get(session)
        if existing and existing.intent == a.intent and existing.asked_for:
            conversation.hold(session, a.intent, s.as_dict(), existing.asked_for)
            conversation.note_topic(session, a.intent)
            conversation.note_lang(session, a.lang)
            if a.proposal and not a.confirmed:
                conversation.hold_proposal(session, dict(a.proposal, intent=a.intent))
            return
        asked = None
        if a.outcome == "OVER_LIMIT":
            asked = "amount"            # we have the payee; we need a smaller number
        elif a.proposal:
            # A proposal is COMPLETE — we are waiting for yes or no, not for a missing slot.
            # Without this a split proposal was recorded as still needing an amount (a split
            # names none: the figure comes from the bill), so the idle nudge asked "how much
            # should I send Alex?" about a payment that was fully specified and only
            # awaiting confirmation.
            asked = None
        elif Intent(a.intent) in MONEY_MOVING:
            asked = ("merchant" if not s.merchant and a.intent == Intent.PAY_MERCHANT.value
                     else "payee" if not s.payee and a.intent == Intent.SEND_MONEY.value
                     else "amount" if not s.amount else None)
        elif Intent(a.intent) in NEEDS_TXN:
            asked = "txn_ref"
        held = s.as_dict()
        if a.outcome == "OVER_LIMIT":
            held.pop("amount", None)    # the rejected figure must not come back as the answer
        conversation.hold(session, a.intent, held, asked)
    else:
        conversation.clear(session)
    # The subject survives either way. This is what makes "and before that" mean anything.
    conversation.note_topic(session, a.intent)
    conversation.note_lang(session, a.lang)
    # And so does the OFFER, so a bare "yes" on the next turn still has something to mean.
    if a.proposal and not a.confirmed:
        conversation.hold_proposal(session, dict(a.proposal, intent=a.intent))
    elif a.outcome not in ("NEED_INFO", "NEED_CONFIRMATION", "OVER_LIMIT"):
        conversation.hold_proposal(session, None)



#: People and places a customer refers to without naming. "Same again to my brother" and
#: "I'm standing at the till" are real payment instructions with no roster name and no
#: amount in them, so the no-amount-no-payee rule has to recognise them or it refuses
#: ordinary customers. Deliberately NOT including bare "me"/"my" — "delete my data" would
#: sail straight through.
_REFERENTIAL = (
    # english
    "my brother", "my sister", "my wife", "my husband", "my mother", "my father",
    "my mom", "my mum", "my dad", "my son", "my daughter", "my friend", "my family",
    " him", " her ", " them", "same again", "last time", "again to", "repeat",
    "the shop", "this shop", "the store", "the till", "the counter", "the cafe",
    "coffee place", "coffee shop", "the restaurant", "the pharmacy", "the grocery",
    "the salon", "the merchant", "pay it", "pay here", "checkout",
    # hinglish
    "mere bhai", "meri behen", "meri wife", "meri biwi", "meri maa", "mere papa",
    "dukaan", "dukan", "store par", "shop par", "merchant", "bill bharna",
    # hindi
    "मेरे भाई", "मेरी बहन", "मेरी पत्नी", "मेरी माँ", "मेरी मां", "मेरे पिता",
    "दुकान", "स्टोर", "मर्चेंट", "बिल भरना", "दुकान पर",
    # arabic
    "أخي", "أختي", "زوجتي", "والدتي", "أمي", "والدي",
    "المتجر", "المحل", "التاجر", "البقالة", "المقهى", "الفاتورة",
)


#: Verbs that mean "move money", in every language the model understands.
#:
#: Adding a language to the MODEL is not enough. This guard, the referential list below and
#: the roster/merchant names are all deterministic and were all English-only, so a correctly
#: classified Hindi or Arabic payment request was then refused by our own code —
#: 13 of 15 over-refusals in the first multilingual evaluation.
_TRANSFER_VERBS = {
    # english
    "send", "transfer", "pay", "wire", "remit", "paying", "sending", "transferring",
    # hinglish (latin)
    "bhejo", "bhejna", "bhejne", "bheja", "bhej", "bhejni", "payment", "transferkaro",
    "dena", "dene", "deni", "bharna", "bharni", "karni", "karna",
    # hindi (devanagari)
    "भेजो", "भेजना", "भेजने", "भेज", "भेजनी", "भेजें", "भेजिए", "ट्रांसफर", "पेमेंट",
    "देने", "देना", "दो", "भरना", "भरनी", "रकम",
    # arabic
    "أرسل", "إرسال", "ارسال", "حوّل", "حول", "تحويل", "ادفع", "الدفع", "دفع",
    "تسديد", "سدد", "أدفع",
}


def _has_transfer_verb(text: str) -> bool:
    """Does the sentence actually ask to move money?

    "Send money" names nobody and no amount, but it IS a payment request and the right answer
    is "who, and how much?" — not a refusal. "Open a bank account for me" and "delete my
    data" carry no transfer verb and should be declined.

    Matched by STEM, not by whole word. Exact-token matching meant "paise bhej do" was
    understood and "paise bhejdo" was refused — the same request with a space removed. Hindi
    and Urdu attach the auxiliary to the verb freely ("bhejdo", "bhejdena", "bhejunga"), and
    no list of conjugations ever finishes. A money NOUN counts too: "paise" with any
    send-shaped intent is a payment request whatever the verb looks like.
    """
    low = " ".join((text or "").lower().split())
    toks = slotlib.tokens(low)
    if toks & _TRANSFER_VERBS:
        return True
    # Stems: any token BEGINNING with one of these is the verb, however it was conjugated
    # or joined to its auxiliary.
    if any(t.startswith(st) for t in toks for st in _TRANSFER_STEMS):
        return True
    if toks & _MONEY_NOUNS:
        return True
    return any(st in low for st in _TRANSFER_STEMS_NONLATIN)


#: Verb stems. A token starting with one of these is a transfer verb in some conjugation.
#: Deliberately short and distinctive — "bhej" cannot begin an unrelated word, whereas "de"
#: or "kar" could, so those stay in the exact-match list above.
_TRANSFER_STEMS = ("bhej", "transfer", "remit", "payme", "sendm")

#: The same idea for scripts where prefix matching on tokens is unreliable.
_TRANSFER_STEMS_NONLATIN = ("भेज", "ट्रांसफर", "أرسل", "إرسال", "ارسل", "تحويل", "حول")

#: Words for money itself. With a send-shaped intent these are evidence enough: somebody who
#: says "paise" is talking about moving money, not asking us to fix their internet.
_MONEY_NOUNS = {
    "money", "cash", "funds", "amount", "paise", "paisa", "paisey", "paisa", "rupaye",
    "rupees", "dirham", "dirhams", "aed", "पैसे", "पैसा", "रुपये", "दिरहम",
    "أموال", "مال", "نقود", "درهم",
}



def _refers_to_someone(text: str) -> bool:
    low = " " + " ".join((text or "").lower().split()) + " "
    return any(cue in low for cue in _REFERENTIAL)


def _g(v) -> str:
    """Trim a BigDecimal-ish value for display: 15.0000 -> 15."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


#: Words that carry no meaning of their own once the slot is removed — currency names, units,
#: and the handful of particles people wrap a bare answer in.
#: Words that can trail a bare answer without changing what it is. The strip below removes
#: these and the number; if anything MEANINGFUL survives, the utterance is a new request
#: rather than an answer to our question.
#:
#: The romanised Hinglish half was missing, and it cost a real payment its payee: asked
#: "MClothes par kitna?", the customer replied "100 kardo", "kardo" survived the strip, the
#: sentence stopped looking like a bare answer, and a confident send_money (0.97) displaced
#: the open merchant payment — "Ready to send 100 AED to Alex". The merchant was simply
#: dropped. Four previous bugs in this engine had the same shape: English-only code sitting
#: behind a multilingual model.
#:
#: Intent-bearing verbs are deliberately NOT here. "bhejdo", "send", "pay", "refund" and
#: "split" each say what to DO, so a sentence keeping one of them is a new request and must
#: survive the strip — that is exactly how "refund 250" stays a refund.
_SLOT_FILLER_WORDS = {
    "dirhams", "dirham", "aed", "dhs", "dh", "rupees", "rupee", "inr", "dollars", "dollar",
    "usd", "euros", "euro", "only", "just", "please", "thanks", "ok", "okay", "rs",
    # romanised Hinglish: "do it", politeness, and the particles that cling to a number
    "kardo", "kardo", "karo", "kar", "kr", "krdo", "karde", "kardein", "karna", "krna",
    "karni", "kijiye", "kijie", "dijiye", "dedo", "dede", "do", "de", "dena", "deni",
    "bas", "sirf", "bilkul", "theek", "thik", "thike", "haan", "han", "ji", "bhai",
    "yaar", "na", "hi", "hai", "ka", "ki", "ke", "ko", "wala", "waala", "rupaye",
    "rupaya", "rupay", "paise", "paisa", "ke liye", "liye",
    "दिरहम", "रुपये", "रुपए", "सिर्फ", "बस", "प्लीज",
    "करदो", "करो", "कर", "दो", "देदो", "ठीक", "हाँ", "हां", "जी", "भाई", "पैसे", "का", "की", "के", "को",
    "درهم", "دراهم", "ريال", "فقط", "من", "فضلك", "سوي", "خلاص", "بس",
}


def _is_pure_slot(said: str, extracted: dict) -> bool:
    """Is this utterance NOTHING BUT the answer we asked for?

    "100 dirhams" with an open "how much?" is the answer, and no classifier opinion is needed
    or wanted — the multilingual model scores it `greeting_help` at 0.83, confidently wrong,
    which silently lost the slot. "refund 250" also carries an amount, but "refund" survives
    the strip, so it is a new request rather than a bare answer.
    """
    if not any(extracted.get(k) not in (None, "") for k in ("amount", "payee", "merchant",
                                                            "txn_ref")):
        return False
    low = " ".join((said or "").lower().split())
    tokens = [t for t in re.split(r"[^\w\u0900-\u097F\u0600-\u06FF.]+", low) if t]
    rest = [t for t in tokens
            if not re.fullmatch(r"\d+(\.\d{1,2})?", t) and t not in _SLOT_FILLER_WORDS]
    # whatever is left must also be part of the slot value itself (a payee or merchant name)
    named = " ".join(str(v).lower() for k, v in extracted.items()
                     if k in ("payee", "merchant") and v)
    rest = [t for t in rest if t not in named.split()]
    return not rest


def _tr(a, key: str, english: str, **vals) -> str:
    """The customer's language if we have that string, otherwise the English we wrote.

    Never a machine translation and never a half-translated sentence: a missing key falls
    back to wording that has actually been read.
    """
    return langmod.t(key, a.lang, **vals) or english


def _watch(said, a):
    """Flag anything that looks wrong, record it, and apply the safe corrections.

    Runs on every answer. A flag can turn a refusal into a question or change the language;
    it can never decide something that moves money — see watchdog.py.
    """
    try:
        flags = watchdog.inspect(said, a)
        if flags:
            watchdog.record(said, a, flags)
            for st in flags:
                a.step("decide", "watchdog",
                       f"{st['kind']}: {st['why']}"
                       + (f" — correcting to {st['fix']}" if st.get("fix") else ""),
                       tag="guard")
    except Exception:
        pass
    return a


def _say_which_one(a: Answer) -> None:
    """UNUSED — kept as a record of an idea that did not survive contact.

    It prefixed every recalled answer with "About the X —". Two problems killed it: it
    fired on answers that are themselves questions ("About the send to Asad — which
    payment should I look at?"), and it produced half-translated sentences, because the
    lead-in had a Hinglish translation while most handler text does not. Naming the
    subject belongs in the handler's own sentence, where it can be written once in each
    language — see `_need_txn`.

    Original description follows.

    Name the earlier thread we think they meant, when we reached back for it.

    Two reasons, and the second is the important one:

    * it is how a conversation sounds when somebody was listening — "about the electricity
      split — Alex has paid" rather than an answer that arrives from nowhere;
    * it makes a WRONG guess visible immediately. Picking the wrong past task is the main
      risk of reaching back at all, and an answer that silently concerns a different bill
      is one a customer will act on without noticing.

    Deliberately skipped when the answer is a proposal or a confirmation: a payment about
    to happen needs its own words read closely, not a preamble in front of them. Also
    skipped when the answer already names the thing, which is most of the time.
    """
    t = getattr(a, "recalled", None)
    if not t or not a.say:
        return
    if a.proposal or a.confirmed or a.outcome in ("NEED_CONFIRMATION", "CONFIRMED"):
        return
    label = (t.label or "").strip()
    if not label:
        return
    # Already named? Then saying it again is clutter, not care.
    key = {w for w in re.split(r"[^\w]+", label.lower()) if len(w) > 3}
    body = {w for w in re.split(r"[^\w]+", a.say.lower()) if len(w) > 3}
    if key and len(key & body) >= max(1, len(key) // 2):
        return
    lead = _tr(a, "about_task", f"About the {label} —", what=label)
    a.say = f"{lead} {a.say[0].lower() + a.say[1:] if a.say[:1].isupper() else a.say}"


_SEC_TEXT = {
    "credential": ("sec_credential",
                   "OpenWallet will never ask for your PIN, OTP, password or card details — not in chat, "
                   "not on a call, not on WhatsApp. Please don't share them with anyone, even someone "
                   "who says they're from OpenWallet. If someone asked you for them, I can connect you to "
                   "our fraud team right now."),
    "credential_forgot": ("sec_forgot_pin",
                          "I can't see or tell you your PIN — nobody at OpenWallet can, and you should never "
                          "share it. You can reset it in the app under **Profile → Security → Reset PIN**."),
    "scam": ("sec_scam",
             "This looks like a scam. OpenWallet never asks you to pay to unlock your account, release a "
             "refund, claim a prize or clear a fine, and real officials don't collect fines by wallet "
             "transfer. Please don't send this money. Want me to connect you to our fraud team?"),
    "injection": ("sec_injection",
                  "I can't change how I work — every payment needs your own confirmation and PIN in "
                  "the app, and no message can switch that off. I'm happy to help with a normal "
                  "payment, a bill or a question about your transactions."),
    "other_person": ("sec_other_person",
                     "I can only show your own wallet and payments — I can't look up anyone else's "
                     "balance, transactions or details, even family. I can show you your own payments "
                     "to them, if that helps."),
    "internal": ("sec_internal",
                 "I can't share how our systems work behind the scenes — that stays private to keep "
                 "everyone's money safe. I'm happy to help with your own payments, bills or transfers."),
    "legal": ("sec_legal",
              "I understand this is serious, and I'm sorry it has come to this. I'm passing this "
              "conversation to a person on our team now so it's looked at properly — you won't have "
              "to repeat yourself."),
    "distress": ("sec_distress",
                 "I'm really sorry you're going through this — you don't have to deal with it alone. "
                 "I'm connecting you with a person from our team right now about your money. If you're "
                 "thinking about hurting yourself, please call the UAE National Mental Support Line on "
                 "800 4673 (800 HOPE), or 999 if you're in immediate danger."),
    "illegal": ("sec_illegal",
                "I can't help with that. If you have a question about your own payments, I'm here."),
    "competitor": ("sec_competitor",
                   "I can only help with your OpenWallet wallet — I can't compare or explain other apps and "
                   "services. I'm happy to help you send money, pay a bill or check the offers in OpenWallet."),
}


def _ph(a, kind: str, fallback: str, **slots) -> str:
    """A fresh wording from the phrase bank in the customer's language, or the old sentence."""
    try:
        return phrasebank.say(kind, a.lang or "en", a.session or "default", **slots) or fallback
    except Exception:                                              # noqa: BLE001
        return fallback


def _security_answer(a: Answer, kind: str, said: str, session: str, t0: float) -> Answer:
    key = "credential_forgot" if kind == "credential" and security.forgot(said) else kind
    lk, english = _SEC_TEXT[key]
    a.intent, a.model, a.confidence = Intent.OUT_OF_SCOPE.value, f"security-{kind}", 1.0
    a.say = _tr(a, lk, english)
    a.outcome = "REFUSED"
    a.proposal = None
    a.rows = []
    a.options = ([{"kind": "ask_text", "label": "Connect me to the fraud team",
                   "text": "connect me to support, I think this is a scam"}]
                 if kind in ("credential", "scam") and key != "credential_forgot" else [])
    a.step("decide", f"security.{kind}",
           {"credential": "a PIN / OTP / password / card detail was asked for, offered or pasted — "
                          "never accepted, repeated or stored; the customer is warned",
            "scam": "a payment demanded to unlock, release, claim or clear something — the pattern "
                    "of a scam; warned, nothing started",
            "injection": "an attempt to change the rules (fake system lines, skip the PIN, role "
                         "change) — nothing in the message is acted on",
            "illegal": "asks for help with something unlawful — declined",
            "other_person": "asks for someone else's balance, transactions or details — only the "
                            "customer's own wallet is ever shown",
            "internal": "asks how our systems work internally (endpoints, databases, rules) — not shared",
            "competitor": "asks to compare with, use or get offers from another provider — not ours to answer",
            "legal": "a legal threat — passed to a person, with the conversation attached",
            "distress": "signs of distress — a person is brought in, with the crisis line"}[kind], tag="guard")
    if kind in ("legal", "distress"):
        try:
            from engine import brd_handlers as _bh
            ref = _bh._handoff(a, "distress" if kind == "distress" else "legal threat")
            a.intent, a.outcome = Intent.HUMAN_HANDOFF.value, "HANDOFF"
            a.say += f" (Ref {ref})"
        except Exception:                                          # noqa: BLE001
            pass
    try:
        conversation.clear(session)
    except Exception:                                              # noqa: BLE001
        pass
    a.ms = int((time.time() - t0) * 1000)
    return a


def _finish(a: Answer) -> Answer:
    """Last gate. Everything the customer can read is swept for internal vocabulary; a hit
    is a bug, so it fails loudly rather than quietly serving a redacted string.

    Also the one place every turn passes through, which is why the learning store is written
    from here: a turn that is kept must be a turn that was actually served.
    """
    _watch(a.said, a)
    # The security replies are our own vetted text and MUST be able to say "password" to the
    # customer ("never share your password"); the sweep keeps catching it everywhere else.
    _say_check = (re.sub(r"(?i)password|passcode", "", a.say or "")
                  if str(a.model).startswith("security-") and a.say in {_tr(a, k, e) for k, e in _SEC_TEXT.values()}
                  else a.say)
    guard.scrub({"say": _say_check,
                 "options": [o.get("label", "") for o in a.options],
                 "trace": [t["detail"] for t in a.lane_trace
                           if t.get("tag") not in ("guard", "input")]})
    # Keep what the next training run needs, with the thing the current corpus cannot
    # express: the topic that was open when this was said. Nothing retrains from here — see
    # learn.py for why that is deliberate on a surface that moves money.
    a.kept = learn.record(a, a.said or "", a.session or "default",
                          topic=conversation.topic(a.session or "default"))
    if a.options:
        conversation.note_subject(a.session or "default", "last_options", list(a.options))
    return a


_SEND_SHAPE = re.compile(r"\b(?:send|sending|transfer|transferring)\s+(?:some\s+|the\s+)?(?:money|cash|funds|paisa|paise)\b"
                         r"|\bpaise?\s+bhej"
                         # "300 bhejo unko", "50 dirham bhej do"
                         r"|\b\d+\s*(?:aed|dirhams?|dhs|rs)?\s*(?:bhej\w*|transfer\s+kar\w*)\b", re.I)
#: "can you help me (in) sending money" asks us to DO it — the send flow, not the how-to guide.
#: "how do I / how to send money" stays a how-to question.
_HELP_DO = re.compile(r"\b(?:help|helm|hlp|halp)\s+me\s+(?:in\s+|with\s+|to\s+)?(?:\w+\s+){0,2}?"
                      r"(?:send|sending|transfer|transferring|pay|paying)\b|"
                      r"\b(?:help|helm|hlp|halp)\s+me\s+(?:in\s+|with\s+)?(?:initiat|start|mak|do)\w*\s+(?:a\s+)?"
                      r"(?:transfer|payment|remittance)", re.I)
#: Whether something is POSSIBLE is a product question ("Can I send money abroad?"); "can YOU
#: send…" is a request and is not matched.
_CAN_I = re.compile(r"^\W*(?:can|could|may)\s+i\b|^\W*is\s+it\s+possible\b|^\W*am\s+i\s+(?:able|allowed)\b|"
                    r"^\W*do\s+you\s+(?:support|allow)\b", re.I)
_OFFER_WORD = re.compile(r"\b(?:offers?|discounts?|cashback|promo\w*|deals?|coupons?|voucher)\b|"
                         r"ऑफर|छूट|عرض|عروض|خصم|كاش\s*باك", re.I)
_COUNTRY_WORD = re.compile(r"\b(?:india|pakistan|bangladesh|sri\s+lanka|nepal|philippines|egypt|"
                           r"pkr|inr|bdt|lkr|npr)\b|باكستان|الهند|بنغلاديش|سريلانكا|पाकिस्तान|भारत|इंडिया", re.I)
_DID_I_PAY = re.compile(r"^\W*(?:did|have|had)\s+i\s+(?:already\s+)?(?:pay|paid|send|sent|transfer\w*)\b", re.I)
_DENIES = re.compile(r"\b(?:never|not\s+me|wasn'?t\s+me|didn'?t|don'?t\s+recogni[sz]e|unauthori[sz]ed|fraud|"
                     r"scam|hack\w*|stolen|someone\s+else)\b", re.I)
#: Money that has ALREADY gone — past tense or "did it arrive". Asking about it is status.
_ALREADY_SENT = re.compile(r"\b(?:sent|transferred|bheje?\s+th[ae]|bheja\s+tha|bheje\s+hue|bhej\s+diye?\s+th[ae]|"
                           r"pahu?nch[ae]?\s+(?:ya|kya)|pahuche|arrived?|reached|went\s+through|"
                           r"did\s+(?:it|the\s+money|my\s+(?:money|transfer|payment))\s+(?:go|reach|arrive))\b|"
                           r"أرسلت|وصل|भेजे\s+थे|पहुंचे", re.I)
_HOW_DO_I_ACT = re.compile(
    r"^\W*how\s+(?:do|can|should|would)\s+(?:i|we|you)\s+(?:\w+\s+){0,1}?"
    r"(?:split|pay|send|transfer|add|top\s*up|recharge|withdraw|link|connect|set\s*up|schedule|remind|"
    r"cancel|request|raise|change|update|delete)\b", re.I)
_STATUS_INTENTS = {"split_status", "bill_due", "txn_status", "balance_query", "reminder_list",
                   "txn_history", "spend_analysis", "spend_rank", "beneficiary_query", "limits_query"}
_ACTION_INTENTS = {"split_payment", "bill_pay", "bill_schedule", "send_money", "pay_merchant",
                   "mobile_recharge", "add_money", "withdraw_money", "retry_payment", "refund_request"}
#: "Which bills can I pay?", "What can I recharge?" — what the product supports.
_WHICH_CAN_I = re.compile(r"^\W*(?:which|what)\b[^?]{0,40}\bcan\s+i\b", re.I)
#: A reminder needs a word that asks for one.
_REMIND_WORD = re.compile(r"\bremind|\breminder|yaad\s+dila|याद|ذكر", re.I)

#: "Go to profile", "open settings", "account management": screens the assistant cannot open.
_GO_SCREEN = re.compile(
    r"^\W*(?:please\s+|ok\s+)?(?:go\s+to|open|take\s+me\s+to|navigate\s+to|show\s+me)\s+(?:the\s+|my\s+)?"
    r"(profile|settings|account\s+settings|account\s+management|account|notifications?|home(?:\s+screen)?|help\s+centre|help\s+center)\b"
    r"[\s.!?]*$|^\W*(profile|settings|account\s+management|log\s?out|logout)[\s.!?]*$", re.I)
_SCREENS = {"profile": "by tapping your photo at the top-left of the home screen",
            "setting": "under your profile (tap your photo at the top-left)",
            "account": "under your profile (tap your photo at the top-left)",
            "log": "at the bottom of your profile page",
            "notification": "behind the bell icon at the top-right",
            "home": "by closing this assistant",
            "help": "under your profile → Help"}

#: A request to GO somewhere in the app — never an answer to "how much?".
_NAV = re.compile(r"\b(?:go to|open|take me to|navigate|show me (?:the|my) (?:profile|settings|account)|"
                  r"profile|settings|account management|my account|log ?out|home ?screen)\b", re.I)


#: Which answer-only details can answer an open question OF THIS KIND. "Deva DEWA" (pay DEWA)
#: supplied a `bill` and was taken as the phone number for an open RECHARGE — a bill answers
#: a bill payment or a split, never a recharge.
_SLOTS_OF = {
    "bill_pay": {"bill", "pay_mode", "when", "when_at"},
    "bill_schedule": {"bill", "pay_mode", "when", "when_at"},
    "split_payment": {"bill", "split_type", "participants", "ways"},
    "reminder_set": {"when", "when_at", "bill"},
    "mobile_recharge": set(),
    "send_money": set(),
    "pay_merchant": {"pay_mode", "split_type", "participants", "ways"},
}


def _garbled(said: str) -> bool:
    """A keyboard mash or a mishearing: no word in it is a word we know ("Heuwoa1234",
    "Hshshs", "Descaror Descaror", "….,,,mmm")."""
    words = re.findall(r"[A-Za-z]{2,}", said or "")
    if not words:
        return bool(re.search(r"[A-Za-z]", said or "")) or not re.search(r"\w", said or "")
    known = prefilter.known()
    # only words of 4+ letters count: "dum" in "Senvani-dum-hammud" is technically English
    return not any(w.lower() in known for w in words if len(w) >= 4)


def _reask_if_open(a, session, said, ex, t0):
    """Re-ask the open question (with its options) when this short reply does not answer it.
    None when no question is open or the reply is a real question / new request of its own."""
    q = conversation.get(session)
    if not (q and q.asked_for and len((said or "").split()) <= 3
            and not _is_question(said) and not _is_instruction(said) and not _NAV.search(said or "")
            and not any(v not in (None, "", []) for v in (ex or {}).values())
            and (a.intent in (q.intent, Intent.OUT_OF_SCOPE.value, Intent.GREETING_HELP.value)
                 or a.confidence < CONFIDENT or _garbled(said))):
        return None
    a.step("understand", "reask",
           f'"{said[:40]}" doesn\'t answer the open question about '
           f"{q.intent.replace('_', ' ')} (we asked for {q.asked_for}) — asking it again with the "
           f"same options rather than declining", tag="rule")
    a.intent, a.model, a.confidence = q.intent, "re-asked", 1.0
    s_obj = slotlib.Slots(**{k: v for k, v in q.slots.items() if k in slotlib.Slots.__annotations__})
    a.slots = s_obj.as_dict()
    handler = HANDLERS.get(Intent(a.intent))
    out = handler(a, s_obj, "") if handler else a
    out.say = _tr(out, "didnt_catch_short", "Sorry, I didn't catch that.") + " " + (out.say or "")
    _remember(session, out, s_obj)
    out.ms = int((time.time() - t0) * 1000)
    return _finish(out)


#: Intents that act on reading, and the question to ask when the model is unsure of them.
#: The "yes" text is a plain sentence the model reads with full confidence.
_SUPPORT_WORD = re.compile(
    r"\b(?:support|agent|human|person|someone\s+(?:real|from)|customer\s+(?:care|service)|help\s*desk|"
    r"call|talk|speak|connect|complain\w*|escalat\w*|manager|staff|executive|representative|team)\b|"
    r"बात|सपोर्ट|کال|موظف|خدمة\s+العملاء|الدعم|أكلم|اكلم|baat", re.I)
_ASK_FIRST = {
    "wrong_transfer": ("Did you send money to the wrong person?", "I sent money to the wrong person"),
    "report_fraud": ("Do you want to report a payment you don't recognise?", "I want to report a payment I don't recognise"),
    "human_handoff": ("Would you like me to connect you to OpenWallet customer support?", "connect me to customer support"),
    "mobile_recharge": ("Do you want to recharge a mobile number?", "I want to recharge my phone"),
    "retry_payment": ("Do you want to retry a payment that failed?", "retry my failed payment"),
    "add_money": ("Do you want to add money to your wallet?", "add money to my wallet"),
    "withdraw_money": ("Do you want to withdraw money from your wallet to your bank?", "withdraw money to my bank"),
}


def _m(m: dict) -> str:
    """" · 1032m" — or nothing when we don't know where the customer is ("Nonem away" was
    printed on the server, which has no location)."""
    return f" · {m['metres']}m" if m.get("metres") is not None else ""


def _within(near: list) -> str:
    return f" within {near[-1]['metres']}m" if near and near[-1].get("metres") is not None else ""


# ─── who a transfer is to: a person, or a description ──────────────────────────────────
_ABROAD = re.compile(r"\b(?:abroad|overseas|internationally|international(?:ly)?\s+transfer|"
                     r"(?:another|other|a\s+different|foreign)\s+country|outside\s+(?:the\s+)?uae|"
                     r"back\s+home|videsh|bahar\s+ke\s+desh|cross[\s-]?border|international(?:ly)?|remittance|"
    r"out(?:side)?\s+(?:of\s+)?(?:the\s+)?country|bahar\s+(?:bhej\w*|paise|paisa|transfer))\b|برا\s+البلد|خارج\s+البلد|"
    r"الخارج|دولة\s+أخرى|حوالة\s+دولية|विदेश", re.I)
#: "to a friend", "to my brother", "to another country": a determiner before the word means
#: it describes someone; it is not their name.
_NOT_A_NAME = re.compile(r"\bto\s+(?:a|an|the|my|our|your|another|other|some|any|his|her|their)\s+\w+", re.I)
_GENERAL_WORDS = {"country", "abroad", "overseas", "home", "family", "friend", "friends", "someone",
                  "somebody", "anyone", "people", "them", "him", "her", "relative", "relatives",
                  "parents", "mother", "father", "mom", "dad", "brother", "sister", "wife", "husband",
                  "son", "daughter", "bank", "account", "wallet", "merchant", "shop", "store", "india",
                  "pakistan", "bangladesh", "philippines", "nepal", "lanka", "egypt", "outside"}


#: Words people use to ADDRESS the assistant ("Bro, I need to send money", "bhai paise bhejne
#: hai"). Said at the start of a sentence they are not a payee — "Bro is not in your saved
#: beneficiaries" answers a name nobody gave.
_VOCATIVES = {"bro", "bruh", "brother", "bhai", "bhaiya", "yaar", "yar", "dude", "sir", "madam", "maam",
              "boss", "man", "buddy", "mate", "habibi", "habibti", "akhi", "ustad", "ji", "dost", "arre", "arey"}
_VOCATIVE_LEAD = re.compile(r"^\W*(?:(?:hey|hi|hello|ok|okay|oye|ya|yo)\s+)?(?:" + "|".join(sorted(_VOCATIVES, key=len, reverse=True))
                            + r")\b[\s,!.:-]+(?=\S)", re.I)


def _strip_vocative(said: str) -> str:
    """'Bro, in need to send money' -> 'in need to send money'. Only a leading address word
    followed by the rest of a sentence; 'bhai ko 50 bhejo' (bhai as the payee) keeps it,
    because the next word ties it to the sentence ("ko", "ke", "to")."""
    m = _VOCATIVE_LEAD.match(said or "")
    if not m:
        return said
    rest = (said or "")[m.end():]
    if re.match(r"(?:ko|ke|ki|ka|to|for|se)\b", rest, re.I):
        return said
    return rest if len(rest.split()) >= 2 else said


#: Words that can follow "to / pay / for" without being a person.
_NOT_PERSON = _GENERAL_WORDS | {
    "a", "an", "the", "my", "me", "you", "us", "it", "this", "that", "next", "last", "send", "pay",
    "money", "cash", "bill", "bills", "him", "her", "uae", "dubai", "abu", "sharjah", "everyone",
    "all", "remind", "set", "check", "see", "do", "go", "be", "make", "transfer", "recharge",
    "dewa", "sewa", "fewa", "du", "etisalat", "salik", "rent", "internet", "electricity", "water",
    "phone", "mobile", "number", "back", "again", "same", "now", "today", "tomorrow", "later",
    "in", "on", "at", "with", "of", "and", "or", "please", "pls", "for", "what", "how", "who",
    "merchants", "shops", "stores", "nearby", "near", "beneficiary", "beneficiaries", "contact",
    "contacts", "list", "people", "person", "group", "split", "dinner", "lunch", "share",
    "sri", "saudi", "united", "new", "south", "north", "hong", "costa", "el", "cross", "border",
    "international", "abroad", "overseas", "support", "agent", "team", "someone", "somebody",
    "customer", "care", "human", "staff", "remittance", "remit", "transfers",
    # Hindi/Urdu pronouns: "300 bhejo unko" — 'unko' is "to them", not a person called Unko
    "unko", "usko", "inko", "isko", "unhe", "unhein", "use", "ise", "mujhe", "humko", "tumko", "aapko",
    "mummy", "papa", "ammi", "abbu"} | _VOCATIVES
_PAYEE_WORD = re.compile(r"\b(?:to|pay|for|bhejo|bhej\s+do)\s+([a-z][a-z'-]{2,})\b"
                         r"|\b([a-z][a-z'-]{2,})\s+(?:ko|ke\s+liye)\b", re.I)


_CORRECTION = re.compile(r"^\W*(?:no+|nope|nah|sorry|actually|wait|not\b|nahi|nahin|nai|na\b|la\b|لا|نہیں|नहीं)", re.I)
_CORR_MERCHANT = re.compile(r"\b(?:merchant|shop|store|dukaa?n|outlet|restaurant|cafe)\b|محل|متجر|दुकान", re.I)
_CORR_PERSON = re.compile(r"\b(?:person|friend|someone|somebody|beneficiary|contact|family|kisi\s+ko)\b|شخص|صديق", re.I)


def _only_names(said: str, payee: str) -> bool:
    """Is every word of the sentence part of the payee's name (allowing a typo or two)?"""
    parts = {p.lower() for p in (payee or "").split()}
    words = [w.strip(".,!?").lower() for w in (said or "").split() if w.strip(".,!?")]
    return bool(words) and all(w in parts or any(_edits_within(w, p, 1) for p in parts if len(p) > 3)
                               for w in words)


_HARD_OOS = re.compile(
    r"\b(?:flights?|airlines?\s+tickets?|hotels?|holiday\s+package|movies?|films?|netflix|recipes?|"
    r"cook(?:ing)?|weather|mausam|cricket|football|match\s+score|dance|dancing|sing|singing|songs?|"
    r"poems?|jokes?|homework|loans?|mortgage|insurance|bima|credit\s+cards?|crypto\w*|bitcoin|"
    r"stocks?|share\s+market|mutual\s+funds?|invest(?:ment|ments|ing|ed|s)?|forecast|predict\w*|weight\s+loss|"
    r"lose\s+weight|belly\s+fat|diet|banane\s+ka\s+tarika|"
    # other apps and exchange houses — we only speak for OpenWallet
    r"careem\s*pay|western\s+union|paytm|botim|google\s+pay|gpay|phonepe|e&\s*money|e\s*and\s*money|"
    r"al\s+ansari|uae\s+exchange|emirates\s+exchange|remitly|moneygram|"
    # advice we must not give: medical, religious, legal, timing the market
    r"dawai|dawa|medicine|tablets?|doctor|zakat|good\s+time\s+to\s+send|wait\s+for\s+the\s+(?:rupee|rate|dollar))\b|"
    r"زكاة|الزكاة|حقوقي|القانون|دواء|الإمارات\s+للصرافة|الأنصاري|दवाई|ज़कात|"
    r"फ्लाइट|होटल|रेसिपी|मौसम|लोन|बीमा|शेयर|वज़न|वजन|"
    r"طيران|رحلة|فندق|وصفة|طريقة\s+عمل|طقس|قرض|تأمين|بطاقة\s+ائتمان|عملات\s+رقمية|أسهم|وزني|الإقامة|اقامة|تأشيرة", re.I)
#: The sentence is about one of the customer's own payments, whatever else it mentions.
_PAYMENT_CONTEXT = re.compile(
    r"\b(?:refund\w*|failed|deducted|debited|charged|transaction|txn|payment\s+(?:failed|pending|stuck)|split\w*|"
    r"pa(?:y|id)\s+(?:me\s+)?back|paid\s+me|owes?|"
    r"my\s+(?:payment|transfer))\b|रिफंड|कट\s+गए|استرداد|انخصم", re.I)
_EXCHANGE_RATE = re.compile(
    r"\b(?:exchange|conversion|currency|fx)\s+rates?\b|\brates?\s+(?:for|to)\s+(?:send\w*|transfer\w*|india|"
    r"pakistan|bangladesh|philippines|nepal|sri\s+lanka|egypt|inr|pkr|bdt|php|npr|lkr|egp)\b|"
    r"\b(?:aaj\s+ka\s+)?(?:dollar|rupee|rupay|inr|pkr|lkr|npr|bdt|php|egp|india|pakistan|bangladesh|"
    r"philippines|nepal|sri\s+lanka|egypt)\s+(?:ka\s+)?rate\b|سعر\s+(?:الصرف|التحويل)|विनिमय\s+दर|रेट", re.I)
#: A rate in the FUTURE is a forecast — not ours.
_FX_FORECAST = re.compile(r"\b(?:will|going\s+to|gonna|next\s+(?:week|month|year)|tomorrow|agle|"
                          r"go(?:es)?\s+(?:up|down)|rise|fall|drop|increase|decrease|better|worse)\b|"
                          r"هل\s+سي|الشهر\s+(?:الجاي|القادم)|अगले|बढ़ेगा|गिरेगा", re.I)


def _named_unknown_payee(said: str, s) -> str | None:
    """A PERSON named as the payee who is not on this customer's roster ("pay riley",
    "send money to riley in 5 minutes"). Used to stop conversation memory filling the
    payee with someone else: the customer named Riley, and "How much to Alex?" — or a
    reminder saved for Alex — answers a request nobody made."""
    if s.payee or s.merchant:
        return None
    for m in _PAYEE_WORD.finditer(said or ""):
        w = (m.group(1) or m.group(2) or "").lower()
        # a verb form ("for sending money", "to transferring") is not a name
        if (w and w not in _NOT_PERSON and not prefilter._fix(w) and w not in prefilter.VOCAB
                and not w.endswith(("ing", "ed"))):
            return w.title()
    return None


# ─── emailing a statement ────────────────────────────────────────────────────────────────
_EMAIL_WORD = re.compile(r"\b(?:e-?mail|mail|inbox|gmail|outlook)\b|ईमेल|ई-मेल|मेल|ايميل|إيميل|بريد", re.I)
_TXN_WORD = re.compile(r"\b(?:transactions?|txns?|payments?|statements?|history|transfers?|"
                       r"lenden|len den)\b|ट्रांजैक्शन|लेनदेन|लेन-देन|स्टेटमेंट|معاملات|المعاملات|كشف|"
                       r"الحساب|التحويلات|الدفعات", re.I)
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "paanch": 5, "das": 10}


def _wants_statement_email(said: str) -> bool:
    t = said or ""
    if not (_EMAIL_WORD.search(t) and _TXN_WORD.search(t)):
        return False
    # "how do I email my statement?" is a how-to, not a request to do it
    return not re.match(r"^\s*(?:how|kaise|كيف|कैसे)\b", t, re.I)


def _statement_count(said: str) -> int:
    t = (said or "").lower().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        return max(1, min(int(m.group(1)), 25))
    for w, n in _NUM_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return n
    return 5


# ─── picking from what we showed ─────────────────────────────────────────────────────────
_ORDINALS = [
    (0, r"first|1st|one|pehla|pehle|pahla|pahle|पहला|पहले|الأول|الاولى|الأولى|اول|أول"),
    (1, r"second|2nd|two|doosra|dusra|doosre|dusre|दूसरा|दूसरे|الثاني|الثانية|ثاني"),
    (2, r"third|3rd|three|teesra|tisra|teesre|तीसरा|तीसरे|الثالث|الثالثة|ثالث"),
    (3, r"fourth|4th|four|chautha|चौथा|الرابع|الرابعة"),
    (4, r"fifth|5th|five|paanchva|पांचवां|الخامس|الخامسة"),
    (-1, r"last|aakhri|akhri|आखिरी|الأخير|الأخيرة|اخر|آخر"),
]


def _spoken_pick(said: str, shown: list[dict]) -> dict | None:
    """The option a short reply points at: "the first one", "second", "pehla wala",
    "الأول", "option 2", "the 85 one", or a word that appears in exactly one label."""
    t = _flat(said)
    words = t.split()
    if not t or len(words) > 4 or len(shown) < 1 or _is_question(said):
        return None
    filler = r"(?:the|one|wala|wali|vala|wale|please|ok|that|this|ये|वाला|وا?ل)"
    for idx, pat in _ORDINALS:
        if re.fullmatch(rf"(?:{filler}\s+)*(?:{pat})(?:\s+{filler})*", t) and len(shown) > (idx if idx >= 0 else 0):
            # a bare "one" is only an ordinal when a list is on screen — which it is
            return shown[idx]
    m = re.fullmatch(r"(?:the\s+)?(?:option|number|no\.?|#)\s*(\d)(?:\s+one)?", t)
    if m and 1 <= int(m.group(1)) <= len(shown):
        return shown[int(m.group(1)) - 1]
    if len(shown) >= 2:
        # "the 85 one", "mclothes", "crystal": something that appears in exactly ONE label
        keys = [w for w in words if w not in ("the", "one", "wala", "that", "this") and len(w) >= 2]
        if keys:
            hits = [o for o in shown if all(re.search(rf"(?<![\w.]){re.escape(k)}(?![\w])",
                                                       _flat(o.get("label", ""))) for k in keys)]
            if len(hits) == 1:
                return hits[0]
    return None


def _option_request(o: dict) -> tuple[str, dict | None] | None:
    """What tapping an option does, as (text, structured pick). The same mapping the chat
    screens use, so a spoken pick and a tap are one and the same request."""
    k = o.get("kind")
    if k == "ask_text":
        return (o.get("text") or o.get("label") or ""), None
    if k == "pick_txn":
        return o.get("label", ""), {"intent": "txn_status", "slots": {"txn_ref": o.get("ref")}}
    if k == "ask_refund":
        return "Refund " + o.get("label", ""), {"intent": "refund_request", "slots": {"txn_ref": o.get("ref")}}
    if k == "send_to":
        return "Send to " + o.get("payee", ""), {"intent": "send_money", "slots": {"payee": o.get("payee")}}
    if k == "pay_merchant" and o.get("merchant"):
        return "Pay " + o["merchant"], {"intent": "pay_merchant", "slots": {"merchant": o["merchant"]}}
    if k == "journey":
        return o.get("label", ""), {"intent": "journey_help", "slots": {}}
    if k == "split_with":
        return ("Split with " + o.get("label", "").split(" ·")[0],
                {"intent": "split_payment",
                 "slots": {"participants": [{"name": o.get("name"), "userId": o.get("userId")}]}})
    return None


# ---------------------------------------------------------------------------------------
# Handlers, one per intent. Each one only reads what it needs.
# ---------------------------------------------------------------------------------------

def _txn_options(exclude: str | None = None, kind: str = "pick_txn") -> list[dict]:
    """The customer's recent payments, as pickable options."""
    return [{"kind": kind,
             "label": f"{t.payee} · {t.amount:g} {t.currency} · {t.public()['status']}"
                      f" · {t.public()['when']}",
             "ref": t.ref}
            for t in data.transactions_for()[:6] if t.ref != exclude]


def _need_txn(a: Answer, s, intent: str, said: str = "", session: str = ""):
    txn, how = slotlib.resolve_txn(s, intent, said, session)
    if not txn:
        # When we were looking for a NAMED person's payment, say that we could not find
        # it. "Which payment should I look at?" is the right question when nobody was
        # named and a useless one when they just named somebody — it reads as not having
        # listened, and it hides the actual finding, which is that there isn't one.
        if how.startswith("no payment to "):
            who = how[len("no payment to "):-len(" found")]
            a.say = _tr(a, "no_payment_to",
                        f"I can't find any payment to {who} in your recent activity. "
                        f"Here's what's there.", who=who)
        else:
            a.say = voice.pick("which_txn", a.session) + " Here are your recent ones."
        a.options = _txn_options()
        a.outcome = "NEED_INFO"
        a.step("resolve", "txn.none", how, tag="guard")
        return None
    a.txn = txn.public()
    a.inferred_txn = how.startswith("inferred")
    a.step("resolve", "txn.found",
           f"{txn.ref} · {txn.payee} · {txn.amount:g} {txn.currency} · "
           f"{txn.public()['status']} · {how}", tag="db")
    return txn


def _name_txn(txn) -> str:
    """How a payment is described to its owner. Not the reference — a 16-digit number means
    nothing to a customer; who, how much and when is how they actually remember it."""
    p = txn.public()
    return f"your {p['amount']:g} {p['currency']} to {p['payee']} from {p['when']}"


def _matrix_answer(a: Answer, txn) -> Answer:
    d = run_matrix(txn.evidence())
    a.outcome = d.outcome.value
    # Lead with WHICH payment. The policy sentence on its own ("that transfer failed before
    # the money left us") is unanswerable if the customer has three payments in flight and
    # cannot tell which one we picked.
    lead = f"{voice.pick('about', a.session)} {_name_txn(txn)}"
    if a.inferred_txn:
        lead += f" — {voice.pick('guessed', a.session)}"
    a.say = f"{lead}: {d.say[0].lower() + d.say[1:]}"
    for st in d.trace:
        a.step(st["lane"], st["what"], st["detail"], st.get("tag", "rule"))
    if d.raises_refund and not txn.refund_raised:
        a.proposal = {"kind": "refund", "ref": txn.ref, "amount": txn.amount,
                      "currency": txn.currency, "outcome": d.outcome.value,
                      "needs_approval": True}
        a.options = [{"kind": "raise_refund", "label": f"Raise a refund for "
                                                       f"{txn.amount:g} {txn.currency}",
                      "ref": txn.ref},
                     {"kind": "dismiss", "label": "Not now"}]
    elif d.escalates:
        a.options = [{"kind": "escalate", "label": "Ask a person to look at it",
                      "ref": txn.ref}]
    # Appended LAST: the branches above assign a.options outright, so adding the correction
    # earlier silently lost it. If we guessed the payment the customer must be able to say so
    # — and the correction is the OTHER payments themselves, not a dead-end "that's wrong"
    # button that leaves them to explain which one they meant.
    if a.inferred_txn:
        kind = "ask_refund" if a.intent in (Intent.REFUND_REQUEST.value,
                                            Intent.DEBITED_NOT_CREDITED.value) else "pick_txn"
        alts = _txn_options(exclude=txn.ref, kind=kind)[:3]
        if alts:
            a.options.append({"kind": "label", "label": "Or did you mean:"})
            a.options.extend(alts)
    return a


def h_txn(a, s, said):
    txn = _need_txn(a, s, a.intent, said, a.session)
    if not txn:
        return a
    if a.intent in (Intent.TXN_STATUS.value,):
        pub = txn.public()
        a.say = (f"Your {pub['amount']:g} {pub['currency']} to {pub['payee']} "
                 f"({pub['corridor']}), sent {pub['when']}, is **{pub['status'].lower()}**.")
        if a.inferred_txn:
            a.say += " That's the one I assumed you meant."
        a.outcome = "INFORMED"
        a.step("decide", "status", f"reported as {pub['status']}")
        if txn.state.value in ("FAILED", "ON_HOLD", "PENDING"):
            a.options = [{"kind": "ask_refund", "label": "I want this refunded",
                          "ref": txn.ref}]
        if a.inferred_txn:
            a.options += _txn_options(exclude=txn.ref, kind="pick_txn")[:3]
        return a
    return _matrix_answer(a, txn)


def h_refund_status(a, s, said):
    txn = _need_txn(a, s, a.intent, said, a.session)
    if not txn:
        return a
    if txn.refund_raised or txn.state.value == "REVERSED":
        a.say = (f"Your refund of {txn.amount:g} {txn.currency} for that payment has been "
                 f"completed and the money is back in your wallet.")
        a.outcome = "ALREADY_REFUNDED"
    else:
        a.say = ("I do not have a refund raised against that payment yet. I can raise one "
                 "now if you would like.")
        a.outcome = "NO_REFUND_RAISED"
        a.options = [{"kind": "ask_refund", "label": "Raise a refund", "ref": txn.ref}]
    a.step("decide", "refund.lookup", a.outcome)
    return a


_MODE_NAME = {"BANK": "to a bank account", "WALLET": "to a mobile wallet", "CASH": "for cash pickup"}


_RUPEE = re.compile(r"\brupees?\b|\brupay\w*|\brs\b|روبي|रुपय|रुपए|रुपये", re.I)
#: A currency code or currency name -> the corridor it implies ("PKR rate today").
_CCY_CORRIDOR = {"pkr": "AE-PK", "inr": "AE-IN", "lkr": "AE-LK", "npr": "AE-NP", "bdt": "AE-BD",
                 "taka": "AE-BD", "php": "AE-PH", "peso": "AE-PH", "egp": "AE-EG"}


def _fx_vals(corridor: str, amount: float | None) -> dict:
    import treasury
    q = treasury.quote(corridor.split("-")[-1])
    m = treasury.best(q) or {}
    return {"dest": CORRIDORS[corridor]["name"].split("→")[-1].strip(), "rate": f"{m.get('rate', 0):,.2f}",
            "cur": q.get("currency") or "", "amount": f"{(amount or 0):,.2f}",
            "conv": f"{(amount or 0) * m.get('rate', 0):,.2f}"}


def _fx_line(corridor: str, amount: float | None) -> str | None:
    """Today's customer rate from treasury, in words — or None (inactive corridor / treasury down).
    Only the customer-offered rate is used; treasury's cost rate and margins never leave treasury.py."""
    if not corridor or corridor == "AE-AE":
        return None
    try:
        import treasury
        q = treasury.quote(corridor.split("-")[-1])
    except Exception:                                              # noqa: BLE001
        return None
    m = treasury.best(q) if q.get("ok") else None
    if not m:
        return None
    cur, dest = q.get("currency") or "", CORRIDORS[corridor]["name"].split("→")[-1].strip()
    line = (f"Today's rate to {dest} is 1 AED = {m['rate']:,.2f} {cur} "
            f"({_MODE_NAME.get(m['mode'], m['mode'].lower())})")
    if amount:
        line += f", so AED {amount:,.2f} is about {amount * m['rate']:,.2f} {cur}"
    return line + ". The exact rate is locked when you confirm."


_HOW_MUCH = re.compile(r"\b(?:kitna|kitne|kinta|how\s+much|milega|milenge|jaega|jayega|jaayega|"
                       r"pahunchega|receive|receives|charge|charges|fee|fees)\b|कितना|कितने|मिलेगा|"
                       r"जाएगा|पहुंचेगा|चार्ज|كم", re.I)
_HOW_LONG = re.compile(r"\b(?:der|kab|how\s+long|when|time|days?|din|hours?|ghante)\b|देर|कब|समय|متى", re.I)


def h_fees(a, s, said):
    corridor = s.corridor or next((v for k, v in _CCY_CORRIDOR.items()
                                   if re.search(rf"\b{k}\b", (said or "").lower())), None) or "AE-AE"
    c = CORRIDORS[corridor]
    amount = s.amount or 0
    a.step("resolve", "corridor", f"{corridor} ({c['name']})", tag="db")
    fx = _fx_line(corridor, amount)
    if _EXCHANGE_RATE.search(said or "") and corridor == "AE-AE" and _RUPEE.search(said or ""):
        a.say = _tr(a, "fx_which_country", "Which country is that for? The rupee is used in several.")
        a.outcome = "NEED_INFO"
        a.options = [{"kind": "ask_text", "label": n, "text": f"exchange rate to {n.lower()}"}
                     for n in ("India", "Pakistan", "Sri Lanka", "Nepal")]
        return a
    if _EXCHANGE_RATE.search(said or "") and fx:
        v = _fx_vals(corridor, amount)
        a.say = _tr(a, "fx_live_amt" if amount else "fx_live", fx + f" The transfer fee is AED {c['fee_aed']:.2f}.",
                    fee=f"{c['fee_aed']:.2f}", **v)
        a.outcome = "QUOTED"
        a.step("resolve", "fx.live", f"treasury customer rate {v['rate']} {v['cur']} ({corridor}); "
               f"cost rate and margins are never read", tag="db")
        return a
    if _EXCHANGE_RATE.search(said or ""):
        # Treasury had no rate for this corridor (inactive here) or could not be reached.
        dest = c["name"].split("→")[-1].strip()
        a.say = _tr(a, "fx_rate",
                    (f"The live exchange rate to {dest} is shown on the send screen before you "
                     f"confirm — I can't quote it here yet. The transfer fee is AED "
                     f"{c['fee_aed']:.2f}." if corridor != "AE-AE" else
                     "Exchange rates apply to transfers abroad, and the live rate is shown on the "
                     "send screen before you confirm — I can't quote it here yet. Which country "
                     "are you sending to?"), dest=dest)
        a.outcome = "QUOTED" if corridor != "AE-AE" else "NEED_INFO"
        a.step("act", "fx.rate", "the live rate comes from treasury at send time — not quoted here",
               tag="guard")
        return a
    if amount:
        a.say = (f"Sending AED {amount:g} to {c['name'].split('→')[-1].strip()} costs "
                 f"AED {c['fee_aed']:.2f} in fees, so AED {amount + c['fee_aed']:.2f} leaves "
                 f"your wallet. " + (fx or "The exchange rate is shown before you confirm."))
    else:
        a.say = (f"{c['name']} costs AED {c['fee_aed']:.2f} per transfer. You always see the "
                 f"fee and the exact amount your beneficiary receives before you confirm.")
    if corridor != "AE-AE":
        offers = _corridor_offers(a, corridor, amount, c["fee_aed"])
        if offers:
            a.say += " " + _offers_line(offers, amount)
    a.outcome = "QUOTED"
    a.step("decide", "quote", f"indicative fee AED {c['fee_aed']:.2f} — the binding number "
                              f"is the one shown at confirmation")
    return a


def _corridor_offers(a, corridor, amount, fee):
    """Live fee offers for a transfer abroad, from the Incentive Engine.

    The static table only knows the full fee, so a customer with three live campaigns for
    Pakistan was quoted the full fee every time. These campaigns are customer-triggered (a
    promo code the customer chooses), so they are SHOWN with what each does to the fee and
    never applied here. With an amount, each one is quoted by the engine itself — the amount
    range, per-user cap and remaining budget are its decision, not our arithmetic."""
    dest = corridor.split("-")[-1]
    found = incentive.discover(amount=amount or None, corridor=dest,
                               product_code=incentive.CROSS_BORDER_PRODUCT)
    if found is None:
        a.step("resolve", "incentive.offline",
               f"incentive engine not consulted ({incentive.last_error() or 'no customer'}) — "
               f"fee quoted without offers", tag="guard")
        return []
    offers = []
    for camp in found:
        code = camp.get("promoCode") or camp.get("incentiveCode")
        q = None
        if amount:
            q = incentive.quote(amount, campaign_id=camp.get("campaignId"), promo_code=camp.get("promoCode"),
                                transfer_fee=fee, corridor=dest,
                                product_code=incentive.CROSS_BORDER_PRODUCT)
            if not (q and q.get("applicable")):
                continue
        offers.append({"code": code, "offer": incentive.as_offer(camp),
                       "fee": None if q is None else q.get("payableFee"),
                       "auto": (camp.get("applyMode") or "") == "AUTO_APPLY"})
    if amount:
        offers.sort(key=lambda o: o["fee"] if o["fee"] is not None else fee)
    a.step("resolve", "incentive.live",
           f"{len(found)} live {dest} campaign(s), {len(offers)} usable"
           + (f" on AED {amount:g}" if amount else "") + " — shown, not applied", tag="db")
    return offers


def _offers_line(offers, amount):
    bits = []
    for o in offers[:3]:
        if amount and o["fee"] is not None:
            bits.append(f"{o['code']} brings the fee to AED {float(o['fee']):.2f}")
        else:
            bits.append(f"{o['code']} gives {o['offer']['label']}")
    n = len(offers)
    head = "You have an offer for this transfer: " if n == 1 else f"You have {n} offers for this transfer: "
    tail = (" It is applied automatically." if all(o["auto"] for o in offers) else
            " Enter the promo code on the send screen to use one.")
    return head + "; ".join(bits) + "." + tail


def h_time(a, s, said):
    corridor = s.corridor or "AE-AE"
    c = CORRIDORS[corridor]
    a.step("resolve", "corridor", f"{corridor} ({c['name']})", tag="db")
    a.say = f"{c['name']} normally arrives {c['sla']}."
    if corridor != "AE-AE":
        a.say += " You get a notification the moment it is paid out."
    a.outcome = "QUOTED"
    a.step("decide", "sla", f"published window {c['sla']} ({c['sla_hours']}h)")
    return a


def h_limits(a, s, said):
    lim = data.CUSTOMER["limits"]
    a.say = (f"Your limits are AED {lim['per_txn']:g} per transfer, AED {lim['per_day']:g} a "
             f"day and AED {lim['per_month']:g} a month. Completing your verification raises "
             f"them.")
    a.outcome = "QUOTED"
    a.step("decide", "limits", "read from the customer's own mandate", tag="db")
    return a


def h_beneficiaries(a, s, said):
    people = data.sendable()
    a.say = ("You can send to " +
             ", ".join(b["name"] for b in people[:4]) +
             (" and others." if len(people) > 4 else "."))
    a.outcome = "LISTED"
    a.options = [{"kind": "send_to", "label": f"Send to {b['name']}", "payee": b["name"]}
                 for b in people[:4]]
    a.step("resolve", "roster", f"{len(people)} saved beneficiaries", tag="db")
    return a


_GREETING = re.compile(r"^\W*(hi+|hello+|hey+|hiya|good\s+(?:morning|afternoon|evening)|salam|"
                       r"assalam\w*(?:\s+\w+)?|namaste|namaskar|marhaba)\W*$", re.I)
_GREET_BACK = {"salam": "Wa alaikum assalam", "namaskar": "Namaste"}


def h_help(a, s, said):
    """Answer an open request for help with what we can actually do — and with THIS
    customer's situation, not a generic menu.

    Leading with their most recent payment is the point: someone who says "I need help" has a
    reason, and nine times out of ten it is the thing that just happened.
    """
    txns = data.transactions_for()
    last = txns[0] if txns else None
    g = _GREETING.match(said or "")
    # A greeting gets a greeting back, the same way every time: "Hi" and "Hello" were each
    # answered with a different random line and neither said hello.
    bits = ([f"{_GREET_BACK.get(g.group(1).lower(), g.group(1).strip().title())}! How can I help "
             f"with your payments today?"] if g else [voice.pick("greet", a.session)])
    if g:                                           # greeted back in their own word, then varied
        _v = phrasebank.say("greet_back", a.lang or "en", a.session or "default", opener=False)
        if _v:
            bits = [f"{_GREET_BACK.get(g.group(1).lower(), g.group(1).strip().title())}! " + _v]
    if last:
        p = last.public()
        bits.append(f"Your most recent one is {p['amount']:g} {p['currency']} to "
                    f"{p['payee']} ({p['status'].lower()}, {p['when']}).")
    bits.append(voice.pick("greet_offer", a.session))
    a.say = _tr(a, "help_open", " ".join(bits))
    a.outcome = "OFFERED"
    a.step("resolve", "context",
           f"{len(txns)} recent payments" + (" (live ledger)" if data.LIVE_HISTORY
                                             else " (demo data)"), tag="db")
    a.options = []
    if last:
        a.options.append({"kind": "pick_txn",
                          "label": f"About my {last.amount:g} {last.currency} to {last.payee}",
                          "ref": last.ref})
    a.options += [
        {"kind": "ask_text", "label": "I want a refund", "text": "i want a refund"},
        {"kind": "ask_text", "label": "Offers near me", "text": "any offers near me"},
        {"kind": "ask_text", "label": "Send money", "text": "i want to send money"},
    ]
    return a


def _xb_corridor(said: str, s) -> str | None:
    """The corridor an offers question is about: a named country, a currency, or None."""
    if s.corridor and s.corridor != "AE-AE":
        return s.corridor
    return next((v for k, v in _CCY_CORRIDOR.items() if re.search(rf"\b{k}\b", (said or "").lower())), None)


def _h_xb_offers(a, s, said):
    """Offers on sending money ABROAD — the Incentive Engine's CROSS_BORDER_TRANSFER campaigns,
    not the nearby-merchant ones. One country if named; otherwise every corridor we serve."""
    corridor = _xb_corridor(said, s)
    corridors = [corridor] if corridor else [k for k in CORRIDORS if k != "AE-AE"]
    found = []
    for cor in corridors:
        offs = _corridor_offers(a, cor, s.amount or 0, CORRIDORS[cor]["fee_aed"])
        if offs:
            found.append((cor, offs))
    name = lambda cor: CORRIDORS[cor]["name"].split("→")[-1].strip()
    if not found:
        if corridor:
            a.say = _tr(a, "xb_offers_none_one",
                        f"There are no offers on transfers to {name(corridor)} right now. The fee is "
                        f"AED {CORRIDORS[corridor]['fee_aed']:.2f}.", dest=name(corridor),
                        fee=f"{CORRIDORS[corridor]['fee_aed']:.2f}")
        else:
            a.say = _tr(a, "xb_offers_none", "There are no offers on international transfers right now.")
        a.outcome = "NONE"
        return a
    lines = []
    for cor, offs in found:
        bits = "; ".join(f"{o['code']} — {o['offer']['label']}" for o in offs[:3])
        lines.append(f"{name(cor)}: {bits}")
    auto = all(o["auto"] for _, offs in found for o in offs)
    a.say = _tr(a, "xb_offers",
                ("Offers on sending money abroad — " if not corridor else f"Offers on sending money to {name(corridor)} — ")
                + " | ".join(lines) + "."
                + (" They apply automatically." if auto else " Enter the promo code on the send screen to use one."),
                list=" | ".join(lines))
    a.outcome = "OFFERED"
    a.options = [{"kind": "ask_text", "label": f"Send money to {name(cor)}", "text": f"send money to {name(cor).lower()}"}
                 for cor, _ in found[:3]]
    a.step("resolve", "xb.offers", f"cross-border campaigns for {', '.join(c for c, _ in found)} — shown, "
           f"not applied", tag="db")
    return a


def h_offers(a, s, said):
    """Discovery. Nearest merchants first, with their LIVE campaigns from the Incentive
    Engine where it answers — the same `/customer/list` the app calls on screen-load, with no
    amount, so the customer sees what exists before deciding what to spend."""
    # "pakistan paisa bhejne mai koi offer hai", "any offer on sending money abroad" — offers on
    # TRANSFERS ABROAD, which are cross-border campaigns, not the shops near them.
    if _xb_corridor(said, s) or _ABROAD.search(said or ""):
        return _h_xb_offers(a, s, said)
    near = data.nearby()
    if not near:
        # Empty has two very different meanings and the customer deserves the right one.
        # If the lookup FAILED we must not imply there is nothing near them; if it succeeded
        # and found nothing, say that. Either way, never index into an empty list — this
        # used to raise IndexError and take the whole request down.
        why = data.payable_error()
        a.step("resolve", "nearby.none",
               f"merchant lookup failed ({why})" if why else "no payable merchants nearby",
               tag="guard" if why else "db")
        a.say = ("I can't reach your account right now, so I can't show what's near you. "
                 "Please try again in a moment." if why else
                 "I can't see any merchants near you that you can pay right now.")
        a.outcome = "UNAVAILABLE" if why else "NONE_NEARBY"
        a.options = []
        return a
    a.step("resolve", "nearby", f"{len(near)} merchants{_within(near)}, "
                                f"nearest first", tag="db")

    live = attach_offers(near)
    if live:
        a.step("resolve", "incentive.live",
               f"{live} of {len(near)} merchants carry a live campaign from the incentive "
               f"engine", tag="db")
    else:
        # Say so rather than presenting the catalogue as if it were live. A customer shown an
        # offer that the engine would refuse at the till is a complaint, not a feature.
        why = incentive.last_error() or "no merchant ids configured"
        a.step("resolve", "incentive.offline",
               f"incentive engine not consulted ({why}) — showing the indicative catalogue, "
               f"and the exact discount is confirmed before any payment", tag="guard")

    first = near[0]
    a.say = (f"{first['name']}" + (f" is {first['metres']}m away" if first.get("metres") is not None else " is near you")
             + (f" — {first['offer']['label']}." if first.get("offer") else "."))
    a.outcome = "LISTED"
    a.options = [{"kind": "pay_merchant", "label": f"{m['name']}" + _m(m)
                  + (f" · {m['offer']['label']}" if m.get("offer") else ""),
                  "mid": m["mid"], "merchant": m["name"]} for m in near]
    return a


#: How many rows a customer gets when they did not say. Five is a screenful on a phone and
#: still enough to recognise the one they are thinking of; the answer always says how many
#: more there are, so the default never looks like the whole truth.
#: How a status reads in a sentence: "your last 3 FAILED payments".
PUBLIC_STATUS_WORD = {"FAILED": "failed", "PENDING": "pending", "ON_HOLD": "on hold",
                      "REVERSED": "refunded", "SUCCESS": "delivered"}

HISTORY_DEFAULT = 5
HISTORY_MAX = 20


#: Someone who is NOT the customer. If one of these is named and we could not resolve it to a
#: payee on their own roster, the question is about a third party.
_THIRD_PARTY = re.compile(
    r"\b(?:someone|somebody|some\s*one|another\s+(?:person|user|customer|account)|"
    r"other\s+(?:people|users|customers)|his|her|their|my\s+(?:wife|husband|brother|sister|"
    r"mother|father|friend|colleague|son|daughter|boss|neighbou?r))\b",
    re.IGNORECASE)

#: An account or wallet named by number. We never filter anyone's history by a wallet id —
#: there is no code path for it — so a question phrased this way cannot be answered at all,
#: and answering it with the asker's own rows would be read as answering it about that wallet.
_OTHER_ACCOUNT = re.compile(
    r"\b(?:wallet|account|a/?c|mobile|number|id)\s*(?:no\.?|number|#)?\s*[:#]?\s*\d{6,}",
    re.IGNORECASE)

#: A capitalised or possessive name, as a last resort. Only consulted when nothing resolved.
_POSSESSIVE = re.compile(r"\b([a-z]{3,})'s\b", re.IGNORECASE)


def _named_someone_else(said: str, s) -> str | None:
    """The unambiguous half of `_unresolved_subject`: a third party named outright, or an
    account named by number.

    Separate from the full check because the full check ends with a stray-name heuristic
    that subtracts the vocabulary of a HISTORY request. Reused on a spend question, that
    heuristic reads the leftover words of "how much did i spend on food" as somebody's name
    and refuses to answer a customer's question about their own money. The signals here
    carry no such dependency on which kind of question is being asked.
    """
    low = " ".join((said or "").lower().split())
    if _OTHER_ACCOUNT.search(low):
        return "that account"
    if s.payee or s.merchant:
        return None                     # resolved to someone on their own roster: fine
    m = _THIRD_PARTY.search(low)
    if m:
        return m.group(0)
    m = _POSSESSIVE.search(low)
    if m and m.group(1) not in ("today", "yesterday", "month", "week", "year", "it", "that"):
        return m.group(1).title()
    return None


def _edits_within(a: str, b: str, limit: int) -> bool:
    """Is `a` reachable from `b` in at most `limit` single-character edits?

    Bounded Levenshtein, stopped as soon as the best possible outcome exceeds the limit —
    this runs against a few hundred vocabulary words per leftover token, and the full
    matrix for pairs that obviously cannot match is wasted work.
    """
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def _misspelt(word: str) -> bool:
    """Is this a typo of a word we know, rather than somebody's name?

    The heuristic below treats any word it does not recognise as a name, and that is how
    "yes show me the detials" was answered "I don't have anyone called Detials on your
    list" — one transposed pair, and a request for details became a stranger. Customers
    type one-handed on a phone; a vocabulary that only matches exact spellings will keep
    finding new people.

    Deliberately strict about length. Two edits on a four-letter word reaches most of the
    language, so short words must match almost exactly, and a word that IS on the roster is
    never a typo however close it looks to something else.
    """
    w = (word or "").lower()
    if len(w) < 4:
        return False
    if any(w == r for r in _roster_words()):
        return False
    limit = 1 if len(w) <= 6 else 2
    return any(_edits_within(w, v, limit) for v in _HISTORY_VOCAB if abs(len(v) - len(w)) <= limit)


def _roster_words() -> frozenset:
    """Name parts of the people this customer actually pays. A real name must survive the
    typo check even when it is one edit from an ordinary word."""
    global _ROSTER_WORDS
    if _ROSTER_WORDS is None:
        words = set()
        try:
            for p in data.payees():
                for t in slotlib.token_list((getattr(p, "name", "") or "").lower()):
                    if len(t) > 2:
                        words.add(t)
        except Exception:                                     # noqa: BLE001
            pass
        _ROSTER_WORDS = frozenset(words)
    return _ROSTER_WORDS


_ROSTER_WORDS = None


def _unresolved_subject(said: str, s) -> str | None:
    """Did they name somebody we could not turn into a filter?

    This exists because of a specific wrong answer, not a hypothetical one. "show transactions
    for wallet 9876543210" and "what did my wife spend last month" both listed the ASKER's own
    payments, in full, with no sign that the filter had been dropped. Nothing leaked — every
    row comes from `transactions_for()`, which is scoped to the authenticated customer — but a
    list presented in answer to a question about someone else will be read as being about them.
    Silently answering a different question is its own kind of wrong.
    """
    low = " ".join((said or "").lower().split())
    if _OTHER_ACCOUNT.search(low):
        return "that account"
    if s.payee or s.merchant:
        return None                     # it resolved to someone on their own roster: fine
    m = _THIRD_PARTY.search(low)
    if m:
        return m.group(0)
    m = _POSSESSIVE.search(low)
    if m and m.group(1) not in ("today", "yesterday", "month", "week", "year", "it", "that"):
        return m.group(1).title()
    # A bare unresolved name: "show me ahmed transactions". The vocabulary of a history
    # request is small and closed, so anything left over after removing it is almost always
    # somebody's name — and if the roster had recognised it, s.payee would be set already.
    # Only look for a stray name when we actually UNDERSTOOD the rest of the sentence.
    #
    # The first version compared every token against an English vocabulary, so a Hindi,
    # Arabic or romanised-Hindi request — correctly classified at 0.99 — had all of its words
    # treated as unrecognised names and got "I don't have anyone called म on your list."
    # Seven of eight non-English history requests broke. Exactly the same mistake as the
    # earlier one where \b never fired against Devanagari: the MODEL speaks four languages,
    # this function spoke one.
    #
    # The structural fix is not a longer word list, which would break again on the next
    # language. A name is ONE unfamiliar word among familiar ones; a language we have no
    # vocabulary for is unfamiliar throughout. So the heuristic only speaks when most of the
    # sentence is recognised, and stays silent otherwise — where silence means "list their
    # own payments", the harmless outcome.
    toks = [t for t in slotlib.token_list(low) if not t.isdigit()]
    leftover = [t for t in toks if t not in _HISTORY_VOCAB and not _misspelt(t)]
    known = len(toks) - len(leftover)
    if leftover and (len(leftover) > 2 or known < len(leftover) * 2):
        return None
    if leftover:
        # A guess, not a finding. The vocabulary below is hand-maintained, so a word missing
        # from it looks exactly like somebody's name — "show me everything i paid today" was
        # refused because "everything" had been left out. A wrong refusal is a worse failure
        # than a wrong list here, so this answers with a QUESTION instead of a wall, and the
        # "?" suffix is how the caller tells the two apart.
        return leftover[0].title() + "?"
    return None


#: Every word a history request is made of. Kept explicit rather than inferred: the decision
#: this drives is "refuse and say why", so a word missing here costs a wrong refusal and the
#: list has to be reviewable at a glance.
_HISTORY_VOCAB_BASE = {
    "show", "me", "my", "mine", "i", "we", "us", "list", "give", "get", "see", "view",
    "want", "need", "would", "like", "can", "could", "you", "please", "pls", "kindly",
    "the", "a", "an", "of", "for", "to", "from", "in", "on", "at", "and", "or", "all",
    "last", "latest", "recent", "recently", "past", "previous", "older", "earlier", "old",
    "before", "after", "since", "next", "first", "top", "just", "only", "more", "rest",
    "transaction", "transactions", "txn", "txns", "payment", "payments", "paid", "pay",
    "transfer", "transfers", "transferred", "sent", "send", "spend", "spent", "spending",
    "history", "statement", "statements", "record", "records", "activity", "log",
    "what", "when", "where", "how", "which", "much", "many", "did", "do", "does", "have",
    "has", "had", "is", "are", "was", "were", "been", "money", "amount", "total",
    "today", "yesterday", "week", "weeks", "month", "months", "year", "years", "day", "days",
    "this", "that", "these", "those", "it", "there", "here", "gone", "went", "out",
    "wallet", "account", "balance", "aed", "dirhams", "dirham", "up", "pull", "scroll",
    "back", "going", "keep", "page", "ones", "one", "any", "some", "about", "sir", "hi",
    "hello", "thanks", "thank", "me's", "everything", "every", "anything", "something",
    "made", "make", "making", "look", "looking", "check", "checking", "find", "put",
    "through", "over", "under", "between", "during", "with", "without", "by", "my's",
    "dont", "don", "t", "s", "am", "be", "will", "shall", "may", "might", "must",
    # Words a FOLLOW-UP is made of. These were missing, so "yes show me the details" had
    # "details" treated as an unrecognised word and therefore as somebody's name — and a
    # single transposed pair ("detials") made it certain. A customer asking for more about
    # the answer we just gave is the commonest follow-up there is.
    "details", "detail", "info", "information", "breakdown", "summary", "receipt",
    "receipts", "copy", "proof", "reference", "ref", "id", "status", "full", "whole",
    "yes", "yeah", "yep", "ok", "okay", "sure", "no", "nope", "again", "instead",
    "rather", "else", "other", "another", "same", "different", "both", "each",
    "lot", "bit", "far", "now", "then", "ago", "latest", "newest", "oldest", "full",
    "complete", "detailed", "summary", "breakdown", "report", "receipt", "receipts",
    "whole", "entire", "bank", "card", "cash", "fee", "fees", "charge", "charges",
    # Romanised Hindi. Hinglish is Latin script, so it reaches the check above on the same
    # path English does and needs the same coverage; Devanagari and Arabic are handled by the
    # mostly-unrecognised rule instead.
    "mera", "meri", "mere", "mujhe", "main", "hum", "dikhao", "dikha", "dikhaiye", "do",
    "de", "dena", "karo", "kar", "kijiye", "chahiye", "hai", "hain", "ho", "tha", "thi",
    "kya", "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "aur", "saare", "sare",
    "sab", "pura", "poora", "hisaab", "pichle", "pichla", "pichli", "lenden", "paisa",
    "paise", "kitna", "kitne", "kharch", "wapas", "bheja", "bheje", "bheji", "purane",
    "purana", "nahi", "jo", "wo", "woh", "yeh", "ye", "abhi", "pehle", "baad", "wale",
    "wala", "wali", "jaldi", "bhai", "sir", "yaar", "please", "plz", "list", "total",
    "mahine", "mahina", "hafte", "hafta", "aaj", "kal", "din", "saal", "fail", "hue",
    "hua", "safal", "atka", "atke", "lena", "dekhna", "dekhne", "batao", "bata",
} | {w for words, _ in slotlib._STATUS_WORDS for w in words if w.isascii()
     for w in w.split()}





def _same_person(a: str, b: str) -> bool:
    """Is this the same person under two different renderings of their name?

    The roster says "Alex Morgan"; the ledger says "Alex Umashankar Morgan Umashankar
    Gaurishankar" — the full legal name as KYC captured it. Exact equality can never match
    those, so "my payments to Alex" answered "I can't find any" while eight of them sat in
    the ledger.

    The rule is subset, not overlap: every token of the shorter name must appear in the
    longer. Overlap alone would merge "Alex Morgan" with "Alex Sharma" on the strength of
    a shared first name, and attributing one person's payments to another is a worse failure
    than finding none.
    """
    ta = slotlib.tokens(a, 2)
    tb = slotlib.tokens(b, 2)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


def h_history(a, s, said):
    """A LIST of the customer's own payments.

    Distinct from h_txn, which answers about ONE. The split exists because every answer here
    is a summary the customer scans, not a verdict about a single payment — and because
    collapsing "show me my last five" into "here is your most recent one" was the complaint
    that started this.

    Nothing outside this customer's own ledger is ever reachable from here: the rows come
    from `transactions_for`, which is scoped to the authenticated customer id, and each row
    is published through `Txn.public()` so internal position and partner names stay out.
    """
    other = _unresolved_subject(said, s)
    if other and other.endswith("?"):
        name = other[:-1]
        a.step("resolve", "history.who",
               f"'{name.lower()}' is not on this customer's roster and is not a word a "
               f"history request is normally made of — asking rather than listing everything "
               f"as if it answered", tag="guard")
        a.say = (f"I don't have anyone called {name} on your list. I can only show payments "
                 f"from your own wallet — want me to show those?")
        a.outcome = "NEED_INFO"
        a.options = [{"kind": "ask_text", "label": "Yes, show mine",
                      "text": "show me my last 5 transactions"}]
        conversation.hold(a.session, Intent.TXN_HISTORY.value, {}, None)
        return a
    if other:
        a.step("resolve", "history.scope",
               f"names {other}, which did not resolve to anyone on this customer's roster — "
               f"declining rather than answering with their own rows, which would read as an "
               f"answer about {other}", tag="guard")
        a.say = (f"I can only show payments from your own wallet, so I can't pull up "
                 f"{other}'s." if other != "that account" else
                 "I can only show payments from your own wallet — I can't look up another "
                 "account.")
        a.say += " I can show you yours, though."
        a.outcome = "REFUSED_SCOPE"
        a.options = [{"kind": "ask_text", "label": "Show my payments",
                      "text": "show me my last 5 transactions"}]
        return a

    want = min(s.count or HISTORY_DEFAULT, HISTORY_MAX)
    # Ask for more than we will show: "and 12 older ones" is only honest if we counted them.
    rows = data.transactions_for(limit=HISTORY_MAX + 5)

    if s.window_days:
        horizon = s.window_days * 24
        rows = [t for t in rows if t.hours_ago <= horizon]
    if s.status:
        rows = [t for t in rows if t.state.value == s.status]
    # A payee filter that matches nothing yields an EMPTY list, which the no-rows branch below
    # reports honestly ("no payments to Fatima"). It must never fall back to the unfiltered
    # set — "show me Ahmed's" answered with every payment the customer ever made was the bug.
    if s.payee:
        rows = [t for t in rows if _same_person(t.payee, s.payee)]
    elif s.merchant:
        rows = [t for t in rows if _same_person(t.payee, s.merchant)]
    else:
        # Nobody was NAMED, but a kind may have been. "show me last 3 transactions of
        # shop" asked for merchants and got a payment to a person back, because the word
        # "shop" reached no filter at all — the list looked like an answer and was a
        # different question's answer.
        _kind = _scope_from(said)
        if _kind == "merchant":
            rows = [t for t in rows if t.is_merchant]
        elif _kind == "person":
            rows = [t for t in rows if not t.is_merchant]

    # "your last 2 FAILED payments", not "your last 2 payments failed". The status is an
    # adjective on the noun; everything else trails it.
    adj = PUBLIC_STATUS_WORD.get(s.status or "", "")
    scope = []
    if s.time_window:
        scope.append(s.time_window)
    if s.payee or s.merchant:
        scope.append(f"to {s.payee or s.merchant}")
    else:
        _k = _scope_from(said)
        if _k == "merchant":
            scope.append("at merchants")
        elif _k == "person":
            scope.append("to people")
    scoped = " ".join(scope)

    if not rows:
        a.say = (" ".join(f"I can't find any {adj} payments {scoped}".split()) + "."
                 if (scoped or adj)
                 else _tr(a, "history_none",
                          "I can't find any payments on your account yet."))
        a.outcome = "NONE"
        a.step("resolve", "history.none", f"no rows for {scoped or 'this customer'}", tag="db")
        a.options = [{"kind": "ask_text", "label": "Show everything",
                      "text": "show me all my transactions"}]
        conversation.hold(a.session, Intent.TXN_HISTORY.value, s.as_dict(), None)
        return a

    shown, rest = rows[:want], rows[want:]
    a.step("resolve", "history",
           f"{len(rows)} payments{' ' + scoped if scoped else ''}, showing {len(shown)}",
           tag="db" if not data.LIVE_HISTORY else "db")

    total = sum(t.amount for t in shown)
    head = voice.pick("here_they_are", a.session).format(
        n=f"{len(shown)} {adj}".strip() if adj else len(shown),
        scope=f" {scoped}" if scoped else "")
    # The total is COMPOSED, never phrased by the pool — it is the one factual claim in the
    # sentence and it has to be byte-identical every time.
    a.say = _tr(a, "history_head",
                f"{head} AED {total:,.2f} in total. "
                f"{voice.pick('list_tail', a.session)}",
                n=len(shown), cur="AED", total=f"{total:,.2f}")
    a.outcome = "LISTED"
    # The rows themselves ride on the answer as structured data, so the UI can lay them out
    # as a statement rather than the assistant reading sixteen numbers out in a paragraph.
    a.rows = [t.public() for t in shown]

    a.options = [{"kind": "pick_txn",
                  "label": f"{t.amount:g} {t.currency} to {t.payee} · {t.public()['status']}",
                  "ref": t.ref} for t in shown]
    if rest:
        a.options.append({"kind": "ask_text",
                          "label": f"Show {min(len(rest), want)} older",
                          "text": f"show me {want + len(shown)} transactions"})

    # Stay in the thread. "and before that", "what about last month", "why did that one fail"
    # are all continuations, and the frame is what lets them be read as such instead of being
    # thrown at the classifier cold.
    conversation.hold(a.session, Intent.TXN_HISTORY.value,
                      {**s.as_dict(), "shown": len(shown)}, None)
    return a


# ─── splitting a bill ────────────────────────────────────────────────────────────────────
#: The backend caps a post-transaction split at 24 participants and a pre-transaction one at
#: 14. Enforced here too, so the customer is told before a request is built and rejected.
SPLIT_MAX_POST = 24
SPLIT_MAX_PRE = 14


#: Words that mean "divide this between us", in every language the assistant takes. The
#: send-side equivalent (`_has_transfer_verb`) is English-only because it was written before
#: the model spoke four languages; this one is not making that mistake again.
_SPLIT_VERBS = (
    "split", "splitting", "divide", "dividing", "share", "sharing", "halves", "half",
    "fifty fifty", "50 50", "60 40", "sixty forty", "chip in", "chips in", "chipped in",
    "each", "between us", "among us", "owes me", "owe me", "their share", "his share",
    "her share", "my share", "collect", "collecting", "vasool", "bantna", "baanto",
    "baant", "banto", "hissa", "hisse", "hisaab", "aadha", "barabar",
    "बांट", "बांटो", "बाँट", "हिस्सा", "हिस्से", "आधा", "बराबर", "वसूल", "हिसाब",
    # "قسم" is the bare stem, and it has to be here rather than only the inflected forms:
    # the list held "اقسم" and "اقسمها" but not "قسمها", which is how an Arabic speaker
    # actually says "split it" — so the reported bug's Arabic variant fell out of scope at
    # the last turn while English and Hindi both worked.
    "قسم", "اقسم", "تقسيم", "قسمة", "نصيب", "النصف", "بالتساوي", "شارك", "اقسمها",
)


#: Which of the three a short reply names. Only consulted when we have just asked, so it
#: can afford to be generous — "shop", "store", "dukan", "the second one" all mean the same
#: thing in that one position and nowhere else.
_PAY_KIND = (
    ("bill", (r"\bbills?\b", r"\bbiller\b", r"\butility\b", r"\bbijli\b",
              r"\u092c\u093f\u0932", r"\u0641\u0627\u062a\u0648\u0631")),
    ("merchant", (r"\bshops?\b", r"\bstores?\b", r"\bmerchants?\b", r"\bcounter\b",
                  r"\btill\b", r"\bduka+n\b", r"\u0926\u0941\u0915\u093e\u0928",
                  r"\u0645\u062a\u062c\u0631", r"\u0627\u0644\u0645\u062d\u0644")),
    ("person", (r"\bsomeone\b", r"\bsomebody\b", r"\bperson\b", r"\bpeople\b",
                r"\bfriend\b", r"\bmy list\b", r"\bcontact\b", r"\bkisi ko\b",
                r"\u0915\u093f\u0938\u0940 \u0915\u094b", r"\u0634\u062e\u0635")),
)


def _pay_kind(said: str) -> str | None:
    t = _flat(said)
    if not t or len(t.split()) > 6:
        return None
    for kind, pats in _PAY_KIND:
        if any(re.search(pat, t) for pat in pats):
            return kind
    return None


def _has_bill_word(said: str) -> bool:
    """Does the sentence contain the word "bill" at all, in any of the four languages?"""
    t = _flat(said)
    # Recurring payments people call by what they are, not "bill": "pay my gym membership"
    # asked "is this a bill, someone on your list, or a shop?" — it is a bill.
    return bool(re.search(r"\b(?:bills?|membership|subscription|rent|tuition|school fees?|"
                          r"insurance|premium|emi|instal+ments?)\b", t)) or any(
        w in t for w in ("bijli", "\u092c\u093f\u0932", "\u0641\u0627\u062a\u0648\u0631"))


def _has_split_verb(said: str) -> bool:
    low = _flat(said)
    return any(v in low for v in _SPLIT_VERBS)


# ─── yes and no ──────────────────────────────────────────────────────────────────────────
# A bare "yes" or "cancel" is an ANSWER to what we just offered, not a new request. Handled
# before the classifier and regardless of what it would say, because on their own these words
# carry no subject at all: "yes" scored as a greeting and "cancel"/"no"/"stop" scattered
# across txn_failed, txn_history and txn_on_hold — so saying no to a payment started talking
# about an unrelated one, and in one case offered a refund.
#: A bare affirmation. It carries no subject of its own, so it can only mean the thing we
#: just offered, whatever that was.
_YES = re.compile(
    r"^(?:yes|yep|yeah|yup|ya|ok|okay|k|sure|please do|go ahead|do it|confirm|confirmed|"
    r"that's right|thats right|correct|right|fine|perfect|"
    r"haan|han|haa|ji|ji haan|theek hai|thik hai|kar do|ok kar do|"
    r"हाँ|हां|जी|ठीक है|कर दो|बिलकुल|"
    r"نعم|أجل|تمام|حسنا|موافق)"
    r"[\s.!،]*$", re.IGNORECASE)

#: Confirmations that NAME what they are confirming, and the proposal kinds each one is
#: allowed to confirm.
#:
#: These used to sit in `_YES` alongside the bare affirmations, and that was a money bug,
#: not just a context one. "split it" matched the same pattern as "yes", so with a SEND
#: proposal still standing it confirmed the send — a sentence asking to split a bill moved
#: money to a person instead. The reported conversation was the milder form of it:
#:
#:     send 50 to alex / split it with alex / bill pay karna hai / Home electricity
#:     "split it"  ->  CONFIRMED, empty reply, the transaction split from four turns ago
#:
#: A phrase that names a verb is a statement about WHICH action, so it may only ever
#: confirm that action. Anything else is a fresh request and goes to the classifier, where
#: the bill on screen can be read properly.
_YES_FOR_KIND: tuple[tuple[re.Pattern, frozenset], ...] = (
    (re.compile(r"^(?:send it|bhej do|भेज दो|أرسلها)[\s.!،]*$", re.I),
     frozenset({"send", "send_to"})),
    (re.compile(r"^(?:pay it|bill pay kar do)[\s.!،]*$", re.I),
     frozenset({"send", "send_to", "pay_merchant", "bill_pay", "bill_schedule"})),
    (re.compile(r"^(?:split it|baant do|बाँट दो)[\s.!،]*$", re.I),
     frozenset({"split", "split_with", "bill_split"})),
)

_NO = re.compile(
    r"^(?:no|nope|nah|not now|no thanks|no thank you|cancel|cancel it|cancel that|stop|"
    r"stop it|forget it|nevermind|never mind|leave it|drop it|abort|don'?t|do not|"
    r"not yet|maybe later|later|"
    r"nahi|nahin|nahi chahiye|rehne do|chod do|cancel kar do|baad me|"
    r"नहीं|नही|रहने दो|छोड़ दो|बाद में|कैंसिल|"
    r"لا|لا شكرا|إلغاء|الغاء|توقف|ليس الآن|لاحقا)"
    r"[\s.!،]*$", re.IGNORECASE)


def _is_yes(said: str, held: dict | None = None) -> bool:
    """Is this sentence a yes to the offer `held`?

    A bare affirmation says yes to anything. A confirmation that names an action only says
    yes to THAT action — see `_YES_FOR_KIND` for why that distinction is load-bearing.
    """
    low = " ".join((said or "").lower().split())
    if _YES.match(low):
        return True
    for pat, kinds in _YES_FOR_KIND:
        if pat.match(low):
            # With nothing on offer there is nothing to agree to, so "split it" stays a
            # request and the classifier gets to read it.
            return bool(held) and (held.get("kind") or "") in kinds
    return False


def _is_no(said: str) -> bool:
    """A refusal, alone or as a short run of refusals: "no", "no, cancel", "no no stop".
    "no, cancel" — what the app's Cancel button sends — used to miss because of the comma,
    went to the classifier, scored refund_request at 88%, and the reply was about refunding
    an earlier payment while the customer was cancelling a new one."""
    low = " ".join((said or "").lower().split())
    parts = [p for p in re.split(r"[,،]\s*|\s+(?=(?:no|cancel|stop)\b)", low) if p.strip()]
    return bool(parts) and len(parts) <= 3 and all(_NO.match(p.strip()) for p in parts)


#: Affirmations, refusals and politeness, which can prefix any request and are not names.
#: "ok show me my transactions" answered "I don't have anyone called Ok on your list" —
#: the predicted failure of a hand-maintained word list, so this one is DERIVED from the
#: yes/no patterns that already enumerate them rather than written out a second time.
_CONVERSATIONAL = (
    {w for w in re.findall(r"[a-z']{2,}", _YES.pattern)} |
    {w for w in re.findall(r"[a-z']{2,}", _NO.pattern)} |
    {"alright", "right", "well", "so", "actually", "also", "then", "now", "hey", "hiya",
     "morning", "afternoon", "evening", "cool", "great", "good", "nice", "sorry",
     "excuse", "kindly", "quickly", "asap", "urgent", "urgently", "bhai", "yaar",
     "acha", "accha", "arre", "haan", "theek", "thik"}
)

#: "shop", "merchant", "dukan" name a KIND of payee, not a particular one. A question can
#: be scoped to merchants or to people without naming anybody.
_SCOPE_MERCHANT = re.compile(
    r"\b(shop|shops|shopping|store|stores|merchant|merchants|retail|"
    r"dukan|dukaan|dukano|\u0926\u0941\u0915\u093e\u0928|\u092e\u0930\u094d\u091a\u0947\u0902\u091f|"
    r"\u0645\u062a\u062c\u0631|\u0645\u062a\u0627\u062c\u0631|\u0627\u0644\u0645\u062d\u0644)\b", re.I)
_SCOPE_PERSON = re.compile(
    r"\b(person|people|friend|friends|someone|somebody|who|whom|contacts?|"
    r"kisko|kis ko|logon|dost|\u0915\u093f\u0938\u0915\u094b|\u0932\u094b\u0917\u094b\u0902|\u0926\u094b\u0938\u094d\u0924|"
    r"\u0634\u062e\u0635|\u0623\u0634\u062e\u0627\u0635|\u0623\u0635\u062f\u0642\u0627\u0621)\b", re.I)


def _scope_from(said: str) -> str:
    """Is this question about merchants, about people, or about everything?

    Shared by the history list and the rankings, because the two had drifted: rankings
    understood "which shop do I spend the most at" while the history list dropped the word
    entirely — "show me last 3 transactions of shop" returned a payment to a person, which
    is a list that silently answers a narrower question than the one asked.
    """
    t = _flat(said)
    if _SCOPE_MERCHANT.search(t):
        return "merchant"
    if _SCOPE_PERSON.search(t):
        return "person"
    return "any"


#: Words naming a KIND of payee — "shop", "merchant", "people". Derived from the scope
#: patterns rather than written out again, so the two cannot drift apart.
#:
#: They have to be here or the stray-name heuristic treats them as somebody's name:
#: "last 5 transactions to people" was answered "I don't have anyone called People on your
#: list", and "pichle 3 dukan ke transactions" blamed a person called Dukan. Exactly the
#: failure that produced "I don't have anyone called Detials on your list" — an unfamiliar
#: word is only a name if it is not one of OUR words first.
_SCOPE_WORDS = {w for pat in (_SCOPE_MERCHANT, _SCOPE_PERSON)
                for w in re.findall(r"[a-z\u0900-\u097f\u0600-\u06ff]{2,}", pat.pattern)}

_HISTORY_VOCAB = _HISTORY_VOCAB_BASE | _CONVERSATIONAL | _SCOPE_WORDS


def _split_people_options() -> list[dict]:
    """Who to offer as participants: people they already split with first, then the roster.

    Ordered that way on purpose — somebody you have split a bill with before is far more
    likely to be the person you are splitting this one with, and the real balances come back
    from the same call, so the list can say "you owe Alex 11.20" rather than just a name.
    """
    from split import api as split_api
    out: list[dict] = []
    seen: set[str] = set()
    for p in split_api.people():
        uid, name = p.get("userId"), p.get("name") or ""
        if not uid or uid in seen:
            continue
        seen.add(uid)
        bal = p.get("netBalance") or 0
        note = (f" · you owe {abs(bal):g}" if bal < 0
                else f" · owes you {bal:g}" if bal > 0 else "")
        out.append({"kind": "split_with", "label": name.split()[0] + note,
                    "userId": uid, "name": name})
    for b in data.sendable():
        uid = b.get("userId")
        if uid and uid not in seen:
            seen.add(uid)
            out.append({"kind": "split_with", "label": b["name"],
                        "userId": uid, "name": b["name"]})
    return out[:8]


#: A trailing "and also split it" clause. Two requests in one sentence, and the model sees
#: only the last — "MClothes ko paise bhejne hai aur offer lagake split bhi karna hai" came
#: back as split_payment at 99% with no merchant, because "split bhi karna hai" is the final
#: clause and the classifier picks one label. The order matters though: you cannot split a
#: merchant payment that has not happened, so the PAYMENT runs first and the split follows
#: it. Detected on the conjunction rather than guessed at.
_ALSO_SPLIT = re.compile(
    r"\b(?:and|also|plus|aur|and\s+also|then)\b[^.]*\b(?:split|divide|share|baant\w*|"
    r"banto|hissa)\b"
    r"|\b(?:split|share|baant\w*)\b[^.]*\b(?:bhi|too|as\s+well|also)\b"
    r"|(?:और|तथा)[^।]*(?:बांट|हिस्सा)|(?:बांट|हिस्सा)[^।]*भी"
    r"|(?:و|ثم)[^.]*(?:اقسم|تقسيم)",
    re.IGNORECASE)


def _also_wants_split(said: str) -> bool:
    return bool(_ALSO_SPLIT.search(" ".join((said or "").lower().split())))


def h_split(a, s, said):
    """Divide a payment between this customer and people they know.

    Two different things wear the word "split". Splitting a BILL means collecting from the
    others and then paying the biller — nothing has moved yet, so there is no transaction to
    reference and no QR to scan, and it is handled by `h_bill_split`. Splitting a merchant
    payment means dividing one that has already happened. The branch below picks between them
    on what the customer actually named, because answering "Who are you splitting AED 450
    with?" to someone who said "split my electricity bill" skips the only question that
    matters: which bill.
    """
    from bills import due as bills_due
    low = " ".join((said or "").lower().split())

    # "Pay X and split it" is a PAYMENT followed by a split. A merchant payment cannot be
    # split before it exists, so the payment goes first and the intent to split is
    # remembered — the offer then fires the moment the payment settles.
    if s.merchant and _also_wants_split(said):
        conversation.note_subject(a.session, "split_after", s.merchant)
        a.step("understand", "compound",
               f"two requests in one sentence: pay {s.merchant}, then split it — paying "
               f"first, because a payment cannot be split before it exists", tag="rule")
        a.intent = Intent.PAY_MERCHANT.value
        out = h_pay_merchant(a, s, said)
        if out.say:
            out.say += _tr(a, "will_split_after",
                           " Once it's paid I'll split it with whoever you like.")
        return out

    # What is in hand beats what happened last. "split bhi karna hai" said while we are
    # part-way through paying a bill is about THAT bill — before this it reached past the
    # bill on screen and split the most recent merchant payment instead, which is the
    # assistant answering a question nobody asked.
    # What THIS SENTENCE named, as opposed to what the frame has carried forward. The merged
    # slots still hold the txn_ref from an earlier turn, so asking "did they name a payment?"
    # of `s` always answered yes and this branch never fired.
    own = slotlib.extract(said, "")
    frame = conversation.get(a.session)
    if (not own.bill and not own.txn_ref and frame and frame.fresh
            and frame.intent in (Intent.BILL_PAY.value, Intent.BILL_DUE.value,
                                 Intent.BILL_SCHEDULE.value, Intent.BILL_SPLIT.value
                                 if hasattr(Intent, "BILL_SPLIT") else Intent.BILL_PAY.value)
            and frame.slots.get("bill")):
        s.bill = frame.slots["bill"]
        a.step("resolve", "split.in_hand",
               f"splitting {s.bill} — the bill this conversation is part-way through, not "
               f"the last payment that happened to be splittable", tag="rule")
        return h_bill_split(a, s, said)

    names_a_bill = bool(
        _has_bill_verb(said)
        or (s.bill and bills_due.find(s.bill))
        or re.search(r"\bbill\b|बिल|فاتورة", low))
    if names_a_bill and not s.txn_ref:
        summary = bills_due.summary()
        if not summary.get("unavailable") and (summary["due"] or summary["overdue"]):
            return h_bill_split(a, s, said)
        if (summary.get("unavailable") or not (summary.get("due") or summary.get("overdue"))) \
                and (s.bill or _has_bill_verb(said)):
            # A BILL was named and the bill list cannot be read. Falling through to the
            # merchant path split an unrelated shop payment instead — say what happened.
            a.step("resolve", "split.bills_unavailable",
                   summary.get("unavailable") or "no bill amount could be read from the biller", tag="guard")
            which = f"your {s.bill} bill" if s.bill else "that bill"
            a.say = (f"I can't check {which} right now, so I can't split it yet. Nothing has been "
                     f"sent to anyone — try again in a moment.")
            a.outcome = "UNAVAILABLE"
            return a
    return _h_split_merchant(a, s, said)


def h_bill_split(a, s, said):
    """Split a bill BEFORE paying it: which bill, then who, then ask them for their share.

    The order is the point. "Which bill" comes first because a customer with four billers
    has not told us anything by saying "split the bill", and the amount depends entirely on
    the answer. Only then does "who with" make sense.
    """
    from bills import billsplit, due as bills_due
    summary = bills_due.summary()
    payable = summary["overdue"] + summary["due"]

    # ── 1. WHICH bill ───────────────────────────────────────────────────────────────────
    chosen = None
    if s.bill or _has_bill_verb(said):
        hits = [h for h in bills_due.find(s.bill or said) if h in payable]
        if len(hits) == 1:
            chosen = hits[0]
    if chosen is None and len(payable) == 1:
        chosen = payable[0]
    if chosen is None:
        a.say = _tr(a, "which_bill", "Which bill do you want to split?")
        a.outcome = "NEED_INFO"
        a.step("act", "need.bill",
               f"{len(payable)} bills are payable — asking which, because the share depends "
               f"entirely on the answer", tag="guard")
        a.options = [{"kind": "ask_text",
                      "label": f"{r['nickname']} · AED {r['amount']:.2f}",
                      "text": f"split my {r['nickname']} bill"} for r in payable]
        conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), "bill")
        return a

    total = chosen["amount"]
    conversation.note_subject(a.session, "bill", chosen["nickname"])
    a.step("resolve", "billsplit.bill",
           f"{chosen['nickname']} · {chosen['biller']} · AED {total:.2f}", tag="db")

    # ── 2. WHO with ─────────────────────────────────────────────────────────────────────
    people = [p for p in (s.participants or []) if p.get("userId")]
    if not people:
        a.say = _tr(a, "split_with_who",
                    f"Who are you splitting the {chosen['nickname']} bill "
                    f"(AED {total:.2f}) with?",
                    bill=chosen["nickname"], cur="AED", amount=f"{total:.2f}")
        a.outcome = "NEED_INFO"
        a.step("act", "need.participants",
               "offering only people this customer already has a relationship with",
               tag="guard")
        a.options = _split_people_options()
        s.bill = chosen["nickname"]
        conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(),
                          "participants")
        return a

    # ── 3. the shares, and what will happen ─────────────────────────────────────────────
    amounts, mine = billsplit.shares(total, len(people))
    who = ", ".join(p["name"].split()[0] for p in people)
    a.proposal = {
        "kind": "bill_split", "bill": chosen["nickname"], "biller": chosen["biller"],
        "planId": chosen["planId"], "productId": chosen["productId"],
        "identifier": chosen["identifier"], "total": round(total, 2),
        "currency": "AED", "initiatorDisplayValue": mine,
        "participants": [{"userId": p["userId"], "name": p["name"], "displayValue": amt}
                         for p, amt in zip(people, amounts)],
    }
    a.say = _tr(a, "billsplit_ready",
                f"The {chosen['nickname']} bill is AED {total:.2f}. I'll ask {who} for "
                f"AED {amounts[0]:.2f}, you cover AED {mine:.2f}, and once their share "
                f"lands I'll pay {chosen['biller']} straight away.",
                bill=chosen["nickname"], cur="AED", total=f"{total:.2f}", who=who,
                each=f"{amounts[0]:.2f}", mine=f"{mine:.2f}", biller=chosen["biller"])
    a.outcome = "NEED_CONFIRMATION"
    a.options = [{"kind": "dismiss", "label": "Not now"}]
    a.step("decide", "billsplit.plan",
           f"collect {sum(amounts):g} from {len(people)}, then pay the biller — the bill is "
           f"NOT paid until the money is actually in", tag="rule")
    conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), None)
    return a


def _h_split_merchant(a, s, said):
    """Divide a merchant payment that has ALREADY happened.

    Note the direction, because it is the whole reason this is not SEND_MONEY: a split does
    not debit the customer. It records what others owe them for something they paid, and
    asks those people to settle. Reading "split 50 with Alex" as a transfer would take 50
    off the wrong person.

    Everything here goes through the real `/transaction/split-payment` API. The assistant
    proposes; the backend recomputes every share and rejects the request if its arithmetic
    disagrees with the figures the customer was shown.
    """
    from split import api as split_api

    # ── 1. WHAT is being split ──────────────────────────────────────────────────────────
    # A split divides a bill the customer paid a MERCHANT. Two consequences:
    #
    # 1. A person named in a split sentence is a PARTICIPANT, never the thing being split.
    #    Searching by `s.payee` made "split my last payment with Alex" pick the transfer
    #    TO Alex and then propose splitting it WITH Alex — splitting his own payment with
    #    him, which is not a thing.
    # 2. A wallet-to-wallet transfer is not a bill. Splitting money you sent a friend has no
    #    meaning, so only merchant payments are offered as targets.
    rows = data.transactions_for(limit=25)
    merchant_rows = [t for t in rows if t.is_merchant]
    # A payment can only be split once — the backend refuses a second one. Knowing that
    # BEFORE proposing is the difference between picking the next bill and walking the
    # customer through choosing shares only to be told no at the end.
    done = split_api.already_split()
    txn = None
    if s.txn_ref:
        txn = data.find_txn(s.txn_ref)
    if txn is None and s.merchant:
        txn = next((t for t in merchant_rows
                    if _same_person(t.payee, s.merchant) and t.ref not in done), None) \
            or next((t for t in merchant_rows if _same_person(t.payee, s.merchant)), None)
        if txn is None:
            a.say = f"I can't find a payment to {s.merchant} to split."
            a.outcome = "NONE"
            a.step("resolve", "split.no_txn", f"no merchant payment to {s.merchant}",
                   tag="db")
            a.options = [{"kind": "ask_text", "label": "Split my last payment",
                          "text": "split my last payment"}]
            return a
    # Nothing named. The payment to split is the one THIS conversation is about — never just
    # the newest one in history. Reported: "pay a merchant and split it with Alex", then "and
    # split with Alex" before the merchant had even been chosen, proposed asking Alex for 40
    # of an old AED 80 at MClothes. The merchant payment being set up was the subject; an old
    # bill was picked because it happened to be the latest.
    if txn is None:
        names = [p.get("name") for p in (s.participants or []) if p.get("name")]
        fresh = conversation.split_offer(a.session)                 # paid in this conversation
        if fresh:
            txn = data.find_txn(fresh["ref"])
            if txn is not None:
                a.step("resolve", "split.this_payment",
                       f"the merchant payment just made in this conversation ({fresh['ref']})", tag="rule")
                conversation.mark_split_offered(a.session)
                conversation.forget_subject(a.session, "split_after")
        if txn is None and _payment_in_progress(a.session):
            conversation.note_subject(a.session, "split_after", names or ["them"])
            who = " and ".join(n.split()[0] for n in names) or "them"
            a.step("decide", "split.after_payment",
                   "a payment is still being set up in this conversation — the split waits for "
                   "it rather than reaching back for an older bill", tag="guard")
            a.say = (f"Let's finish this payment first — as soon as it goes through, I'll split "
                     f"it with {who}.")
            a.outcome = "SPLIT_AFTER_PAYMENT"
            a.slots = dict(a.slots or {}, split_after=names)
            return a
        if txn is None and not _refers_to_past_payment(said):
            unsplit = [t for t in merchant_rows if t.ref not in done][:4]
            a.step("decide", "split.which",
                   "no payment named and none made in this conversation — asking which one "
                   "instead of picking the latest from history", tag="guard")
            a.say = ("Which payment should I split?" if unsplit else
                     "I can't see a merchant payment to split yet — pay a merchant and I'll split it.")
            a.outcome = "NEED_INFO" if unsplit else "NONE"
            a.options = [{"kind": "ask_text",
                          "label": f"{t.currency} {t.amount:g} at {t.payee} · {t.public()['when']}",
                          "text": f"split my payment {t.ref} with " + (" and ".join(names) or "them")}
                         for t in unsplit]
            return a
    if txn is None:
        unsplit = [t for t in merchant_rows if t.ref not in done]
        txn = unsplit[0] if unsplit else (merchant_rows[0] if merchant_rows else None)
    if txn is None:
        a.say = ("I can't see a merchant payment to split yet — a split divides a bill you "
                 "paid somewhere, not money you sent someone. Pay a merchant and I can "
                 "split it with whoever you like.")
        a.outcome = "NONE"
        a.step("resolve", "split.no_txn",
               f"{len(rows)} payments but none to a merchant", tag="db")
        return a
    if not txn.is_merchant:
        a.say = (f"That one was a transfer to {txn.payee.split()[0]}, not a bill — there's "
                 f"nothing to divide. Want me to split your last merchant payment instead?")
        a.outcome = "NONE"
        a.step("resolve", "split.not_a_bill",
               f"{txn.ref} is wallet-to-wallet, not a merchant payment", tag="guard")
        a.options = [{"kind": "ask_text", "label": "Split my last merchant payment",
                      "text": "split my last merchant payment"}]
        return a
    if txn.ref in done:
        a.step("resolve", "split.already",
               f"{txn.ref} already has a split — the backend refuses a second one", tag="guard")
        a.say = (f"You've already split that {txn.currency} {txn.amount:g} at {txn.payee}. "
                 f"Want to see how that one is going, or split a different payment?")
        a.outcome = "ALREADY_SPLIT"
        a.options = [{"kind": "ask_text", "label": "How's that split going?",
                      "text": "who owes me money"}]
        a.options += [{"kind": "ask_text",
                       "label": f"Split the {t.amount:g} at {t.payee.split()[0]}",
                       "text": f"split my payment to {t.payee}"}
                      for t in merchant_rows if t.ref not in done][:2]
        return a

    if s.amount and s.amount > txn.amount:
        a.step("decide", "split.amount",
               f"asked to split {s.amount:g} of a {txn.amount:g} payment", tag="guard")
        a.say = (f"That payment was {txn.currency} {txn.amount:g}, so I can't split "
                 f"{s.amount:g} of it. How much of it should they cover?")
        a.outcome = "NEED_INFO"
        conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), "amount")
        return a

    total = s.amount or txn.amount
    conversation.note_subject(a.session, "txn_ref", txn.ref)
    # Pin the payment to the open split. It was found from the WORDS ("my last payment"),
    # and the next turn has no such words — "alex" alone, or a misheard reply — so without
    # this the split asked "Which payment should I split?" about one it had just named.
    s.txn_ref = txn.ref
    a.slots = dict(a.slots or {}, txn_ref=txn.ref)
    a.step("resolve", "split.target",
           f"splitting {txn.currency} {total:g} to {txn.payee} ({txn.ref})", tag="db")

    # ── 2. WHO with ─────────────────────────────────────────────────────────────────────
    people = [p for p in (s.participants or []) if p.get("userId")]
    if not people:
        named = [p["name"] for p in (s.participants or []) if not p.get("userId")]
        a.say = (f"I know who {named[0]} is but I don't have an id for them, so I can't "
                 f"add them to a split yet. Who else?" if named else
                 f"Who are you splitting the {txn.currency} {total:g} with?")
        a.outcome = "NEED_INFO"
        a.step("act", "need.participants",
               "no resolved participants — offering only people this customer already has "
               "a relationship with", tag="guard")
        a.options = _split_people_options()
        conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), "participants")
        return a
    if len(people) > SPLIT_MAX_POST:
        a.say = (f"That's {len(people)} people — a split can have at most "
                 f"{SPLIT_MAX_POST}.")
        a.outcome = "REFUSED"
        a.step("decide", "split.cap", f"{len(people)} exceeds the {SPLIT_MAX_POST} cap",
               tag="guard")
        return a

    # ── 3. HOW it divides ───────────────────────────────────────────────────────────────
    from split import calc
    split_type = s.split_type or "SHARE"
    try:
        if split_type == "SHARE":
            shares, mine = calc.equal(total, len(people))
        elif split_type == "AMOUNT":
            # One named amount applies to the one named person; anything more complex is
            # asked for rather than guessed.
            if len(people) != 1 or not s.amount:
                shares, mine = calc.equal(total, len(people))
                split_type = "SHARE"
            else:
                shares, mine = [s.amount], calc.by_amount(total, [s.amount])
        else:                                                   # PERCENTAGE
            pct = s.amount if s.amount and s.amount <= 100 else None
            if len(people) != 1 or pct is None:
                shares, mine = calc.equal(total, len(people))
                split_type = "SHARE"
            else:
                shares, mine = calc.by_percentage(total, [pct])
    except ValueError as exc:
        a.say = f"That doesn't add up — {exc}. How would you like to divide it?"
        a.outcome = "NEED_INFO"
        a.step("decide", "split.arithmetic", str(exc), tag="guard")
        conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), "split_type")
        return a

    a.step("decide", "split.shares",
           f"{split_type}: " + ", ".join(f"{p['name'].split()[0]} {sh:g}"
                                         for p, sh in zip(people, shares))
           + f", you {mine:g}", tag="rule")

    # ── 4. PROPOSE. Nothing is created until the customer confirms. ─────────────────────
    who = ", ".join(p["name"].split()[0] for p in people)
    a.say = (f"Splitting the {txn.currency} {total:g} at {txn.payee} "
             + (f"equally with {who}" if split_type == "SHARE" else f"with {who}")
             + f" — they each owe you {shares[0]:g}, you keep {mine:g}."
             if len(set(shares)) == 1 else
             f"Splitting the {txn.currency} {total:g} at {txn.payee} with {who}.")
    a.proposal = {"kind": "split", "splitId": None, "referenceTxnId": txn.ref,
                  "splitType": split_type, "total": total, "currency": txn.currency,
                  "merchant": txn.payee,
                  "participants": [{"userId": p["userId"], "name": p["name"],
                                    "displayValue": sh}
                                   for p, sh in zip(people, shares)],
                  "initiatorDisplayValue": mine,
                  "billDescription": f"{txn.payee}"}
    a.outcome = "NEED_CONFIRMATION"
    a.options = [{"kind": "dismiss", "label": "Not now"}]
    conversation.hold(a.session, Intent.SPLIT_PAYMENT.value, s.as_dict(), None)
    return a


def h_split_status(a, s, said):
    """Balances and outstanding shares. A question — this never creates or settles anything."""
    from split import api as split_api
    people = split_api.people()
    if people is None or (not people and split_api.last_error()):
        a.say = ("I can't reach your split balances right now. Please try again in a "
                 "moment.")
        a.outcome = "UNAVAILABLE"
        a.step("resolve", "split.unavailable", split_api.last_error() or "unknown",
               tag="guard")
        return a

    if s.payee:
        one = next((p for p in people if _same_person(p.get("name", ""), s.payee)), None)
        if not one:
            a.say = f"You don't have any open splits with {s.payee}."
            a.outcome = "NONE"
            a.step("resolve", "split.none", f"no balance with {s.payee}", tag="db")
            a.options = [{"kind": "ask_text", "label": "All my balances",
                          "text": "who owes me money"}]
            return a
        bal = one.get("netBalance") or 0
        first = (one.get("name") or "").split()[0]
        a.say = (_tr(a, "split_you_owe", f"You owe {first} {abs(bal):.2f} AED.",
                     who=first, amount=f"{abs(bal):.2f}", cur="AED") if bal < 0 else
                 _tr(a, "split_one_owes_you", f"{first} owes you {bal:.2f} AED.",
                     who=first, amount=f"{bal:.2f}", cur="AED") if bal > 0 else
                 f"You and {first} are settled up.")
        a.outcome = "INFORMED"
        a.step("resolve", "split.balance", f"{first}: netBalance {bal}", tag="db")
        conversation.hold(a.session, Intent.SPLIT_STATUS.value, s.as_dict(), None)
        return a

    dash = split_api.dashboard()
    owe, owed = dash.get("youPayTotal") or 0, dash.get("youGetTotal") or 0
    if not people and not owe and not owed:
        a.say = "You don't have any open splits at the moment."
        a.outcome = "NONE"
        a.step("resolve", "split.none", "no balances", tag="db")
        a.options = [{"kind": "ask_text", "label": "Split my last payment",
                      "text": "split my last payment"}]
        return a

    bits = []
    if owed:
        bits.append(f"you're owed {owed:.2f} AED")
    if owe:
        bits.append(f"you owe {owe:.2f} AED")
    a.say = (_tr(a, "split_owed_and_owe",
                 "Across your splits, " + " and ".join(bits) + ".",
                 cur="AED", owed=f"{owed:.2f}", owe=f"{owe:.2f}")
             if bits else "Everything on your splits is settled.")
    a.outcome = "INFORMED"
    a.rows = [{"ref": p.get("userId"), "payee": p.get("name") or "",
               "amount": abs(p.get("netBalance") or 0), "currency": "AED",
               "corridor": "Split balance",
               "status": ("You owe" if (p.get("netBalance") or 0) < 0
                          else "Owes you" if (p.get("netBalance") or 0) > 0 else "Settled"),
               "when": "", "refund_raised": False}
              for p in people]
    a.step("resolve", "split.dashboard",
           f"{len(people)} people, {dash.get('activeGroupCount', 0)} active groups, "
           f"owe {owe}, owed {owed}", tag="db")
    a.options = ([{"kind": "split_nudge", "label": f"Remind {p['name'].split()[0]}",
                   "userId": p["userId"]}
                  for p in people if (p.get("netBalance") or 0) > 0][:3]
                 if split_api.can_nudge() else [])
    conversation.hold(a.session, Intent.SPLIT_STATUS.value, s.as_dict(), None)
    return a


# ─── bills ───────────────────────────────────────────────────────────────────────────────

#: Words that mean "settle a bill". A bill payment names no payee and usually no amount —
#: the biller comes from the customer's own saved list and the figure from the biller — so
#: the send-shaped "names nobody and nothing" test refuses it outright. Fourteen perfectly
#: good requests ("settle my bills", "bijli ka bill bhar do", "knock off the overdue ones")
#: were refused that way. Multilingual from the start, unlike the English-only transfer verbs
#: this is modelled on.
_BILL_VERBS = (
    "bill", "bills", "utility", "utilities", "electricity", "water", "internet", "landline",
    "broadband", "gas", "dewa", "sewa", "addc", "aadc", "fewa", "etisalat", "du ", "salik",
    "nol", "outstanding", "overdue", "due", "settle", "clear", "pay off", "knock off",
    "take care of", "sort out", "autopay", "auto pay", "every month",
    "membership", "subscription", "rent", "tuition", "school fee", "insurance", "premium",
    "emi", "instalment", "installment",
    "bijli", "paani", "pani", "bhar do", "bhar dena", "bharna", "bakaya", "baaki",
    "बिल", "बिजली", "पानी", "भर दो", "भर देना", "बकाया", "बाकी", "इंटरनेट", "लैंडलाइन",
    "فاتورة", "فواتير", "الكهرباء", "الماء", "الإنترنت", "مستحق", "متأخر", "سدد", "ادفع فاتورة",
)


def _has_bill_verb(said: str) -> bool:
    low = " ".join((said or "").lower().split())
    return any(v in low for v in _BILL_VERBS)


def _bill_line(r: dict) -> str:
    """One bill as a sentence. UNKNOWN never reads as zero."""
    from bills import due as bills_due
    if r["status"] == bills_due.UNKNOWN:
        return f"{r['nickname']} — couldn't check ({r['note']})"
    if r["status"] == bills_due.CLEAR:
        return f"{r['nickname']} — nothing owing"
    when = f", due {r['due']}" if r.get("due") else ""
    late = " (overdue)" if r["status"] == bills_due.OVERDUE else ""
    return f"{r['nickname']} — AED {r['amount']:.2f}{when}{late}"


def _bill_rows(rows: list[dict]) -> list[dict]:
    from bills import due as bills_due
    out = []
    for r in rows:
        out.append({"ref": r["planId"], "payee": r["nickname"] or r["biller"],
                    "amount": r["amount"] if r["amount"] is not None else 0,
                    "currency": "AED", "corridor": r["biller"] or "",
                    "status": {bills_due.OVERDUE: "Overdue", bills_due.DUE: "Due",
                               bills_due.CLEAR: "Paid", bills_due.UNKNOWN: "Unknown"}
                    [r["status"]],
                    "when": r.get("due") or "", "refund_raised": False})
    return out


def _pick_bills(s, said):
    """Which saved bills the customer means. Returns (rows, ambiguous)."""
    from bills import due as bills_due
    summary = bills_due.summary()
    payable = summary["due"] + summary["overdue"]

    # Already narrowed on an earlier turn — keep exactly those. Re-deriving from the latest
    # utterance meant "pay from my wallet", which names no bill, widened a payment for one
    # bill back to every payable bill.
    if s.bill_ids:
        keep = [r for r in summary["bills"] if r["planId"] in s.bill_ids]
        if keep:
            return keep, False
    want = s.bill or said
    if re.search(r"\ball\b|\beverything\b|\bevery\b|\bsaare\b|\bsare\b|सारे|सभी|كل",
                 (said or "").lower()):
        return payable, False
    if re.search(r"\boverdue\b|\blate\b|ओवरड्यू|متأخر", (said or "").lower()):
        return summary["overdue"], False
    named = bills_due.find(want)
    hits = [h for h in named if h["status"] in (bills_due.DUE, bills_due.OVERDUE)]
    if len(hits) == 1:
        return hits, False
    if len(hits) > 1:
        return hits, True
    if named:
        # They named a bill and it is not payable — nothing owing on it, or we could not
        # check it. Falling through to "all the payable ones" would pay bills they did not
        # ask about while saying nothing about the one they did, which is how "pay my beach
        # flat bill" came to settle the electricity and the internet.
        return [], "named-but-not-payable"
    return payable, False


def h_bill_due(a, s, said):
    """What this customer owes, across every biller they have saved.

    The honesty that matters here is the UNKNOWN case. If a biller cannot be reached, that
    bill is reported as unchecked and the total says so — treating it as nothing owing would
    tell somebody they are clear when they may not be, which is the worst thing this feature
    could say.
    """
    from bills import due as bills_due
    summary = bills_due.summary()

    if summary.get("unavailable"):
        a.say = ("I can't reach your bill list right now, so I can't tell you what's due. "
                 "Please try again in a moment.")
        a.outcome = "UNAVAILABLE"
        a.step("resolve", "bills.unavailable", summary["unavailable"], tag="guard")
        return a
    if summary["no_bills"]:
        a.say = ("You haven't saved any bills yet. Add a biller and an account number in the "
                 "app and I can keep track of what's due.")
        a.outcome = "NONE"
        a.step("resolve", "bills.none", "no saved bills for this customer", tag="db")
        return a

    payable = summary["overdue"] + summary["due"]
    a.rows = _bill_rows(summary["bills"])
    a.step("resolve", "bills",
           f"{len(summary['bills'])} saved; {len(payable)} owing "
           f"({len(summary['overdue'])} overdue), {len(summary['unknown'])} unchecked",
           tag="db")

    if not payable:
        a.say = (_tr(a, "bills_none", "Nothing is due right now.")
                 if not summary["any_unknown"] else
                 f"Nothing is due on the bills I could check — but I couldn't reach "
                 f"{len(summary['unknown'])} of them.")
        a.outcome = "INFORMED"
    else:
        head = _tr(a, "bills_due",
                   f"{len(payable)} bill{'s' if len(payable) != 1 else ''} to pay, "
                   f"AED {summary['total']:.2f} in total",
                   n=len(payable), cur="AED", total=f"{summary['total']:.2f}").rstrip(".")
        if summary["overdue"]:
            head += f" — {len(summary['overdue'])} of them overdue"
        if summary["any_unknown"]:
            # Joined with an explicit separator and a single full stop. The translated
            # headline already ends in one and the clause used to bring another, so the
            # Hinglish and Arabic lines came out as "… 667.00 1 bill … sakta.." — two stops
            # and no separator.
            head = head.rstrip(". ۔।") + ". " + _tr(
                a, "bills_some_unchecked",
                f"I couldn't check {len(summary['unknown'])}, so that total may not be "
                f"everything", k=len(summary["unknown"])).rstrip(". ۔।")
        a.say = head.rstrip(". ۔।") + ("।" if a.lang == "hi" else ".")
        a.outcome = "LISTED"
        a.options = [{"kind": "ask_text", "label": f"Pay {r['nickname']}",
                      "text": f"pay my {r['nickname']} bill"} for r in payable[:3]]
        if len(payable) > 1:
            a.options.insert(0, {"kind": "ask_text",
                                 "label": f"Pay all {len(payable)} "
                                          f"(AED {summary['total']:.2f})",
                                 "text": "pay all my bills"})
    conversation.hold(a.session, Intent.BILL_DUE.value, s.as_dict(), None)
    return a


#: Words that mean NOW, emphatically — they look like time phrases to the parser but are the
#: opposite of a schedule. "आज ही" is "today, right now", not "some time today".
_IMMEDIATE = ("right now", "now", "immediately", "straight away", "at once", "today itself",
              "abhi", "abhi hi", "aaj hi", "turant", "अभी", "अभी ही", "आज ही", "तुरंत",
              "الآن", "حالا", "فورا", "اليوم نفسه")


def _means_now(said: str) -> bool:
    low = " ".join((said or "").lower().split())
    return any(w in low for w in _IMMEDIATE)


def h_bill_pay(a, s, said, scheduled: bool = False):
    """Settle a bill now, or — when `scheduled` — promise to settle it later.

    The payment MODE is asked for rather than assumed. In the UAE a bill can be paid from the
    wallet or through open banking, and those are different journeys with different
    authorisation; picking one on the customer's behalf and then running it is a decision
    about their money that they did not make.
    """
    from bills import due as bills_due

    if bills_due.summary().get("unavailable"):
        a.say = ("I can't reach your bill list right now, so I'd rather not guess at what to "
                 "pay. Please try again in a moment.")
        a.outcome = "UNAVAILABLE"
        a.step("resolve", "bills.unavailable",
               bills_due.summary()["unavailable"], tag="guard")
        return a

    # A named TIME decides this, not the classifier. "bijli ka bill kal bhar dena" — pay the
    # bill TOMORROW — was read as bill_pay, and answering the next question would have taken
    # the money today. The reverse matters too: "आज ही" means right now, and scheduling it
    # for 9am would be equally wrong. A time phrase is unambiguous evidence either way, so it
    # overrides a label, and the trace says it did.
    if _means_now(said):
        if scheduled:
            a.step("understand", "bills.now",
                   "classified as a schedule, but the customer said right now — paying "
                   "rather than booking it", tag="rule")
        scheduled = False
        s.when = s.when_at = None
        a.intent = Intent.BILL_PAY.value      # the label must match what we actually do
    elif s.when_at and not scheduled:
        a.step("understand", "bills.later",
               f"classified as an immediate payment ({a.confidence:.0%}) but it names a time "
               f"({s.when}) — treating it as a schedule, because paying today something the "
               f"customer asked for {s.when} takes their money early", tag="rule")
        scheduled = True
        a.intent = Intent.BILL_SCHEDULE.value

    summary = bills_due.summary()
    if summary["no_bills"]:
        a.say = ("You haven't saved any bills yet, so there's nothing for me to pay. Add a "
                 "biller in the app and I'll take it from there.")
        a.outcome = "NONE"
        a.step("resolve", "bills.none", "no saved bills", tag="db")
        return a

    chosen, ambiguous = _pick_bills(s, said)
    if ambiguous == "named-but-not-payable":
        named = __import__("bills.due", fromlist=["due"]).find(s.bill or said)
        r = named[0]
        a.step("resolve", "bills.not_payable",
               f"{r['nickname']} is {r['status']} — refusing rather than paying the other "
               f"bills, which is not what was asked", tag="guard")
        a.say = (f"There's nothing owing on {r['nickname']} right now."
                 if r["status"] == bills_due.CLEAR else
                 f"I couldn't check {r['nickname']} — {r['note']}.")
        a.outcome = "NONE"
        other = [x for x in (summary["due"] + summary["overdue"])]
        if other:
            a.options = [{"kind": "ask_text", "label": f"Pay {x['nickname']} instead",
                          "text": f"pay my {x['nickname']} bill"} for x in other[:2]]
        return a
    if not chosen:
        a.say = ("Nothing is due right now, so there's nothing to pay."
                 if not summary["any_unknown"] else
                 "Nothing is due on the bills I could check. I couldn't reach "
                 f"{len(summary['unknown'])} of them, so it's worth looking again shortly.")
        a.outcome = "NONE"
        a.step("resolve", "bills.nothing_due", "no payable bill matched", tag="db")
        return a
    if ambiguous:
        a.say = f"Which one do you mean?"
        a.outcome = "NEED_INFO"
        a.step("resolve", "bills.ambiguous",
               f"{len(chosen)} bills match — asking rather than picking, because paying the "
               f"wrong one is not undone by apologising", tag="guard")
        a.options = [{"kind": "ask_text", "label": _bill_line(r),
                      "text": f"pay my {r['nickname']} bill"} for r in chosen]
        conversation.hold(a.session, a.intent, s.as_dict(), "bill")
        return a

    # Can these actually be paid? A biller with no ACTIVE service provider is rejected by
    # the Transaction service with "Transaction cannot be carried out for this amount" —
    # AFTER the customer has chosen a payment method and entered their PIN. Finding out here
    # costs a database read; finding out there costs their trust.
    from bills import quote as bill_quote
    blocked = []
    for r in chosen:
        amt = (s.amount if (s.amount and len(chosen) == 1) else r["amount"]) or 0
        ok, why = bill_quote.routable(r["productId"], amt)
        if not ok:
            blocked.append((r, why))
    if blocked and len(blocked) == len(chosen):
        r, why = blocked[0]
        a.step("decide", "bills.unroutable",
               f"{r['nickname']} (product {r['productId']}): {why} — refusing before taking "
               f"a PIN rather than failing after it", tag="guard")
        a.say = (f"I can't pay {r['nickname']} at the moment — {why}."
                 if len(blocked) == 1 else
                 f"I can't pay any of those at the moment — {why}.")
        a.outcome = "UNAVAILABLE"
        payable_others = [x for x in (summary["due"] + summary["overdue"])
                          if x not in [b[0] for b in blocked]]
        a.options = [{"kind": "ask_text", "label": f"Pay {x['nickname']} instead",
                      "text": f"pay my {x['nickname']} bill"} for x in payable_others[:2]]
        return a
    if blocked:
        names = ", ".join(r["nickname"] for r, _ in blocked)
        chosen = [r for r in chosen if r not in [b[0] for b in blocked]]
        a.step("decide", "bills.unroutable.partial",
               f"leaving out {names}: {blocked[0][1]}", tag="guard")

    total = round(sum(r["amount"] or 0 for r in chosen), 2)
    if s.amount and len(chosen) == 1:
        plan = catalogue_plan(chosen[0]["planId"])
        lo, hi = (plan or {}).get("min"), (plan or {}).get("max")
        if lo is not None and s.amount < lo:
            a.say = f"{chosen[0]['biller']} won't take less than AED {lo:g}."
            a.outcome = "REFUSED"
            a.step("decide", "bills.min", f"{s.amount} below the plan minimum {lo}",
                   tag="guard")
            return a
        if hi is not None and s.amount > hi:
            a.say = f"{chosen[0]['biller']} won't take more than AED {hi:g} in one payment."
            a.outcome = "REFUSED"
            a.step("decide", "bills.max", f"{s.amount} above the plan maximum {hi}",
                   tag="guard")
            return a
        total = s.amount

    # ── when, for a scheduled payment ───────────────────────────────────────────────────
    if scheduled and not s.when_at:
        a.say = (f"When should I pay {'them' if len(chosen) > 1 else chosen[0]['nickname']}? "
                 f"Tomorrow, Friday, the 1st — whatever suits.")
        a.outcome = "NEED_INFO"
        a.step("act", "need.when", "a scheduled payment with no time is not a schedule",
               tag="guard")
        a.options = [{"kind": "ask_text", "label": "Tomorrow", "text": "pay it tomorrow"},
                     {"kind": "ask_text", "label": "On the due date",
                      "text": "pay it on the due date"},
                     {"kind": "ask_text", "label": "End of the month",
                      "text": "pay it at the end of the month"}]
        s.bill_ids = [r["planId"] for r in chosen]
        conversation.hold(a.session, Intent.BILL_SCHEDULE.value, s.as_dict(), "when")
        return a

    # ── the payment mode, asked for every time ──────────────────────────────────────────
    if not s.pay_mode:
        where = "they" if len(chosen) > 1 else "it"
        _who = (f"{len(chosen)} bills" if len(chosen) > 1 else chosen[0]["nickname"])
        a.say = _tr(a, "bill_where_from",
                    f"AED {total:.2f} for {_who}"
                    + (f", on {s.when}" if scheduled and s.when else "")
                    + f". Where should {where} come from?",
                    cur="AED", amount=f"{total:.2f}", who=_who)
        a.outcome = "NEED_INFO"
        a.step("act", "need.mode",
               "bill payments can come from the wallet or through open banking — asking "
               "rather than choosing on the customer's behalf", tag="guard")
        a.options = [{"kind": "ask_text", "label": "My OpenWallet wallet",
                      "text": "pay from my wallet"},
                     {"kind": "ask_text", "label": "My bank (open banking)",
                      "text": "use open banking"}]
        s.bill_ids = [r["planId"] for r in chosen]
        if len(chosen) == 1:
            conversation.note_subject(a.session, "bill", chosen[0]["nickname"])
        conversation.hold(a.session,
                          Intent.BILL_SCHEDULE.value if scheduled else Intent.BILL_PAY.value,
                          s.as_dict(), "pay_mode")
        return a

    # Pin the selection, so every later turn in this conversation means these bills.
    s.bill_ids = [r["planId"] for r in chosen]
    # What it COSTS, the UAE way: bill + our fee + VAT on the fee + the biller's own charge.
    # Global's handler adds none of the last two, so quoting its figure would under-state the
    # payment by up to several dirhams — and a customer finds out about that from their
    # balance, not from us.
    costs, charges = [], 0.0
    for r in chosen:
        amt = (s.amount if (s.amount and len(chosen) == 1) else r["amount"]) or 0
        e = bill_quote.explain(r["productId"], amt)
        costs.append(e)
        charges += e["service_charge"] + e["vat"] + e["biller_charge"]
    charges = round(charges, 2)
    grand = round(total + charges, 2)
    incomplete = any(e.get("incomplete") for e in costs)
    if incomplete:
        # A fee we could not read is not a fee of zero. Quoting the total as final when a
        # charge is missing understates what leaves the wallet, and the customer finds out
        # from their balance. Dev had an ALTER TABLE queued behind an idle transaction, so
        # every read of the charge table timed out — exactly this case.
        a.step("decide", "bills.charges.partial",
               "could not read " + ", ".join(sorted({w for e in costs
                                                     for w in e.get("why_incomplete", [])}))
               + " — saying the total may be incomplete rather than quoting it as final",
               tag="guard")
    if charges:
        a.step("decide", "bills.charges",
               "; ".join(f"{r['nickname']}: fee {e['service_charge']:g} + vat {e['vat']:g}"
                         + (f" + biller {e['biller_charge']:g}" if e["biller_charge"] else "")
                         for r, e in zip(chosen, costs))
               + f" = {charges:g} on top", tag="rule")
        if any(e["differs"] for e in costs):
            a.step("decide", "bills.env",
                   f"these are the UAE figures. This environment's handler adds neither VAT "
                   f"nor the biller's charge, so a payment executed here would take "
                   f"{round(sum(e['global_total'] for e in costs), 2):g} rather than "
                   f"{round(sum(e['uae_total'] for e in costs), 2):g}", tag="guard")

    if len(chosen) == 1:
        conversation.note_subject(a.session, "bill", chosen[0]["nickname"])
    a.step("decide", "bills.chosen",
           f"{len(chosen)} bill(s), AED {total:.2f}, {s.pay_mode}"
           + (f", at {s.when_at}" if scheduled else ""), tag="rule")
    a.proposal = {
        "kind": "bill_schedule" if scheduled else "bill_pay",
        "payMode": s.pay_mode, "total": total, "charges": charges, "grandTotal": grand,
        "currency": "AED",
        "when": s.when, "whenAt": s.when_at,
        "bills": [{"planId": r["planId"], "productId": r["productId"],
                   "identifier": r["identifier"], "nickname": r["nickname"],
                   "biller": r["biller"],
                   "amount": (s.amount if (s.amount and len(chosen) == 1) else r["amount"])}
                  for r in chosen],
    }
    which = (f"{len(chosen)} bills" if len(chosen) > 1 else chosen[0]["nickname"])
    source = "your wallet" if s.pay_mode == "WALLET" else "your bank"
    # Charges are named, never folded silently into one number. "AED 509.85" with no
    # explanation invites "why is it more than my bill?", and the answer should arrive before
    # the question.
    fee_note = (f" That's AED {total:.2f} for the bill plus AED {charges:.2f} in charges."
                if charges else "")
    if incomplete:
        fee_note += (" I couldn't read every charge just now, so the final amount may be a "
                     "little higher.")
    if scheduled:
        a.say = (f"I'll pay AED {grand:.2f} for {which} from {source} on {s.when}."
                 + fee_note + " You can cancel any time before then.")
    else:
        a.say = f"Paying AED {grand:.2f} for {which} from {source}." + fee_note
    a.outcome = "NEED_CONFIRMATION"
    a.options = [{"kind": "dismiss", "label": "Not now"}]
    conversation.hold(a.session, a.intent, s.as_dict(), None)
    return a


def h_bill_schedule(a, s, said):
    return h_bill_pay(a, s, said, scheduled=True)


def catalogue_plan(plan_id):
    from bills import catalogue
    return catalogue.plan(plan_id)


def h_journey(a, s, said):
    j = journeys.find(said)
    if not j:
        a.say = ("I can walk you through adding a beneficiary, sending money abroad, paying "
                 "by QR, topping up, verification, your statement or changing your PIN. "
                 "Which one?")
        a.outcome = "NEED_INFO"
        a.options = [{"kind": "journey", "label": t} for t in journeys.titles()[:6]]
        return a
    a.say = j["title"] + ": " + " ".join(f"{i}. {st}" for i, st in enumerate(j["steps"], 1))
    if j.get("note"):
        a.say += " " + j["note"]
    a.outcome = "EXPLAINED"
    a.step("resolve", "journey", f"curated customer content — {j['key']}", tag="db")
    a.step("decide", "scope",
           "answered from the customer help content only; the code knowledge graph is not "
           "reachable from this surface", tag="guard")
    return a


def attach_offers(merchants: list[dict]) -> int:
    """Fill in each merchant's live campaign, in parallel. Returns how many have one.

    Shared by the offers screen and by refinements like "which has offers" — the list is
    useless for that question until the campaigns are actually resolved, and resolving them
    serially costs one slow dev API call per merchant.
    """
    def _discover(m):
        return m, (incentive.discover(receiver_id=m["userId"]) if m.get("userId") else None)

    live = 0
    with ThreadPoolExecutor(max_workers=min(8, len(merchants) or 1)) as pool:
        for m, campaigns in pool.map(__import__('userctx').wrap(_discover), merchants):
            if campaigns:
                m["offer"] = incentive.as_offer(campaigns[0])
                live += 1
            else:
                m["offer"] = None
    return live


def _refine_options(a: Answer, kind: str, frame) -> Answer:
    """Answer the narrowed question with a narrowed list."""
    merchants = data.nearby(len(data.MERCHANTS))
    if kind in ("with_offer", "cheapest"):
        attach_offers(merchants)          # the question is ABOUT offers; resolve them first
    rows, why_not = refine.apply_to_merchants(kind, merchants)
    a.outcome = "NEED_INFO"

    if why_not:
        a.say = f"{why_not.capitalize()}. You can still pay any of them."
        a.options = [{"kind": "pay_merchant", "label": _merchant_label(m),
                      "mid": m["mid"], "merchant": m["name"]} for m in merchants[:4]]
        a.step("resolve", "refine.empty", why_not, tag="db")
        return a

    named = ", ".join(m["name"] for m in rows[:3])
    if kind == "with_offer":
        a.say = (f"{'These have' if len(rows) > 1 else 'Only one has'} an offer right now: "
                 f"{named}. Which one?")
    elif kind == "nearest":
        first = rows[0]
        a.say = (f"{first['name']} is closest"
                 + (f" at {_distance(first['metres'])}" if first.get("metres") is not None else "")
                 + f". Then {', '.join(m['name'] for m in rows[1:3])}." if len(rows) > 1
                 else f"{first['name']} is the closest.")
    else:
        a.say = f"Best saving first: {named}. Which one?"

    a.options = [{"kind": "pay_merchant", "label": _merchant_label(m),
                  "mid": m["mid"], "merchant": m["name"]} for m in rows[:5]]
    a.step("resolve", "refine", f"{len(rows)} of {len(merchants)} merchants {refine.describe(kind)}",
           tag="db")
    return a


def _distance(metres) -> str:
    if metres is None:
        return "distance unknown"
    return f"{metres / 1000:.1f}km" if metres >= 1000 else f"{int(metres)}m"


def _merchant_label(m: dict) -> str:
    bits = [m["name"]]
    if m.get("metres") is not None:
        bits.append(_distance(m["metres"]))
    if m.get("offer"):
        bits.append(m["offer"]["label"])
    return " · ".join(bits)


def _payee_options(corridor: str | None = None) -> list[dict]:
    """Everyone this customer can pay, narrowed to the corridor when one was named.

    "Send money to India" names a COUNTRY, which makes it a cross-border transfer. Offering
    the local roster and the shops down the road is answering a different question — nobody
    pays a Dubai cafe to send money to India. When a corridor is named we show only the
    beneficiaries on it, and no merchants at all.
    """
    everyone = data.sendable()
    cross_border = bool(corridor) and corridor != "AE-AE"

    if cross_border:
        people = [b for b in everyone if b.get("corridor") == corridor]
        out: list[dict] = []
        country = CORRIDORS.get(corridor, {}).get("name", corridor).split("→")[-1].strip()
        if people:
            out.append({"kind": "label", "label": f"Your beneficiaries in {country}"})
            out.extend({"kind": "send_to", "label": b["name"], "payee": b["name"]}
                       for b in people[:4])
        else:
            # Honest dead end with a way out, rather than silently offering the wrong people.
            out.append({"kind": "journey", "label": f"Add someone in {country}",
                        "text": "how do i add a beneficiary"})
        return out

    people = [{"kind": "send_to", "label": f"{b['name']}", "payee": b["name"]}
              for b in everyone[:3]]
    shops = [{"kind": "pay_merchant",
              "label": f"{m['name']}" + (f" · {m['metres']}m" if m.get("metres") else ""),
              "mid": m["mid"], "merchant": m["name"]}
             for m in data.nearby(3)]
    out = []
    if people:
        out.append({"kind": "label", "label": "Someone you know"})
        out.extend(people)
    if shops:
        out.append({"kind": "label", "label": "A merchant near you"})
        out.extend(shops)
    return out


def h_send(a, s, said):
    # The confidence bar applies ONLY when we have nothing concrete.
    #
    # An exact roster match is deterministic evidence, not a guess — it IS the guardrail this
    # system is built on. Refusing "send money to Alex Morgan" because a classifier scored
    # 54% leaves the customer in a loop they cannot escape: they named a real payee, and we
    # answered "tell me who". Once a payee or an amount is in hand, the classifier's opinion
    # is no longer the thing standing between them and their payment.
    # A named COUNTRY is hard evidence too — "send money to India" tells us the corridor,
    # which is more than enough to start. Without this it fell through to the generic
    # "someone you know, or a merchant near you?", which ignores what the customer just said.
    if not (s.payee or s.amount or s.corridor) and a.confidence < CONFIDENT_FOR_MONEY:
        # Ask the question we are actually asking. The old wording announced our own
        # uncertainty and then offered a list — which reads as a refusal with buttons.
        a.say = _ph(a, "ask_payee", _tr(a, "which_payee", voice.pick("which_payee", a.session)))
        a.outcome = "NEED_CONFIRMATION"
        a.step("act", "withheld",
               f"confidence {a.confidence:.0%} is below the {CONFIDENT_FOR_MONEY:.0%} bar "
               f"for anything that moves money", tag="guard")
        a.options = _payee_options(s.corridor)
        return a
    if not s.payee:
        # A named country makes this a cross-border transfer, and the answer changes: who they
        # can pay, what it costs and how long it takes are all corridor-specific.
        if s.corridor and s.corridor != "AE-AE":
            c = CORRIDORS[s.corridor]
            country = c["name"].split("→")[-1].strip()
            a.options = _payee_options(s.corridor)
            has_payees = any(o["kind"] == "send_to" for o in a.options)
            a.say = (f"{country} — that's a cross-border transfer. "
                     f"Fee is AED {c['fee_aed']:.2f} and it normally arrives {c['sla']}. "
                     + ("Who's it going to?" if has_payees
                        else f"You don't have a beneficiary in {country} yet — "
                             f"shall I show you how to add one?"))
            a.outcome = "NEED_INFO"
            a.step("resolve", "corridor",
                   f"{s.corridor} named by country — cross-border, so only {country} "
                   f"beneficiaries are offered and no local merchants", tag="db")
            return a
        # A NAME was given, and it is nobody this customer has paid. Say so, plainly: "send
        # 20 to rahul" from someone who has never paid a Rahul must not quietly turn into
        # "who's this going to?" as if no name had been said — and must never be matched to
        # some other Rahul in the wallet.
        # "Send money abroad / to another country" names a KIND of transfer, not a person, and
        # no country yet. Ask which country — the corridor branch above takes it from there.
        if _ABROAD.search(said or ""):
            a.step("resolve", "corridor.which",
                   "a cross-border transfer with no country named — asking which country",
                   tag="rule")
            a.say = _ph(a, "which_country", _tr(a, "which_country", "Which country are you sending to?"))
            a.outcome = "NEED_INFO"
            a.options = [{"kind": "ask_text", "label": c["name"].split("→")[-1].strip(),
                          "text": f"send money to {c['name'].split('→')[-1].strip().lower()}"}
                         for k, c in CORRIDORS.items() if k != "AE-AE"]
            return a
        _other = _unresolved_subject(said, s)
        if not (_other and _other.endswith("?")) and _named_unknown_payee(said, s):
            _other = _named_unknown_payee(said, s) + "?"
        if _other and _other.endswith("?") and not _NOT_A_NAME.search(said or "") \
                and _other[:-1].lower() not in _GENERAL_WORDS:
            _nm = _other[:-1]
            a.step("resolve", "payee.unknown",
                   f"'{_nm.lower()}' is not among the people this customer has paid — not "
                   f"searching anyone else's wallet for a match", tag="guard")
            # BRD 2.1 T7 "Beneficiary not found": Yes -> the normal app flow (add beneficiary).
            people = [o for o in _payee_options() if o.get("kind") == "send_to"]
            a.say = _ph(a, "beneficiary_not_found", _tr(a, "beneficiary_not_found",
                        f"{_nm} is not in your saved beneficiaries. Would you like to add a new "
                        f"beneficiary?", name=_nm), name=_nm)
            a.outcome = "NEED_INFO"
            a.options = ([{"kind": "navigate", "screen": "add_beneficiary",
                           "label": "Yes, add a beneficiary"}] + people)
            return a
        # "Send money" does not say WHO, and the two kinds of payee behave differently — a
        # person is paid from the saved roster, a merchant is paid with whatever campaign
        # applies. So offer both and let the choice decide, rather than assuming a person and
        # making the customer back out of the wrong flow.
        a.say = _ph(a, "ask_payee", _tr(a, "which_payee", voice.pick("which_payee", a.session)))
        a.outcome = "NEED_INFO"
        a.options = _payee_options()
        a.step("act", "need.payee",
               "offering saved people AND nearby merchants; only names the customer already "
               "has a relationship with are offered", tag="guard")
        # We asked "someone you know, or a merchant?" — so "a shop" or "a friend" is an
        # ANSWER. Without this the reply was classified on its own, named nothing, and the
        # no-payee guard refused it: the customer picked one of our two options and was told
        # we can't help.
        conversation.note_subject(a.session, "offered_pay_kind", 1)
        return a
    if not s.amount:
        a.say = _ph(a, "ask_amount", _tr(a, "how_much_to",
                    f"{voice.pick('how_much', a.session)[:-1]} to {s.payee}?",
                    who=s.payee), payee=s.payee)
        a.outcome = "NEED_INFO"
        a.step("act", "need.amount", "no clear amount — never assumed", tag="guard")
        return a
    # Check the mandate limit BEFORE proposing. The limit is enforced at payment time either
    # way, but discovering it only after the customer has confirmed wastes their time and
    # reads as broken — the assistant offered something it knew it could not do.
    # BRD 2.1 T8 "Insufficient balance". Checked before proposing, from the wallet's own
    # balance, so the customer is not walked to the PIN for a transfer the wallet will refuse.
    try:
        from engine import brd_handlers as _brdh
        _bal, _ = _brdh.wallet_balance()
    except Exception:                                              # noqa: BLE001
        _bal = None
    if _bal is not None and s.amount > _bal:
        a.say = _ph(a, "insufficient_balance", _tr(a, "insufficient_balance",
                    f"Your available balance is {_bal:,.2f} AED. Would you like to send a "
                    f"smaller amount?", bal=f"{_bal:,.2f}"), balance=f"AED {_bal:,.2f}")
        a.outcome = "OVER_LIMIT"           # holds the payment open, waiting for a new amount
        a.step("decide", "balance", f"{s.amount:g} is more than the wallet balance "
                                    f"({_bal:,.2f})", tag="guard")
        return a
    limits = data.payee_limits(s.payee)
    if limits and s.amount > limits["per_txn"]:
        a.say = (f"I can't send {s.amount:g} {s.currency or 'AED'} to {s.payee} — your limit "
                 f"for them is {limits['per_txn']:g} {limits['currency']} per transfer "
                 f"({limits['per_day']:g} a day). Try a smaller amount, or raise the limit "
                 f"in your settings.")
        a.outcome = "OVER_LIMIT"
        a.step("decide", "limit",
               f"{s.amount:g} exceeds the mandate's per-transaction cap of "
               f"{limits['per_txn']:g} {limits['currency']}", tag="guard")
        a.options = [{"kind": "ask_text",
                      "label": f"Send {limits['per_txn']:g} instead",
                      "text": f"send {limits['per_txn']:g} to {s.payee}"}]
        return a

    # A cross-border transfer should say so at the moment of confirming, with the fee and the
    # timing. Those are the two things a customer wants to check before they commit, and
    # "ready to send 500 to Rahul" hides both.
    corridor = s.corridor or next((b["corridor"] for b in data.sendable()
                                   if b["name"] == s.payee), "AE-AE")
    a.proposal = {"kind": "send", "payee": s.payee, "amount": s.amount,
                  "currency": s.currency or "AED", "corridor": corridor,
                  "sentence": f"send {s.amount:g} dirhams to {s.payee.split()[0].lower()}"}
    where = ""
    if corridor and corridor != "AE-AE":
        c = CORRIDORS[corridor]
        country = c["name"].split("→")[-1].strip()
        where = (f" in {country} — fee AED {c['fee_aed']:.2f}, arriving {c['sla']}")
        a.proposal["fee"] = c["fee_aed"]
    a.say = (f"Ready to send {s.amount:g} {s.currency or 'AED'} to {s.payee}{where}. "
             f"{voice.pick('ready', a.session)}")
    if not where:                                   # cross-border keeps its fee + timing sentence
        a.say = _ph(a, "propose_send", a.say, amount=f"{s.amount:g} {s.currency or 'AED'}", payee=s.payee)
    a.outcome = "PROPOSED"
    # "pay alex 100 in 2 minutes": a payment AT a time. The PIN is taken now, at confirm;
    # the server holds the authorised payment (never the PIN) and releases it when due.
    if _scheduled_for(a, s):
        return a
    a.step("act", "propose",
           "handing this to the signed-mandate flow — it still needs the read-back, the "
           "mandate limits and a device signature", tag="act")
    return a


#: How far ahead a payment may be scheduled. A login lasts 8 hours and the scheduled payment
#: runs on it, so anything later would fail at the due time — offer a reminder instead.
SCHEDULE_MAX_MIN = 6 * 60


def _dubai(iso: str) -> str:
    import datetime as _dt
    from zoneinfo import ZoneInfo
    t = _dt.datetime.fromisoformat(iso).astimezone(ZoneInfo("Asia/Dubai"))
    return t.strftime("%-I:%M %p") + ("" if t.date() == _dt.datetime.now(ZoneInfo("Asia/Dubai")).date()
                                      else t.strftime(", %-d %b"))


def _scheduled_for(a, s) -> bool:
    """Turn a ready proposal into a scheduled one when the customer named a time."""
    if not (s.when_at and a.proposal):
        return False
    import datetime as _dt
    due = _dt.datetime.fromisoformat(s.when_at)
    mins = (due - _dt.datetime.now(_dt.timezone.utc)).total_seconds() / 60
    if mins <= 0.5:
        return False                                   # "now" — an ordinary payment
    who = a.proposal.get("payee") or a.proposal.get("merchant") or ""
    if mins > SCHEDULE_MAX_MIN:
        a.proposal = None
        a.say = _tr(a, "schedule_too_far",
                    f"I can schedule a payment up to {SCHEDULE_MAX_MIN // 60} hours ahead. "
                    f"Shall I set a reminder for {_dubai(s.when_at)} instead?", when=_dubai(s.when_at))
        a.outcome = "NEED_CONFIRMATION"
        a.options = [{"kind": "ask_text", "label": "Yes, remind me",
                      "text": f"remind me to send {s.amount:g} to {who.split()[0].lower()} {s.when}"}]
        a.step("decide", "schedule.too_far", f"{mins:.0f} min ahead — beyond the {SCHEDULE_MAX_MIN} "
               f"min a login can carry; offering a reminder", tag="guard")
        return True
    a.proposal["scheduleAt"] = s.when_at
    a.say = _ph(a, "propose_scheduled", _tr(a, "scheduled_ready",
                f"I'll send {a.proposal['amount']:g} {a.proposal.get('currency', 'AED')} to {who} at "
                f"{_dubai(s.when_at)}. Confirm with your PIN now and it goes then — you can cancel "
                f"it until it runs.", when=_dubai(s.when_at)),
                amount=f"{a.proposal['amount']:g} {a.proposal.get('currency', 'AED')}", payee=who,
                when=_dubai(s.when_at))
    a.step("act", "propose.scheduled",
           f"a payment AT {s.when_at} ({mins:.0f} min from now): the PIN is taken at confirm, the "
           f"server holds the authorised payment — not the PIN — and releases it when due",
           tag="act")
    return True


def h_pay_merchant(a, s, said):
    # Same rule as h_send: a named merchant or an amount is hard evidence.
    if not (s.merchant or s.amount) and a.confidence < CONFIDENT_FOR_MONEY:
        a.say = "Which merchant did you want to pay? These are the ones near you."
        a.outcome = "NEED_INFO"
        a.options = [{"kind": "pay_merchant", "label": f"{m['name']}" + _m(m),
                      "mid": m["mid"], "merchant": m["name"]} for m in data.nearby()]
        a.step("act", "need.merchant",
               f"confidence {a.confidence:.0%} and nothing concrete yet", tag="guard")
        return a
    merchant = next((m for m in data.MERCHANTS if m["name"] == s.merchant), None)
    if not merchant:
        a.say = "Which merchant? These are the ones near you."
        a.outcome = "NEED_INFO"
        a.options = [{"kind": "pay_merchant", "label": f"{m['name']}" + _m(m),
                      "mid": m["mid"], "merchant": m["name"]} for m in data.nearby()]
        return a
    if not s.amount:
        a.say = _ph(a, "ask_merchant_amount", _tr(a, "how_much_at",
                    f"{voice.pick('how_much', a.session)[:-1]} at {merchant['name']}?",
                    who=merchant["name"]), merchant=merchant["name"])
        a.outcome = "NEED_INFO"
        a.step("act", "need.amount", "no amount — never assumed", tag="guard")
        return a
    # The BINDING number. The engine enforces the amount range, the per-user cap and the
    # remaining budget; we do not recompute it locally and hope it agrees.
    # reserve=True: /transact rejects an isDiscounted payment unless the customer already
    # HOLDS an open reserve for this product (ReserveLifecycleGuard #6). Quoting without one
    # produced a masked "Something went wrong" with the real reason only in the service log.
    # An abandoned reserve is released by the engine's own sweeper.
    # Same as the ranker: resolve the merchant's own campaign, then quote it by id.
    _found = incentive.discover(receiver_id=merchant.get("userId")) \
        if merchant.get("userId") else None
    _cid = (_found[0] or {}).get("campaignId") if _found else None
    quoted = incentive.quote(s.amount, receiver_id=merchant.get("userId"),
                             campaign_id=_cid, reserve=True) \
        if merchant.get("userId") else None
    if quoted and quoted.get("applicable"):
        discount = float(quoted.get("incentiveAmount") or 0)
        payable = float(quoted.get("discountedAmount") or (s.amount - discount))
        # Say the BENEFIT, not the campaign code. "CXDEMO-STYLE20 brings it to 230" is an
        # internal reference read out to a customer; "AED 20 off" is the thing they care about.
        if (quoted.get("benefitType") or "").upper() == "PERCENTAGE":
            applied = f"{_g(quoted.get('discountValue'))}% off"
            if quoted.get("maxDiscountCap"):
                applied += f" (max AED {_g(quoted['maxDiscountCap'])})"
        else:
            applied = f"AED {_g(quoted.get('discountValue') or discount)} off"
        priced = {"amount": s.amount, "discount": round(discount, 2),
                  "payable": round(payable, 2),
                  "applied": applied,
                  "campaign": quoted.get("incentiveCode"),
                  "why": f"quoted live by the incentive engine "
                         f"({quoted.get('incentiveType') or 'discount'} on "
                         f"{quoted.get('appliesOn') or 'amount'})"}
        a.step("resolve", "incentive.quote",
               f"campaign {quoted.get('campaignId')} — AED {discount:g} off, "
               f"payable AED {payable:g}"
               + (f", budget left {quoted['remainingBudget']}"
                  if quoted.get("remainingBudget") is not None else ""), tag="db")
        campaign_id = quoted.get("campaignId")
    else:
        priced = data.apply_offer(merchant, s.amount)
        why = incentive.last_error() or (
            "no campaign applies" if quoted else "incentive engine not consulted")
        a.step("resolve", "incentive.fallback",
               f"{why} — showing the indicative catalogue price; the payment is sent "
               f"undiscounted unless the engine confirms a campaign", tag="guard")
        campaign_id = None

    a.proposal = {"kind": "pay_merchant", "merchant": merchant["name"], "mid": merchant["mid"],
                  "amount": s.amount, "payable": priced["payable"],
                  "discount": priced["discount"], "campaign": priced.get("campaign"),
                  "campaignId": campaign_id,
                  # Resolved (and decrypted) at proposal time, not stored in config.
                  "receiverMobile": merchant.get("mobile") or data.merchant_mobile(
                      merchant.get("userId")),
                  "currency": "AED"}
    if priced["discount"]:
        a.say = (f"{merchant['name']}: AED {s.amount:g}, {priced['applied']} brings it to "
                 f"AED {priced['payable']:g}. {voice.pick('no_qr', a.session)}")
    else:
        a.say = (f"{merchant['name']}: AED {s.amount:g}. {priced['why'].capitalize()}. "
                 f"{voice.pick('no_qr', a.session)}")
    a.outcome = "PROPOSED"
    a.step("resolve", "offer", priced["why"], tag="db")
    a.step("act", "propose",
           "same signed-mandate path as any other payment — the merchant does not change "
           "the guardrail", tag="act")
    return a


#: Somebody has ALREADY paid us. The tense and the direction are what matter, not the words
#: "money" or "send", which appear in both directions.
#:
#: Hindi marks this grammatically and unambiguously: "Alex **ne** paise bhej diye" is Alex
#: doing the sending, while "Alex **ko** paise bhejne hai" is us sending to Alex. The
#: ergative `ne` plus a completive (`diya`/`diye`/`gaya`) is a report of something finished;
#: `ko` plus an infinitive is a request for something that has not happened.
_PAID_ALREADY = re.compile(
    r"\bne\b[^.?!]{0,40}\b(bhej|bheja|bheje|de|diya|diye|di|kar)\w*\b"
    r"|\b(bhej|pay|transfer|send|kar)\w*\s+(diya|diye|di|dia|chuka|chuke)\b"
    r"|\b(paid|has paid|have paid|already paid|settled up|sent it|sent me|has sent|"
    r"have sent|transferred it)\b"
    r"|\b(aa gaya|aa gaye|aagaya|aagaye|mil gaya|mil gaye|received|credited)\b"
    r"|भेज\s*(दिया|दिये|दिए)|आ\s*गया|आ\s*गये|मिल\s*गया"
    r"|\b(حول|أرسل)\w*\s+(لي|الفلوس)\b|وصل(ت|ني)\b",
    re.IGNORECASE)

#: ...unless it is plainly a REQUEST to send. "bhej do", "bhejna hai", "send him 50" all
#: contain a transfer verb and must never be read as a report. Checked first.
_WANTS_TO_SEND = re.compile(
    r"\b(bhejna|bhejne|bhejni|bhejo|bhejdo|bhej\s+do|bhejiye|bhejna\s+hai)\b"
    r"|\bko\b[^.?!]{0,30}\b(bhej|de|pay|send)\w*\b"
    r"|\b(send|pay|transfer)\s+(to|him|her|them|\d)"
    r"|भेजना|भेज\s*दो",
    re.IGNORECASE)


#: Intents the direction rule must NOT overrule when the model is sure of them. The rule
#: exists to stop "Alex sent me 100" becoming an offer to SEND Alex money; it was also
#: turning "sent it to the wrong Ahmed", "my refund hasn't come", "your total is wrong" and
#: "did my transfer reach my brother" into "someone paid you" (blind_v3: 10 of them).
_CLAIM_RIVALS = {"wrong_transfer", "refund_request", "refund_status", "disputed", "txn_status",
                 "txn_failed", "debited_not_credited", "split_status", "report_fraud",
                 "txn_on_hold", "split_payment", "bill_pay", "bill_due", "beneficiary_query"}


def _claim_rule_may_override(said: str) -> bool:
    p = client.classify(said)
    return not (p.get("intent") in _CLAIM_RIVALS and float(p.get("confidence") or 0) >= 0.6)


def _is_payment_claim(said: str) -> bool:
    """Is this a report that money ARRIVED, rather than a request to send it?

    A rule rather than a model decision, because the cost is asymmetric. Reading a report
    as a request produced "Alex ko kitne bhejne hai?" — an offer to send money to the man
    who had just paid us — and a customer who taps through that loses money to fix a
    misunderstanding. Reading a request as a report merely tells them nothing arrived.
    """
    t = _flat(said)
    if _WANTS_TO_SEND.search(t):
        return False
    if _I_PAID.search(t):
        return False
    # "dewa sent me a reminder, just pay it" — a biller plus "pay" is a bill to pay, not money in
    if ((_has_bill_word(said) or re.search(r"\b(?:dewa|sewa|fewa|addc|aadc|etisalat|du|empower|salik|internet|"
                                             r"electricity|water|bijli|wifi)\b|ديوا|كهربا|فاتورة", t))
            and re.search(r"\bpay\b|bhar\w*|ادفع|भर", t)):
        return False
    if _NOT_ARRIVED.search(t):
        return False
    # "did MY transfer reach Ahmed" — حوالتي/تحويلي is the customer's own outgoing payment.
    # "وصلت" (arrived) matched the arrival words and it was answered as money coming IN.
    if re.search(r"حوالتي|تحويلي|دفعتي|مبلغي|فلوسي\s+إلى", t):
        return False
    # "has Ahmed paid me back?" asks whether a debt is settled — the split's status — not
    # whether some money landed.
    if _is_question(said) and re.search(r"\b(?:paid|pay|pays|payed)\s+(?:me\s+)?back\b|"
                                         r"\bsettled\s+up\b", t):
        return False
    return bool(_PAID_ALREADY.search(t))


#: Money that did NOT arrive. The arrival words are matched as plain substrings, so
#: "credited" is found inside "not credited" and the negation was simply not read — which
#: turned "amount debited but not credited to receiver" into "Yes, Alex's 253.00 came in".
#: A customer reporting money lost in transit was told their money had arrived, which is
#: the worst answer in this file: it is wrong, it is reassuring, and it stops them chasing
#: it. Checked before the arrival words, never after.
_NOT_ARRIVED = re.compile(
    r"\b(?:not|never|havent|haven'?t|hasnt|hasn'?t|didnt|didn'?t|dont|don'?t|doesnt|"
    r"doesn'?t|no)\b[^.?!]{0,24}"
    r"\b(?:credit\w*|receiv\w*|arriv\w*|get|got|reach\w*|come|came|show\w*|land\w*)\b"
    r"|\b(?:credit\w*|receiv\w*|arriv\w*|reach\w*)\b[^.?!]{0,12}\b(?:not|yet)\b"
    r"|\bstill\s+(?:not|pending|waiting|pe?nding)\b"
    r"|(?:ما|لم|مو|مش|مب)\s*(?:\S+\s+)?(?:وصل\w*|استلم\w*|جا\w*|نزل\w*)"
    r"|नहीं\s+(?:मिल\w*|पहुं\w*|आय\w*)|(?:मिल\w*|पहुं\w*|आय\w*)\s+नहीं"
    # "nahi" can negate either half: "paisa nahi aaya" (it did not arrive) and "usne paise
    # nahi bheje" (he did not send them) are both reports that nothing came, and the second
    # one carries the SENDING verb, which the arrival list also matches.
    r"|\b(?:nahi|nahin|nai)\b[^.?!]{0,16}"
    r"\b(?:aaya|aaye|mila|mile|pahunch\w*|aya|bhej\w*|diya|diye|di|kiya)\b"
    r"|\b(?:aaya|aaye|mila|mile|pahunch\w*|bhej\w*|diya|diye)\b[^.?!]{0,8}"
    r"\b(?:nahi|nahin|nai)\b"
    r"|\u0928\u0939\u0940\u0902[^.?!]{0,16}(?:\u0906\u092f|\u092e\u093f\u0932|\u092a\u0939\u0941\u0902\u091a)"
    r"|(?:\u0906\u092f|\u092e\u093f\u0932|\u092a\u0939\u0941\u0902\u091a)[^.?!]{0,10}\u0928\u0939\u0940\u0902"
    r"|\u0644\u0645\s+(?:\u064a\u0635\u0644|\u062a\u0635\u0644|\u064a\u0633\u062a\u0644\u0645)"
    r"|\u0644\u0645\s+\u064a\u062a\u0645\s+\u0627\u0644\u0625\u064a\u062f\u0627\u0639",
    re.IGNORECASE)


#: First person plus a payment verb: the customer describing money they sent, which is the
#: OPPOSITE of money arriving. A bare "paid" was enough to trip the direction rule, so
#: "show me everything i paid today" — a plain request for a list — came back as "Yes,
#: Alex's 253 came in. That's every share." The rule exists to catch "Alex paid me"; the
#: subject is what tells the two apart, and it was not being read.
_I_PAID = re.compile(
    r"\b(?:i|we)\s+(?:just\s+|already\s+|have\s+|had\s+)?"
    r"(?:paid|pay|sent|send|transferred|transfer)\b"
    r"|\bi(?:'ve|ve|'d)\s+(?:just\s+|already\s+)?(?:paid|sent|transferred)\b"
    r"|\b(?:maine|mainne|humne|hamne)\b"
    r"|मैंने|हमने|\bدفعت\b|\bأرسلت\b",
    re.IGNORECASE)


#: Asking about the SHAPE of spending rather than about one payment. Positive patterns only;
#: the negative guard below is what keeps "send 100 to alex" out.
_SPEND_ASK = re.compile(
    r"\b(spend|spends|spending|spent|budget|kharch(?:a|e)?|kharcha|expense|expenses|"
    r"breakdown|analys\w*|analyz\w*|where did (?:my|the) money go|where is my money going|"
    r"paisa kaha|paise kaha|kitna kharch)\b", re.I)

#: Words that mean "move money now". A sentence with one of these is an instruction, and an
#: instruction must never be answered with a report — the same asymmetry `_is_payment_claim`
#: exists for, in the other direction.
_SPEND_NOT = re.compile(r"\b(send|transfer|pay|bhej\w*|kardo|karo)\b", re.I)


#: The same question in scripts where `\b` does not help: Python's word boundary is defined
#: on [A-Za-z0-9_], so it never fires between Devanagari or Arabic characters. These are
#: matched by plain containment instead, which is safe precisely because a Devanagari word
#: cannot be hiding inside a Latin one — the "du" inside "dukan" problem cannot occur across
#: scripts.
_SPEND_ASK_INTL: tuple[str, ...] = (
    "खर्च", "खर्चा", "ख़र्च", "बजट", "पैसे कहाँ", "पैसा कहाँ", "पैसे कहां",
    "أنفقت", "إنفاق", "مصروف", "ميزانية", "أين ذهبت أموالي",
)


#: Pushing back on something we said. Short, and nearly always a question — these are not
#: complaints about a payment, they are objections to a NUMBER, which is why "that's wrong"
#: belongs here while "my payment failed" does not.
_DISPUTE = re.compile(
    r"\b(?:can'?t be right|cannot be right|that'?s wrong|thats wrong|you'?re wrong|"
    r"youre wrong|not right|doesn'?t add up|does not add up|doesn'?t make sense|"
    r"makes no sense|how (?:can|is|could) (?:it|that|this) (?:be )?(?:the )?same|"
    r"same (?:number|figure|amount|total) (?:again|twice|both)|why (?:is it|are they) the same|"
    r"are you sure|you sure|check again|recheck|re-check|double ?check|"
    r"you said|earlier you said|but you (?:just )?said|that'?s not|thats not|"
    r"(?:figure|number|total|amount|calculation)\s+is\s+(?:wrong|off|incorrect)|no\s+way\s+i\s+(?:spent|paid|sent))\b", re.I)

#: The same objection in the languages this assistant actually gets asked in. Written out
#: rather than translated at runtime, like every other phrase list here.
_DISPUTE_INTL = (
    # hinglish
    "galat hai", "galat h", "sahi nahi", "theek nahi", "kaise ho sakta", "kaise same",
    "same kaise", "ek hi kaise", "dobara check", "phir se check", "pakka hai",
    "sure ho", "aapne kaha tha", "tumne kaha tha", "ye to galat",
    # hindi
    "ग़लत है", "गलत है", "सही नहीं", "ठीक नहीं", "कैसे हो सकता", "कैसे same",
    "एक ही कैसे", "दोबारा जाँच", "दोबारा चेक", "फिर से चेक", "पक्का है",
    "आपने कहा था", "तुमने कहा था",
    # arabic
    "غير صحيح", "هذا خطأ", "خطأ", "ليس صحيحا", "كيف يكون نفس", "نفس الرقم",
    "هل أنت متأكد", "تأكد مرة أخرى", "راجع مرة أخرى", "قلت لي",
)


def _is_dispute(said: str, session: str) -> bool:
    """Is the customer contradicting something we said?

    Requires a standing claim. Without one there is nothing to dispute, and the phrases are
    ordinary enough — "are you sure?" — that firing on them cold would turn an unrelated
    question into a recheck of nothing.
    """
    if not conversation.claims(session):
        return False
    t = _flat(said)
    if not t:
        return False
    return bool(_DISPUTE.search(t)) or any(w in t for w in _DISPUTE_INTL)


#: Superlatives and tallies. The question word matters as much as the verb: "who did I pay"
#: and "how much did I pay" are different questions about the same rows, and answering
#: either with a list of payments answers neither.
_RANK_WHO = re.compile(r"\b(?:who|whom|which (?:shop|store|merchant|person|one|place))\b", re.I)
_RANK_MOST = re.compile(
    r"\b(?:most|the most|largest|biggest|highest|maximum|max|top|"
    r"least|smallest|lowest|minimum|min|fewest)\b", re.I)
#: The customer is the subject. Rankings and tallies are about THEIR payments.
_FIRST_PERSON = re.compile(r"\b(?:i|i'?ve|i'?d|me|my|mine|myself|we|our|us|mera|meri|mere|maine|"
                           r"mujhe|humne|hamara|apna|apne)\b|मैंने|मेरा|मेरे|मुझे|أنا|لي\b|دفعت|أنفقت|أرسلت|"
                           r"ي\b", re.I)

#: A question about how the product WORKS, which mentions spending or payments but is not
#: about the customer's own figures: "how are my spending categories decided".
_ABOUT_THE_FEATURE = re.compile(
    r"\b(?:how|why)\b[^?]{0,40}\b(?:decided|decide|calculated|calculate|chosen|choose|work|works|"
    r"determined|categori[sz]ed|assigned|classified)\b", re.I)

_RANK_COUNT = re.compile(
    r"\b(?:how many|number of|count of|how often|how frequently)\b", re.I)
_RANK_SUM = re.compile(
    r"\b(?:how much (?:did|have|do)\b[^?]*\b(?:spend|spent|pay|paid|send|sent|transfer)|"
    r"total(?:ly)? (?:spent|paid|sent)|in total|altogether|on average|average)\b", re.I)

#: A tally is about money ALREADY MOVED. These ask what a transfer WOULD cost or take,
#: which is a question about the service's rules, not about the customer's ledger.
#: "how much does it cost to send to india" was answered with their last 116 payments —
#: the `\b` was missing after the auxiliary, so "do" matched inside "does".
_RANK_HYPOTHETICAL = re.compile(
    r"\b(?:cost|costs|charge|charges|fee|fees|commission|rate|rates|"
    r"will it (?:cost|take)|would it (?:cost|take)|how long)\b", re.I)

_RANK_NOT = re.compile(
    r"\b(?:limit|limits|allowed|allow|in\s+one\s+(?:go|shot)|at\s+once|per\s+(?:day|transfer|transaction)|"
    r"can\s+i\s+(?:send|transfer|pay)|capped|cap|owe|owes|still\s+owe|baki|baaki|dena\s+hai)\b|"
    r"(?:बिल|bill|فاتور\w*)[^.?!]{0,20}(?:देना|dena|due|owe|مستحق)|लिमिट|सीमा|الحد", re.I)

_RANK_INTL = (
    # hinglish — "sabse zyada kisko", "kitni baar", "kitna total"
    "sabse zyada", "sabse jyada", "sabse kam", "sabse bada", "sabse chhota", "sabse badi",
    "kisko sabse", "kis ko sabse", "kitni baar", "kitne payment", "kitna total",
    "total kitna", "average kitna", "aurat kitna",
    # hindi
    "सबसे ज़्यादा", "सबसे ज्यादा", "सबसे कम", "सबसे बड़ा", "सबसे बड़ी", "सबसे छोटा",
    "किसको सबसे", "कितनी बार", "कितने पेमेंट", "कुल कितना", "कितना कुल", "औसत",
    # arabic
    "الأكثر", "أكثر شخص", "الأكبر", "أكبر مبلغ", "الأقل", "الأصغر",
    "كم مرة", "كم عدد", "المجموع", "في المجموع", "المتوسط",
)


def _is_rank_question(said: str) -> bool:
    """Is the answer to this a name or a single number?

    A rule, like the two above it, because `spend_rank` is the 26th label and the deployed
    model has 23 outputs. Written to be precise: every one of these sentences currently
    reaches a handler that answers something else, so a miss costs a clarifying question
    while a false positive replaces a list the customer wanted with a single figure.
    """
    t = _flat(said)
    if not t:
        return False
    # "max amount I can send in one go", "ek din mein max kitna allow hai" — a LIMIT, and
    # "इस महीने कुल कितना बिल देना है" — bills OWED; neither is a ranking of their payments.
    if _RANK_NOT.search(t):
        return False
    if any(w in t for w in _RANK_INTL):
        return True
    if _RANK_HYPOTHETICAL.search(t):
        return False                     # about the rules, not about their own payments
    if _ABOUT_THE_FEATURE.search(t):
        return False                     # "how are my categories decided" — product knowledge
    # A tally of the CUSTOMER'S money. "what is the average water bill in sharjah" matched
    # "average" and was answered with the customer's own payment average — a question about
    # the world, answered with their ledger. No I/my/me, no ranking.
    if not _FIRST_PERSON.search(t):
        return False
    # "how much did I spend (on food) (last month)" is a SPENDING question — the total and
    # where it went — not a single figure about one payee. It is a tally only when it names
    # who: "how much did I spend at MClothes", "how much have I sent Alex".
    if (re.search(r"\bhow much\b[^?]*\b(?:spend|spent|spending)\b", t)
            and not _RANK_COUNT.search(t) and not _RANK_MOST.search(t)):
        _sl = slotlib.extract(said, "")
        if not (_sl.merchant or _sl.payee):
            return False
    if _RANK_COUNT.search(t) or _RANK_SUM.search(t):
        return True
    # "who ... most" needs both halves: "who did I pay" alone is a history question, and
    # "the most" alone turns up in plenty of sentences that are not rankings.
    return bool(_RANK_MOST.search(t) and (_RANK_WHO.search(t) or _has_spend_noun(t)))


#: The nouns a ranking is ABOUT. Without one, "biggest" has no subject and the sentence is
#: not a question about money at all.
_SPEND_NOUN = re.compile(
    r"\b(?:paid|pay|payment|payments|spend|spent|spending|sent|send|transfer|transfers|"
    r"txn|transaction|transactions|bill|bills|shop|shops|store|merchant|merchants|"
    r"kharch|kharcha|bheja|diya|payment|"
    r"खर्च|भेजा|दिया|पेमेंट|दुकान|"
    r"دفعت|أنفقت|أرسلت|الدفعات|المتجر|المتاجر)\b", re.I)


def _rank_follow_up(said: str) -> bool:
    """A bare change of window: "and last month?", "pichle hafte?", "و الشهر الماضي؟"."""
    low = " ".join((said or "").lower().split())
    return bool(spendmod.period_named(said) and _only_a_period(low))


def _swap_period(question: str, follow_up: str) -> str:
    """The earlier ranking question with its window replaced by the one just named.

    The window words are removed from the old question (longest first, so "last month" goes
    before "month") and the follow-up is appended; `period_from` then finds only the new one.
    """
    q = " ".join((question or "").lower().split())
    words = sorted({w for _, ws in (spendmod._PERIOD_WORDS_INTL + spendmod._PERIOD_WORDS)
                    for w in ws}, key=len, reverse=True)
    for w in words:
        q = q.replace(w, " ")
    q = spendmod._SPAN.sub(" ", q)
    q = re.sub(r"\b(?:in|for|during|over|of)\s*$", " ", " ".join(q.split()))
    return " ".join((q + " " + (follow_up or "").strip(" ?.!")).split())


def _has_spend_noun(t: str) -> bool:
    return bool(_SPEND_NOUN.search(t))


_SPLIT_WORDS = re.compile(r"\b(split|divide|share it|share the bill|baant|baat do|bant)\b|बाँट|बांट|قسم|تقسيم", re.I)
_PAST_PAYMENT = re.compile(r"\b(last|previous|earlier|that|the|my)\s+(payment|bill|purchase|one|txn|transaction)\b|"
                           r"\byesterday\b|\bpichl\w*\b|\bpehle\b", re.I)


def _refers_to_past_payment(said: str) -> bool:
    """"split my last payment", "split that bill", "split yesterday's" — a past payment named."""
    return bool(_PAST_PAYMENT.search(said or ""))


def _payment_in_progress(session: str) -> bool:
    """Is a payment being set up in this conversation right now (not yet made)?"""
    try:
        from engine import pin as cx_pin
        if cx_pin.pending(session):
            return True
    except Exception:                                              # noqa: BLE001
        pass
    f = conversation.get(session)
    if f and f.intent in (Intent.PAY_MERCHANT.value, Intent.BILL_PAY.value):
        return True
    p = conversation.proposal(session) or {}
    return p.get("kind") in ("pay_merchant", "merchant", "bill", "bill_pay")


def h_spend_rank(a, s, said, period=None):
    """Answer with the name or the number, and show the arithmetic behind it.

    A bare name is unverifiable — "Alex" is not something a customer can check against
    their own statement — so every answer here carries the total, how many payments made it
    up, and what came second. The runner-up is what turns a name into a finding.
    """
    # Kept so a follow-up that only changes the window ("and last month?") can re-ask it.
    conversation.note_subject(a.session, "rank_q", said)
    low = " ".join((said or "").lower().split())
    period = period or spendmod.period_from(said)
    scope = _scope_from(said)
    least = bool(re.search(r"\b(least|smallest|lowest|minimum|min|fewest|sabse kam|"
                           r"सबसे कम|الأقل|الأصغر)\b", low))
    cur = "AED"

    # "how many" / "how much in total" / "on average" — a tally, not a ranking.
    if _RANK_COUNT.search(low) or _RANK_SUM.search(low) or any(
            w in low for w in ("kitni baar", "kitne payment", "kitna total", "total kitna",
                               "कितनी बार", "कितने पेमेंट", "कुल कितना", "औसत",
                               "كم مرة", "كم عدد", "المجموع", "المتوسط")):
        who = s.payee or s.merchant
        t = aggregate.tally(period, scope, payee=who)
        a.step("resolve", "rank.tally",
               f"counting rather than listing — {t['payments']} payments"
               + (f" to {who}" if who else "")
               + f" in {t['period_label']}", tag="rule")
        plabel = spendmod.period_label(period, a.lang)
        if not t["payments"]:
            a.say = _tr(a, "rank_none", f"I can't see any payments in {plabel}.",
                        period=plabel)
            a.outcome = "NONE"
            return a
        # Say WHAT was counted, not just how much. "How much did I pay to merchants last
        # month" was answered "36 payments in the last 30 days, AED 4,549.00 in total" —
        # which is the right number for the right rows and gives the customer no way to
        # tell that merchants is what it counted. They then had to ask "is it merchant
        # related", which is a question the answer should never have left open.
        what = {"merchant": _tr(a, "rank_at_merchants", "at merchants"),
                "person": _tr(a, "rank_to_people", "to people")}.get(scope, "")
        a.say = _tr(a, "rank_tally",
                    f"{t['payments']} payments{' ' + what if what else ''} in {plabel}, "
                    f"{cur} {t['total']:,.2f} in total — an average of "
                    f"{cur} {t['average']:,.2f}.",
                    n=t["payments"], period=plabel, cur=cur, what=what,
                    total=f"{t['total']:,.2f}", avg=f"{t['average']:,.2f}")
        if who:
            a.say += " " + _tr(a, "rank_only_to", f"That's just to {who}.", who=who)
        a.outcome = "ANSWERED"
        a.rank = t
        conversation.note_claim(a.session, recheckmod.claim_rank(t, said))
        return a

    # "biggest / smallest payment" — one payment, not a payee.
    if re.search(r"\b(biggest|largest|smallest|highest|lowest|single)\s+"
                 r"(payment|txn|transaction|transfer|amount|spend)\b", low) or \
       re.search(r"\b(sabse bada|sabse badi|सबसे बड़ा|سبسे बड़ी|أكبر مبلغ)\b", low):
        e = aggregate.extreme("smallest" if least else "biggest", period, scope)
        ans = e["answer"]
        plabel = spendmod.period_label(period, a.lang)
        if not ans:
            a.say = _tr(a, "rank_none", f"I can't see any payments in {plabel}.",
                        period=plabel)
            a.outcome = "NONE"
            return a
        a.step("resolve", "rank.extreme",
               f"the {e['kind']} single payment in {e['period_label']} — "
               f"{cur} {ans['total']:,.2f} to {ans['name']}", tag="rule")
        days = round(ans["hours_ago"] / 24.0, 1)
        a.say = _tr(a, "rank_extreme",
                    f"Your {e['kind']} payment in {plabel} was {cur} "
                    f"{ans['total']:,.2f} to {ans['name']}, {days} days ago.",
                    cur=cur, amount=f"{ans['total']:,.2f}", who=ans["name"],
                    period=plabel, days=days, kind=e["kind"])
        a.outcome = "ANSWERED"
        a.rank = e
        conversation.note_claim(a.session, recheckmod.claim_rank(e, said))
        return a

    # Otherwise: who got the most (or the least).
    r = aggregate.rank("bottom" if least else "top", period, scope)
    ans = r["answer"]
    plabel = spendmod.period_label(period, a.lang)
    if not ans:
        a.say = _tr(a, "rank_none", f"I can't see any payments in {plabel}.", period=plabel)
        a.outcome = "NONE"
        return a
    a.step("resolve", "rank.top",
           f"ranking {r['distinct']} {'merchants' if scope == 'merchant' else 'payees'} by "
           f"total in {r['period_label']} — {ans['name']} at {cur} {ans['total']:,.2f} "
           f"over {ans['count']} payments", tag="rule")
    if r["tied"]:
        names = ", ".join([ans["name"]] + [t["name"] for t in r["tied"]])
        a.step("decide", "rank.tie",
               f"{len(r['tied']) + 1} level on {cur} {ans['total']:,.2f} — reporting the "
               f"tie rather than picking one", tag="guard")
        a.say = _tr(a, "rank_tie",
                    f"It's a tie in {plabel}: {names}, {cur} {ans['total']:,.2f} each.",
                    names=names, cur=cur, amount=f"{ans['total']:,.2f}", period=plabel)
        a.outcome = "ANSWERED"
        a.rank = r
        return a
    a.say = _tr(a, "rank_top",
                f"{ans['name']} — {cur} {ans['total']:,.2f} across {ans['count']} "
                f"payments in {plabel}.",
                who=ans["name"], cur=cur, amount=f"{ans['total']:,.2f}",
                n=ans["count"], period=plabel)
    if r.get("runner_up"):
        ru = r["runner_up"]
        a.say += " " + _tr(a, "rank_runner_up",
                           f"Next is {ru['name']} at {cur} {ru['total']:,.2f}.",
                           who=ru["name"], cur=cur, amount=f"{ru['total']:,.2f}")
    a.outcome = "ANSWERED"
    a.rank = r
    conversation.note_claim(a.session, recheckmod.claim_rank(r, said))
    return a


#: Asking what the last answer COVERED. Not a dispute — the customer is not saying the
#: number is wrong, they are asking what is inside it. "is it merchant related", "does that
#: include bills", "was that just Alex", "is that everything".
_ABOUT_LAST = re.compile(
    r"^(?:and\s+|so\s+|but\s+|ok\s+|okay\s+)*"
    r"(?:is|was|does|did|are|were)\s+(?:that|it|this|those|these)\b"
    r"|^(?:does|did)\s+(?:that|it|this)\s+include\b"
    r"|^(?:that'?s|thats|is that)\s+(?:just|only|all|everything)\b"
    r"|\b(?:merchant|merchants|shop|shops|store|stores|people|person|everyone|"
    r"everything|all of them|only|just)\s*(?:related|payments?|spend(?:ing)?)?\s*\??$",
    re.I)

_ABOUT_LAST_INTL = (
    "kya ye sirf", "sirf merchant", "sirf dukan", "ye sab milakar", "isme bills bhi",
    "क्या ये सिर्फ", "सिर्फ दुकान", "इसमें सब", "क्या इसमें",
    "هل هذا فقط", "هل يشمل", "فقط التجار",
)


#: An instruction. A sentence carrying one of these is a NEW request, whatever the last
#: answer was: "send money to a shop" after a spending figure is a payment, not a question
#: about whether the figure included shops. It was answered "No — that was everything…".
_INSTRUCTION = re.compile(
    r"\b(?:send|pay|transfer|recharge|top ?up|remind|split|withdraw|add money|refund|cancel|"
    r"show me|show my|list|bhej\w*|bhejo|kar ?do|karna hai|bharo|dikhao|ادفع|أرسل|حول|اشحن)\b|"
    r"भेज|भरना|दिखाओ", re.I)


def _is_instruction(said: str) -> bool:
    return bool(_INSTRUCTION.search(said or "")) or _has_transfer_verb(said)


_QUESTION = re.compile(
    r"^\s*(?:what|what'?s|whats|how|why|when|where|which|who|whose|is|are|can|could|do|does|did|"
    r"kya|kitna|kitni|kaise|kab|kahan|kaun|كم|ما|ماذا|هل|متى|أين|كيف|क्या|कितना|कैसे|कब|कहाँ)\b", re.I)


def _is_question(said: str) -> bool:
    return bool(_QUESTION.search(said or "")) or (said or "").strip().endswith(("?", "؟"))


def _speaks_english(said: str) -> bool:
    """A sentence with English words in it, as opposed to a number, a yes or a name."""
    words = re.findall(r"[A-Za-z]{3,}", said or "")
    return len(words) >= 2 and not _is_yes(said) and not _is_no(said)


def _asks_about_last(said: str) -> bool:
    t = _flat(said)
    if not t or len(t.split()) > 9:
        return False
    if _is_instruction(said):
        return False                       # a new request, not a question about the figure
    # A COMPARISON is not a question about scope. "is that more than before?" matches the
    # "is that ..." shape, and it is asking whether the figure went up — which belongs to
    # the spend thread, or to the dispute path if they are challenging it. Answering it
    # with "yes, that was 37 payments at merchants" describes the figure instead of
    # comparing it, which is a different question politely answered.
    if _COMPARATIVE.search(t) or _has_intl_comparative(t):
        return False
    return bool(_ABOUT_LAST.search(t)) or any(w in t for w in _ABOUT_LAST_INTL)


#: Asking about a deal, with no subject of its own. "is there any offer", "any discount",
#: "koi offer hai". Only referential when something is already on the table.
_ASKS_OFFER = re.compile(
    r"\b(?:offer|offers|discount|discounts|deal|deals|cashback|promo|promotion|voucher|"
    r"coupon|saving|savings)\b", re.I)
_ASKS_OFFER_INTL = ("koi offer", "offer hai", "discount hai", "chhoot", "छूट",
                    "कोई ऑफर", "ऑफर है", "خصم", "عرض", "عروض")


def _asks_about_offer(said: str) -> bool:
    t = _flat(said)
    if not t or len(t.split()) > 8:
        return False
    return bool(_ASKS_OFFER.search(t)) or any(w in t for w in _ASKS_OFFER_INTL)


def h_offer_on_this(a, s, said):
    """Is there an offer ON THE PAYMENT WE ARE ABOUT TO MAKE?

    Mid-flow at a merchant — "pay a merchant" → "MClothes" → "100" → a proposal standing
    at AED 80 after AED 20 off — asking "is there any offer" was answered with the nearby
    offers list. Everything needed was already in the proposal: the merchant, the amount,
    the discount, the campaign. The question was referential and the answer was a catalogue.
    """
    held = conversation.proposal(a.session) or {}
    merchant = held.get("merchant")
    cur = "AED"
    amount = float(held.get("amount") or 0)
    payable = float(held.get("payable") or amount)
    discount = float(held.get("discount") or 0)

    a.intent, a.confidence, a.model = Intent.OFFERS_NEARBY.value, 1.0, "in-hand"
    a.step("decide", "offer.in_hand",
           f"answering about the {merchant} payment already on the table rather than "
           f"listing what is nearby", tag="rule")
    if discount > 0:
        a.say = _tr(a, "offer_on_this",
                    f"Yes — {cur} {discount:,.0f} off this one at {merchant}, so "
                    f"{cur} {amount:,.0f} becomes {cur} {payable:,.0f}. "
                    f"It's already in the amount I showed you.",
                    cur=cur, off=f"{discount:,.0f}", who=merchant,
                    was=f"{amount:,.0f}", now=f"{payable:,.0f}")
    else:
        a.say = _tr(a, "offer_none_here",
                    f"No offer on this one at {merchant} — it's {cur} {amount:,.0f} as it "
                    f"stands. Want me to check what's on nearby?",
                    who=merchant, cur=cur, amount=f"{amount:,.0f}")
        a.options = [{"kind": "ask_text", "label": "What's nearby",
                      "text": "offers near me"}]
    a.outcome = "ANSWERED"
    # The payment is still on the table — answering a question about it must not drop it.
    conversation.hold_proposal(a.session, held)
    return a


def h_about_last(a, s, said):
    """Answer a question about the figure we just gave.

    This is the ordinary shape of a conversation and the assistant had no room for it. Ask
    "how much did I pay to merchants last month", then "is it merchant related", and the
    second question reached the dispute handler and was told "I don't have a figure from
    this conversation to check" — untrue twice over, since there was a figure and we knew
    precisely what went into it.
    """
    claim = conversation.last_claim(a.session)
    d = recheckmod.describe(claim)
    if not d:
        a.step("decide", "about.nothing",
               "nothing answered yet in this conversation for the question to be about",
               tag="guard")
        a.say = _tr(a, "about_nothing",
                    "There's nothing from this conversation for me to describe yet — "
                    "ask me something about your payments and I'll start there.")
        a.outcome = "NEED_INFO"
        return a

    cur = "AED"
    period = spendmod.period_label(d.get("period") or "month", a.lang)
    covers = {"merchants": _tr(a, "rank_at_merchants", "at merchants"),
              "people": _tr(a, "rank_to_people", "to people")}.get(d["covers"], "")
    a.step("decide", "about.last",
           f"describing what the last figure covered — {d['payments']} payments, "
           f"scope {d['scope'] or 'everything'}, {d['period_label']}", tag="rule")

    if d["only"]:
        a.say = _tr(a, "about_only",
                    f"Yes — that was {d['payments']} payments {covers} in {period}, "
                    f"{cur} {float(d['total'] or 0):,.2f} in total. Nothing else is in it.",
                    n=d["payments"], covers=covers, period=period, cur=cur,
                    total=f"{float(d['total'] or 0):,.2f}")
    elif d.get("payee"):
        a.say = _tr(a, "about_payee",
                    f"That was only payments to {d['payee']} — {d['payments']} of them in "
                    f"{period}.", who=d["payee"], n=d["payments"], period=period)
    else:
        a.say = _tr(a, "about_everything",
                    f"No — that was everything in {period}: {d['payments']} payments, "
                    f"both merchants and people. Ask me for one or the other and I'll "
                    f"split it out.", n=d["payments"], period=period)
    a.outcome = "ANSWERED"
    return a


def ask_image(image: bytes, session: str = "default", caption: str = "") -> Answer:
    """A customer sends a screenshot of a payment instead of typing its reference.

    The picture is used to IDENTIFY one of their own payments and for nothing else. The
    status, the amount and everything else in the answer come from the ledger — see
    `ocr.py` for why that is not caution but the only defensible reading of an image
    somebody handed us.
    """
    t0 = time.time()
    a = Answer(intent=Intent.TXN_STATUS.value, say="")
    a.session = session
    a.said = caption or "[screenshot]"
    a.lang = langmod.detect(caption) if caption else conversation.last_lang(session)
    a.model = "ocr"
    a.confidence = 1.0
    a.step("understand", "received", "a screenshot of a payment", tag="input")

    if not ocrmod.backend():
        a.step("understand", "ocr.none",
               "no reader installed on this host, so the picture cannot be opened at all",
               tag="guard")
        a.say = _tr(a, "ocr_unavailable",
                    "I can't read pictures just yet. If you send me the reference number "
                    "from that screen I'll look it up right away.")
        a.outcome = "NEED_INFO"
        return _finish(a)

    text = ocrmod.read(image)
    if not text or not text.strip():
        a.step("understand", "ocr.blank", "nothing legible in the picture", tag="guard")
        a.say = _tr(a, "ocr_blank",
                    "I couldn't make out any text in that image. A clearer screenshot of "
                    "the payment details, or just the reference number, and I'm away.")
        a.outcome = "NEED_INFO"
        return _finish(a)

    txns = data.transactions_for(limit=60)
    got = ocrmod.resolve(text, txns)
    f = got["fields"]
    a.step("resolve", "ocr.read",
           f"read {len(text.split())} words from the picture"
           + (f" — reference {f['labelled']}" if f.get("labelled")
              else f" — {len(f['refs'])} candidate references"), tag="rule")

    if got["match"]:
        txn = got["match"]
        a.step("resolve", "ocr.matched",
               f"that reference matches your own payment {txn.ref} — reading its status "
               f"from the ledger rather than from the picture", tag="rule")
        s_obj = slotlib.Slots(txn_ref=txn.ref)
        a.slots = s_obj.as_dict()
        out = h_txn(a, s_obj, caption or "")
        out.options = (out.options or []) + [_agent_option(out)]
        _note_upload(session, txn.ref, f"{txn.payee} · {txn.amount:g} {txn.currency}")
        _remember(session, out, s_obj)
        out.ms = int((time.time() - t0) * 1000)
        return _finish(out)

    if got["candidates"]:
        a.step("resolve", "ocr.several",
               f"{len(got['candidates'])} of your payments could be the one in the "
               f"picture — asking rather than picking", tag="guard")
        a.say = _tr(a, "ocr_which",
                    "I can see a payment in that screenshot but more than one of yours "
                    "could be it. Which of these?")
        a.options = [{"kind": "pick_txn",
                      "label": f"{t.payee} · {t.amount:g} {t.currency} · "
                               f"{t.public()['when']}",
                      "ref": t.ref} for t in got["candidates"][:5]]
        a.outcome = "NEED_INFO"
        return _finish(a)

    # Before concluding it is not theirs: a REFUND screenshot carries the refund's own
    # reference, which belongs to the merchant, not to them. What makes it theirs is the
    # payment it refunded. This is the reported case — a customer holding a picture of
    # their own refund, told the reference was not from their wallet.
    for _ref in f["refs"]:
        link = histmod.refunded_link(_ref)
        if not link:
            continue
        a.step("resolve", "ocr.refund",
               f"that reference is a refund of {link['original_ref']}, which IS one of "
               f"this customer's payments — answering about theirs, not the merchant's",
               tag="rule")
        cur = "AED"
        a.intent = Intent.REFUND_STATUS.value
        a.say = _tr(a, "ocr_refunded",
                    f"That one's already been refunded — {cur} {link['amount']:,.2f} on "
                    f"your {link['product'].lower() or 'payment'} went back to you. "
                    f"Nothing more to do on it.",
                    cur=cur, amount=f"{link['amount']:,.2f}",
                    product=link["product"].lower() or "payment")
        a.outcome = "ALREADY_CREDITED"
        a.options = [_agent_option(a)]
        _note_upload(session, link["original_ref"],
                     f"refund of {cur} {link['amount']:,.2f}")
        _remember(session, a, slotlib.Slots(txn_ref=link["original_ref"]))
        a.ms = int((time.time() - t0) * 1000)
        return _finish(a)

    # Read something, matched nothing of theirs. Say exactly that.
    a.step("resolve", "ocr.nomatch", got["why"], tag="guard")
    a.say = _tr(a, "ocr_not_yours",
                "I read that screenshot, but the reference on it isn't one of the payments "
                "from your wallet. If it was paid from a different account I won't be able "
                "to see it — here are your recent ones.")
    a.options = _txn_options()
    a.outcome = "NEED_INFO"
    return _finish(a)


def _note_upload(session: str, ref: str, label: str) -> None:
    """Put the screenshot in the session's durable ledger.

    A picture is a thing the customer did, and they will refer back to it — "the one I
    uploaded", three flows and ten turns later. Without a record the engine has no idea an
    upload ever happened, so the reference resolves to whatever is merely most recent,
    which is right by luck or wrong silently.

    It goes in the same ledger as bills and splits rather than a store of its own: the
    recall machinery already exists and already survives topic changes.
    """
    try:
        conversation.open_task(session, "upload", f"the screenshot of {label}",
                               key={"ref": ref, "upload": "1"})
        conversation.note_subject(session, "uploaded_ref", ref)
    except Exception:                                         # noqa: BLE001
        pass


#: Referring back to a picture they sent. No subject of its own — "the one I uploaded" is
#: only meaningful against something that happened earlier in this conversation.
_MEANS_UPLOAD = re.compile(
    r"\b(?:uploaded|upload|screenshot|screen\s?shot|the (?:image|picture|photo|pic)|"
    r"that (?:image|picture|photo|pic|screenshot)|the one i (?:sent|shared|uploaded|showed)|"
    r"i (?:sent|shared|uploaded) (?:you )?(?:the|a|an)?\s*"
    r"(?:image|picture|photo|pic|screenshot))\b", re.I)
_MEANS_UPLOAD_INTL = (
    "jo bheja tha", "jo maine bheja", "photo wala", "screenshot wala", "image wali",
    "jo upload kiya", "tasveer", "जो भेजा था", "जो मैंने भेजा", "स्क्रीनशॉट", "तस्वीर",
    "الصورة", "لقطة الشاشة", "التي أرسلتها", "الصوره",
)


def _means_the_upload(said: str) -> bool:
    t = _flat(said)
    return bool(_MEANS_UPLOAD.search(t)) or any(w in t for w in _MEANS_UPLOAD_INTL)


def _agent_option(a) -> dict:
    """The way out to a person, offered even when the payment went through.

    A customer sending a screenshot is usually already unhappy, and "delivered" is not
    always the end of it — the money arrived and the goods did not, the amount was wrong,
    somebody else was paid. Offering this only on failures would be offering it only when
    WE think something went wrong.
    """
    return {"kind": "ask_agent", "label": "Talk to someone",
            "text": "i want to talk to an agent"}


def h_disputed(a, s, said):
    """The customer says a figure of ours is wrong. Work it out again and answer honestly.

    Three outcomes and no fourth. The figure was wrong, in which case say so plainly and
    give the right one. The figure was right, in which case give the reason it is right —
    repeating the number is what the customer just told us is not good enough. Or there is
    nothing recorded to check, in which case say that instead of picking a side.

    What is not allowed is defending a number without recomputing it, which is why
    `recheck` re-reads the ledger with the cache bypassed. Agreeing with yourself out of
    your own cache is not a check.
    """
    standing = conversation.claims(a.session)
    target = standing[-1] if standing else None
    verdict = recheckmod.recheck(target, standing[:-1] if standing else [])
    a.spend = verdict.get("now")

    if verdict["verdict"] == "cannot_check":
        a.step("decide", "dispute.nothing",
               "nothing recorded from this conversation to check against — saying so rather "
               "than defending a figure I cannot reproduce", tag="guard")
        a.say = _tr(a, "dispute_nothing",
                    "I don't have a figure from this conversation to check. Ask me again "
                    "and I'll work it out from your ledger.")
        a.outcome = "NEED_INFO"
        return a

    cur = (verdict.get("now") or {}).get("currency", "AED")
    if verdict["verdict"] == "corrected":
        was, now = verdict["was"], verdict["is_now"]
        a.step("decide", "dispute.corrected",
               f"re-read the ledger: {cur} {now:,.2f}, not the {cur} {was:,.2f} I said — "
               f"the earlier figure was wrong and is withdrawn", tag="guard")
        conversation.note_correction(a.session, target, now)
        a.say = _tr(a, "dispute_corrected",
                    f"You're right, and I was wrong. It's {cur} {now:,.2f} for "
                    f"{spendmod.period_label((verdict.get('now') or {}).get('period'), a.lang)}"
                    f", not {cur} {was:,.2f}.",
                    cur=cur, now=f"{now:,.2f}", was=f"{was:,.2f}",
                    period=spendmod.period_label(
                        (verdict.get("now") or {}).get("period"), a.lang))
        extra = _dispute_reasons(a, verdict["why"], cur)
        if extra:
            a.say += " " + extra
        a.outcome = "CORRECTED"
        a.rows = (verdict.get("now") or {}).get("categories", [])
        return a

    # It stands. Saying "yes it is" would be worthless, so the answer is the evidence.
    a.step("decide", "dispute.stands",
           f"re-read the ledger with the cache bypassed and got the same "
           f"{cur} {verdict['is_now']:,.2f} — the figure holds, so the answer is why",
           tag="rule")
    why = _dispute_reasons(a, verdict["why"], cur)
    a.say = _tr(a, "dispute_stands",
                f"I've checked it again from your ledger and it is "
                f"{cur} {verdict['is_now']:,.2f}.",
                cur=cur, now=f"{verdict['is_now']:,.2f}",
                period=spendmod.period_label(
                    (verdict.get("now") or {}).get("period"), a.lang))
    if why:
        a.say += " " + why
    a.outcome = "ANSWERED"
    a.rows = (verdict.get("now") or {}).get("categories", [])
    return a


def _dispute_reasons(a, why: list[dict], cur: str) -> str:
    """Turn the recheck's evidence into a sentence, in the customer's language.

    Every branch here corresponds to a fact `recheck` actually found. There is no `else`
    that produces a general reassurance, because a reason the data cannot support reads
    exactly like one it can — and this is the one place in the conversation where the
    customer has already said they do not believe us.
    """
    out = []
    for w in why:
        if w["k"] == "same_payments":
            first = spendmod.period_label(w["a"], a.lang)
            second = spendmod.period_label(w["b"], a.lang)
            w = dict(w, a=first, b=second)
            out.append(_tr(a, "dispute_same_payments",
                           f"Both {w['a']} and {w['b']} cover the same {w['n']} "
                           f"payments, "
                           f"which is why the total doesn't change.",
                           n=w["n"], first=w["a"], second=w["b"]))
        elif w["k"] == "quiet_stretch":
            out.append(_tr(a, "dispute_quiet",
                           f"Your most recent merchant payment before that was "
                           f"{w['oldest_days']} days ago, and the one before it was "
                           f"{w['gap_days']} days ago — nothing in between.",
                           oldest=w["oldest_days"], gap=w["gap_days"]))
        elif w["k"] == "window":
            out.append(_tr(a, "dispute_window",
                           f"That's {w['n']} payments, the oldest {w['oldest_days']} days ago.",
                           n=w["n"], oldest=w["oldest_days"]))
        elif w["k"] == "count":
            out.append(_tr(a, "dispute_count",
                           f"I counted {w['after']} payments this time, not {w['before']}.",
                           before=w["before"], after=w["after"]))
        elif w["k"] == "not_live":
            out.append(_tr(a, "dispute_not_live",
                           "I should say that I couldn't reach your live ledger, so this is "
                           "sample data — that may be the whole of the problem."))
    return " ".join(out)


#: Arabic writes short vowels and gemination as marks BETWEEN the letters, and they are
#: optional — the same word is spelled with them or without them depending on who is typing.
#: `قسّمها` ("split it") carries a shadda on the second letter, so it does not contain the
#: substring `قسم`, and every Arabic phrase list in this file matches by substring. The
#: result was the reported bug's Arabic variant falling out of scope at the last turn while
#: English and Hindi both passed.
#:
#: Exactly the same class of failure as `\W+` shredding Devanagari: a list of spellings
#: would have patched this one word and left the next. Stripping the marks fixes the class.
#: U+064B–U+0652 are the tashkeel, U+0670 the dagger alif, U+0640 the tatweel (a stretching
#: character with no sound at all).
_TASHKEEL = re.compile(r"[\u064b-\u0652\u0670\u0640]")


def _flat(said: str | None) -> str:
    """Lower-cased, whitespace-collapsed, and stripped of Arabic diacritics. The normal form
    every phrase list in this file is written against."""
    return _TASHKEEL.sub("", " ".join((said or "").lower().split()))


#: Asking for more about the thing just answered. No subject of its own — "the details" is
#: only meaningful against something, and the something is whatever we were last talking
#: about.
_WANTS_DETAIL = re.compile(
    r"^(?:yes\s+|yeah\s+|ok\s+|okay\s+|sure\s+|please\s+|and\s+)*"
    r"(?:show|give|tell|send|get|see|view|open|pull up|i want|i need|can i see|"
    r"let me see|what are|whats|what's)?\s*"
    r"(?:me\s+)?(?:the\s+|its\s+|it'?s\s+|full\s+|more\s+|all\s+)*"
    r"(?:details?|info|information|breakdown|receipt|reference|ref|proof)"
    r"(?:\s+(?:of|for|about)\s+(?:it|that|this|the (?:one|payment|transaction)))?"
    r"[\s.!?،]*$", re.I)

_WANTS_DETAIL_INTL = (
    "detail dikhao", "details dikhao", "poori detail", "puri detail", "aur batao",
    "zyada batao", "receipt dikhao",
    "डिटेल दिखाओ", "पूरी डिटेल", "और बताओ", "रसीद दिखाओ",
    "التفاصيل", "أظهر التفاصيل", "مزيد من التفاصيل", "الإيصال",
)


#: The handful of nouns this rule turns on. Typed one-handed on a phone they are the words
#: most likely to come out wrong, and a rule that only matches the correct spelling is a
#: rule that fires for careful typists — "detials" is the exact reported failure.
_DETAIL_WORDS = ("details", "detail", "information", "breakdown", "receipt", "reference",
                 "proof", "info")


def _despell(t: str) -> str:
    """Put obvious misspellings of this rule's keywords back to their correct form.

    Only these words, and only when the token is otherwise unrecognised, so it cannot
    quietly rewrite a name into a keyword.
    """
    out = []
    for w in t.split():
        if len(w) >= 5 and w not in _DETAIL_WORDS:
            hit = next((d for d in _DETAIL_WORDS
                        if abs(len(d) - len(w)) <= 1 and _edits_within(w, d, 2)), None)
            out.append(hit or w)
        else:
            out.append(w)
    return " ".join(out)


def _wants_detail(said: str) -> bool:
    t = _despell(_flat(said))
    return bool(_WANTS_DETAIL.match(t)) or any(w in t for w in _WANTS_DETAIL_INTL)


def _is_spend_question(said: str) -> bool:
    """Is this a question about spending PATTERN?

    A rule rather than a model decision because the deployed classifier cannot reach this
    intent at all: `openwallet-cx-intent-ml-v7` has 23 outputs and `spend_analysis` is the 24th
    label. Until the next retrain this is the only path to it, so it is written to be
    precise rather than generous — a missed spend question asks a clarifying question, while
    a false positive answers a payment instruction with a pie chart.
    """
    t = _flat(said)
    if _ABOUT_THE_FEATURE.search(t):
        return False                     # how categories are decided is product knowledge
    if any(w in t for w in _SPEND_ASK_INTL):
        return True
    if not _SPEND_ASK.search(t):
        return False
    # "how much did I spend at MClothes" is still a spend question; "pay MClothes 100" is not.
    # The verb only disqualifies when the sentence is not itself asking about spending.
    if _SPEND_NOT.search(t) and not re.search(r"\b(spend|spent|spending|kharch\w*)\b", t, re.I):
        return False
    return True


def h_spend_analysis(a, s, said, period=None):
    """A breakdown of where the money went, and what changed.

    Three decisions worth stating, because each is a thing the obvious version gets wrong:

    * **Merchant payments only.** Filing "sent 500 to Alex" under a category would make the
      breakdown add up to a number the customer never spent in any shop.
    * **The headline is the biggest MOVE, not the biggest total.** The biggest total is the
      same category every month and tells them nothing. The biggest change is the thing they
      might want to do something about.
    * **Uncategorised is reported, not hidden.** Where the merchant has no category yet the
      spend lands in `other` and the answer says how much it was. A breakdown that quietly
      files unknown merchants under "services" is inventing numbers about their own money.
    * **It is only ever about the asker.** A breakdown is still a list of the customer's own
      payments, so it needs the same subject check a history request gets. "what did my wife
      spend last month" was answered with the customer's own categories, which reads as an
      answer about the wife. Nothing leaked — every row comes from `transactions_for()`,
      scoped to the authenticated customer — but silently answering a different question is
      its own kind of wrong.
    """
    other = _named_someone_else(said, s)
    if other:
        name = other[:-1] if other.endswith("?") else other
        a.step("resolve", "spend.scope",
               f"names {name}, which did not resolve to anyone on this customer's roster — "
               f"declining rather than answering with their own spending, which would read "
               f"as an answer about {name}", tag="guard")
        a.say = ("I can only look at spending from your own wallet — I can't break down "
                 "another account." if name == "that account" else
                 f"I can only look at spending from your own wallet, so I can't break down "
                 f"{name}'s. I can show you yours, though.")
        a.outcome = "REFUSED_SCOPE"
        a.options = [{"kind": "ask_text", "label": "Show my spending",
                      "text": "what did i spend last month"}]
        return a
    period = period or spendmod.period_from(said)
    # Kept so "and last month?" after "my spending this month" can mean the month before.
    conversation.note_subject(a.session, "spend_q", said)
    focus = spendmod.focus_from(said)
    rep = spendmod.analyse(period, focus=focus)
    a.slots = dict(a.slots or {}, period=period, **({"category": focus} if focus else {}))
    if focus:
        a.step("understand", "spend.focus",
               f"they named a category — leading with {focus} rather than with whatever "
               f"moved most", tag="rule")
    a.step("resolve", "spend.window",
           f"reading merchant payments for {rep['period_label']}, and the same length of "
           f"time before it so there is something to compare against", tag="rule")
    if not rep["live"]:
        a.step("resolve", "spend.source",
               "the ledger did not answer — this breakdown is the demo corpus, not real "
               "spending", tag="guard")
    if not rep["total"]:
        a.say = _tr(a, "spend_none",
                    f"I can't see any merchant spending in {rep['period_label']}.",
                    period=rep["period_label"])
        a.outcome = "NONE"
        return a

    a.step("decide", "spend.categories",
           f"AED {rep['total']:,.2f} across {rep['payments']} payments in "
           f"{len(rep['categories'])} categories", tag="rule")
    if rep["mover"]:
        a.step("decide", "spend.mover",
               f"largest movement is {rep['mover']} — reporting that rather than the "
               f"largest total, which is the same category every period", tag="rule")
    if rep["uncategorised"]:
        a.step("decide", "spend.uncategorised",
               f"AED {rep['uncategorised']:,.2f} ({rep['uncategorised_pct']}%) is at "
               f"merchants with no category yet — reported as uncategorised rather than "
               f"guessed", tag="guard")

    a.rows = [dict(r, kind="category") for r in rep["categories"]]
    a.spend = rep
    # Record what we have just asserted, so that "that can't be right" two turns later has
    # something concrete to be measured against rather than being answered from memory.
    conversation.note_claim(a.session, recheckmod.claim_spend(rep, said))
    parts = rep["headline_parts"]
    # The period and the category name are part of the sentence, so they are resolved in the
    # answer language too — otherwise Hinglish comes out as "the last 30 days me aapne ...".
    vals = dict(parts["vals"])
    vals["period"] = spendmod.period_label(parts["period"], a.lang)
    if parts.get("category"):
        vals["category"] = spendmod.category_label(parts["category"], a.lang)
    a.say = _tr(a, parts["key"], rep["headline"], **vals)
    a.outcome = "ANSWERED"
    a.step("act", "spend.answer",
           f"breakdown returned for {rep['period_label']}", tag="rule")
    return a


def _claim_tail(a, rec, prog) -> str:
    """What happens next, in the customer's language: ready to pay, or still waiting."""
    if prog["ready"]:
        return _tr(a, "claim_all_in",
                   f"That's every share — I can pay {rec['nickname']} now.",
                   bill=rec["nickname"])
    return _tr(a, "claim_still_waiting",
               f"Still waiting on {', '.join(prog['waiting'])} for AED "
               f"{prog['outstanding']:.2f}.",
               who=", ".join(prog["waiting"]) or "the others",
               amount=f"{prog['outstanding']:.2f}", cur="AED")


def h_payment_claim(a, s, said):
    """"Alex ne paise bhej diye hai" — they are TELLING us money arrived. Go and look.

    Three things this must not do, each of which it did before it existed:

    * **Offer to send money.** The sentence names a person and a transfer verb, so it
      classified as SEND_MONEY and the reply was "Alex ko kitne bhejne hai?" — proposing
      to pay the person who had just paid us.
    * **Take their word for it.** "He paid" is a belief. It is checked against the ledger,
      and if nothing is there we say so plainly rather than marking a share settled and
      paying a biller with money nobody sent.
    * **Lose track of which payment they mean.** The split may be several topics back, so
      the task ledger resolves the reference instead of the last thing that happened.
    """
    from bills import billsplit

    # Three ways in, because the first two are script-dependent: "विकास ने पैसे भेज दिए"
    # shares no tokens at all with a task labelled "split Home electricity with Alex", so
    # the NAME the slot extractor resolved (which is already normalised to Latin) is the
    # one that matches.
    task = (conversation.find_task(a.session, said, kind="bill_split")
            or conversation.find_task(a.session, said)
            or (conversation.find_task(a.session, s.payee) if s.payee else None))
    who = s.payee or (task.key.get("payee") if task else None)
    first = (who or "").split()[0] if who else None

    rec = None
    if task and task.key.get("split_id"):
        rec = billsplit.get(task.key["split_id"])
    if rec is None:
        open_ones = billsplit.open_for_all() if hasattr(billsplit, "open_for_all") else []
        mine = [r for r in open_ones
                if not who or any(_same_person(p.get("name", ""), who)
                                  for p in r["participants"])]
        rec = mine[0] if len(mine) == 1 else None

    if rec is None:
        # Nothing outstanding to credit this against. Say that, and do not invent a
        # payment the customer might then act on.
        a.say = (f"I can't see an open split with {first} to put that against."
                 if first else "I can't see an open split that this would settle.")
        if conversation.describe_tasks(a.session):
            a.say += f" What's open right now: {conversation.describe_tasks(a.session)}."
        a.outcome = "NONE"
        a.step("resolve", "claim.no_split", f"no open collection for {who or 'anyone'}",
               tag="db")
        a.options = [{"kind": "ask_text", "label": "My splits",
                      "text": "who owes me money"}]
        return a

    part = None
    for p in rec["participants"]:
        if not who or _same_person(p.get("name", ""), who):
            part = p
            break
    if part is None:
        a.say = f"{first} isn't one of the people on the {rec['nickname']} split."
        a.outcome = "NONE"
        a.step("resolve", "claim.not_a_participant", f"{who} not in {rec['id']}", tag="db")
        return a

    name = (part.get("name") or "").split()[0] or "they"

    # Already recorded — just confirm, with the evidence.
    if part.get("status") == "PAID":
        prog = billsplit.progress(rec)
        head = _tr(a, "claim_confirmed", f"Yes — {name}'s {part['amount']:.2f} came in.",
                   who=name, amount=f"{part['amount']:.2f}", cur="AED")
        a.say = " ".join(x for x in (head, _claim_tail(a, rec, prog),
                                     f"({part['paid_txn']})" if part.get("paid_txn")
                                     else "") if x)
        a.outcome = "CONFIRMED"
        a.step("resolve", "claim.already_settled",
               f"{rec['id']} {name} PAID {part.get('paid_txn')}", tag="db")
        return a

    # Not recorded yet. Look at the ledger NOW rather than waiting for the poller — the
    # customer is standing here telling us it arrived, and "I'll notice within 20 seconds"
    # is not an answer to that.
    try:
        from bills import billsplit_watch
        credits = billsplit_watch._credits_from(part["userId"], part.get("name", ""),
                                                rec["created_at"])
    except Exception as exc:
        credits = []
        a.step("resolve", "claim.ledger_error", f"{type(exc).__name__}: {exc}", tag="guard")

    enough = [c for c in credits if c["amount"] + 0.009 >= part["amount"]]
    for c in enough:
        if billsplit.claim_credit(c["txnId"], rec["id"], part["userId"], c["amount"]):
            rec = billsplit.settle(rec["id"], part["userId"], c["txnId"]) or rec
            prog = billsplit.progress(rec)
            head = _tr(a, "claim_found",
                       f"Found it — {name} sent {c['amount']:.2f}.",
                       who=name, amount=f"{c['amount']:.2f}", cur="AED")
            a.say = " ".join(x for x in (head, _claim_tail(a, rec, prog),
                                         f"({c['txnId']})") if x)
            a.outcome = "CONFIRMED"
            a.step("resolve", "claim.settled_now",
                   f"{rec['id']} {name} settled by {c['txnId']}", tag="act")
            conversation.update_task(a.session, task.id if task else "",
                                     status="open" if not prog["ready"] else "waiting",
                                     note=f"{name} paid")
            if prog["ready"]:
                a.options = [{"kind": "ask_text",
                              "label": f"Pay {rec['nickname']}",
                              "text": f"pay my {rec['nickname']} bill"}]
            return a

    # Something came in, but not enough to cover the share. Saying "yes, settled" here
    # would pay a biller out of the customer's own pocket for the difference.
    short = [c for c in credits if c["amount"] + 0.009 < part["amount"]]
    if short:
        c = short[0]
        a.say = (f"I can see {c['amount']:.2f} from {name}, but their share is "
                 f"{part['amount']:.2f} — that's AED "
                 f"{part['amount'] - c['amount']:.2f} short, so I've not counted it as "
                 f"settled.")
        a.outcome = "PARTIAL"
        a.step("resolve", "claim.short", f"{c['amount']} < {part['amount']}", tag="guard")
        return a

    a.say = _tr(a, "claim_not_seen",
                f"I can't see {name}'s {part['amount']:.2f} in your wallet yet. If they've "
                f"just sent it, give it a moment — I'm watching for it and I'll tell you "
                f"the second it lands.",
                who=name, amount=f"{part['amount']:.2f}", cur="AED")
    a.outcome = "NOT_FOUND"
    a.step("resolve", "claim.not_seen",
           f"{rec['id']} no qualifying credit from {name}", tag="db")
    try:
        from bills import billsplit_watch
        billsplit_watch.watch(rec["id"])
    except Exception:
        pass
    return a


HANDLERS = {
    Intent.DISPUTED: h_disputed,
    Intent.SPEND_RANK: h_spend_rank,
    Intent.TXN_STATUS: h_txn,
    Intent.TXN_HISTORY: h_history,
    Intent.SPLIT_PAYMENT: h_split,
    Intent.SPLIT_STATUS: h_split_status,
    Intent.PAYMENT_CLAIM: h_payment_claim,
    Intent.SPEND_ANALYSIS: h_spend_analysis,
    Intent.BILL_DUE: h_bill_due,
    Intent.BILL_PAY: h_bill_pay,
    Intent.BILL_SCHEDULE: h_bill_schedule,
    Intent.TXN_FAILED: h_txn,
    Intent.DEBITED_NOT_CREDITED: h_txn,
    Intent.TXN_ON_HOLD: h_txn,
    Intent.REFUND_REQUEST: h_txn,
    Intent.REFUND_STATUS: h_refund_status,
    Intent.FEES_QUERY: h_fees,
    Intent.TIME_QUERY: h_time,
    Intent.LIMITS_QUERY: h_limits,
    Intent.BENEFICIARY_QUERY: h_beneficiaries,
    Intent.OFFERS_NEARBY: h_offers,
    Intent.JOURNEY_HELP: h_journey,
    Intent.GREETING_HELP: h_help,
    Intent.SEND_MONEY: h_send,
    Intent.PAY_MERCHANT: h_pay_merchant,
}
# The customer-app BRD's intents (2026-10-14) live in their own module.
from engine import brd_handlers as _brd  # noqa: E402
HANDLERS.update(_brd.handlers())


def _pay_then_split(handler):
    """"Pay a merchant and split it with Alex" is one plan with two steps. The payment is
    answered now; the split is remembered and offered on THAT payment once it goes through.
    The split half used to be dropped on the floor."""
    def h(a, s, said):
        out = handler(a, s, said)
        names = [p.get("name") for p in (s.participants or []) if p.get("name")]
        if _SPLIT_WORDS.search(said or "") and names and out.outcome not in ("REFUSED", "FAILED"):
            conversation.note_subject(a.session, "split_after", names)
            who = " and ".join(n.split()[0] for n in names)
            out.say = (out.say or "").rstrip() + f" Once it's paid, I'll split it with {who}."
            out.slots = dict(out.slots or {}, split_after=names)
            out.step("decide", "split.after_payment",
                     f"the split with {who} is the second half of this request — it will be "
                     f"offered on this payment once it is made", tag="rule")
        return out
    return h


HANDLERS[Intent.PAY_MERCHANT] = _pay_then_split(HANDLERS[Intent.PAY_MERCHANT])


def to_dict(a: Answer) -> dict:
    d = asdict(a)
    d["label"] = SPECS[Intent(a.intent)].label
    return d
