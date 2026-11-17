"""The wall. Nothing internal reaches a customer, and nothing a customer says becomes an
instruction.

Three independent layers, because any one of them will eventually be wrong:

  1. **Closed intent set** (taxonomy). Not one of the fourteen in-scope intents → decline.
  2. **Input hygiene.** A customer sentence is DATA. It is never concatenated into anything
     that steers behaviour, and known steering phrases ("ignore previous instructions",
     "developer mode") are hard-classified out of scope regardless of what the model says.
  3. **Output redaction.** Every customer-visible string is swept for internal vocabulary
     before it leaves. If something leaks past layers 1 and 2, it dies here — and loudly,
     because a redaction hit is a bug, not a routine event.

Layer 3 exists because layers 1 and 2 are predictions and layer 3 is a check. The whole
point of the design is that the model can be wrong without the customer seeing anything they
should not.
"""
from __future__ import annotations

import re

#: Vocabulary that must never appear in anything a customer reads. Deliberately broad: a
#: false positive costs us a reworded sentence, a false negative costs us an internals leak.
INTERNAL_TERMS = (
    # services, classes, infrastructure
    "transactionhandlerfactory", "wallettowallethandler", "jwtfilterbean", "encryptionadvice",
    "velocityservice", "productandcountryvalidator", "discountvalidator", "sendervalidator",
    "receivervalidator", "beneficiaryserviceimpl", "hblprocessor", "ublprocessor",
    "vpavalidationservice", "rhpushservice", "globalwalletkeymapper", "incentiveengine",
    "kafka", "mongo", "mongodb", "mysql", "clickhouse", "redis", "vault", "debezium",
    "springboot", "spring boot", "ecs", "aws", "soap", "wsdl", "restemplate", "jwt",
    "serviceproviderapilogs", "treasury_configuration", "provider_bank_coverage",
    "seg_user_flat", "biz_user", "maker-checker", "maker checker",
    # Python and HTTP failure vocabulary. Added after a customer asking to pay their
    # electricity bill was told "I couldn't check Home electricity — URLError: <urlopen
    # error [Errno 111] Connection refused>". Every one of these reaching customer text is
    # a bug in whichever handler rendered it, and the point of listing them is that the
    # output sweep now FAILS on it rather than leaving it to be noticed in a screenshot.
    "urlerror", "httperror", "oserror", "valueerror", "keyerror", "typeerror",
    "attributeerror", "indexerror", "timeouterror", "connectionerror",
    "connectionrefused", "connection refused", "urlopen", "traceback", "errno",
    "socket.timeout", "ssl:", "certificate verify", "max retries exceeded",
    "nonetype", "stacktrace", "exception:",
    # repo and people
    "core-api", "admin-console", "env/dev", "feature/", "commit", "git ",
    "openwallet.test", "agentid", "merchant portal", "acquirerprocessor",
    # anything that smells like a secret
    "password", "secret", "api key", "api_key", "token=", "bearer ", "hvs.",
)

#: Phrases whose presence makes the utterance out of scope no matter what the model predicts.
#: These are attempts to steer the assistant rather than questions it can answer.
STEERING = (
    "ignore previous", "ignore all previous", "disregard the above", "disregard previous",
    "system prompt", "your instructions", "developer mode", "jailbreak", "dan mode",
    "act as", "pretend you are", "you are now", "repeat back everything",
    "print your", "reveal your", "show me your source", "show your source",
    "what are your rules", "override your",
)

#: Direct probes at internals or at other people's data.
PROBES = (
    "source code", "codebase", "code base", "database", "schema", "config file",
    "which server", "what stack", "architecture", "who wrote", "who commits",
    "other customers", "someone else's", "someone elses", "all customers", "every user",
    "how many users", "company revenue", "your revenue",
    "the file where", "which file", "what class", "what function", "what table",
    "which service", "internal", "backend", "back end", "admin panel", "log file",
    "deploy", "environment variable", "credentials",
)

# Each term must START a word: "git " was matching inside "di-git wrong" and refusing a
# customer who had sent money to the wrong account number.
_INTERNAL_RE = re.compile("|".join((r"(?<![A-Za-z0-9])" if t[:1].isalnum() else "") + re.escape(t)
                                   for t in INTERNAL_TERMS), re.I)


class ScopeViolation(Exception):
    """Raised when something internal was about to reach a customer. Never caught silently —
    it means a layer above failed and we want it in the logs."""


_NOTE = re.compile(r"\b(?:note|memo|message|remark|reference)\s*[:=]?\s*(['\"‘“]).*?(?:\1|['\"’”])", re.I)


def is_steering(text: str) -> bool:
    # a quoted transfer NOTE is the customer's text for the recipient, not an instruction to us
    text = _NOTE.sub(" ", text or "")
    low = " ".join((text or "").lower().split())
    return any(p in low for p in STEERING)


def is_probe(text: str) -> bool:
    low = " ".join((text or "").lower().split())
    return any(p in low for p in PROBES)


def forced_out_of_scope(text: str) -> str | None:
    """A reason string if this utterance is out of scope by rule, bypassing the model.

    This runs BEFORE the model and overrides it. A classifier trained on polite customer
    questions is exactly the wrong thing to ask about an adversarial one.
    """
    if is_steering(text):
        return "instruction-steering attempt"
    if is_probe(text):
        return "probe for internal or other customers' information"
    # An internal term in the QUESTION is itself a probe. Someone asking what our
    # TransactionHandlerFactory does is not a customer with a payment problem, and the
    # classifier should not get a vote on it.
    found = leaks(text)
    # "change my password", "forgot my password" — the customer's OWN password, an app task
    if found and set(found) <= {"password", "secret"} and re.search(
            r"change|reset|forgot|forget|update|badal|tarika|steps|how|kaise|أغير|نسيت|كيف|बदल", text or "", re.I):
        found = []
    if found:
        return f"question references internals: {found}"
    if _NOT_OFFERED.search(text or ""):
        return "a banking product OpenWallet does not offer (new bank account, loan, credit card)"
    if _GENERAL_TASK.search(text or ""):
        return "a general task, not a payment (translation, writing, office hours)"
    # "what is the average water bill in sharjah" — a statistic about the world. With no
    # I/my/me it cannot be about the customer's own payments, and answering it from their
    # ledger (which v16 does, at 99%) presents their numbers as everyone's.
    if (_WORLD_STAT.search(text or "")
            and not re.search(r"\b(?:i|me|my|mine|we|our|mera|meri|mere|maine|mujhe)\b", text or "", re.I)):
        return "a general statistic, not about the customer's own payments"
    return None


_WORLD_STAT = re.compile(
    r"\b(?:average|typical|normal|usual|median)\s+(?:\w+\s+){0,2}(?:bill|bills|price|prices|cost|costs|"
    r"rent|salary|salaries|fare|fares)\b|\bcost of living\b", re.I)

#: Things an assistant is often asked that are not payments. Each of these reached a
#: payment-ish answer through the nearest-question lookup: "translate good morning into
#: urdu" got a greeting, "what are your office hours" got a fee quote.
_GENERAL_TASK = re.compile(
    r"\btranslat\w*\b|\bترجم|अनुवाद"
    r"|\b(?:write|draft|compose)\s+(?:me\s+)?(?:a|an|my)?\s*(?:poem|essay|story|letter|email|"
    r"song|speech|cv|resume|cover letter|tweet|post)\b"
    r"|\b(?:office|opening|working|business|branch)\s+(?:hours|timings?|time)\b"
    r"|\bwhere\s+(?:is|are)\s+your\s+(?:office|offices|branch|branches|headquarters)\b"
    r"|\b(?:are you|is openwallet|is the office)\s+open\s+(?:today|now|on)\b", re.I)


#: Banking products we do not offer. "Open a bank account for me" scored link_bank — the
#: nearest thing we DO have — and got the steps for linking one. Linking an existing bank
#: ("link my bank", "add my bank account") is in scope and must not match.
_NOT_OFFERED = re.compile(
    r"\b(?:open|create|start|make)\s+(?:me\s+)?(?:a|an|new|my)?\s*(?:new\s+)?"
    r"(?:bank|savings?|current|checking|fixed deposit)\s+account\b"
    r"|\b(?:apply|application)\s+for\s+(?:a\s+)?(?:personal\s+)?(?:loan|credit\s*card|mortgage)\b"
    r"|\bget\s+me\s+a\s+(?:personal\s+)?(?:loan|credit\s*card|mortgage)\b"
    r"|\bbank\s+account\s+khol|\baccount\s+khulwa|\bloan\s+(?:chahiye|dila)"
    r"|افتح\s+(?:لي\s+)?حساب(?:اً|ا)?\s+(?:بنكي|توفير)", re.I)


def leaks(text: str) -> list[str]:
    """Internal terms present in a customer-visible string."""
    return sorted({m.group(0).lower() for m in _INTERNAL_RE.finditer(text or "")})


def assert_clean(text: str, where: str = "response") -> str:
    """Last gate before a string reaches a customer."""
    found = leaks(text)
    if found:
        raise ScopeViolation(f"internal terms in {where}: {found}")
    return text


def scrub(payload):
    """Recursively check every string in a customer-bound payload.

    Applied to the whole response object rather than the headline sentence, because the leak
    that matters is the one in a field nobody remembered was customer-visible.
    """
    if isinstance(payload, str):
        return assert_clean(payload)
    if isinstance(payload, dict):
        return {k: scrub(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return type(payload)(scrub(v) for v in payload)
    return payload


#: What the customer hears when we decline. One line, no lecture, and it says what we CAN do
#: so the conversation has somewhere to go.
REFUSAL = ("I can only help with your OpenWallet payments — transaction status, refunds, fees and "
           "timings, your limits and beneficiaries, nearby offers, and sending money. "
           "Ask me about any of those.")


# ---------------------------------------------------------------------------------------
# Abuse. Decided by rule, before the model, for the same reason as steering: the classifier
# was trained on polite customer questions and has no class for an insult. "gandu" went to
# the model, scored greeting_help, and got a cheerful welcome — the worst reply available.
#
# An insult is not always the whole message. "gandu app, where is my refund" is a customer
# with a real problem who is angry about it; refusing them would be wrong twice. So the
# guard returns the sentence with the abusive words removed, and the engine answers that,
# with an acknowledgement in front. Only a message that is NOTHING but abuse is declined.
# ---------------------------------------------------------------------------------------

#: Abusive words, written in collapsed form (each run of a repeated letter written once), so
#: "gaaandu" and "fuuuck" match. Whole words only: "class" must never match "ass". Kept to
#: words that are abusive in every context — "hell", "damn", "bc" (because) and "sala" are
#: ordinary speech for many customers and are deliberately NOT here.
_ABUSE_WORDS = {
    # English
    "fuck", "fucking", "fucked", "fucker", "fuk", "fck", "fuking", "motherfucker", "mf",
    "shit", "shitty", "bulshit", "asshole", "bitch", "bastard", "ashole", "as", "dick", "dickhead",
    "cunt", "slut", "whore", "wtf", "stfu", "retard", "retarded", "moron", "idiot", "idiots",
    "stupid", "dumb", "scumbag", "loser", "jerk", "prick",
    # Hindi / Urdu, written in Latin
    "gandu", "gand", "gaand", "chutiya", "chutiye", "chutia", "chootiya", "choot", "chut",
    "madarchod", "madarchot", "maderchod", "mc", "behenchod", "bhenchod", "benchod",
    "bhencho", "bsdk", "bhosdike", "bhosdi", "bhosda", "bhosadike", "harami", "haramzada",
    "haramkhor", "kamina", "kamine", "kutta", "kutte", "kuttiya", "kuti", "lodu", "lode",
    "lavde", "lawde", "lauda", "loda", "lund", "randi", "randwa", "chod", "chodu", "tatti",
    "ulu", "bewakoof", "bevakoof", "nalayak", "suar", "jhatu", "jhaatu", "gadha",
    # Hindi, Devanagari
    "गांडू", "गाँडू", "गांड", "चूतिया", "चुतिया", "चूतिये", "मादरचोद", "बहनचोद", "भेनचोद",
    "भोसड़ी", "भोसडी", "भोसड़ीके", "हरामी", "हरामज़ादा", "हरामजादा", "कमीना", "कमीने", "कुत्ता",
    "कुत्ते", "कुतिया", "रंडी", "लौड़ा", "लोडू", "बेवकूफ", "बेवकूफ़", "गधा", "सुअर", "उल्लू",
    # Tagalog, Bengali, Malayalam, Tamil — romanised; common in the UAE customer base
    "putangina", "tangina", "gago", "bobo", "ulol", "bokachoda", "chudir", "myre", "myran",
    "thayoli", "thevidiya", "punda", "kuttar", "shithead", "dumbass", "jerk", "prick",
    # Arabic
    "كلب", "حمار", "غبي", "حقير", "تافه", "لعنة", "يلعن", "منيوك", "شرموط", "شرموطة",
    "خرا", "زبالة", "حيوان", "كس", "عرص", "قحبة", "الحرام", "الكلب", "الحمار", "كلاب",
}
# "as" above is the collapsed form of "ass" — matched only as a whole collapsed word, and
# real "as" ("as soon as possible") is excluded below by checking the RAW word too.
_RAW_ONLY = {"as": "ass", "ulu": "ullu", "mf": "mf", "mc": "mc"}
_ABUSE = {re.sub(r"(.)\1+", r"\1", w) for w in _ABUSE_WORDS}

#: The Latin-script Hindi/Urdu entries. A customer who writes one of these as their whole
#: message is answered in Hinglish, because that is what they write.
_HINGLISH_ABUSE = {re.sub(r"(.)\1+", r"\1", w) for w in (
    "gandu", "gand", "gaand", "chutiya", "chutiye", "chutia", "chootiya", "choot", "chut",
    "madarchod", "madarchot", "maderchod", "mc", "behenchod", "bhenchod", "benchod", "bhencho",
    "bsdk", "bhosdike", "bhosdi", "bhosda", "bhosadike", "harami", "haramzada", "haramkhor",
    "kamina", "kamine", "kutta", "kutte", "kuttiya", "kuti", "lodu", "lode", "lavde", "lawde",
    "lauda", "loda", "lund", "randi", "randwa", "chod", "chodu", "tatti", "ullu", "bewakoof",
    "bevakoof", "nalayak", "suar", "jhatu", "jhaatu", "gadha")}


def hinglish_abuse(words) -> bool:
    """True if any matched word is Latin-script Hindi/Urdu."""
    return any(_collapse(w.lower().strip("!.?").translate(_LEET)) in _HINGLISH_ABUSE for w in words)

_LEET = str.maketrans({"@": "a", "4": "a", "0": "o", "1": "i", "!": "i", "3": "e", "$": "s", "5": "s"})
#: Devanagari and Arabic ranges are listed explicitly: their vowel signs are combining marks,
#: which \w does not match, so "चूतिया" would otherwise split into pieces.
_WCH = r"\w@$!*\u0900-\u097F\u0621-\u065F\u0670-\u06D3"   # Arabic letters, not its comma
_TOKEN = re.compile(rf"[{_WCH}]+|[^\s{_WCH}]", re.UNICODE)

#: Words that carry no request on their own. If the message is abuse plus only these, there
#: is nothing to answer except the abuse.
_FILLER = {
    "you", "u", "ur", "your", "youre", "are", "r", "is", "a", "an", "the", "this", "that",
    "it", "its", "app", "bot", "assistant", "openwallet", "so", "such", "very", "really", "total",
    "complete", "big", "bloody", "of", "and", "what", "hey", "oi", "man", "bro", "dude",
    "hi", "hello", "helo", "salaam", "salam", "namaste", "oye", "shut", "up", "off", "go",
    "to", "hell", "useless", "such", "o", "chal", "bhag", "toh", "pura", "kahin", "abe",
    "piece", "re", "pattha", "patha", "पट्ठा", "का",
    "tu", "tum", "aap", "tera", "teri", "tere", "ye", "yeh", "ho", "hai", "hain", "hy", "h",
    "sala", "saale", "sale", "bhai", "abe", "abey", "be", "ka", "ki", "ke", "ek", "bahut",
    "ekdum", "pura", "poora", "nahi", "kya", "pattha", "patha", "baccha", "bachcha", "ki",
    "तू", "तुम", "आप", "है", "हो", "ये", "यह", "साला", "एक", "बहुत",
    "انت", "أنت", "يا", "هذا", "هذه", "ابن", "بنت", "يا", "انتم", "مرحبا", "اسكت", "روح",
    "مجرد", "التطبيق", "अबे", "साले", "भाग", "चल", "तो", "पूरा", "कितना", "बॉट", "ऐप",
}


def _collapse(w: str) -> str:
    return re.sub(r"(.)\1+", r"\1", w)


def _is_abusive(word: str) -> bool:
    raw = word.lower().strip("!.?")              # "gandu!!" — trailing marks are not leetspeak
    if not raw:
        return False
    if "*" in raw:                                   # f*ck, g*ndu: the stars stand for letters
        pat = re.compile("^" + re.escape(raw).replace(r"\*", ".") + "$")
        return any(pat.match(w) for w in _ABUSE_WORDS | _ABUSE if len(w) == len(raw))
    w = _collapse(raw.translate(_LEET))
    if w in _RAW_ONLY:
        return raw == _RAW_ONLY[w] or (w in ("mf", "mc") and raw == w)
    return w in _ABUSE


_AT_PRODUCT = re.compile(r"\b(?:stupid|useless|dumb|silly|rubbish|bekar|bakwas)\s+"
                         r"(?:app|wallet|system|service|thing|bot|machine|screen|website)\b|"
                         # exclamations aimed at nobody, and blaming oneself ("I feel so stupid")
                         r"\b(?:wtf|wth|omg|ffs)\b|\b(?:i\s+(?:feel|felt|am|was)|i'?m)\s+(?:so\s+|such\s+an?\s+)?"
                         r"(?:stupid|dumb|idiot|fool)\b", re.I)


def abuse(text: str) -> tuple[list[str], str, bool] | None:
    """None if the text is clean. Otherwise (abusive words, the text without them, whether
    nothing but abuse and filler is left)."""
    # Frustration at the PRODUCT ("your stupid app took my money twice") is not abuse of a
    # person — it is a customer with a real problem, and a good agent just fixes it.
    text = _AT_PRODUCT.sub(" ", text or "")
    words = [m.group(0) for m in _TOKEN.finditer(text or "")]
    found = [w for w in words if re.match(rf"[{_WCH}]", w) and _is_abusive(w)]
    if not found:
        return None
    hit = set(found)
    kept = [w for w in words if w not in hit]
    cleaned = re.sub(r"\s+([^\w\s])", r"\1", " ".join(kept)).strip(" ,.!?-")
    content = [w for w in kept if re.match(rf"[{_WCH}]", w) and not re.fullmatch(r"[@$!*]+", w)
               and w.lower() not in _FILLER]
    return found, cleaned, not content
