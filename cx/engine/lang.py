"""Which language the customer is writing in, and the words to answer them with.

A customer who types "paise bhejne hai" and is answered in English has been understood but
not met. The model already reads four languages; this is the other half of that.

## How the language is decided

Script first, because it is unambiguous: Devanagari means Hindi, Arabic script means Arabic.
Latin script is the hard case — it carries both English and Hinglish — so it is decided by
MARKER WORDS rather than by a classifier. "bhejne", "chahiye", "kitna", "wala" do not occur
in English, and a handful of them is a far more reliable signal than anything statistical on
a four-word sentence.

The default is English. Guessing Hinglish from a sentence that happens to contain "do" would
answer an English speaker in a language they may not read.

## Coverage, stated plainly

The composed sentences — amounts, references, statuses, the refund policy — are NOT
translated. They are built from data and must stay byte-identical in every language, the same
reason `voice.py` refuses to let a model reword them. What IS translated is the wrapper: the
questions, the acknowledgements, the refusal, the headline of each answer.

So a Hindi speaker gets Hindi questions and Hindi framing with the figures in digits. That is
the honest 80%, and it is marked per-string so the gaps are visible rather than guessed at.
Anything without a translation falls back to English rather than to a machine rendering
nobody has read.
"""
from __future__ import annotations

import re

EN, HI, HINGLISH, AR = "en", "hi", "hinglish", "ar"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ARABIC = re.compile(r"[؀-ۿ]")

#: Romanised Hindi/Urdu words that do not occur in English. One is enough — these are not
#: words an English sentence produces by accident, unlike "do", "me", "to" or "hai".
_HINGLISH_MARKERS = {
    "paise", "paisa", "paisey", "rupaye", "bhejo", "bhejna", "bhejne", "bheja", "bhej",
    "bhejni", "bhejiye", "chahiye", "chahiya", "kitna", "kitne", "kitni", "mera", "meri",
    "mere", "mujhe", "mujh", "tumhe", "aapko", "kaise", "kaisa", "kyun", "kyu", "kahan",
    "kaun", "kaunsa", "kaunse", "wala", "wali", "wale", "karo", "kardo", "kijiye",
    "dikhao", "dikha", "dikhaiye", "batao", "bata", "baaki", "bakaya", "hisaab", "hissa",
    "baanto", "baant", "banto", "wapas", "wapis", "nahi", "nahin", "haan", "theek",
    "thik", "acha", "accha", "jaldi", "abhi", "pichle", "pichla", "agle", "saare",
    "sare", "sabhi", "bhar", "bharo", "bharna", "bharni", "lenden", "purane", "pura",
    "bhai", "yaar", "kya", "bhi", "hua", "hui", "gaya", "gayi", "diya", "liya",
    "karna", "karne", "karni", "karo", "kardo", "lena", "dena", "dene", "deni",
    "saath", "sath", "kiske", "kisko", "kisse", "usse", "iske", "unka", "apna", "apne",
    "mahine", "hafte", "tareekh", "bijli", "paani", "dukaan", "aadha", "barabar",
}

_WORD = re.compile(r"[^\wऀ-ॿ؀-ۿ]+")


def detect(text: str) -> str:
    """The language to answer in. Defaults to English."""
    t = text or ""
    if _DEVANAGARI.search(t):
        return HI
    if _ARABIC.search(t):
        return AR
    words = {w for w in _WORD.split(t.lower()) if w}
    if words & _HINGLISH_MARKERS:
        return HINGLISH
    return EN


#: Translations, keyed by a short id. A key with no entry for a language falls back to
#: English on purpose: a missing translation should read as English, never as a guess.
#:
#: `{}` placeholders are filled with COMPOSED values — amounts, names, references — which are
#: never translated. The figures a customer acts on must be identical in every language.
T: dict[str, dict[str, str]] = {
    # ── asking who / how much ───────────────────────────────────────────────────────────
    "which_payee": {
        HI: "किसे भेजना है — आपकी लिस्ट में से कोई, या पास की कोई दुकान?",
        HINGLISH: "Kisko bhejna hai — aapki list me se koi, ya paas ki koi dukaan?",
        AR: "إلى من تريد الإرسال — شخص من قائمتك أم متجر قريب؟",
    },
    "email_statement": {
        HI: "मैं आपके आखिरी {n} ट्रांजैक्शन {to} पर ईमेल कर दूँगा। कन्फर्म करें, फिर मैं आपका PIN माँगूँगा।",
        HINGLISH: "Aapke last {n} transactions {to} pe email kar dunga. Confirm karo, phir PIN maangunga.",
        AR: "سأرسل آخر {n} معاملات إلى {to}. أكّد، ثم سأطلب رمز PIN.",
    },
    "didnt_catch_short": {
        HI: "माफ़ कीजिए, मैं समझ नहीं पाया।",
        HINGLISH: "Sorry, samajh nahi aaya.",
        AR: "عذراً، لم أفهم ذلك.",
    },
    "which_person": {
        HI: "आपकी लिस्ट में से किसे?",
        HINGLISH: "Aapki list me se kisko?",
        AR: "لمن من قائمتك؟",
    },
    "how_much_to": {
        HI: "{who} को कितना भेजना है?",
        HINGLISH: "{who} ko kitna bhejna hai?",
        AR: "كم تريد أن ترسل إلى {who}؟",
    },
    "ready_send": {
        HI: "{amount} {cur} {who} को भेजने के लिए तैयार है। कन्फर्म कीजिए और यह आगे बढ़ जाएगा।",
        HINGLISH: "{amount} {cur} {who} ko bhejne ke liye ready hai. Confirm kijiye.",
        AR: "جاهز لإرسال {amount} {cur} إلى {who}. أكّد وسيتم التنفيذ.",
    },
    # ── somebody has paid us ────────────────────────────────────────────────────────────
    #: These carry a transaction reference and an amount. Both are COMPOSED values and are
    #: never translated — a figure the customer checks against their wallet must read the
    #: same in every language.
    "claim_confirmed": {
        HI: "हाँ — {who} के {amount} {cur} आ चुके हैं।",
        HINGLISH: "Haan — {who} ke {amount} {cur} aa chuke hai.",
        AR: "نعم — وصل مبلغ {amount} {cur} من {who}.",
    },
    "claim_found": {
        HI: "मिल गया — {who} ने {amount} {cur} भेजे हैं।",
        HINGLISH: "Mil gaya — {who} ne {amount} {cur} bhej diye hai.",
        AR: "وجدته — أرسل {who} مبلغ {amount} {cur}.",
    },
    "claim_not_seen": {
        HI: "{who} के {amount} {cur} अभी आपके वॉलेट में नहीं दिख रहे। अभी भेजे हैं तो थोड़ा रुकिए — मैं नज़र रखे हूँ और आते ही बता दूँगा।",
        HINGLISH: "{who} ke {amount} {cur} abhi aapke wallet me nahi dikh rahe. Abhi bheje hai to thoda ruko — main dekh raha hoon, aate hi bata doonga.",
        AR: "لا أرى مبلغ {amount} {cur} من {who} في محفظتك بعد. إن أرسله للتو فأمهله قليلاً — أنا أراقب وسأخبرك فور وصوله.",
    },
    #: The second half of a claim answer, kept separate so the translated version keeps the
    #: reference and the next step instead of ending at "yes, it arrived".
    "claim_all_in": {
        HI: "सबका हिस्सा आ गया — अब मैं {bill} भर सकता हूँ।",
        HINGLISH: "Sabka hissa aa gaya — ab main {bill} bhar sakta hoon.",
        AR: "اكتملت كل الحصص — يمكنني دفع {bill} الآن.",
    },
    "claim_still_waiting": {
        HI: "{who} की ओर से {amount} {cur} अभी बाकी हैं।",
        HINGLISH: "{who} ke {amount} {cur} abhi baaki hai.",
        AR: "ما زلنا ننتظر {amount} {cur} من {who}.",
    },
    #: A follow-up with nothing to follow. Said cold, in whatever language they opened in.
    "no_thread": {
        HI: "अभी तुलना करने के लिए कुछ नहीं है — इस बातचीत में हमने अभी कुछ देखा ही नहीं। "
            "अपने खर्च या पेमेंट्स के बारे में पूछिए, वहाँ से आगे बढ़ेंगे।",
        HINGLISH: "Abhi compare karne ke liye kuch nahi hai — is baat me humne abhi kuch "
                  "dekha hi nahi. Apne kharche ya payments ke baare me puchiye, wahan se "
                  "aage badhenge.",
        AR: "لا يوجد شيء لأقارنه به بعد — لم نطّلع على أي شيء في هذه المحادثة. اسألني عن "
            "إنفاقك أو مدفوعاتك وسأبدأ من هناك.",
    },
    # ── being told we are wrong ─────────────────────────────────────────────────────────
    #: The admission comes FIRST in every language. "You're right, and I was wrong" before
    #: the number, because a correction that leads with the new figure reads as a restatement
    #: and the customer has to work out for themselves that we changed our mind.
    "dispute_corrected": {
        HI: "आप सही हैं, मुझसे ग़लती हुई। {period} के लिए यह {cur} {now} है, {cur} {was} नहीं।",
        HINGLISH: "Aap sahi hain, mujhse galti hui. {period} ke liye ye {cur} {now} hai, "
                  "{cur} {was} nahi.",
        AR: "أنت محق، وقد أخطأت. المبلغ {cur} {now} عن {period}، وليس {cur} {was}.",
    },
    "dispute_stands": {
        HI: "मैंने आपके लेज़र से दोबारा जाँचा — यह {cur} {now} ही है।",
        HINGLISH: "Maine aapke ledger se dobara check kiya — ye {cur} {now} hi hai.",
        AR: "راجعته مرة أخرى من سجلك — المبلغ {cur} {now} فعلاً.",
    },
    "dispute_same_payments": {
        HI: "{first} और {second}, दोनों में वही {n} पेमेंट आते हैं — इसीलिए कुल रकम नहीं बदलती।",
        HINGLISH: "{first} aur {second}, dono me wahi {n} payments aate hain — isiliye total "
                  "nahi badalta.",
        AR: "كل من {first} و{second} يشمل نفس الـ{n} عملية، ولهذا لا يتغير المجموع.",
    },
    "dispute_quiet": {
        HI: "उससे पहले आपका आख़िरी मर्चेंट पेमेंट {oldest} दिन पहले था, और उससे पिछला "
            "{gap} दिन पहले — बीच में कुछ नहीं।",
        HINGLISH: "Usse pehle aapka last merchant payment {oldest} din pehle tha, aur usse "
                  "pichla {gap} din pehle — beech me kuch nahi.",
        AR: "آخر عملية لديك لدى تاجر كانت قبل {oldest} يوماً، والتي قبلها قبل {gap} يوماً — "
            "ولا شيء بينهما.",
    },
    "dispute_window": {
        HI: "यह {n} पेमेंट हैं, सबसे पुराना {oldest} दिन पहले।",
        HINGLISH: "Ye {n} payments hain, sabse purana {oldest} din pehle.",
        AR: "هذه {n} عملية، أقدمها قبل {oldest} يوماً.",
    },
    "dispute_count": {
        HI: "इस बार मैंने {after} पेमेंट गिने, {before} नहीं।",
        HINGLISH: "Is baar maine {after} payments gine, {before} nahi.",
        AR: "أحصيت {after} عملية هذه المرة، وليس {before}.",
    },
    "dispute_not_live": {
        HI: "एक बात बतानी चाहिए — मैं आपका लाइव लेज़र नहीं पढ़ पाया, तो यह सैंपल डेटा है। "
            "शायद पूरी गड़बड़ इसी की है।",
        HINGLISH: "Ek baat batani chahiye — main aapka live ledger nahi padh paya, to ye "
                  "sample data hai. Shayad poori gadbad isi ki hai.",
        AR: "يجب أن أوضح أنني لم أتمكن من قراءة سجلك المباشر، لذا هذه بيانات تجريبية — "
            "وقد تكون هذه هي المشكلة كلها.",
    },
    "dispute_nothing": {
        HI: "इस बातचीत में मेरा दिया हुआ कोई आँकड़ा नहीं है जिसे जाँच सकूँ। फिर से पूछिए, "
            "मैं आपके लेज़र से निकालकर बताऊँगा।",
        HINGLISH: "Is baat me mera diya hua koi figure nahi hai jise check kar sakun. Phir "
                  "se puchiye, main aapke ledger se nikaal kar bataunga.",
        AR: "ليس لدي رقم من هذه المحادثة لأراجعه. اسألني مرة أخرى وسأحسبه من سجلك.",
    },
    # ── where the money went ────────────────────────────────────────────────────────────
    #: The spend breakdown headline. Four shapes rather than one with a {direction} slot:
    #: "more" and "less" are exactly the words that must be translated.
    "spend_none": {
        HI: "{period} में कोई मर्चेंट खर्च नहीं दिख रहा।",
        HINGLISH: "{period} me koi merchant kharcha nahi dikh raha.",
        AR: "لا أرى أي إنفاق لدى التجار في {period}.",
    },
    "spend_flat": {
        HI: "{period} में आपने दुकानों पर {cur} {total} खर्च किए।",
        HINGLISH: "{period} me aapne shops par {cur} {total} kharch kiye.",
        AR: "أنفقت {cur} {total} لدى التجار في {period}.",
    },
    "spend_more": {
        HI: "{period} में आपने {cur} {total} खर्च किए — {category} पर पिछली बार से {cur} {delta} ज़्यादा।",
        HINGLISH: "{period} me aapne {cur} {total} kharch kiye — {category} par pichli baar se {cur} {delta} zyada.",
        AR: "أنفقت {cur} {total} في {period} — بزيادة {cur} {delta} على {category} مقارنة بالفترة السابقة.",
    },
    "spend_less": {
        HI: "{period} में आपने {cur} {total} खर्च किए — {category} पर पिछली बार से {cur} {delta} कम।",
        HINGLISH: "{period} me aapne {cur} {total} kharch kiye — {category} par pichli baar se {cur} {delta} kam.",
        AR: "أنفقت {cur} {total} في {period} — بنقصان {cur} {delta} على {category} مقارنة بالفترة السابقة.",
    },
    "spend_more_blame": {
        HI: "{period} में आपने {cur} {total} खर्च किए — {category} पर पिछली बार से {cur} {delta} ज़्यादा। इसमें ज़्यादातर {merchant} है।",
        HINGLISH: "{period} me aapne {cur} {total} kharch kiye — {category} par pichli baar se {cur} {delta} zyada. Iska zyada hissa {merchant} hai.",
        AR: "أنفقت {cur} {total} في {period} — بزيادة {cur} {delta} على {category}. معظمها لدى {merchant}.",
    },
    "spend_less_blame": {
        HI: "{period} में आपने {cur} {total} खर्च किए — {category} पर पिछली बार से {cur} {delta} कम। इसमें ज़्यादातर {merchant} है।",
        HINGLISH: "{period} me aapne {cur} {total} kharch kiye — {category} par pichli baar se {cur} {delta} kam. Iska zyada hissa {merchant} hai.",
        AR: "أنفقت {cur} {total} في {period} — بنقصان {cur} {delta} على {category}. معظمها لدى {merchant}.",
    },
    "no_payment_to": {
        HI: "{who} को किया गया कोई पेमेंट आपकी हाल की एक्टिविटी में नहीं मिला। ये रहे बाकी।",
        HINGLISH: "{who} ko kiya hua koi payment aapki recent activity me nahi mila. Ye rahe baaki.",
        AR: "لم أجد أي دفعة إلى {who} في نشاطك الأخير. هذه آخر العمليات.",
    },
    # ── picking an earlier thread back up ───────────────────────────────────────────────
    #: Said when the customer reaches back past a topic change. It is not decoration: it
    #: shows WHICH of several open things we think they mean, so a wrong guess is visible
    #: in one glance instead of being buried in an answer about the wrong bill.
    "about_task": {
        HI: "{what} के बारे में —",
        HINGLISH: "{what} ke baare me —",
        AR: "بخصوص {what} —",
    },
    # ── bills ───────────────────────────────────────────────────────────────────────────
    "bills_due": {
        HI: "{n} बिल बाकी हैं, कुल {cur} {total}।",
        HINGLISH: "{n} bill baaki hai, total {cur} {total}.",
        AR: "{n} فاتورة مستحقة، المجموع {cur} {total}.",
    },
    "bills_none": {
        HI: "अभी कोई बिल बाकी नहीं है।",
        HINGLISH: "Abhi koi bill baaki nahi hai.",
        AR: "لا توجد فواتير مستحقة الآن.",
    },
    "bills_some_unchecked": {
        HI: "{k} बिल चेक नहीं हो पाए, तो यह कुल पूरा नहीं हो सकता",
        HINGLISH: "{k} bill check nahi ho paye, to ye total poora nahi ho sakta",
        AR: "لم أتمكن من التحقق من {k}، لذا قد لا يكون المجموع كاملاً",
    },
    "bill_where_from": {
        HI: "{who} के लिए {cur} {amount}। कहाँ से लें — वॉलेट या बैंक?",
        HINGLISH: "{who} ke liye {cur} {amount}. Kahan se le — wallet ya bank?",
        AR: "{cur} {amount} لـ {who}. من أين؟ المحفظة أم البنك؟",
    },
    # ── splits ──────────────────────────────────────────────────────────────────────────
    "split_owed_and_owe": {
        HI: "आपके स्प्लिट में, आपको {owed} {cur} मिलने हैं और आपको {owe} {cur} देने हैं।",
        HINGLISH: "Aapke splits me, aapko {owed} {cur} milne hai aur aapko {owe} {cur} dene hai.",
        AR: "في تقسيماتك، لك {owed} {cur} وعليك {owe} {cur}.",
    },
    "split_one_owes_you": {
        HI: "{who} को आपको {amount} {cur} देने हैं।",
        HINGLISH: "{who} ko aapko {amount} {cur} dene hai.",
        AR: "{who} يدين لك بـ {amount} {cur}.",
    },
    "split_you_owe": {
        HI: "आपको {who} को {amount} {cur} देने हैं।",
        HINGLISH: "Aapko {who} ko {amount} {cur} dene hai.",
        AR: "أنت مدين لـ {who} بـ {amount} {cur}.",
    },
    # ── history ─────────────────────────────────────────────────────────────────────────
    "history_head": {
        HI: "आपके पिछले {n} पेमेंट — कुल {cur} {total}।",
        HINGLISH: "Aapke pichle {n} payment — total {cur} {total}.",
        AR: "آخر {n} دفعات لك — المجموع {cur} {total}.",
    },
    "history_none": {
        HI: "आपके अकाउंट पर अभी कोई पेमेंट नहीं मिली।",
        HINGLISH: "Aapke account par abhi koi payment nahi mili.",
        AR: "لم أجد أي دفعات على حسابك.",
    },
    # ── the wall, and the common refusals ───────────────────────────────────────────────
    "refusal": {
        HI: "मैं सिर्फ आपके OpenWallet पेमेंट में मदद कर सकता हूँ — ट्रांजैक्शन का स्टेटस, "
            "रिफंड, फीस और समय, आपकी लिमिट, और पास के ऑफर। इनमें से कुछ भी पूछ लीजिए।",
        HINGLISH: "Main sirf aapke OpenWallet payments me help kar sakta hu — transaction status, "
                  "refund, fees aur timing, aapki limits, aur paas ke offers. "
                  "Inme se kuch bhi puchh lijiye.",
        AR: "أستطيع المساعدة في مدفوعات OpenWallet فقط — حالة المعاملات، المبالغ المستردة، "
            "الرسوم والتوقيت، حدودك، والعروض القريبة. اسألني عن أي منها.",
    },
    # ── abuse: firm, calm, and still offering help ──────────────────────────────────────
    # ── the customer-app BRD's flows (2026-10-14) ──────────────────────────────────────
    "balance_is": {
        AR: "رصيد محفظتك **{amount} درهم**.",
        HI: "आपके वॉलेट में **AED {amount}** हैं।",
        HINGLISH: "Aapke wallet me **AED {amount}** hai.",
    },
    "balance_unavailable": {
        AR: "لا أستطيع قراءة رصيدك الآن. حاول بعد قليل — لن أخمّن رقماً.",
        HI: "अभी आपका बैलेंस नहीं पढ़ पा रहा। थोड़ी देर में कोशिश करें — मैं अंदाज़ा नहीं लगाऊँगा।",
        HINGLISH: "Abhi balance nahi padh pa raha. Thodi der me try karein — main guess nahi karunga.",
    },
    "recharge_saved_or_new": {
        AR: "أي رقم؟ رقمك المحفوظ ({mask}) أم رقم جديد؟",
        HI: "कौन सा नंबर? आपका सेव किया नंबर ({mask}) या नया?",
        HINGLISH: "Kaunsa number? Aapka saved number ({mask}) ya naya?",
    },
    "recharge_which_number": {AR: "ما هو الرقم؟", HI: "नंबर क्या है?", HINGLISH: "Number kya hai?"},
    "recharge_how_much": {AR: "كم المبلغ؟", HI: "कितने का?", HINGLISH: "Kitne ka?"},
    "recharge_operator": {
        AR: "هل هذا رقم du أم اتصالات (e&)؟",
        HI: "यह du का नंबर है या Etisalat (e&) का?",
        HINGLISH: "Ye du ka number hai ya Etisalat (e&) ka?",
    },
    "recharge_bad_number": {
        AR: "هذا الرقم لا يبدو صحيحاً — رقم الجوال في الإمارات يبدأ بـ 05 ومن 10 أرقام. تحقق منه وحاول مجدداً.",
        HI: "यह नंबर सही नहीं लगता — UAE का मोबाइल 05 से शुरू होता है और 10 अंकों का होता है।",
        HINGLISH: "Ye number sahi nahi lagta — UAE mobile 05 se shuru hota hai, 10 digits ka.",
    },
    "recharge_confirm": {
        AR: "شحن {op} {mask} بمبلغ {amount} درهم؟",
        HI: "{op} {mask} पर AED {amount} का रिचार्ज करें?",
        HINGLISH: "{op} {mask} pe AED {amount} ka recharge karein?",
    },
    "reminder_what": {
        AR: "بماذا أذكّرك — فاتورة، أم إرسال مال لشخص؟",
        HI: "किस बात की याद दिलाऊँ — कोई बिल, या किसी को पैसे भेजना?",
        HINGLISH: "Kis cheez ka reminder — koi bill, ya kisi ko paise bhejna?",
    },
    "reminder_when": {AR: "متى أذكّرك؟", HI: "कब याद दिलाऊँ?", HINGLISH: "Kab yaad dilaun?"},
    "reminder_day": {AR: "في أي يوم من الشهر؟", HI: "महीने की किस तारीख को?", HINGLISH: "Mahine ki kis tareekh ko?"},
    "reminder_set": {
        AR: "تم ضبط التذكير: {what}، {when}. سيكتفي بتنبيهك — أنت من يؤكد كل دفعة.",
        HI: "रिमाइंडर सेट: {what}, {when}। यह सिर्फ़ याद दिलाएगा — हर पेमेंट आप ही कन्फर्म करेंगे।",
        HINGLISH: "Reminder set: {what}, {when}. Ye sirf yaad dilayega — har payment aap hi confirm karenge.",
    },
    "reminder_none": {
        AR: "لا توجد لديك تذكيرات. هل أضبط واحداً؟",
        HI: "आपका कोई रिमाइंडर नहीं है। एक सेट कर दूँ?",
        HINGLISH: "Aapka koi reminder nahi hai. Ek set kar doon?",
    },
    "retry_unknown": {
        AR: "دفعتك بمبلغ {amount} درهم إلى {who} لا تزال قيد التأكيد. إعادة المحاولة الآن قد ترسل المال مرتين، لذلك لن أفعل — سأبلغك فور اكتمالها.",
        HI: "{who} को AED {amount} का पेमेंट अभी कन्फर्म हो रहा है। अभी दोबारा करने से पैसे दो बार जा सकते हैं, इसलिए मैं नहीं करूँगा।",
        HINGLISH: "{who} ko AED {amount} ka payment abhi confirm ho raha hai. Abhi retry karne se paise do baar ja sakte hain, isliye main nahi karunga.",
    },
    "retry_succeeded": {
        AR: "دفعتك الأخيرة بمبلغ {amount} درهم إلى {who} تمت بنجاح. المرجع: {ref}. لا حاجة لإعادة المحاولة.",
        HI: "{who} को AED {amount} का आख़िरी पेमेंट हो चुका है। रेफ़रेंस: {ref}। दोबारा करने की ज़रूरत नहीं।",
        HINGLISH: "{who} ko AED {amount} ka last payment ho chuka hai. Reference: {ref}. Retry ki zaroorat nahi.",
    },
    "retry_failed": {
        AR: "دفعتك السابقة بمبلغ {amount} درهم إلى {who} فشلت ولم يُخصم أي مبلغ. هل تريد المحاولة مجدداً؟",
        HI: "{who} को AED {amount} का पिछला पेमेंट फेल हुआ और कोई पैसा नहीं कटा। फिर से कोशिश करें?",
        HINGLISH: "{who} ko AED {amount} ka pichla payment fail hua, koi paisa nahi kata. Phir se try karein?",
    },
    "cancel_done": {AR: "تم إلغاء الدفع. لم يُرسل أي مبلغ.", HI: "पेमेंट रद्द। कोई पैसा नहीं भेजा गया।",
                    HINGLISH: "Payment cancel. Koi paisa nahi bheja gaya."},
    "cancel_nothing": {AR: "لا يوجد شيء قيد التنفيذ — لم يتحرك أي مبلغ.", HI: "कुछ भी चल नहीं रहा — कोई पैसा नहीं गया।",
                       HINGLISH: "Kuch bhi chal nahi raha — koi paisa nahi gaya."},
    "handoff": {
        AR: "سأوصلك بخدمة عملاء OpenWallet، وقد أرفقت هذه المحادثة حتى لا تضطر للتكرار. رقم المرجع {ref}.",
        HI: "मैं आपको OpenWallet कस्टमर सपोर्ट से जोड़ रहा हूँ। यह बातचीत साथ भेज दी है। रेफ़रेंस {ref}।",
        HINGLISH: "Main aapko OpenWallet customer support se jod raha hoon. Ye baatcheet saath bhej di hai. Reference {ref}.",
    },
    "report_fraud": {
        AR: "سأوصلك بفريق الاحتيال فوراً ({ref}). لا تشارك رمزك السري أو رمز التحقق مع أحد — OpenWallet لن تطلبهما أبداً.",
        HI: "मैं आपको तुरंत फ्रॉड टीम से जोड़ रहा हूँ ({ref})। अपना PIN या OTP किसी को न बताएँ।",
        HINGLISH: "Main aapko turant fraud team se jod raha hoon ({ref}). Apna PIN ya OTP kisi ko na batayein.",
    },
    "knowledge_none": {
        AR: "ليس لدي إجابة معتمدة لهذا بعد. يمكنني توصيلك بالدعم، أو مساعدتك في دفعة أو فاتورة أو معاملاتك.",
        HI: "इसका स्वीकृत जवाब अभी मेरे पास नहीं है। मैं सपोर्ट से जोड़ सकता हूँ, या पेमेंट, बिल या लेनदेन में मदद कर सकता हूँ।",
        HINGLISH: "Iska approved jawab abhi mere paas nahi hai. Main support se jod sakta hoon, ya payment, bill ya transactions me madad kar sakta hoon.",
    },
    "nav_add_money": {AR: "أفتح **إضافة أموال** — اشحن محفظتك من بطاقتك أو بنكك.",
                      HI: "**Add Money** खोल रहा हूँ — कार्ड या बैंक से वॉलेट टॉप-अप करें।",
                      HINGLISH: "**Add Money** khol raha hoon — card ya bank se wallet top-up karein."},
    "nav_withdraw_money": {AR: "أفتح **سحب الأموال** — حوّل أموال المحفظة إلى بنكك.",
                           HI: "**Withdraw Money** खोल रहा हूँ — वॉलेट से बैंक में पैसे भेजें।",
                           HINGLISH: "**Withdraw Money** khol raha hoon — wallet se bank me paise bhejein."},
    "nav_link_bank": {AR: "أفتح **ربط البنك** — اربط حسابك البنكي بمحفظتك.",
                      HI: "**Connect Bank** खोल रहा हूँ — अपना बैंक खाता जोड़ें।",
                      HINGLISH: "**Connect Bank** khol raha hoon — apna bank account jodein."},
    "small_talk": {
        AR: "أنا بخير، شكراً لسؤالك! يمكنني فقط مساعدتك في مدفوعات OpenWallet — إرسال المال والفواتير ومعاملاتك ورصيدك. ماذا تود أن تفعل؟",
        HI: "मैं ठीक हूँ, पूछने के लिए धन्यवाद! मैं सिर्फ़ आपके OpenWallet पेमेंट्स में मदद कर सकता हूँ — पैसे भेजना, बिल, आपके लेनदेन और बैलेंस। आप क्या करना चाहेंगे?",
        HINGLISH: "Main badhiya hoon, poochne ke liye shukriya! Main sirf aapke OpenWallet payments me madad kar sakta hoon — paise bhejna, bills, transactions aur balance. Kya karna hai?",
    },
    "didnt_catch": {
        AR: "عذراً، لم أفهم ذلك — ماذا تود أن تفعل؟",
        HI: "माफ़ कीजिए, समझ नहीं आया — आप क्या करना चाहेंगे?",
        HINGLISH: "Sorry, samajh nahi aaya — kya karna hai?",
    },
    "abuse_boundary": {
        HI: "मैं यहाँ आपकी मदद के लिए हूँ, पर कृपया सम्मान से बात करें। अगर किसी पेमेंट में "
            "दिक्कत है, तो बताइए क्या हुआ — मैं देखता हूँ।",
        HINGLISH: "Main yahan aapki madad ke liye hoon, par please izzat se baat karein. Agar kisi "
                  "payment me dikkat hai, to bataiye kya hua — main dekhta hoon.",
        AR: "أنا هنا لمساعدتك، لكن أرجو أن نتحدث باحترام. إذا كانت هناك مشكلة في أي دفعة، "
            "أخبرني بما حدث وسأتحقق منها.",
    },
    "abuse_prefix": {
        HI: "कृपया सम्मान से बात करें।",
        HINGLISH: "Please izzat se baat karein.",
        AR: "أرجو أن نتحدث باحترام.",
    },
    "abuse_stopped": {
        HI: "अपशब्दों की वजह से मैं यह बातचीत आगे नहीं बढ़ा सकता। अगर किसी पेमेंट में दिक्कत है, "
            "तो कृपया OpenWallet कस्टमर सपोर्ट से संपर्क करें — वे इसे देखेंगे।",
        HINGLISH: "Gaali-galoch ki wajah se main ye baatcheet aage nahi badha sakta. Agar kisi "
                  "payment me dikkat hai, to please OpenWallet customer support se sampark karein — "
                  "wo ise dekhenge.",
        AR: "لا يمكنني متابعة هذه المحادثة بسبب الإساءة. إذا كانت هناك مشكلة في أي دفعة، يرجى "
            "التواصل مع خدمة عملاء OpenWallet وسيتابعونها.",
    },
    "cancelled": {
        HI: "भुगतान रद्द कर दिया गया। कोई पैसा नहीं भेजा गया।",
        HINGLISH: "Payment cancel ho gaya. Koi paisa nahi gaya.",
        AR: "تم إلغاء الدفع. لم يتم إرسال أي أموال.",
    },
    "cancelled_other": {
        HI: "ठीक है, छोड़ दिया। कुछ भी नहीं हुआ।",
        HINGLISH: "Theek hai, chod diya. Kuch bhi nahi hua.",
        AR: "حسناً، تركتها. لم يتحرك شيء.",
    },
    "which_country": {
        HI: "आप किस देश में पैसे भेज रहे हैं?",
        HINGLISH: "Aap kis desh me paise bhej rahe ho?",
        AR: "إلى أي دولة تريد الإرسال؟",
    },
    "reminders_clear_confirm": {
        HI: "आपके सभी {n} रिमाइंडर हटा दूँ? यह वापस नहीं होगा।",
        HINGLISH: "Aapke saare {n} reminders delete kar du? Ye wapas nahi hoga.",
        AR: "هل أحذف كل التذكيرات ({n})؟ لا يمكن التراجع عن ذلك.",
    },
    "reminders_cleared": {
        HI: "हो गया — आपके सभी {n} रिमाइंडर हटा दिए गए।",
        HINGLISH: "Ho gaya — aapke saare {n} reminders delete kar diye.",
        AR: "تم — حُذفت كل تذكيراتك ({n}).",
    },
    "sec_credential": {
        HI: "OpenWallet कभी भी आपका PIN, OTP, पासवर्ड या कार्ड की जानकारी नहीं माँगता — न चैट में, न कॉल पर, न WhatsApp पर। इन्हें किसी को न बताएँ, चाहे वह खुद को OpenWallet का बताए। अगर किसी ने माँगा है, तो मैं आपको अभी फ़्रॉड टीम से जोड़ सकता हूँ।",
        HINGLISH: "OpenWallet kabhi bhi aapka PIN, OTP, password ya card details nahi maangta — na chat me, na call par, na WhatsApp par. Inhe kisi ko mat batayein, chahe wo khud ko OpenWallet ka bataye. Agar kisi ne maanga hai to main abhi aapko fraud team se jod sakta hu.",
        AR: "OpenWallet لن تطلب منك أبداً الرقم السري أو رمز التحقق أو كلمة المرور أو بيانات البطاقة — لا في المحادثة ولا بالاتصال ولا عبر واتساب. لا تشاركها مع أي شخص، حتى لو قال إنه من OpenWallet. إذا طلبها منك أحد، أستطيع توصيلك بفريق مكافحة الاحتيال الآن.",
    },
    "sec_forgot_pin": {
        HI: "मैं आपका PIN न देख सकता हूँ न बता सकता हूँ — OpenWallet में कोई भी नहीं, और इसे कभी किसी को न बताएँ। आप ऐप में Profile → Security → Reset PIN से इसे रीसेट कर सकते हैं।",
        HINGLISH: "Main aapka PIN na dekh sakta hu na bata sakta hu — OpenWallet me koi bhi nahi, aur ise kabhi kisi ko mat batayein. App me Profile → Security → Reset PIN se reset kar sakte hain.",
        AR: "لا أستطيع رؤية رقمك السري أو إخبارك به — لا أحد في OpenWallet يستطيع، ولا تشاركه مع أحد أبداً. يمكنك إعادة تعيينه من التطبيق: الملف الشخصي ← الأمان ← إعادة تعيين الرقم السري.",
    },
    "sec_scam": {
        HI: "यह धोखाधड़ी लग रही है। OpenWallet कभी भी खाता खोलने, रिफंड छुड़ाने, इनाम पाने या जुर्माना भरने के लिए पैसे नहीं माँगता, और असली अधिकारी वॉलेट ट्रांसफ़र से जुर्माना नहीं लेते। कृपया यह पैसा न भेजें। क्या मैं आपको फ़्रॉड टीम से जोड़ दूँ?",
        HINGLISH: "Ye scam lag raha hai. OpenWallet kabhi bhi account unlock karne, refund release karne, prize lene ya fine bharne ke liye paise nahi maangta, aur asli officials wallet transfer se fine nahi lete. Please ye paisa mat bhejiye. Kya main aapko fraud team se jod du?",
        AR: "يبدو هذا احتيالاً. OpenWallet لا تطلب أبداً دفع مبلغ لفتح حسابك أو للإفراج عن مبلغ مسترد أو لاستلام جائزة أو لسداد مخالفة، والجهات الرسمية لا تحصّل المخالفات بتحويل من المحفظة. من فضلك لا ترسل هذا المبلغ. هل أوصلك بفريق مكافحة الاحتيال؟",
    },
    "sec_injection": {
        HI: "मैं अपने काम करने का तरीका नहीं बदल सकता — हर भुगतान के लिए ऐप में आपकी अपनी पुष्टि और PIN ज़रूरी है, और कोई संदेश इसे बंद नहीं कर सकता। सामान्य भुगतान, बिल या आपके लेनदेन के सवाल में मैं ख़ुशी से मदद करूँगा।",
        HINGLISH: "Main apna kaam karne ka tareeka nahi badal sakta — har payment ke liye app me aapka apna confirm aur PIN zaroori hai, aur koi message ise band nahi kar sakta. Normal payment, bill ya transactions ke sawaal me main khushi se madad karunga.",
        AR: "لا أستطيع تغيير طريقة عملي — كل دفعة تحتاج تأكيدك ورقمك السري في التطبيق، ولا توجد رسالة تستطيع إلغاء ذلك. يسعدني مساعدتك في دفعة عادية أو فاتورة أو سؤال عن معاملاتك.",
    },
    "sec_other_person": {
        HI: "मैं सिर्फ़ आपका अपना वॉलेट और भुगतान दिखा सकता हूँ — किसी और का बैलेंस, लेनदेन या जानकारी नहीं, परिवार की भी नहीं। चाहें तो उन्हें किए गए आपके अपने भुगतान दिखा दूँ।",
        HINGLISH: "Main sirf aapka apna wallet aur payments dikha sakta hu — kisi aur ka balance, transactions ya details nahi, family ka bhi nahi. Chahein to unhe kiye gaye aapke apne payments dikha du.",
        AR: "أستطيع عرض محفظتك ومدفوعاتك أنت فقط — لا أستطيع الاطلاع على رصيد أو معاملات أو بيانات أي شخص آخر، حتى من العائلة. أستطيع عرض مدفوعاتك أنت لهم إن أردت.",
    },
    "sec_internal": {
        HI: "मैं यह नहीं बता सकता कि हमारे सिस्टम अंदर से कैसे काम करते हैं — सबके पैसे की सुरक्षा के लिए यह निजी रहता है। आपके अपने भुगतान, बिल या ट्रांसफ़र में मदद करके ख़ुशी होगी।",
        HINGLISH: "Main ye nahi bata sakta ki hamare systems andar se kaise kaam karte hain — sabke paise ki safety ke liye ye private rehta hai. Aapke apne payments, bills ya transfers me madad karke khushi hogi.",
        AR: "لا أستطيع مشاركة كيف تعمل أنظمتنا من الداخل — تبقى خاصة لحماية أموال الجميع. يسعدني مساعدتك في مدفوعاتك أو فواتيرك أو تحويلاتك.",
    },
    "sec_legal": {
        HI: "मैं समझता हूँ कि यह गंभीर है, और मुझे खेद है कि बात यहाँ तक पहुँची। मैं यह बातचीत अभी हमारी टीम के एक व्यक्ति को दे रहा हूँ ताकि इसे ठीक से देखा जाए — आपको दोबारा नहीं बताना पड़ेगा।",
        HINGLISH: "Main samajhta hu ki ye serious hai, aur mujhe afsos hai ki baat yahan tak pahunchi. Main ye conversation abhi hamari team ke ek insaan ko de raha hu taaki ise theek se dekha jaye — aapko dobara nahi batana padega.",
        AR: "أتفهم أن الأمر جدّي، وآسف أنه وصل إلى هنا. سأحوّل هذه المحادثة الآن إلى شخص من فريقنا لينظر فيها كما يجب — لن تحتاج إلى تكرار كلامك.",
    },
    "sec_distress": {
        HI: "मुझे बहुत दुख है कि आप इससे गुज़र रहे हैं — आपको अकेले इसका सामना नहीं करना है। मैं आपको अभी आपके पैसे के बारे में हमारी टीम के एक व्यक्ति से जोड़ रहा हूँ। अगर आपके मन में ख़ुद को नुकसान पहुँचाने का ख़्याल आ रहा है, तो कृपया UAE नेशनल मेंटल सपोर्ट लाइन 800 4673 (800 HOPE) पर कॉल करें, या तुरंत ख़तरे में हों तो 999 पर।",
        HINGLISH: "Mujhe bahut afsos hai ki aap isse guzar rahe hain — aapko akele iska saamna nahi karna. Main aapko abhi aapke paise ke baare me hamari team ke ek insaan se jod raha hu. Agar aapke mann me khud ko nuksaan pahunchane ka khayal aa raha hai, to please UAE National Mental Support Line 800 4673 (800 HOPE) par call karein, ya turant khatre me hon to 999 par.",
        AR: "أنا آسف جداً لما تمر به — لست وحدك في هذا. سأوصلك الآن بشخص من فريقنا بخصوص أموالك. إذا كانت تراودك أفكار بإيذاء نفسك، من فضلك اتصل بالخط الوطني للدعم النفسي في الإمارات 800 4673 (800 HOPE)، أو 999 إذا كنت في خطر مباشر.",
    },
    "sec_illegal": {
        HI: "मैं इसमें मदद नहीं कर सकता। अपने भुगतानों के बारे में कोई सवाल हो तो मैं यहीं हूँ।",
        HINGLISH: "Main isme madad nahi kar sakta. Apne payments ke baare me koi sawaal ho to main yahin hu.",
        AR: "لا أستطيع المساعدة في ذلك. إذا كان لديك سؤال عن مدفوعاتك، أنا هنا.",
    },
    "xb_offers": {
        HI: "विदेश पैसे भेजने पर ऑफ़र — {list}। इस्तेमाल करने के लिए भेजने वाली स्क्रीन पर प्रोमो कोड डालें।",
        HINGLISH: "Bahar paise bhejne par offers — {list}. Use karne ke liye send screen par promo code daalein.",
        AR: "عروض التحويل للخارج — {list}. أدخل رمز العرض في شاشة الإرسال لاستخدامه.",
    },
    "xb_offers_none_one": {
        HI: "अभी {dest} भेजने पर कोई ऑफ़र नहीं है। फ़ीस AED {fee} है।",
        HINGLISH: "Abhi {dest} bhejne par koi offer nahi hai. Fee AED {fee} hai.",
        AR: "لا توجد عروض على التحويل إلى {dest} حالياً. الرسوم {fee} درهم.",
    },
    "xb_offers_none": {
        HI: "अभी विदेश पैसे भेजने पर कोई ऑफ़र नहीं है।",
        HINGLISH: "Abhi bahar paise bhejne par koi offer nahi hai.",
        AR: "لا توجد عروض على التحويلات الدولية حالياً.",
    },
    "fx_live": {
        HI: "आज {dest} का रेट 1 AED = {rate} {cur} है। सही रेट कन्फर्म करते समय लॉक होता है। ट्रांसफर फीस AED {fee} है।",
        HINGLISH: "Aaj {dest} ka rate 1 AED = {rate} {cur} hai. Exact rate confirm karte waqt lock hota hai. Transfer fee AED {fee} hai.",
        AR: "سعر اليوم إلى {dest}: 1 درهم = {rate} {cur}. يُثبَّت السعر النهائي عند التأكيد. رسوم التحويل {fee} درهم.",
    },
    "fx_live_amt": {
        HI: "आज {dest} का रेट 1 AED = {rate} {cur} है, यानी AED {amount} लगभग {conv} {cur} होगा। सही रेट कन्फर्म करते समय लॉक होता है। ट्रांसफर फीस AED {fee} है।",
        HINGLISH: "Aaj {dest} ka rate 1 AED = {rate} {cur} hai, yaani AED {amount} lagbhag {conv} {cur} hoga. Exact rate confirm karte waqt lock hota hai. Transfer fee AED {fee} hai.",
        AR: "سعر اليوم إلى {dest}: 1 درهم = {rate} {cur}، أي أن {amount} درهم تساوي تقريباً {conv} {cur}. يُثبَّت السعر النهائي عند التأكيد. رسوم التحويل {fee} درهم.",
    },
    "fx_rate": {
        HI: "{dest} का लाइव एक्सचेंज रेट भेजने वाली स्क्रीन पर कन्फर्म से पहले दिखता है — मैं अभी यहाँ नहीं बता सकता।",
        HINGLISH: "{dest} ka live exchange rate send screen par confirm se pehle dikhta hai — main abhi yahan nahi bata sakta.",
        AR: "يظهر سعر الصرف المباشر إلى {dest} في شاشة الإرسال قبل التأكيد — لا أستطيع ذكره هنا الآن.",
    },
    "fx_which_country": {
        HI: "किस देश के लिए रेट चाहिए? रुपया कई देशों में चलता है।",
        HINGLISH: "Kis country ka rate chahiye? Rupee kai countries me chalta hai.",
        AR: "لأي دولة تريد السعر؟ الروبية مستخدمة في أكثر من دولة.",
    },
    "scheduled_ready": {
        HI: "मैं {when} पर पैसे भेज दूँगा। अभी अपने PIN से कन्फर्म करें — चलने से पहले आप इसे रद्द कर सकते हैं।",
        HINGLISH: "Main {when} par paise bhej dunga. Abhi apne PIN se confirm karein — chalne se pehle aap ise cancel kar sakte hain.",
        AR: "سأرسل المبلغ الساعة {when}. أكّد برقمك السري الآن وسيُرسَل في وقته — يمكنك الإلغاء قبل ذلك.",
    },
    "schedule_too_far": {
        HI: "मैं भुगतान ज़्यादा से ज़्यादा 6 घंटे आगे शेड्यूल कर सकता हूँ। क्या {when} के लिए रिमाइंडर सेट कर दूँ?",
        HINGLISH: "Main payment zyada se zyada 6 ghante aage schedule kar sakta hu. Kya {when} ke liye reminder set kar du?",
        AR: "أستطيع جدولة الدفعة حتى 6 ساعات مقدماً فقط. هل أضبط تذكيراً للساعة {when} بدلاً من ذلك؟",
    },
    "beneficiary_not_found": {
        HI: "{name} आपके सेव किए गए लाभार्थियों में नहीं है। क्या आप नया लाभार्थी जोड़ना चाहेंगे?",
        HINGLISH: "{name} aapke saved beneficiaries me nahi hai. Kya aap naya beneficiary add karna chahenge?",
        AR: "{name} ليس ضمن المستفيدين المحفوظين لديك. هل تريد إضافة مستفيد جديد؟",
    },
    "insufficient_balance": {
        HI: "आपका उपलब्ध बैलेंस {bal} AED है। क्या आप कम राशि भेजना चाहेंगे?",
        HINGLISH: "Aapka available balance {bal} AED hai. Kya aap kam amount bhejna chahenge?",
        AR: "رصيدك المتاح هو {bal} درهم. هل تريد إرسال مبلغ أقل؟",
    },
    "greeting": {
        HI: "नमस्ते। मैं आपके पेमेंट में मदद कर सकता हूँ — कोई पेमेंट कहाँ पहुँची, रिफंड, "
            "फीस और समय, आपकी लिमिट, और पास के ऑफर। क्या चाहिए?",
        HINGLISH: "Namaste. Main aapke payments me help kar sakta hu — koi payment kahan "
                  "pahunchi, refund, fees aur timing, aapki limits, aur paas ke offers. "
                  "Kya chahiye?",
        AR: "مرحباً. أستطيع مساعدتك في مدفوعاتك — إلى أين وصلت دفعة، المبالغ المستردة، "
            "الرسوم والتوقيت، حدودك، والعروض القريبة. ماذا تحتاج؟",
    },
    "which_bill": {
        HI: "कौन सा बिल बांटना है?",
        HINGLISH: "Kaunsa bill split karna hai?",
        AR: "أي فاتورة تريد تقسيمها؟",
    },
    "split_with_who": {
        HI: "{bill} का बिल ({cur} {amount}) किसके साथ बांटना है?",
        HINGLISH: "{bill} ka bill ({cur} {amount}) kiske saath split karna hai?",
        AR: "مع من تريد تقسيم فاتورة {bill} ({cur} {amount})؟",
    },
    "billsplit_ready": {
        HI: "{bill} का बिल {cur} {total} है। मैं {who} से {cur} {each} मांगूंगा, "
            "आप {cur} {mine} देंगे, और उनका हिस्सा आते ही मैं {biller} को भर दूंगा।",
        HINGLISH: "{bill} ka bill {cur} {total} hai. Main {who} se {cur} {each} maangunga, "
                  "aap {cur} {mine} denge, aur unka hissa aate hi main {biller} ko bhar "
                  "dunga.",
        AR: "فاتورة {bill} هي {cur} {total}. سأطلب من {who} {cur} {each}، وتدفع أنت "
            "{cur} {mine}، وبمجرد وصول نصيبهم سأدفع لـ {biller}.",
    },
    "help_open": {
        HI: "बिल्कुल। पेमेंट से जुड़ी कोई भी बात पूछिए — कोई पेमेंट कहाँ पहुँची, रिफंड, "
            "फीस, आपकी लिमिट, बिल, या पास के ऑफर।",
        HINGLISH: "Bilkul. Payment se judi koi bhi baat puchh lijiye — koi payment kahan "
                  "pahunchi, refund, fees, aapki limits, bill, ya paas ke offers.",
        AR: "بالتأكيد. اسأل عن أي شيء يتعلق بمدفوعاتك — إلى أين وصلت دفعة، المبالغ "
            "المستردة، الرسوم، حدودك، الفواتير، أو العروض القريبة.",
    },
    "how_much_at": {
        HI: "{who} पर कितना?",
        HINGLISH: "{who} par kitna?",
        AR: "كم المبلغ في {who}؟",
    },
    "will_split_after": {
        HI: " पेमेंट होने के बाद मैं इसे जिसके साथ कहें बांट दूंगा।",
        HINGLISH: " Payment hone ke baad main isko jiske saath kahein baant dunga.",
        AR: " بعد الدفع سأقسمها مع من تريد.",
    },
    "enter_pin": {
        HI: "कन्फर्म करने के लिए अपना PIN डालिए।",
        HINGLISH: "Confirm karne ke liye apna PIN daaliye.",
        AR: "أدخل رقمك السري للتأكيد.",
    },
}


def t(key: str, lang: str, **vals) -> str | None:
    """The translated string, or None when we do not have one for this language.

    None rather than a fallback so the caller keeps its own English wording — which has been
    read and checked — instead of this module inventing one.
    """
    entry = T.get(key)
    if not entry or lang == EN:
        return None
    s = entry.get(lang)
    if not s:
        return None
    try:
        return s.format(**vals)
    except (KeyError, IndexError):
        return None
