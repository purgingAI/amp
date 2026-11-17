"""The first gate: credentials, scams, instruction tampering, illegal requests.

Runs before classification, on the raw sentence. A payments assistant that is polite to an
attacker is still a liability, so these four never reach the model or a handler:

  credential   asked for, offered, pasted or "save it for me": PIN, OTP, password, CVV,
               card number. OpenWallet never asks for them — say so, never repeat the code back,
               offer the fraud team (and the reset path if they just forgot it).
  scam         a payment demanded to unlock an account, release a refund, claim a prize,
               clear a "fine", pay a "fee", return an "overpayment", move to a "safe
               account", or the "new number" family scam. Warn; never start the payment.
  injection    "ignore your instructions", fake SYSTEM/ADMIN lines, "skip the PIN", "you are
               now DAN", "print your prompt". The rules don't change; nothing is acted on —
               not even the ordinary-looking part of the same message.
  illegal      laundering, structuring under limits, fake receipts, using someone else's
               wallet, hacking. A firm no.

Every pattern is in four languages (en / hinglish / hi / ar). Detection only — the reply
text lives in lang.py and the engine.
"""
from __future__ import annotations

import re

_I = re.I

# ── credentials ─────────────────────────────────────────────────────────────────────────
_CRED_WORD = (r"\b(?:pin|otp|one[\s-]?time\s+(?:password|code)|password|passcode|cvv|cvc|"
              r"verification\s+code|security\s+code|login\s+code|card\s+(?:number|details))\b|"
              r"الرقم\s+السري|رمز\s+(?:ال)?(?:تحقق|سري)|الرمز|رمزي|الكود|كلمة\s+(?:المرور|السر)|الـ?\s*otp|الـ?\s*cvv|رقم\s+البطاقة|"
              r"\b(?:sms|verification|6-digit|six[\s-]digit)\s+code\b|"
              r"पिन|ओटीपी|पासवर्ड|सीवीवी")
CRED = re.compile(_CRED_WORD, _I)
#: Talking ABOUT a credential in a way that matters: asked for, shared, stored, used, forgotten.
CRED_CONTEXT = re.compile(
    r"\b(?:what(?:'s|\s+is)|tell|give|send|share|reply|type|enter|read|store|save|remember|keep|use|"
    r"verify|confirm|forgot|forget|is\s+\d|here|asking|asked|wants?|need|batao|bata|de\s+du|do|"
    r"bhej|save|yaad|kya\s+hai|maanga|maang)\b|\d{4,}|"
    r"ارسل|أرسل|ابعت|اعطي|أعطي|أعطيه|عطني|اعطني|قول|قولي|اكتب|احفظ|خزّ?ن|نسيت|يبي|يطلب|هو|أحتاج|احتاج|محتاج|أبي|ابي|"
    r"remind\s+me\s+what|"
    r"बताइए|बताओ|बता|दो|भेजो|सेव|याद|माँगा|मांगा|क्या\s+है|है", _I)
#: Card number (13-19 digits, grouped or not).
CARD = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}(?!\d)|"
                  r"\b(?:card|cvv|cvc|expiry|exp)\b[^.]{0,20}(?<!\d)\d{13,19}(?!\d)|(?<!\d)\d{13,19}(?!\d)[^.]{0,20}\b(?:card|cvv|exp)\b", re.I)

# ── scams ───────────────────────────────────────────────────────────────────────────────
SCAM = re.compile(
    r"\b(?:unlock|unfreeze|frozen|reactivat\w*|account\s+(?:will\s+be\s+)?(?:closed|blocked|suspended|frozen)|"
    r"(?:agent|processing|release|clearance|activation|renewal|registration|visa\s+processing|handling)\s+fee|"
    r"release\s+(?:my|your|the)\s+refund|refund\s+(?:is\s+)?stuck|lottery|lucky\s+draw|prize|won\s+a|"
    r"(?:police|traffic|court)\s+(?:fine|case)|customs|parcel|overpaid|paid\s+you\s+(?:extra|by\s+mistake)|"
    r"send\s+(?:it\s+)?back|safe\s+account|test\s+transaction|return\s+(?:you\s+)?(?:double|\d+)|"
    r"new\s+number|hey\s+mum|hi\s+mum|hey\s+dad|recruiter|job\s+offer|kyc\s+(?:expired|update|renewal)|"
    r"band\s+ho\s+jayega|block\s+ho\s+jayega|lottery\s+jeet\w*|inaam|safe\s+account)\b|"
    r"سيتم\s+(?:إغلاق|ايقاف|إيقاف)|حسابك\s+(?:بيتقفل|موقوف|مجمد)|رسوم\s+(?:إفراج|تخليص|معالجة)|جائزة|يانصيب|"
    r"مخالفة|الشرطة|جمارك|طرد|رقم\s*(?:ي)?\s*الجديد|رقمي\s+الجديد|حساب\s+آمن|رسوم\s+وكيل|"
    r"الهوية\s+منتهية|تجديد\s+الهوية|تجريبي|يرجعون\s+لك|"
    r"\bpolice\b[^.]{0,40}\bfine\b|सुरक्षित\s+खाते|"
    r"\b(?:double|doubles|doubling|guarantee[sd]?\s+\d+%|30%\s+monthly|lock\s+my\s+slot)\b|double\s+kar|dogun|दोगुना|يضاعف|"
    r"\b(?:relief|donat\w*|charity|earthquake|flood)\b|जुर्माना|تبرع|زلزال|خيرية|"
    r"cashback\s+(?:activat\w*|ke\s+liye)|activate\s+(?:it|the\s+cashback)|كاش\s*باك[^.]{0,30}(?:تفعيل|حول)|"
    r"\b(?:token|deposit|advance|booking)\b[^.]{0,30}\b(?:before|pehle)\b|عربون|قبل\s+المعاينة|"
    r"redelivery|re-?delivery|courier[^.]{0,30}(?:fee|link)|parcel\s+chhuda|पार्सल\s+छुड़ा|إعادة\s+توصيل|الشحنة\s+متعطلة|"
    r"(?:girl|boy)friend[^.]{0,40}(?:visa|stuck|maang)|online\s+mila|ladki\s+se\s+online|"
    r"tax\s+authority|penalty|recruitment|visa\s+fee|رسوم\s+فيزا|شركة\s+توظيف|"
    r"security\s+(?:team|department)[^.]{0,30}(?:move|shift|transfer)|المحفظة\s+الآمن|من\s+الأمن|"
    r"खाता\s+बंद|ब्लॉक\s+हो\s+जाएगा|लॉटरी|इनाम|जुर्माना|कस्टम|नया\s+नंबर", _I)
#: ...and it must be about paying someone, or it's just a question about scams.
MONEY_ASK = re.compile(
    r"\b(?:send|pay|transfer|bhej\w*|de\s+do|dena|bharna|bhar\w*|chahiye)\b|\d{2,}|"
    r"ارسل|أرسل|حول|ادفع|ابعت|भेज|भुगतान|पैसे|दिरहम", _I)

# ── instruction tampering ───────────────────────────────────────────────────────────────
INJECTION = re.compile(
    r"\bignore\b[^.]{0,30}\b(?:instructions?|rules?|above|previous|prompt)\b|"
    r"\b(?:system|admin|developer|root)\s*(?:message|prompt|override|mode)?\s*[:：]|"
    r"#{2,}\s*admin|admin\s+override|developer\s+mode|dev\s+mode|jailbreak|\bdan\b|"
    r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bpretend\s+(?:to\s+be|you)\b|act\s+as\s+(?:a|an|my)\b|"
    r"(?:skip|bypass|disable|without)\s+(?:the\s+)?(?:pin|otp|confirmation|verification|checks?|limits?)|"
    r"pin\s*=\s*skip|confirm\s*=\s*true|approve\s+all|"
    r"(?:print|show|repeat|reveal|copy)\b[^.]{0,25}\b(?:prompt|instructions|system|above|hidden|settings)|"
    r"new\s+conversation\s*:|end\s+of\s+conversation|<\|im_start\|>|\[inst\]|"
    r"(?:pin|otp)\s+(?:already|pehle\s+se)\s+(?:verified|entered|ho\s+gaya)|already\s+(?:verified|entered)\s+(?:the\s+)?pin|"
    r"bina\s+pin|pin\s+ke\s+bina|rule\s+nahi|instructions?\s+bhool|bhool\s+jao|"
    r"approve[\s_-]*all|skip[\s_-]*pin|ignore[\s_-]*rules|bypass[\s_-]*\w+|"
    r"(?:do|follow)\s+what\s+(?:it|this)\s+says|jo\s+likha\s+hai\s+(?:wo\s+)?karo|upar\s+jo\s+(?:bhi\s+)?likha|"
    r"role[\s-]?play|maan\s+lo\s+tum|imagine\s+you(?:'re|\s+are)|kahani\s+sunao\s+jisme|"
    r"\btum\s+ab\s+[^.]{0,25}\s+ho\b|confirm\s+nahi\s+chahiye|doesn'?t\s+need\s+confirmations?|"
    r"وافق[_\s]*على[_\s]*كل|تخيل\s+انك|تخيل\s+أنك|تخيل\s+إنك|افترض|مث[ّ]?ل\s+دور|"
    r"\b(?:decode|decrypt|rot13|base64)\b[^.]{0,30}\b(?:execute|run|do|act)\b|\brot13\s*:|base64\s*:|"
    r"\"tool\"\s*:|\"function\"\s*:|\"args\"\s*:|\"pin_?\w*\"\s*:|تم_التحقق|\"أمر\"|نفّ?ذ\s+(?:هالكود|الكود|هذا)|فك\s+(?:هالتشفير|التشفير)|"
    r"maintenance\s+mode|pre[-\s]?approved|confirmation\s+(?:step\s+)?removed|system\s+update|"
    r"(?:the\s+)?developer\s+(?:said|says|told)|team\s+ne\s+approve|admin\s+mode|"
    r"act\s+on\s+it|complete\s+this\s+sentence|no\s+limits|bina\s+limit|"
    r"socho\s+tum|suppose\s+you(?:'re|\s+are)|المطو[ّ]?ر\s+قال|وضع\s+الصيانة|معتمدة\s+مسبقاً|اقفل\s+الحماية|"
    r"فريق\s+التطوير|تتخطى|مدير\s+OpenWallet|صلاحية\s+ترفع|"
    r"मान\s+लो\s+तुम|"
    r"تجاهل|انسى?\s+(?:التعليمات|انك)|بدون\s+(?:رمز|الرقم\s+السري|تأكيد)|وضع\s+المطو|النظام\s*[:：]|"
    r"رسالة\s+من\s+الإدارة|كرر\s+الكلام|انت\s+(?:الحين|الآن)\s+dan|"
    r"निर्देश\s+भूल|बिना\s+(?:पिन|PIN)|नियम\s+वाले|सिस्टम\s*[:：]|पुष्टि\s+छोड़", _I)

# ── illegal ─────────────────────────────────────────────────────────────────────────────
ILLEGAL = re.compile(
    r"\blaunder\w*|money\s+laundering|black\s+money|under\s+the\s+(?:reporting\s+)?limit|"
    r"so\s+(?:nobody|no\s+one)\s+(?:notices|finds\s+out)|avoid\s+(?:tax|reporting)|"
    r"fake\s+(?:receipt|screenshot|payment|invoice)|without\s+(?:them|him|her)\s+knowing|"
    r"without\s+(?:them|anyone)\s+checking|where\s+it\s+came\s+from|"
    r"someone\s+else'?s\s+(?:wallet|account)\s+without|hack\w*\s+(?:an?\s+)?account|buy\s+drugs|"
    r"chhote\s+chhote\s+transfer|limit\s+se\s+bachne|bina\s+pata\s+chale|"
    r"غسيل|اغسل|أغسل|مزور|حد\s+الإبلاغ|بدون\s+ما\s+يعرف|"
    r"काले\s+धन|सफ़ेद\s+कैसे|चुपके\s+से", _I)


# ── someone else's data ─────────────────────────────────────────────────────────────────
_REL = (r"(?:wife|husband|son|daughter|brother|sister|friend|employee|staff|mom|mum|dad|father|mother|flatmate|roommate|room\s*mate|"
        r"girlfriend|boyfriend|gf|bf|fianc\w*|landlord|tenant|गर्लफ्रेंड|बॉयफ्रेंड|صديقتي|خطيبتي|"
        r"boss|colleague|neighbou?r|biwi|patni|pati|bhai|behen|dost|beta|beti|"
        r"زوجتي|زوجي|أخوي|اخوي|أختي|اختي|ولدي|بنتي|صاحبي|موظف\w*|"
        r"पत्नी|पति|भाई|बहन|बेटा|बेटी|दोस्त|कर्मचारी)")
_DATA = (r"(?:balance|spend\w*|spent|transactions?|history|statement|kharch\w*|kharid\w*|earn\w*|salary|"
         r"رصيد|صرف\w*|معاملات|"
         r"बैलेंस|खर्च|लेनदेन)")
OTHER = re.compile(
    rf"\b(?:my|meri|mera|mere|uski|uska)\s+{_REL}\b[^.?!]{{0,40}}{_DATA}|{_DATA}[^.?!]{{0,30}}\b(?:my|of\s+my|meri|mere)\s+{_REL}\b|"
    rf"{_REL}[^.?!]{{0,25}}(?:ne|ki|ka|ke)\s+[^.?!]{{0,20}}{_DATA}|"
    r"\b(?!(?:this|last|next|today|yesterday|month|week|year|day)\b)[A-Za-z]+(?:'s|\s+ka|\s+ki)\s+(?:balance|transactions?|history)\b|"
    r"(?:wallet|account)\s*(?:of\s+|no\.?\s*)?0?5\d{7,8}|0?5\d{7,8}\s*(?:wale|registered)|"
    r"(?:transactions?|balance|history|name|statement)\s+(?:of|on|for)\s+(?:wallet\s+|number\s+)?0?5\d{7,8}|"
    r"(?:who|kis\w*|مين)\b[^.?!]{0,30}(?:sent|bheje|حو[ّ]?ل)[^.?!]{0,30}(?:0?5\d{7,8}|هالرقم|this\s+number)|"
    r"\b(?:everyone|anyone)\s+who\s+has\b[^.?!]{0,20}\bsaved\b|\bmy\s+(?:flatmate|roommate|ex|girlfriend|boyfriend|landlord)\b[^.?!]{0,40}"
    r"\b(?:pay|paid|paying|using|spend|send|sent)\b|\b(?:meri|mera)\s+(?:girlfriend|boyfriend|ex)\s+ne\b|"
    r"\b[A-Z][a-z]+\s+ne\s+kis\s+kis\s+ko\b|show\s+me\s+his\s+transaction|"
    r"التحويلات\s+حق\s+(?:زوجتي|زوجي|أخوي|اخوي)|حساب\s+جاري|इस\s+नंबर\s+वाले|"
    r"\b(?:all|saare|sab|sabhi)\s+(?:the\s+)?(?:users|customers|accounts|grahak\w*)\b|سبھی|"
    r"सभी\s+(?:ग्राहक|यूज़र)|كل\s+(?:العملاء|المستخدمين)|"
    r"\bmerchant\b[^.?!]{0,30}\b(?:earn|income|revenue|sales)\b|"
    # what someone ELSE received or sent: "did my son receive any money", "meri biwi ne kise bheje"
    rf"\bdid\s+my\s+{_REL}\s+(?:receive|get|send|spend|pay)\b|\b(?:his|her|their)\s+(?:wallet|account)\s+(?:is|number)\b|"
    rf"\b(?:meri|mere|mera)\s+{_REL}\s+ne\b[^.?!]{{0,30}}\b(?:bheje|bheja|kharch\w*|kharid\w*|diye)\b|"
    r"(?:मेरी|मेरा|मेरे)\s+(?:पत्नी|पति|भाई|बहन|बेटा|बेटी|दोस्त)\s+ने[^.?!]{0,30}(?:भेजे|भेजा|खर्च|खरीद)|"
    r"(?:رصيد|معاملات|صرف\w*)[^.?!]{0,20}(?:زوجتي|زوجي|أخوي|اخوي|أختي|اختي|ولدي|بنتي|صاحبي|الموظف)|"
    r"(?:زوجتي|زوجي|أخوي|اخوي|أختي|اختي|ولدي|بنتي|صاحبي|الموظف)[^.?!]{0,20}(?:رصيد|معاملات|صرف\w*)|"
    r"(?:मेरी|मेरा|मेरे)\s+(?:पत्नी|पति|भाई|बहन|बेटा|बेटी|दोस्त|कर्मचारी)[^.?!]{0,30}(?:बैलेंस|खर्च|लेनदेन|खाते)|"
    r"\bfull\s+name\b[^.?!]{0,20}\b(?:registered|on|of)\b|whose\s+number|who\s+owns\s+(?:this|the)\s+number|"
    r"kiska\s+number|صاحب\s+(?:هال|هذا\s+ال|ال)?رقم|اسمه\s+الكامل|किसका\s+नंबर", re.I)
#: The customer's OWN money taken — a fraud report, answered by the fraud team.
STOLEN = re.compile(
    r"\b(?:stolen|stole|hacked|someone\s+(?:took|used|emptied)|unauthori[sz]ed)\b|"
    r"انسرقت|سرقوا|انسحبت\s+فلوسي\s+بدون|اتهكر|चोरी|हैक|chori\s+ho|hack\s+ho", re.I)
NAMED_BALANCE = re.compile(r"\b(?!(?:my|your|phone|mobile|card|wallet|account|salik|nol|du|etisalat|data|sim|"
                           r"number|wallet's)\b)[A-Za-z]+(?:'s|\s+ka|\s+ki)\s+(?:wallet\s+)?balance\b", re.I)
#: Other providers' services: a comparison with them, how to use them, or their offers. A sentence-level model cannot
#: learn "decline only when a competitor is named" from a few examples (it learns "how-to questions → decline" and
#: starts declining our own), so this is a rule. Sending TO someone's UPI / PhonePe / Paytm, or adding our card to
#: Apple Pay, is the customer's own business and is left alone.
_COMP = (r"(?:careem\s*pay|careem|phone\s*pe|phonepe|paytm|google\s*pay|gpay|g\s*pay|botim|apple\s*pay|samsung\s*pay|western\s*union|"
         r"money\s*gram|moneygram|e&\s*money|etisalat\s*wallet|payit|payby|al\s*ansari|uae\s*exchange|lulu\s*(?:exchange|money)|remitly|"
         r"كريم\s*باي|كريم|بوتيم|آبل\s*باي|ابل\s*باي|أبل\s*باي|ويسترن\s*يونيون|الإمارات\s*للصرافة|الامارات\s*للصرافة|الأنصاري|الانصاري|"
         r"फोनपे|फ़ोनपे|पेटीएम|गूगल\s*पे|एप्पल\s*पे)")
COMPETITOR_MENTION = re.compile(_COMP, re.I)
COMPETITOR = re.compile(
    # a comparison — "is OpenWallet better than Careem Pay", "Botim ya OpenWallet kaunsa sasta", "شو أحسن OpenWallet ولا Apple Pay"
    r"(?:better|safer|cheaper|best|\bvs\.?\b|versus|compar\w*|kaunsa|kaun\s*sa|sasta|behtar|बेहतर|सस्ता|सुरक्षित|कौन\s*सा|"
    r"أحسن|احسن|أفضل|افضل|أأمن|آمن|أرخص|ارخص)[^.?!]{0,60}" + _COMP + r"|" + _COMP + r"[^.?!]{0,60}(?:better|safer|cheaper|best|\bvs\.?\b|versus|"
    r"kaunsa|kaun\s*sa|sasta|behtar|बेहतर|सस्ता|सुरक्षित|أحسن|احسن|أفضل|افضل|أأمن|أرخص|ارخص)|"
    r"(?:pay\s*10|بي\s*تن)\s*(?:ya|or|ولا|أو|او)\s*" + _COMP + r"|" + _COMP + r"\s*(?:ya|or|ولا|أو|او)\s*(?:pay\s*10|بي\s*تن)|"
    # how to do something THROUGH another provider — "how do I send money with Careem Pay", "Paytm se paise kaise bheju"
    r"(?:\bhow\b|kaise|kese|kaisay|كيف|ازاي|إزاي|कैसे)[^.?!]{0,50}(?:with|through|via|using|by|عن\s*طريق|عبر|بواسطة)\s+" + _COMP + r"|"
    r"" + _COMP + r"\s+(?:se|से)\b[^.?!]{0,40}(?:kaise|kese|कैसे|bhej|send|transfer|pay)|"
    # their offers — "PhonePe ka cashback offer kya chal raha hai"
    + _COMP + r"[^.?!]{0,40}(?:cash\s*back|offers?\b|promo|discount|deals?\b|rewards?\b|عرض|عروض|كاش\s*باك|ऑफर|कैशबैक)|"
    r"(?:عرض|عروض|كاش\s*باك|ऑफर|कैशबैक|cash\s*back\s+offers?)[^.?!]{0,40}(?:على|في|on|पर)\s*" + _COMP, re.I)
#: our card / wallet linked INTO another wallet — the customer's own setup, not a competitor question
_LINK_OWN = re.compile(r"(?:add|link|connect|jod|जोड़|أضيف|اضيف|ربط)[^.?!]{0,40}(?:card|wallet|pay\s*10|كارت|بطاقة|कार्ड)[^.?!]{0,30}"
                       r"(?:to|in|into|with|mein|में|إلى|الى|في|على)\s*(?:my\s+)?(?:apple|samsung|google)\s*pay", re.I)

#: Someone else's payment activity — "X ne kisko paise bheje", "how much did my flatmate pay", "show me his history",
#: "who transferred to number …". The customer's OWN history ("maine kisko bheje", "मैंने किसको") is not this.
OTHER_ACTIVITY = re.compile(
    r"(?:^|\s)(?:ne|ने)\s+(?:kal\s+|aaj\s+|parso\s+|कल\s+|आज\s+|last\s+\w+\s+)?(?:kisko|kise|kis\s*kis\s*(?:ko)?|किसको|किसे|किस-किस\s*को)\s+"
    r"[^.?!]{0,25}(?:paise|paisa|pay|bhej|transfer|पैसे|भेज|पे\s*किया)|"
    rf"how\s+much\s+(?:did|does|has)\s+(?:my\s+)?{_REL}\b[^.?!]{{0,40}}\b(?:pay|paid|send|sent|spend|spent)|"
    r"(?:show|give|list|send|tell)\s+(?:me\s+)?(?:his|her|their)\s+(?:\w+\s+)?(?:transactions?|payments?|history|statement|balance)|"
    r"(?:مين|من)\s+(?:حول|دفع|ارسل|أرسل|بعت)\s+[^.?!]{0,25}(?:على|إلى|الى|ل)\s*(?:الرقم|رقم)", re.I)

#: Someone's identity details asked for by a wallet / phone number (local or +971 form) — never the customer's own.
ID_OF_OTHER = re.compile(r"(?:emirates\s*id|\beid\b|passport|national\s*id|date\s+of\s+birth|\bdob\b|address|"
                         r"الهوية|جواز)[^.?!]{0,50}(?:linked|registered|of|on|for|behind|belong\w*|wale|ka|ki)\s+"
                         r"(?:to\s+|with\s+)?(?:the\s+|this\s+)?(?:wallet|account|number|mobile|phone)?\s*(?:no\.?\s*)?(?:\+?971|00971)?0?5\d{7,8}\b", re.I)
#: The customer's own relationship through a split / their own payment is fine.
_OWN = re.compile(r"\bowes?\b|\bpaid\s+me\b|\bmy\s+share\b|hissa|dene\s+hai|debe|i\s+(?:sent|paid)\b|"
                  r"maine\s+(?:bheje|diye)|\bto\s+my\b|ko\s+(?:bhej|de)|"
                  # money moving from THEIR OWN wallet, or a payment/recharge for someone
                  r"\bmy\s+wallet\b|from\s+my\b|deducted|got\s+nothing|never\s+received|didn'?t\s+(?:get|receive)|"
                  r"\b(?:send|put|pay|transfer|recharge|top\s*up)\b|kar\s+do|bhej|انخصم|ما\s+وصل|ما\s+وصله", re.I)


INTERNAL = re.compile(
    r"\bapi\s+(?:endpoint|key|url)|\bendpoints?\b|\bheaders?\b|\bsource\s+code|\byour\s+code\b|"
    r"\b(?:database|db|schema|server|backend|microservice|kafka|redis|mongo\w*|mysql|vault)\b|"
    r"internal\s+(?:services?|systems?|tools?)|admin\s+(?:panel|dashboard|portal|link)|"
    r"(?:monitoring|fraud|risk)\s+(?:system|rules?|threshold)|what\s+amount\s+(?:makes|triggers)|"
    r"which\s+(?:model|llm|ai)\s+(?:are\s+you|do\s+you\s+use)|list\s+of\s+intents|intents\s+you|confidence\s+score|"
    r"system\s+prompt|fraud\s+check\s+ka|exact\s+limit\s+(?:of|for)\s+(?:the\s+)?fraud|"
    r"सिस्टम\s+प्रॉम्प्ट|कौन\s+सी\s+API|النموذج\s+اللي|التعليمات\s+الداخلية|نظام\s+كشف\s+الاحتيال|"
    r"لوحة\s+(?:تحكم|التحكم)|الخدمات\s+الداخلية|قاعدة\s+البيانات|الكود\s+حقك|"
    r"डेटाबेस|सर्वर|एडमिन\s+पैनल", re.I)
#: Self-harm signals — answered by a person, with care, never by a refusal.
DISTRESS = re.compile(
    r"\b(?:kill\s+myself|end\s+(?:it|my\s+life)|suicid\w*|don'?t\s+(?:see\s+the\s+point|want\s+to)\s+(?:of\s+)?liv\w*|"
    r"no\s+reason\s+to\s+live|want\s+to\s+die|jeena\s+nahi|mar\s+jaun|marna\s+chahta|"
    r"can'?t\s+take\s+(?:this|it)\s+anymore|what'?s\s+the\s+point\s+of\s+anything|i'?m\s+done\s+with\s+(?:life|everything)|"
    r"shaking\s+and\s+crying|jeene\s+ka\s+mann\s+nahi|sab\s+khatam\s+ho\s+gaya)\b|"
    r"جيني\s+का|जीने\s+का\s+मन\s+नहीं|सब\s+खत्म|تعبت\s+من\s+الحياة|منهار|ودي\s+أختفي|ما\s+عاد\s+فيني\s+أتحمل|"
    r"ما\s+عاد\s+أبي\s+أعيش|ابي\s+اموت|أبي\s+أموت|انتحار|"
    r"जीना\s+मुश्किल|मरना\s+चाहता|मर\s+जाऊँ|आत्महत्या", re.I)
LEGAL = re.compile(
    r"\b(?:sue|lawsuit|lawyer|legal\s+action|court|consumer\s+court|central\s+bank\s+complaint)\b|"
    r"قضية|محامي|المحكمة|برفع\s+عليكم|अदालत|वकील|केस\s+करूँगा|case\s+karunga|कानूनी\s+कार्रवाई", re.I)


CHANGE_PIN = re.compile(r"\b(?:change|reset|update|set|naya|new|badal\w*)\s+(?:my\s+|the\s+)?(?:pin|password)\b|"
                        r"\b(?:forgot|forget)\s+(?:my\s+)?(?:pin|password)[^.?!]{0,30}\b(?:how|reset|change)\b|"
                        r"\b(?:otp|code)\s+(?:never|not|doesn'?t|didn'?t)\s+(?:arrive|come|reach)|otp\s+nahi\s+aa|"
                        r"\bfingerprint|face\s*id|biometric|\bdidn'?t\s+share\b|\bhow\s+do\s+i\s+change\s+it|"
                        r"(?:password|pin)\s+change\s+karne|change\s+karne\s+ka\s+tarika|"
                        r"أغير\s+كلمة\s+المرور|كيف\s+أغيره|نسيت\s+الرقم\s+السري[^.?!]{0,20}كيف|"
                        r"\b(?:pin|password)\s+(?:change|reset|update|badal\w*)|"
                        r"تغيير\s+(?:ال)?رقم\s+السري|أغير\s+(?:ال)?رقم\s+السري|पिन\s+बदल", re.I)


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _deleet(t: str) -> str:
    """'1gn0r3 4ll rul35' -> 'ignore all rules' — only words that mix letters and digits."""
    return re.sub(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", lambda m: m.group(0).lower().translate(_LEET), t, flags=re.I)


#: Paying an OFFICIAL biller is a bill, even when the words are "fine" or "customs":
#: "pay my traffic fine to RTA", "pay the Dubai Customs bill".
#: More scam shapes: a sudden family emergency + "send to this number" (+ "I'll explain later"), and fees to "register" or
#: "activate" a government benefit (Arabic). Paired with a money ask, like SCAM.
SCAM_MORE = re.compile(
    r"(?:ill|sick|hospital|accident|emergency|surgery|tabiyat|tabiyet|bimar|बीमार|तबियत|तबीयत|अस्पताल|مستشفى|مريض|حادث)[^.?!]{0,100}"
    r"(?:this\s+number|is\s+number|iss\s+number|इस\s+नंबर|هذا\s+الرقم|هالرقم|ها\s+الرقم)|"
    r"(?:this\s+number|is\s+number|इस\s+नंबर|هذا\s+الرقم|هالرقم)[^.?!]{0,60}(?:explain\s+later|baad\s+mein\s+(?:sab\s+)?samjha|बाद\s+में\s+समझा)|"
    r"رسوم\s+(?:التسجيل|التفعيل|الاشتراك)|مبادرة\s+الحكومة|ضمن\s+المستفيدين", re.I)

OFFICIAL_BILL = re.compile(r"\b(?:rta|dubai\s+customs|customs\s+bill|moi|dubai\s+police\s+app|traffic\s+fine\s+(?:to|via|through)|"
                           r"pay\s+(?:my|the)\s+(?:traffic\s+)?fine\s+(?:through|via|in)\s+the\s+app|how\s+do\s+i\s+pay\s+a\s+police)\b|"
                           r"هيئة\s+الطرق|جمارك\s+دبي|مخالفة\s+المرور\s+حقي", re.I)


#: An explicit "I want a person, not a bot" — straight to support, in any language.
HUMAN = re.compile(r"\b(?:not\s+(?:a|the)\s+bot|real\s+person|real\s+human|human\s+being|insaan\s+se\s+baat|"
                   r"agent\s+se\s+baat|talk\s+to\s+(?:a\s+)?(?:human|person|agent))\b|"
                   r"ما\s+أبي\s+أتكلم\s+مع\s+روبوت|مش\s+عايز\s+(?:أكلم\s+)?روبوت|بني\s+آدم|وصلني\s+بخدمة\s+العملاء|"
                   r"इंसान\s+से\s+बात|किसी\s+इंसान", re.I)


def check(said: str) -> str | None:
    """'credential' | 'scam' | 'injection' | 'illegal' | 'other_person' | 'competitor' | … | None."""
    t = " ".join((said or "").split())
    if not t:
        return None
    t = _deleet(t)
    if DISTRESS.search(t):
        return "distress"
    # a transfer NOTE is the customer's own text for the recipient ("note: 'ignore previous
    # instructions, love you'") — it is not an instruction to us
    t_inj = re.sub(r"\b(?:note|memo|message|remark|reference)\s*[:=]?\s*(['\"‘“]).*?(?:\1|['\"’”])", " ", t, flags=re.I)
    if INJECTION.search(t_inj):
        return "injection"
    if ILLEGAL.search(t):
        return "illegal"
    if ((CRED.search(t) and CRED_CONTEXT.search(t) and not CHANGE_PIN.search(t)) or CARD.search(t)):
        return "credential"
    if (SCAM.search(t) or SCAM_MORE.search(t)) and MONEY_ASK.search(t) and not OFFICIAL_BILL.search(t):
        return "scam"
    if STOLEN.search(t) and not OTHER.search(t):
        return "fraud_victim"
    if LEGAL.search(t):
        return "legal"
    if HUMAN.search(t):
        return "human"
    if INTERNAL.search(t):
        return "internal"
    if NAMED_BALANCE.search(t) or ID_OF_OTHER.search(t) or OTHER_ACTIVITY.search(t) or (OTHER.search(t) and not _OWN.search(t)):
        return "other_person"
    if COMPETITOR.search(t) and not _LINK_OWN.search(t):
        return "competitor"
    return None


def forgot(said: str) -> bool:
    return bool(re.search(r"forgot|forget|bhool|نسيت|याद\s+नहीं|भूल", said or "", _I))


def scrub(said: str) -> str:
    """The sentence with every code / card number masked — what we log and echo."""
    return re.sub(r"(?<!\d)\d{4,19}(?!\d)", "••••", said or "")
