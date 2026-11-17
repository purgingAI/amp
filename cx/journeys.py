"""Customer-facing journey content. Hand-written, product-level, and deliberately dumb.

This is the complete answer to "tell me the flow". It is a flat dictionary of steps a
customer can see in the app, written in the second person, with no internal vocabulary in
it at all.

**Why it is not generated from the code knowledge graph.** We have a graph of this codebase
in ~/ace-go that knows module names, commit authors, file ownership and service topology.
Pointing a customer assistant at it would be the single fastest way to leak internals: one
"explain how refunds work in detail" and the model is reciting handler class names, Kafka
topics and the names of the people who wrote them. So there is no connection. The engine
never imports the graph, the graph is not reachable from any customer endpoint, and
`test_scope.py` fails the build if an internal term ever appears in a customer-visible
string. The wall is structural, not a prompt instruction — prompt instructions are a request,
and this needed a guarantee.
"""
from __future__ import annotations

JOURNEYS: dict[str, dict] = {
    "add_beneficiary": {
        "title": "Add someone new to send money to",
        "match": ("add a beneficiary", "add beneficiary", "add new beneficiary", "add someone",
                  "new recipient", "add a payee", "save a recipient", "adding someone new"),
        "steps": [
            "Open **Send Money** from the home screen.",
            "Tap **Add New Beneficiary**.",
            "Pick the country you are sending to, then the bank or wallet.",
            "Enter their account or wallet number. We check the name with their bank and "
            "show you the account title that comes back — confirm it matches who you mean.",
            "Enter their mobile number and your relationship to them.",
            "Save. They appear in your list straight away and you can send immediately.",
        ],
        "note": "If the name we get back from the bank looks wrong, stop and check the "
                "account number before sending anything.",
    },
    "kyc": {
        "title": "Complete or upgrade your KYC",
        "match": ("kyc", "do kyc", "upgrade kyc", "verify my identity", "verification",
                  "finish kyc", "complete kyc"),
        "steps": [
            "Go to **Profile → Verification**.",
            "Have your Emirates ID and passport ready.",
            "Scan the front and back of your Emirates ID.",
            "Take a selfie when prompted — it is matched against your ID photo.",
            "Submit. Most checks finish in a few minutes; some need a manual review.",
        ],
        "note": "Your sending limits go up once verification is approved.",
    },
    "send_local": {
        "title": "Send money to someone in the UAE",
        "match": ("how to send money", "how do i send money", "how to transfer money", "send money to a friend",
                  "teach me how to send money", "how can i send money", "how to pay someone", "wallet to wallet"),
        "steps": [
            "Open **Send Money** from the home screen.",
            "Pick someone from your recent people, or enter their mobile number.",
            "Enter the amount in dirhams — the fee, if any, is shown before you confirm.",
            "Review the summary, then confirm with your PIN.",
            "The money arrives in their OpenWallet wallet straight away; you both get a notification.",
        ],
        "note": "You can also just tell me: \"send 50 dirhams to Alex\".",
    },
    "pay_bill": {
        "title": "Pay a bill",
        "match": ("pay bill", "pay a bill", "pay my bill", "bill payment", "pay bills", "pay dewa",
                  "pay my dewa", "pay etisalat", "pay my etisalat", "pay du bill", "pay utility"),
        "steps": [
            "Open **Pay Bills** from the home screen.",
            "Choose the biller (DEWA, Etisalat, du, SEWA…) or one of your saved bills.",
            "Enter the account number the first time — it is saved for next time.",
            "Check the amount due, choose how much to pay, and confirm with your PIN.",
            "You get a receipt straight away.",
        ],
        "note": "Or just tell me: \"pay my DEWA bill\".",
    },
    "split_bill": {
        "title": "Split a bill with friends",
        "match": ("how to split", "how do i split", "split a bill", "steps to split", "where do i tap to split"),
        "steps": [
            "Pay the bill at the shop with OpenWallet as usual.",
            "Open the payment in **History** and tap **Split**.",
            "Pick the people to split with and how — equally, by share or by amount.",
            "They get a request in their app; you see who has paid as it happens.",
        ],
        "note": "Or right after paying, tell me: \"split it with Alex\".",
    },
    "send_abroad": {
        "title": "Send money to another country",
        "match": ("send money abroad", "send abroad", "international transfer", "cross border",
                  "send overseas", "send to another country", "start a transfer",
                  "sending abroad", "send money to pakistan", "send money to india",
                  "transfer money abroad", "transfer abroad", "money abroad", "send internationally",
                  "transfer to another country", "abroad", "how to transfer money abroad",
                  "how to send money abroad", "how do i send money abroad", "how do i transfer money abroad"),
        "steps": [
            "Open **Send Money** and choose a saved beneficiary, or add a new one.",
            "Enter the amount in dirhams. You will see the exchange rate, the fee and "
            "exactly what lands on the other side before you confirm anything.",
            "Choose the purpose of the transfer.",
            "Review the summary, then confirm with your PIN or biometrics.",
            "You get a reference number straight away, and a notification when it is paid out.",
        ],
        "note": "The amount your beneficiary receives is shown before you confirm, so there "
                "are no surprises at the other end.",
    },
    "qr_payment": {
        "title": "Pay a shop with a QR code",
        "match": ("scan a qr", "qr payment", "pay with qr", "scan to pay", "qr flow",
                  "pay a shop", "qr code"),
        "steps": [
            "Tap **Scan** on the home screen.",
            "Point the camera at the merchant's QR code.",
            "The merchant's name appears — check it before continuing.",
            "Enter the amount if the code does not already carry one.",
            "Any offer you qualify for is applied automatically and shown on the "
            "confirmation screen.",
            "Confirm with your PIN or biometrics.",
        ],
        "note": "Nearby merchants can also be paid without scanning at all — ask about "
                "offers near you.",
    },
    "top_up": {
        "title": "Add money to your wallet",
        "match": ("add money", "top up", "load my wallet", "add funds", "cash in",
                  "put money in", "topup", "add money to my wallet"),
        "steps": [
            "Open **Wallet → Add Money**.",
            "Choose your card, a bank transfer, or a cash agent.",
            "Enter the amount.",
            "Confirm. Card top-ups land instantly; bank transfers can take a little longer.",
        ],
    },
    "balance": {
        "title": "Check your balance",
        "match": ("check my balance", "see my balance", "my balance", "balance"),
        "steps": [
            "Your balance is on the home screen at the top.",
            "Tap it to see the full breakdown and your recent activity.",
        ],
    },
    "statement": {
        "title": "See or download your statement",
        "match": ("statement", "see my statement", "transaction history", "download statement",
                  "my transactions", "activity"),
        "steps": [
            "Open **Wallet → Statement**.",
            "Pick a date range.",
            "Tap any row for the full detail of that payment.",
            "Use **Export** to email yourself a PDF.",
        ],
    },
    "change_pin": {
        "title": "Change your PIN",
        "match": ("change my pin", "change pin", "reset pin", "new pin", "forgot pin",
                  "change the pin", "forgot my pin", "reset my pin", "pin change", "pin badal",
                  "change my password", "change password", "reset password", "reset my password",
                  "forgot password", "forgot my password", "password change", "password badal",
                  "أغير الرقم السري", "تغيير الرقم السري", "نسيت الرقم السري", "أغير كلمة المرور",
                  "كلمة المرور", "पिन बदल", "पासवर्ड बदल", "नया पिन"),
        "steps": [
            "Go to **Profile → Security → Change PIN**.",
            "Enter your current PIN, then the new one twice.",
            "If you have forgotten it, use **Forgot PIN** and verify with the code we send "
            "to your registered mobile.",
        ],
    },
    "refund_how": {
        "title": "How refunds work",
        "match": ("how do refunds work", "refund process", "how to get a refund",
                  "refund policy"),
        "steps": [
            "If a transfer fails before the money leaves us, it returns to your wallet "
            "automatically — usually within minutes.",
            "If it already reached the payout partner, we have to confirm with them first "
            "before returning it.",
            "You can ask for a refund from the transaction itself, or just ask here.",
            "Every refund is reviewed and approved by our team before the money moves back.",
        ],
        "note": "You will always get a reference and a notification when a refund completes.",
    },
}


def find(phrase: str) -> dict | None:
    """Best matching journey, or None. Longest match wins so that 'add money to my wallet'
    does not get captured by the shorter 'add money'... which it would, were the table
    scanned in insertion order."""
    text = " ".join((phrase or "").lower().split())
    if not text:
        return None
    best, best_len = None, 0
    for key, journey in JOURNEYS.items():
        for needle in journey["match"]:
            if needle in text and len(needle) > best_len:
                best, best_len = dict(journey, key=key), len(needle)
    return best


def titles() -> list[str]:
    return [j["title"] for j in JOURNEYS.values()]
