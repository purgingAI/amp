"""Real OpenWallet dev transaction client, as a library for the PayI -> OpenWallet demo.

Everything here mirrors the app exactly; the contract details that are easy to get
wrong carry a comment, because each one cost a round trip to discover.

Envelope (EncryptionAdvice + EncryptionUtilsImpl): AES/GCM/NoPadding, 128-bit tag,
no AAD, Base64, IV = the first 16 CHARS of the key as UTF-8.
    K       = 32-char random key
    encKey  = AESGCM(K,             key=staticEncKey, iv=staticEncKey[:16])
    encData = AESGCM(json(payload), key=K,            iv=K[:16])
"""
import base64, hashlib, hmac, json, os, secrets, time, urllib.error, urllib.request

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS = os.path.join(HERE, ".secrets.json")
BASE = os.environ.get("OPENWALLET_BASE", "https://api.example.com")
PATH = "/transaction/ext/transact"


def cfg():
    return json.load(open(SECRETS))


_KEYS: dict = {}


def me() -> dict:
    """The customer this request is for: {user_id, mobile, name, staticEncKey}.

    The configured demo sender reads `.secrets.json` exactly as before. Anyone who logged in
    through the app is resolved from their login (user id, mobile) and their wallet's own key
    record — `globalWalletKeyMapper.staticEncKey`, the same key the app's envelope uses and
    the same record `.secrets.json` was originally copied from.
    """
    import sys
    cx = os.path.join(os.path.dirname(HERE), "cx")
    if cx not in sys.path:
        sys.path.insert(0, cx)
    import userctx
    u = userctx.current()
    c = cfg()
    if u.get("configured") or u.get("user_id") == c.get("sender"):
        p = c["people"][c["sender"]]
        return {"user_id": c["sender"], "mobile": p["mobile"], "name": p.get("name"),
                "staticEncKey": p["staticEncKey"]}
    uid = u["user_id"]
    if uid not in _KEYS:
        import history
        db, _ = history._mongo()
        doc = db["globalWalletKeyMapper"].find_one({"entityKey": uid}) or {}
        if not doc.get("staticEncKey"):
            raise RuntimeError("no wallet key record for this customer")
        _KEYS[uid] = doc["staticEncKey"]
    return {"user_id": uid, "mobile": u.get("mobile"), "name": u.get("name"),
            "staticEncKey": _KEYS[uid]}


def token_for(who: dict | None = None, ttl: int = 900) -> str:
    """The HS512 customer token the wallet services verify, for `who` (default: me())."""
    who = who or me()
    now = int(time.time())
    return jwt_hs512({"sub": who["user_id"], "username": who["mobile"], "userKey": who["user_id"],
                      "role": "CUSTOMER", "role_type": "CUSTOMER", "log": 3, "locale": "en",
                      "iat": now, "exp": now + ttl}, cfg()["jwt_secret"])


# ---------- crypto ----------

def gcm_encrypt(plaintext: str, key: str) -> str:
    e = Cipher(algorithms.AES(key.encode()),
               modes.GCM(key[:16].encode(), min_tag_length=16)).encryptor()
    return base64.b64encode(e.update(plaintext.encode()) + e.finalize() + e.tag).decode()


def gcm_decrypt(blob_b64: str, key: str) -> str:
    raw = base64.b64decode(blob_b64)
    body, tag = raw[:-16], raw[-16:]
    d = Cipher(algorithms.AES(key.encode()),
               modes.GCM(key[:16].encode(), tag, min_tag_length=16)).decryptor()
    return (d.update(body) + d.finalize()).decode()


def jwt_hs512(claims: dict, secret: str) -> str:
    """RequestContext verifies with Algorithm.HMAC512(secretKey.getBytes()) — the secret's
    RAW UTF-8 bytes. Base64-decoding it yields 'JWT has been tampered'."""
    def seg(o):
        return base64.urlsafe_b64encode(json.dumps(o, separators=(",", ":")).encode()).rstrip(b"=")
    signing = seg({"alg": "HS512", "typ": "JWT"}) + b"." + seg(claims)
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), signing, hashlib.sha512).digest()).rstrip(b"=")
    return (signing + b"." + sig).decode()


# ---------- the call ----------

def build_payload(amount, reference=None, purpose="FAMILY_SUPPORT", remarks="PayI voice payment",
                  receiver_mobile=None, discount=None, product=None, bill=None):
    """`receiver_mobile` overrides the configured peer so a MERCHANT can be paid; `discount`
    is the quote the Incentive Engine returned, and it is passed through rather than
    recomputed — the engine decides what comes off, we only report it."""
    c = cfg()
    who = me()
    if not receiver_mobile and not bill and not who.get("configured_receiver_ok", who["user_id"] == c.get("sender")):
        # The configured peer belongs to the configured demo sender only. Any other customer
        # must name who they are paying; we never default a real customer's money to it.
        raise ValueError("no receiver for this payment")
    payload = {
        # SEND_MONEY -> WalletToWalletHandler. SEND_MONEY_CUSTOMER is a real product_master
        # row but has NO handler, so it falls back to wallet-to-BANK and demands a beneficiary.
        # MERCHANT_PAYMENT when paying a merchant — WalletToWalletHandler maps both it and
        # SEND_MONEY. The product must match the one the incentive reserve was taken under,
        # because ReserveLifecycleGuard looks the reserve up per product.
        "product": product or "SEND_MONEY",
        "amount": amount,
        "totalAmount": amount,
        "country": "AE",
        "receiver": c["people"][c["receiver"]]["mobile"],   # resolved BY MOBILE, not by id
        "senderMobileNumber": who["mobile"],
        "referenceId": reference or ("PAYI" + secrets.token_hex(8).upper()),
        "purpose": purpose,
        "remarks": remarks,
        "isDiscounted": False,
        "location": {"latitude": "25.2048", "longitude": "55.2708"},
        # Deliberately NO initiatedFrom: QRCodeValidator reads any non-blank value as
        # "this came from a QR scan" and rejects the txn with "QR Token is null".
    }
    if receiver_mobile:
        payload["receiver"] = receiver_mobile
    if bill:
        # A bill payment. `product` stays BILL_PAYMENTS, which is the ROUTING key —
        # TransactionHandlerFactory sends it to WalletToUtilityHandler and then replaces the
        # product with `subProductId`, so the biller's own slab prices the transaction:
        #
        #     if (BILL_PAYMENTS.equals(transactionType))
        #         transactionRequestDTO.setProduct(transactionRequestDTO.getSubProductId());
        #
        # planId and identifier are stored on the transaction by the handler
        # (setUtilityPlanId / setUtilityIdentifier) — they are how the biller knows which
        # account was paid. There is no beneficiary and no receiver mobile in this flow.
        payload["subProductId"] = bill["subProductId"]
        payload["planId"] = bill["planId"]
        payload["identifier"] = bill["identifier"]
        payload.pop("receiver", None)
    if discount and discount.get("campaignId"):
        # DiscountValidator re-evaluates the campaign itself and requires BOTH a campaignId
        # and a positive discountAmount; without the latter it rejects with "campaignId and a
        # positive discountAmount are required", which /transact masks as a generic
        # "Something went wrong".
        #
        # `amount` here is the DISCOUNTED amount the customer actually pays. The backend
        # reconstructs the merchant's full bill as `amount + discountAmount` and re-quotes
        # that, so the two must be consistent with what the customer was shown or it fails
        # with "Discount amount mismatch".
        payload["isDiscounted"] = True
        payload["campaignId"] = discount["campaignId"]
        if discount.get("discountAmount") is not None:
            payload["discountAmount"] = float(discount["discountAmount"])
        if discount.get("promoCode"):
            payload["promoCode"] = discount["promoCode"]
    return payload


def transact(amount, on_stage=None, receiver_mobile=None, discount=None, product=None,
             bill=None):
    """Run one real transaction. on_stage(name, detail) fires as each step completes.

    `receiver_mobile` pays a merchant instead of the configured peer; `discount` carries the
    Incentive Engine's quote so the transaction is flagged isDiscounted against that campaign.
    """
    def stage(name, **detail):
        if on_stage:
            on_stage(name, detail)

    c = cfg()
    who = me()
    receiver = c["people"][c["receiver"]]
    static_key = who["staticEncKey"]

    payload = build_payload(amount, receiver_mobile=receiver_mobile, discount=discount,
                            product=product, bill=bill)
    if receiver_mobile:
        receiver = {"mobile": receiver_mobile, "name": (discount or {}).get("payeeName", "MERCHANT")}
    stage("payload.build", product=payload["product"], amount=amount,
          reference=payload["referenceId"], receiver="***" + receiver["mobile"][-3:],
          isDiscounted=payload["isDiscounted"],
          campaignId=payload.get("campaignId", "-"),
          discountAmount=payload.get("discountAmount", "-"))

    K = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
    envelope = {"encKey": gcm_encrypt(K, static_key),
                "encData": gcm_encrypt(json.dumps(payload), K)}
    stage("envelope.encrypt", algorithm="AES-256-GCM", layers=2, iv="key[:16]",
          enc_data=envelope["encData"][:44] + "…")

    token = token_for(who)
    stage("jwt.sign", algorithm="HS512",
          claims="username, userKey, role, log, locale")

    req = urllib.request.Request(BASE + PATH, data=json.dumps(envelope).encode(), method="POST")
    for h, v in (("Content-Type", "application/json"), ("Authorization", "Bearer " + token),
                 ("x-latitude", "25.2048"), ("x-longitude", "55.2708"),
                 ("x-device-id", "payi-laptop"), ("X-Device-Model", "PayI-Demo"),
                 ("X-Manufacturer", "OpenWallet"), ("Accept", "application/json")):
        req.add_header(h, v)
    stage("http.request", method="POST", url=BASE + PATH)

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw, code = r.read().decode(), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode(), e.code
    ms = int((time.time() - t0) * 1000)
    stage("http.response", status=code, ms=ms, bytes=len(raw))

    body = json.loads(raw)
    if body.get("encData"):
        # The RESPONSE is wrapped the same way the request was: EncryptionAdvice does
        # encResponseKey = encrypt(decResponseKey, staticEncKey), so unwrap encKey first
        # and use THAT to open encData. Using the static key directly fails with InvalidTag.
        resp_key = gcm_decrypt(body["encKey"], static_key) if body.get("encKey") else K
        inner = json.loads(gcm_decrypt(body["encData"], resp_key))
    else:
        inner = body
    stage("envelope.decrypt", status=inner.get("status"), message=inner.get("message"))
    return inner


if __name__ == "__main__":
    import sys
    amt = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    out = transact(amt, on_stage=lambda n, d: print("  %-18s %s" % (n, d)))
    print(json.dumps(out, indent=2)[:1200])
