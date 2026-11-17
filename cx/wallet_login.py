"""Real customer login against the wallet's own User service — the calls the OpenWallet app makes.

    1. GET  /user/otp            headers x-mobile, x-device-id          -> OTP token (UUID)
    2. POST /user/login          {"encData": RSA-OAEP-SHA256(json)}     -> access/refresh token
                                 json = {"otp", "encKey" (32 chars, ours), "token"}
                                 response sealed with that encKey
    3. POST /user/pin-validate   ENC_KEY envelope {"pin"} + Bearer      -> token again; the
                                 wallet checks the PIN hash, counts attempts, freezes at 5

Mirrors `customer-app-global`: enter_otp_remote_data_source.dart (login body),
IncryptionFunction.dart (RSA OAEP SHA-256 with the app's `public_key 1.der`),
header_builder.dart (x-mobile, x-device-id, Accept-Language, Authorization).

Side effect worth knowing: step 2 registers OUR encKey for the customer, exactly as a new
phone does, so the phone app that was logged in is logged out. That is the wallet's own
one-device rule, not something this module adds.

Nothing here stores the PIN or the OTP. The wallet tokens live only in the server's memory
(userctx) for the length of the login.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "openwallet"))

import openwallet_api as _api  # noqa: E402

BASE = os.environ.get("OPENWALLET_BASE", _api.BASE)
DEVICE = os.environ.get("CX_LOGIN_DEVICE_ID", "openwallet-assistant-demo")
PUBLIC_KEY = os.environ.get("CX_LOGIN_PUBLIC_KEY", os.path.join(HERE, "app", "certs", "wallet_public_key.der"))


class LoginError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _headers(mobile: str, token: str | None = None, user_id: str | None = None) -> dict:
    # As the app's dio interceptor (injection_container_common.dart): x-mobile, x-device-id,
    # x-usertype CUSTOMER, Accept-Language, and — on every non-public call once logged in —
    # x-user-id. The PIN service builds its user lookup from these.
    h = {"x-mobile": mobile, "x-device-id": DEVICE, "x-usertype": "CUSTOMER",
         "Content-Type": "application/json", "Accept-Language": "en", "Accept": "application/json",
         "X-Device-Model": "OpenWallet-Assistant", "X-Manufacturer": "OpenWallet"}
    if token:
        h["Authorization"] = "Bearer " + token
    if user_id:
        h["x-user-id"] = user_id
    return h


def _http(method: str, path: str, headers: dict, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, OSError) as e:
        raise LoginError("The OpenWallet login service could not be reached. Try again in a moment.") from e


def _inner(d: dict) -> dict:
    """{response:{status,data,message}} or {status,data,message} -> the inner object."""
    return d.get("response") if isinstance(d.get("response"), dict) else d


def _open(d: dict, key: str) -> dict:
    """Unseal a response: encKey = AES(responseKey, our key), encData = AES(json, responseKey)."""
    if not d.get("encData"):
        return _inner(d)
    rk = _api.gcm_decrypt(d["encKey"], key) if d.get("encKey") else key
    return _inner(json.loads(_api.gcm_decrypt(d["encData"], rk)))


def _seal(payload: dict, key: str) -> dict:
    k = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
    return {"encKey": _api.gcm_encrypt(k, key), "encData": _api.gcm_encrypt(json.dumps(payload), k)}


def _rsa(plaintext: str) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    pub = serialization.load_der_public_key(open(PUBLIC_KEY, "rb").read())
    ct = pub.encrypt(plaintext.encode(), padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                      algorithm=hashes.SHA256(), label=None))
    return base64.b64encode(ct).decode()


def _message(inner: dict, fallback: str) -> str:
    m = (inner or {}).get("message")
    return m if isinstance(m, str) and m and len(m) < 160 else fallback


def normalise_mobile(raw: str) -> str | None:
    """The 9 digits starting with 5, EXACTLY as the OpenWallet app sends them in x-mobile.

    Not "971…". The app's own validator is ^(0)?(50|52|…)[0-9]{7}$ and the wallet stores the
    9-digit form. "971544244401" is a DIFFERENT number to the wallet, and /user/login CREATES
    a customer for a number it does not know — the first version of this created a duplicate
    dev account that way, which then had no PIN ("User Info Not Present")."""
    d = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if d.startswith("00971"):
        d = d[2:]
    if d.startswith("971"):
        d = d[3:]
    if d.startswith("0"):
        d = d[1:]
    return d if len(d) == 9 and d.startswith("5") else None


def send_otp(mobile: str) -> str:
    code, d = _http("GET", "/user/otp", _headers(mobile))
    inner = _inner(d)
    if code == 425:
        raise LoginError("An OTP was sent less than 90 seconds ago — use that one, or wait a moment.", code)
    if code == 451:
        raise LoginError("This number isn't registered with OpenWallet.", code)
    tok = inner.get("data") if isinstance(inner, dict) else None
    if code != 200 or not isinstance(tok, str) or not tok:
        raise LoginError(_message(inner, "Could not send the OTP. Check the number and try again."), code)
    return tok


def verify_otp(mobile: str, otp: str, token: str) -> dict:
    """-> {"encKey", "accessToken", "refreshToken", "userId"}"""
    enc_key = base64.b64encode(secrets.token_bytes(32)).decode()[:32]
    body = {"encData": _rsa(json.dumps({"otp": otp, "encKey": enc_key, "token": token}))}
    code, d = _http("POST", "/user/login", _headers(mobile), body)
    try:
        inner = _open(d, enc_key)
    except Exception:                                              # noqa: BLE001
        inner = _inner(d)
    data = inner.get("data") if isinstance(inner, dict) else None
    if code != 200 or not isinstance(data, dict) or not data.get("accessToken"):
        raise LoginError(_message(inner, "That OTP is not right, or it has expired."), code)
    return {"encKey": enc_key, "accessToken": data["accessToken"],
            "refreshToken": data.get("refreshToken"),
            "userId": data.get("userId") or _claim(data["accessToken"], "userKey")}


def validate_pin(mobile: str, enc_key: str, access_token: str, pin: str, user_id: str | None = None) -> dict:
    """-> {"accessToken", "refreshToken"} on success; LoginError with the wallet's reason."""
    code, d = _http("POST", "/user/pin-validate", _headers(mobile, access_token, user_id),
                    _seal({"pin": pin, "loginEvent": True}, enc_key))
    try:
        inner = _open(d, enc_key)
    except Exception:                                              # noqa: BLE001
        inner = _inner(d)
    data = inner.get("data") if isinstance(inner, dict) else None
    if code != 200 or (isinstance(inner, dict) and inner.get("status") not in (None, 200)):
        err = LoginError(_message(inner, "That PIN is not right."), code)
        err.wallet = {"http": code, "status": (inner or {}).get("status") if isinstance(inner, dict) else None,
                      "message": _message(inner, "")}           # never the PIN
        raise err
    if isinstance(data, dict) and data.get("accessToken"):
        return {"accessToken": data["accessToken"], "refreshToken": data.get("refreshToken")}
    return {"accessToken": access_token}


def key_registered(user_id: str, enc_key: str) -> bool | None:
    """Did /user/login store OUR session key for this customer? (The PIN call is sealed with
    it; if the wallet holds a different one, it refuses before looking at the PIN.)"""
    try:
        import history
        db, _ = history._mongo()
        doc = db["globalWalletKeyMapper"].find_one({"entityKey": user_id}, {"encKey": 1}) or {}
        return doc.get("encKey") == enc_key
    except Exception:                                              # noqa: BLE001
        return None


def _claim(jwt: str, name: str):
    try:
        seg = jwt.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))).get(name)
    except Exception:                                              # noqa: BLE001
        return None


def profile(user_id: str) -> dict:
    """Name for the greeting, read-only from the wallet DB. Mobile is never re-read here."""
    try:
        from db import db_conn
        cur = db_conn().cursor()
        cur.execute("select first_name, last_name from user where id=%s", (user_id,))
        r = cur.fetchone() or {}
        import pii
        parts = []
        for k in ("first_name", "last_name"):
            v = r.get(k)
            if v:
                dv = pii.decrypt(v)
                parts.append(dv if dv else v)
        return {"name": " ".join(p for p in parts if p).strip()}
    except Exception:                                              # noqa: BLE001
        return {"name": ""}
