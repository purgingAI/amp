"""Reminders: kept in Redis with a TTL, and delivered as a real app push when they fall due.

    set     "remind me to pay Alex in 5 minutes"
              -> openwalletcx:reminder:<id>          the reminder (JSON), TTL = until due + 1 day
              -> openwalletcx:reminders:due          sorted set, score = due time (epoch seconds)
              -> openwalletcx:reminders:user:<uid>   sorted set, this customer's reminders
    fire    a loop in the engine (every 10 s) takes everything due from the sorted set. ZREM
            is the claim: whichever process removes the id is the one that sends it, so two
            engines on the same Redis (laptop + server) never notify twice.
    notify  exactly what the wallet's own services publish for a push
            (SenderValidator.sendAppPushNotification): a NotificationRequestDTO on the
            `notification-response` topic, type APP-NOTIFY, udf1 = the customer's FCM token,
            udf2/udf3 = title/body, udf4 = INFO (a PushNotificationType the Notification
            service accepts — there is no REMINDER value, and an unknown one throws),
            udf6 = CUSTOMER, udf7 = mobile, checkConfiguration = true so the customer's own
            notification settings are respected. Spring's __TypeId__ header is set, or the
            consumer rejects the payload as untyped.

A reminder NOTIFIES. It never pays — paying from it goes through the normal confirmation
card and PIN (BRD 2.6 M8).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PREFIX = "openwalletcx:"
DUE = PREFIX + "reminders:due"
GRACE = 24 * 3600            # a reminder lives this long after it was due, then Redis drops it
TOPIC = "notification-response"
TYPE_ID = b"com.openwallet.utility.dto.NotificationRequestDTO"

_R = None
_P = None
_LOCK = threading.Lock()


class StoreDown(Exception):
    pass


def _cfg() -> dict:
    from devdb import _vault_config
    return _vault_config()


def redis():
    global _R
    if _R is None:
        import redis as _redis
        c = _cfg()
        host = os.environ.get("CX_REDIS_HOST") or c.get("redis.standalone.host") or "127.0.0.1"
        port = int(os.environ.get("CX_REDIS_PORT") or c.get("redis.standalone.port") or 6379)
        _R = _redis.Redis(host=host, port=port, socket_timeout=3, socket_connect_timeout=3,
                          decode_responses=True)
    return _R


def _key(rid: str) -> str:
    return f"{PREFIX}reminder:{rid}"


def _ukey(uid: str) -> str:
    return f"{PREFIX}reminders:user:{uid}"


def add(user: dict, what: str, due: dt.datetime, recur: str | None = None,
        amount: float | None = None) -> dict:
    """Store one reminder; returns it. Raises StoreDown if Redis is unreachable."""
    rid = "rmd_" + secrets.token_hex(6)
    due_ts = int(due.timestamp())
    rec = {"id": rid, "what": what, "due": due_ts, "recur": recur, "amount": amount,
           "user_id": user["user_id"], "mobile": user.get("mobile"), "name": user.get("name") or "",
           "created": int(time.time())}
    ttl = max(60, due_ts - int(time.time()) + GRACE)
    try:
        r = redis()
        p = r.pipeline()
        p.set(_key(rid), json.dumps(rec), ex=ttl)
        p.zadd(DUE, {rid: due_ts})
        p.zadd(_ukey(user["user_id"]), {rid: due_ts})
        p.expire(_ukey(user["user_id"]), ttl + 90 * 24 * 3600)
        p.execute()
    except Exception as e:                                         # noqa: BLE001
        raise StoreDown(str(e)) from e
    return rec


def mine(user_id: str) -> list[dict]:
    """This customer's live reminders, soonest first. Ids whose record has expired are tidied."""
    try:
        r = redis()
        ids = r.zrange(_ukey(user_id), 0, -1)
        out, gone = [], []
        for rid in ids:
            raw = r.get(_key(rid))
            if raw:
                out.append(json.loads(raw))
            else:
                gone.append(rid)
        if gone:
            r.zrem(_ukey(user_id), *gone)
        return sorted(out, key=lambda x: x["due"])
    except Exception as e:                                         # noqa: BLE001
        raise StoreDown(str(e)) from e


def delete(user_id: str, rid: str) -> bool:
    try:
        r = redis()
        raw = r.get(_key(rid))
        if not raw or json.loads(raw).get("user_id") != user_id:
            return False                     # only the owner can delete it
        p = r.pipeline()
        p.delete(_key(rid)); p.zrem(DUE, rid); p.zrem(_ukey(user_id), rid)
        p.execute()
        return True
    except Exception as e:                                         # noqa: BLE001
        raise StoreDown(str(e)) from e


# ── delivery ─────────────────────────────────────────────────────────────────────────────
def _producer():
    global _P
    if _P is None:
        from kafka import KafkaProducer
        _P = KafkaProducer(bootstrap_servers=_cfg()["spring.kafka.bootstrap-servers"],
                           value_serializer=lambda v: json.dumps(v).encode(),
                           acks="all", linger_ms=0, request_timeout_ms=15000)
    return _P


def _fcm_token(user_id: str) -> str | None:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "openwallet"))
    from db import db_conn
    cur = db_conn().cursor()
    cur.execute("select fcm_token from user where id=%s", (user_id,))
    row = cur.fetchone() or {}
    return row.get("fcm_token") or None


def message(rec: dict, token: str) -> dict:
    body = rec["what"].rstrip(".")
    if rec.get("amount"):
        body += f" (AED {float(rec['amount']):g})"
    return {"type": "APP-NOTIFY", "udf1": token, "udf2": "Payment reminder",
            "udf3": f"Reminder: {body}. Open OpenWallet to do it — nothing is paid automatically.",
            "udf4": "INFO", "udf6": "CUSTOMER", "udf7": rec.get("mobile"),
            "notificationType": "INFO", "checkConfiguration": True,
            "recipientName": rec.get("name") or ""}


def notify(rec: dict) -> tuple[bool, str]:
    """Publish the push. (sent?, reason)."""
    token = _fcm_token(rec["user_id"])
    if not token:
        return False, "no app push token on this customer (never logged in to the app?)"
    fut = _producer().send(TOPIC, value=message(rec, token), headers=[("__TypeId__", TYPE_ID)])
    meta = fut.get(timeout=15)
    return True, f"published to {meta.topic}[{meta.partition}]@{meta.offset}"


def _next(rec: dict) -> int | None:
    """The next occurrence of a recurring reminder, or None."""
    d = dt.datetime.fromtimestamp(rec["due"])
    if rec.get("recur") == "weekly":
        return int((d + dt.timedelta(days=7)).timestamp())
    if rec.get("recur") == "monthly":
        y, m = (d.year + (d.month == 12), d.month % 12 + 1)
        for day in (d.day, 30, 29, 28):
            try:
                return int(d.replace(year=y, month=m, day=day).timestamp())
            except ValueError:
                continue
    return None


def fire_due(log=print) -> int:
    """Send everything that is due. Returns how many were sent."""
    sent = 0
    try:
        r = redis()
        ids = r.zrangebyscore(DUE, "-inf", int(time.time()), start=0, num=50)
    except Exception:                                              # noqa: BLE001
        return 0
    for rid in ids:
        if not r.zrem(DUE, rid):             # someone else claimed it
            continue
        raw = r.get(_key(rid))
        if not raw:
            continue
        rec = json.loads(raw)
        try:
            ok, why = notify(rec)
        except Exception as e:                                     # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {e}"
        log("reminder", rid, rec["what"], "→", "SENT" if ok else "NOT SENT", why)
        sent += ok
        nxt = _next(rec)
        if nxt:
            rec["due"] = nxt
            ttl = max(60, nxt - int(time.time()) + GRACE)
            p = r.pipeline()
            p.set(_key(rid), json.dumps(rec), ex=ttl)
            p.zadd(DUE, {rid: nxt}); p.zadd(_ukey(rec["user_id"]), {rid: nxt})
            p.execute()
        else:
            rec["fired"] = int(time.time())
            r.set(_key(rid), json.dumps(rec), ex=GRACE)   # kept a day so "show my reminders" can say it fired
            r.zrem(_ukey(rec["user_id"]), rid)
    return sent


def start_scheduler(interval: float = 10.0, log=print) -> threading.Thread:
    def loop():
        while True:
            try:
                fire_due(log)
            except Exception as e:                                 # noqa: BLE001
                log("reminder scheduler:", type(e).__name__, e)
            time.sleep(interval)
    t = threading.Thread(target=loop, name="reminders", daemon=True)
    t.start()
    return t
