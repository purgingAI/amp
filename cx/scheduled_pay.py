"""Payments AT a time: "pay Alex 100 in 2 minutes".

The customer confirms and enters the PIN NOW. The wallet verifies the PIN at that moment and
the payment is authorised; what we keep until the due time is the authorised ACTION — payee,
amount, the sentence — and the customer's login, never the PIN. When the time comes the same
`cx_act` path runs it as that customer, exactly as if they had pressed confirm then.

Deliberately narrow:
  * in memory only, like logins: a restart drops scheduled payments together with the logins
    they run on. The Redis copy exists only so a restarted server can TELL the customer their
    payment did not run (and nothing moved) instead of losing it silently;
  * at most SCHEDULE_MAX_MIN ahead (engine.SCHEDULE_MAX_MIN), inside the 8-hour login;
  * a payment due after its login has expired is NOT run — the customer is told;
  * cancellable until it fires (`cancel`), and claimed once (a dict pop under a lock).
"""
from __future__ import annotations

import datetime as dt
import json
import secrets
import threading
import time

import userctx

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_RKEY = "openwalletcx:schedpay:"


def _redis():
    try:
        import reminders_store
        return reminders_store.redis()
    except Exception:                                              # noqa: BLE001
        return None


def _push(user_id: str, title: str, body: str, log=print) -> None:
    """Best-effort app push, same rail as reminders."""
    try:
        import reminders_store as rs
        token = rs._fcm_token(user_id)
        if not token:
            log(f"no push token for {user_id[:8]} — {body}")
            return
        msg = {"type": "APP-NOTIFY", "udf1": token, "udf2": title, "udf3": body, "udf4": "INFO",
               "udf6": "CUSTOMER", "notificationType": "INFO", "checkConfiguration": True}
        rs._producer().send(rs.TOPIC, value=msg, headers=[("__TypeId__", rs.TYPE_ID)]).get(timeout=15)
    except Exception as e:                                         # noqa: BLE001
        log(f"push failed ({type(e).__name__}) — {body}")


def schedule(req: dict, max_min: int) -> dict:
    """Hold an authorised payment until req['scheduleAt']. Returns the reply for the app."""
    due = dt.datetime.fromisoformat(req["scheduleAt"])
    now = dt.datetime.now(dt.timezone.utc)
    mins = (due - now).total_seconds() / 60
    if mins > max_min + 2:
        return {"ok": False, "error": f"I can schedule a payment up to {max_min // 60} hours ahead."}
    user = dict(userctx.current())
    jid = "spay_" + secrets.token_hex(6)
    action = {k: v for k, v in req.items() if k not in ("scheduleAt", "pin")}
    job = {"id": jid, "due": due.timestamp(), "user": user, "action": action}
    with _LOCK:
        _JOBS[jid] = job
    r = _redis()
    if r is not None:
        try:
            r.set(_RKEY + jid, json.dumps({"user_id": user.get("user_id"), "due": due.isoformat(),
                                           "amount": action.get("amount"),
                                           "who": action.get("payee") or action.get("merchant")}),
                  ex=int(max(60, (due - now).total_seconds() + 86400)))
        except Exception:                                          # noqa: BLE001
            pass
    return {"ok": True, "scheduled": True, "id": jid, "due": due.isoformat()}


def mine(user_id: str) -> list[dict]:
    with _LOCK:
        return [{"id": j["id"], "due": j["due"], "amount": j["action"].get("amount"),
                 "who": j["action"].get("payee") or j["action"].get("merchant")}
                for j in _JOBS.values() if j["user"].get("user_id") == user_id]


def cancel(user_id: str, jid: str) -> bool:
    with _LOCK:
        j = _JOBS.get(jid)
        if not j or j["user"].get("user_id") != user_id:
            return False
        _JOBS.pop(jid, None)
    r = _redis()
    if r is not None:
        try:
            r.delete(_RKEY + jid)
        except Exception:                                          # noqa: BLE001
            pass
    return True


def _fire(job: dict, run, log) -> None:
    user, act = job["user"], job["action"]
    who = act.get("payee") or act.get("merchant") or "your payee"
    amt = act.get("amount")
    uid = user.get("user_id") or ""
    # no "at" = the local configured customer, who has no login to expire
    if user.get("at") is not None and time.time() - float(user["at"]) > userctx.LOGIN_TTL:
        log(f"{job['id']}: login expired before {who} AED {amt} was due — not run")
        _push(uid, "Scheduled payment not sent",
              f"Your scheduled payment of AED {amt} to {who} did not run because your session "
              f"expired. Nothing was sent.", log)
        return
    try:
        out = userctx.run_as(user, run, dict(act, _pin_ok=True)) or {}
    except Exception as e:                                         # noqa: BLE001
        out = {"ok": False, "error": type(e).__name__}
    ok = bool(out.get("ok"))
    log(f"{job['id']}: AED {amt} to {who} — " + ("started" if ok else f"failed: {out.get('error')}"))
    _push(uid, "Scheduled payment" if ok else "Scheduled payment failed",
          (f"Your scheduled payment of AED {amt} to {who} is going through now."
           if ok else f"Your scheduled payment of AED {amt} to {who} could not be sent "
                      f"({out.get('error') or 'unknown error'}). Nothing was sent."), log)


def start(run, interval: float = 5.0, log=print) -> threading.Thread:
    """Release due payments through `run(action)` (the server's cx_act)."""
    # A restart lost every held payment along with its login: tell those customers.
    r = _redis()
    if r is not None:
        try:
            for k in r.scan_iter(_RKEY + "*"):
                rec = json.loads(r.get(k) or "{}")
                r.delete(k)
                if rec.get("user_id"):
                    _push(rec["user_id"], "Scheduled payment not sent",
                          f"Your scheduled payment of AED {rec.get('amount')} to {rec.get('who')} "
                          f"was cancelled by a service restart. Nothing was sent.", log)
        except Exception:                                          # noqa: BLE001
            pass

    def loop():
        while True:
            now = time.time()
            with _LOCK:
                due = [j for j in _JOBS.values() if j["due"] <= now]
                for j in due:
                    _JOBS.pop(j["id"], None)              # claimed exactly once
            for j in due:
                if r is not None:
                    try:
                        r.delete(_RKEY + j["id"])
                    except Exception:                              # noqa: BLE001
                        pass
                _fire(j, run, log)
            time.sleep(interval)

    t = threading.Thread(target=loop, name="scheduled-pay", daemon=True)
    t.start()
    return t
