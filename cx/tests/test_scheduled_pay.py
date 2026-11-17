#!/usr/bin/env python3
"""Payments AT a time — "pay alex 100 in 2 minutes". No money moves: the release is a stub.

Run: ~/amp/.venv/bin/python cx/tests/test_scheduled_pay.py
"""
import datetime as dt
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import engine as E  # noqa: E402
import scheduled_pay  # noqa: E402

FAILED = []


def check(name, ok, got=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f"    got: {got}"))
    if not ok:
        FAILED.append(name)


def ask(t):
    return E.ask(t, want_second_opinion=False, session="sp-" + uuid.uuid4().hex[:6])


a = ask("send 100 to alex in 2 minutes")
p = a.proposal or {}
check("a payment with a time carries scheduleAt", bool(p.get("scheduleAt")), (a.outcome, p))
check("the amount is 100, not the 2 of '2 minutes'", p.get("amount") == 100, p.get("amount"))
check("the reply says when", bool(__import__("re").search(r"\d{1,2}:\d{2}", a.say or "")), a.say)
a = ask("To send 30 dirhams to Alex in next 1 minute")
check("'in next 1 minute' schedules too", bool((a.proposal or {}).get("scheduleAt")), (a.outcome, a.say[:80]))
a = ask("send 100 to alex tomorrow")
check("beyond the window offers a reminder, holds no payment",
      not a.proposal and "reminder" in (a.say or "").lower(), (a.proposal, a.say[:80]))
a = ask("send 100 to alex")
check("no time = an ordinary payment", not (a.proposal or {}).get("scheduleAt"), a.proposal)

# hold, cancel, release — with a stub instead of the real payment
ran = []
soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=2)).isoformat()
req = {"kind": "send", "payee": "Alex Morgan", "amount": 100, "scheduleAt": soon, "pin": "0000"}
got = scheduled_pay.schedule(req, E.SCHEDULE_MAX_MIN)
check("schedule holds it", got.get("ok") and got.get("id"), got)
job = scheduled_pay._JOBS.get(got["id"]) or {}
check("the PIN is never kept", "pin" not in job.get("action", {}) and "scheduleAt" not in job.get("action", {}),
      job.get("action"))
uid = job.get("user", {}).get("user_id")
check("another customer cannot cancel it", not scheduled_pay.cancel("someone-else", got["id"]))
check("its owner can", scheduled_pay.cancel(uid, got["id"]) and got["id"] not in scheduled_pay._JOBS)
got = scheduled_pay.schedule(req, E.SCHEDULE_MAX_MIN)
job = scheduled_pay._JOBS.pop(got["id"])
scheduled_pay._fire(job, lambda act: ran.append(act) or {"ok": True}, log=lambda *x: None)
check("released through the payment path, marked PIN-verified, without scheduleAt",
      len(ran) == 1 and ran[0].get("_pin_ok") and not ran[0].get("scheduleAt"), ran)
far = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).isoformat()
check("more than the window ahead is refused", not scheduled_pay.schedule(dict(req, scheduleAt=far),
                                                                          E.SCHEDULE_MAX_MIN).get("ok"))
expired = dict(job, user=dict(job["user"], at=0))
ran.clear()
scheduled_pay._fire(expired, lambda act: ran.append(act) or {"ok": True}, log=lambda *x: None)
check("not run once the login has expired", not ran)

print("all passed" if not FAILED else f"{len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
