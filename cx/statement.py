"""Email a customer their recent transactions — to the address on THEIR wallet profile.

    "send my last 5 transactions to my email"
        -> the engine offers it, naming the (masked) address and the count
        -> the customer confirms, and enters their PIN
        -> the server sends it here, through the company SMTP relay in Vault

The recipient is ALWAYS resolved here, on the server, from the customer's own profile. It is
never taken from the request: a statement is personal data, and an address in a payload
would let any caller send a customer's history anywhere they liked.

Profile email: `user.email_id`, encrypted at rest (pii.decrypt). The dev test customer's
row holds a value that does not decrypt to an address, so in dev a fallback is used —
CX_STATEMENT_FALLBACK_EMAIL, the account owner's WORK address. Statements carry transaction
data and belong inside OpenWallet, not in a personal mailbox. In production there is no
fallback: no verified address on the profile means no email, and the customer is told why.
"""
from __future__ import annotations

import csv
import html
import io
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "openwallet"))

import data  # noqa: E402

MAX_ROWS = 25
FALLBACK = os.environ.get("CX_STATEMENT_FALLBACK_EMAIL", "dev@example.com")


def recipient() -> tuple[str | None, str]:
    """(address, where it came from). Address is None when there is nowhere safe to send."""
    try:
        from db import db_conn
        import openwallet_api
        import pii
        cur = db_conn().cursor()
        import userctx
        cur.execute("select email_id from user where id=%s", (userctx.user_id(),))
        r = cur.fetchone()
        addr = pii.decrypt(r["email_id"]) if r and r.get("email_id") else None
        if addr and addr.count("@") == 1 and "." in addr.split("@")[1]:
            return addr.strip(), "profile"
    except Exception:                                              # noqa: BLE001
        pass
    import userctx
    # The dev fallback is the CONFIGURED demo customer's own work address — never anyone
    # else's. A logged-in customer without a readable email gets a polite no, not a copy of
    # their statement in someone else's inbox.
    if FALLBACK and os.environ.get("CX_ENV", "dev") != "prod" and userctx.is_configured_sender():
        return FALLBACK, "dev fallback (profile email does not decrypt)"
    return None, "no verified email on the profile"


def mask(addr: str | None) -> str:
    if not addr or "@" not in addr:
        return "your email"
    local, dom = addr.split("@", 1)
    return f"{local[:2]}{'*' * max(3, len(local) - 2)}@{dom}"


def rows(count: int) -> list[dict]:
    """Newest first, with a real date and time — "7 min ago" is wrong by the time it is read."""
    n = max(1, min(int(count or 5), MAX_ROWS))
    out = []
    for t in data.transactions_for(limit=n)[:n]:
        r = t.public()
        r["when"] = time.strftime("%d %b %Y, %H:%M", time.localtime(time.time() - t.hours_ago * 3600))
        out.append(r)
    return out


def _bodies(items: list[dict], name: str) -> tuple[str, str, bytes]:
    total = sum(r["amount"] for r in items if r["status"] != "Failed")
    lines = [f"Hi {name.split()[0]},", "",
             f"Here are your last {len(items)} OpenWallet transactions.", ""]
    for r in items:
        lines.append(f"  {r['when']:>18}  {r['amount']:>9,.2f} {r['currency']}  "
                     f"{r['payee'][:40]:<40}  {r['status']}")
    lines += ["", f"Total (excluding failed): {total:,.2f} AED", "",
              "You asked for this in the OpenWallet app and confirmed it with your PIN.",
              "If that wasn't you, contact OpenWallet support straight away."]
    text = "\n".join(lines)
    trs = "".join(
        f"<tr><td>{html.escape(r['when'])}</td><td>{html.escape(r['payee'])}</td>"
        f"<td>{html.escape(r['corridor'])}</td><td style='text-align:right'>"
        f"{r['amount']:,.2f} {html.escape(r['currency'])}</td><td>{html.escape(r['status'])}</td>"
        f"<td style='color:#888'>{html.escape(r['ref'])}</td></tr>" for r in items)
    page = f"""<div style="font-family:system-ui,Arial,sans-serif;color:#1d1d1f;max-width:720px">
<p>Hi {html.escape(name.split()[0])},</p>
<p>Here are your last {len(items)} OpenWallet transactions.</p>
<table cellpadding="8" style="border-collapse:collapse;font-size:14px;width:100%">
<tr style="background:#f2f2f7;text-align:left"><th>When</th><th>To</th><th>Type</th>
<th style="text-align:right">Amount</th><th>Status</th><th>Reference</th></tr>{trs}</table>
<p><b>Total (excluding failed): {total:,.2f} AED</b></p>
<p style="color:#666;font-size:12px">You asked for this in the OpenWallet app and confirmed it with
your PIN. If that wasn't you, contact OpenWallet support straight away. A CSV copy is attached.</p></div>"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["when", "to", "type", "amount", "currency", "status", "reference"])
    for r in items:
        w.writerow([r["when"], r["payee"], r["corridor"], f"{r['amount']:.2f}",
                    r["currency"], r["status"], r["ref"]])
    return text, page, buf.getvalue().encode("utf-8")


def send(count: int) -> dict:
    """Send it. Returns {"ok", "to" (masked), "count"} or {"ok": False, "error"}."""
    to, source = recipient()
    if not to:
        return {"ok": False, "error": "There's no verified email address on your profile yet, "
                                      "so I can't send it. Add one in Settings and ask again."}
    items = rows(count)
    if not items:
        return {"ok": False, "error": "I couldn't find any transactions to send."}
    import userctx
    text, page, attachment = _bodies(items, userctx.name() or data.CUSTOMER.get("name", "there"))
    from devdb import _vault_config
    cfg = _vault_config()
    host, port = cfg["email-config.host"], int(cfg["email-config.port"])
    user, pw = cfg["email-config.sender-email"], cfg["email-config.sender-password"]
    msg = EmailMessage()
    msg["Subject"] = f"Your last {len(items)} OpenWallet transactions"
    msg["From"] = f"{cfg.get('email-config.sender-name') or 'OpenWallet'} <{user}>"
    msg["To"] = to
    msg.set_content(text)
    msg.add_alternative(page, subtype="html")
    msg.add_attachment(attachment, maintype="text", subtype="csv",
                       filename=f"openwallet-transactions-{time.strftime('%Y%m%d')}.csv")
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as s:
            s.login(user, pw)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            if str(cfg.get("email-config.enable", "true")).lower() != "false":
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
            s.login(user, pw)
            s.send_message(msg)
    return {"ok": True, "to": mask(to), "count": len(items), "source": source}
