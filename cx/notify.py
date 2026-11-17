"""Email the deployment result, using the company SMTP relay from Vault.

The relay is `email-config.*` in dev Vault — the same one `sp-log-evidence` sends its QA
reports through. Credentials are read at send time and never written anywhere.

Goes to the WORK address deliberately. The body names internal hostnames, dev VPC addresses,
which ports are open and which live-payment flags are set; that belongs inside OpenWallet, not in
a personal mailbox.

    python notify.py            send the report
    python notify.py --dry      print it without sending
"""
from __future__ import annotations

import os
import smtplib
import ssl
import subprocess
import sys
from email.message import EmailMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TO = os.environ.get("CX_NOTIFY_TO", "dev@example.com")
HOST = os.environ.get("CX_DEPLOY_HOST", "riley@127.0.0.1")
DIR = os.environ.get("CX_DEPLOY_DIR", "/home/riley/openwallet-cx")
PORT = os.environ.get("CX_DEPLOY_PORT", "8440")


def _ssh(cmd: str, timeout: int = 25) -> str:
    try:
        out = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", HOST, cmd],
            capture_output=True, text=True, timeout=timeout).stdout
        return "\n".join(l for l in out.splitlines()
                         if "post-quantum" not in l and "store now" not in l
                         and "openssh.com/pq" not in l).strip()
    except Exception as exc:
        return f"(could not check: {type(exc).__name__})"


def gather() -> dict:
    ip = HOST.split("@")[-1]
    facts = {"host": ip, "dir": DIR, "url": f"http://{ip}:{PORT}/assist"}
    facts["ports"] = _ssh("for p in 8440 8450 8480; do printf '%s=' $p; "
                          "ss -ltn 2>/dev/null | grep -q \":$p \" && echo up || echo down; done")
    facts["spec"] = _ssh("echo \"$(nproc) cores, $(free -m | awk '/Mem:/{print $2\"MB RAM\"}')\"")
    facts["dev"] = _ssh("for hp in 127.0.0.1:3306 127.0.0.1:27017 "
                        "127.0.0.1:8200; do h=${hp%:*}; p=${hp#*:}; printf '%s=' $hp; "
                        "timeout 4 bash -c \"cat < /dev/null > /dev/tcp/$h/$p\" 2>/dev/null "
                        "&& echo open || echo CLOSED; done")
    facts["model"] = _ssh("curl -s -m 5 http://127.0.0.1:8450/health 2>/dev/null "
                          "| head -c 220 || echo '(sidecar not answering)'")
    # Prove it answers, from here, as a client would.
    try:
        import json
        import urllib.request
        req = urllib.request.Request(
            f"http://{ip}:{PORT}/api/cx/ask", method="POST",
            data=json.dumps({"text": "how many bills are due",
                             "session": "deploy-check"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.loads(r.read())
        facts["probe"] = (f"{d.get('intent')} ({d.get('confidence', 0):.0%}) -> "
                          f"{(d.get('say') or '')[:150]}")
    except Exception as exc:
        facts["probe"] = f"FAILED: {type(exc).__name__}: {exc}"
    return facts


def body(f: dict) -> str:
    return f"""The OpenWallet CX assistant is deployed and answering.

  Open it:  {f['url']}

WHERE IT RUNS
  {f['host']}  ({f['spec']})
  Path: {f['dir']}

  This is Riley's shared server, chosen because it sits INSIDE the dev VPC and so
  reaches MySQL, Mongo and Vault directly:

    {f['dev']}

  That means no VPN and no reverse tunnel, and your laptop does not need to stay awake.
  Our EC2 box could never have done this: it is in eu-north-1, dev is in ap-south-1,
  and the VPCs are not peered, so it could only ever have proxied back to the laptop.

SERVICES
  {f['ports']}
    8440  the assistant  (/assist, /watch, /reco, /merchant)
    8450  the model sidecar, bound to localhost only
    8480  the DTOne stand-in

  Model: {f['model']}

VERIFIED FROM OUTSIDE
  "how many bills are due"
    {f['probe']}

WHAT IS REAL AND WHAT IS NOT
  Real: your bills, campaigns, ledger, balances, split payments, the PIN check against
        the wallet's own /user/pin-validate, and the payment path itself.
  Simulated: exactly one call - DTOne's bill-due lookup. Its credentials live in the UAE
        Vault, which is not reachable, and the dev Vault values return 401. The client
        speaks DTOne's real contract, so switching to live is one environment variable.

FLAGS ON THAT HOST
  CX_REQUIRE_PIN=1     every payment asks for your PIN and validates it against the wallet
  CX_BILL_PAY_LIVE=0   live bill payments are OFF - deliberately, on a shared server

PLEASE REMOVE IT WHEN THE DEMO IS DONE
  cd ~/amp/cx && ./deploy.sh remove

  Two reasons this matters. It is not your server. And /assist has no authentication, so
  anyone who can reach the host can open it and read this customer's real transactions,
  balances and bills. The deployed tree also contains .secrets.json - a JWT secret, the
  Mongo URI and per-user encryption keys - which is the part that really needs deleting
  rather than just stopping the processes.

  ./deploy.sh status   shows what is running
  ./deploy.sh stop     stops the services but leaves the tree
"""


def main() -> None:
    f = gather()
    text = body(f)
    if "--dry" in sys.argv:
        print(text)
        return
    from devdb import _vault_config
    cfg = _vault_config()
    host = cfg["email-config.host"]
    port = int(cfg["email-config.port"])
    user = cfg["email-config.sender-email"]
    pw = cfg["email-config.sender-password"]
    name = cfg.get("email-config.sender-name") or "OpenWallet"

    msg = EmailMessage()
    msg["Subject"] = "OpenWallet CX assistant - deployed on 127.0.0.1 (please remove after the demo)"
    msg["From"] = f"{name} <{user}>"
    msg["To"] = TO
    msg.set_content(text)

    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(),
                              timeout=30) as s:
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
    print(f"sent to {TO} via {host}:{port} as {user}")


if __name__ == "__main__":
    main()
