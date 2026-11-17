"""Read-only dev lookups for wiring the demo to real merchants and campaigns.

Credentials come from Vault via the AppRole already in the shell environment and are never
printed, never written to disk and never passed on a command line. Same recipe the treasury
guard uses.

    python devdb.py "SELECT ..."          # ad-hoc read
"""
from __future__ import annotations

import json
import re
import socket
import os
import subprocess
import time
import urllib.request


_CFG: dict | None = None


def _vault_config() -> dict:
    """Cached. Every call does an AppRole login plus a secret read, and doing that per query
    turned a ten-merchant lookup into forty seconds of round trips."""
    global _CFG
    if _CFG is not None:
        return _CFG
    _CFG = _fetch_vault_config()
    return _CFG


class Unreachable(RuntimeError):
    """The internal network is not reachable — almost always a disconnected VPN.

    Distinguished from every other error on purpose. "Vault said no" and "Vault is on the
    other side of a tunnel that is down" need completely different responses: the first is a
    bug, the second is a laptop that walked out of the office. Callers can tell them apart
    and say so, instead of showing a customer an empty screen."""


#: Cached reachability. (ok, when). A dead internal network takes 20-30s to time out, and
#: paying that on EVERY request is what turned "VPN is down" into "the demo is unusable" —
#: ten merchants meant ten timeouts. One cheap TCP probe fronts the whole thing.
_REACH: tuple[bool, float] = (True, 0.0)
_REACH_TTL = 20.0
_PROBE_TIMEOUT = 1.5


def reachable(force: bool = False) -> bool:
    """Can we open a TCP connection to Vault at all? 1.5s, not 30."""
    global _REACH
    ok, when = _REACH
    if not force and (time.time() - when) < _REACH_TTL:
        return ok
    host, port = _vault_host_port()
    if not host:
        _REACH = (False, time.time())
        return False
    sk = socket.socket()
    sk.settimeout(_PROBE_TIMEOUT)
    try:
        sk.connect((host, port))
        ok = True
    except OSError:
        ok = False
    finally:
        sk.close()
    _REACH = (ok, time.time())
    return ok


def _vault_host_port() -> tuple[str | None, int]:
    uri = _env()[0]
    m = re.match(r"https?://([^:/]+):?(\d+)?", uri or "")
    if not m:
        return None, 0
    return m.group(1), int(m.group(2) or (443 if uri.startswith("https") else 80))


_ENV: tuple | None = None


def _env() -> tuple[str, str, str, str]:
    """VAULT_URI|ROLE_ID|SECRET_ID|ENV, read once. Sourcing ~/.bashrc costs ~100ms and the
    reachability probe would otherwise pay it on every call."""
    global _ENV
    if _ENV is None and os.environ.get("VAULT_URI"):
        # Passed in by ~/amp/up.sh. A systemd service has no login shell to inherit from,
        # and ~/.bashrc returns early when it is not interactive — which left Vault unknown,
        # PII undecryptable, and every payee shown as "someone".
        _ENV = (os.environ["VAULT_URI"].rstrip("/"), os.environ.get("VAULT_ROLE_ID", ""),
                os.environ.get("VAULT_SECRET_ID", ""), os.environ.get("VAULT_ENV") or "dev")
    if _ENV is None:
        out = subprocess.run(
            ["bash", "-lc",
             'set -a; source ~/.bashrc >/dev/null 2>&1; '
             'echo "$VAULT_URI|$VAULT_ROLE_ID|$VAULT_SECRET_ID|$VAULT_ENV"'],
            capture_output=True, text=True).stdout.strip()
        uri, rid, sid, env = (out.split("|") + ["", "", "", ""])[:4]
        # a trailing slash makes login 302 and drop the body
        _ENV = (uri.rstrip("/"), rid, sid, env or "dev")
    return _ENV


def _fetch_vault_config() -> dict:
    if not reachable():
        raise Unreachable(
            f"cannot reach Vault at {_vault_host_port()[0]} — internal network is not "
            f"routable from here (is the VPN connected?)")
    uri, rid, sid, env = _env()
    req = urllib.request.Request(
        f"{uri}/v1/auth/approle/login",
        data=json.dumps({"role_id": rid, "secret_id": sid}).encode(),
        headers={"Content-Type": "application/json"})
    tok = json.load(urllib.request.urlopen(req, timeout=8))["auth"]["client_token"]
    req = urllib.request.Request(f"{uri}/v1/{env}/data/all-config",
                                 headers={"X-Vault-Token": tok})
    return json.load(urllib.request.urlopen(req, timeout=8))["data"]["data"]


def conn():
    cfg = _vault_config()
    m = re.match(r"jdbc:mysql://([^:/]+):?(\d+)?/([^?]+)", cfg["spring.datasource.url"])
    import pymysql
    return pymysql.connect(
        host=m.group(1), port=int(m.group(2) or 3306),
        user=cfg["spring.datasource.username"], password=cfg["spring.datasource.password"],
        database=m.group(3), cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        # A READ timeout, not just a connect timeout. Dev had an ALTER TABLE queued behind an
        # idle-in-transaction connection from another service, so every SELECT on that table
        # waited on a metadata lock — for 31 minutes and counting. With no read timeout the
        # assistant inherited that wait: a request hung indefinitely and the customer saw
        # nothing at all, which is the worst way to fail.
        #
        # autocommit is on so a plain SELECT does not open a transaction of its own. The
        # blocker here was exactly that pattern on someone else's connection, and this
        # module is read-only in normal use — writes pass an explicit connection.
        connect_timeout=8, read_timeout=12, write_timeout=12)


def q(sql: str, args=None, n: int = 50) -> list[dict]:
    with conn() as c, c.cursor() as cur:
        # execute(sql) WITHOUT an args tuple: pymysql treats the query as a format string
        # whenever args is passed, so a bare LIKE '%merchant%' blows up on the percent signs.
        cur.execute(sql, args) if args else cur.execute(sql)
        return cur.fetchmany(n)


def show(sql: str, args=None, n: int = 50) -> None:
    rows = q(sql, args, n)
    print(f"{len(rows)} rows")
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    import sys
    show(sys.argv[1], n=int(sys.argv[2]) if len(sys.argv) > 2 else 50)
