"""Double-entry ledger for AMP payments.

Three properties the rest of the protocol leans on:

  Append-only      entries are never updated or deleted. A correction is a new, compensating
                   pair that references the original. A ledger you can edit is not a ledger.
  Derived balances a balance is the sum of entries, never a stored number that code mutates.
                   Stored balances drift; derived balances cannot disagree with history.
  Two phases       `authorize` places a hold and validates; `capture` posts the entries. Between
                   them the monitor writes its decision. If anything fails after authorize, the
                   hold expires and nothing moved — a single-phase design has no safe failure point.

Idempotency is not a nicety here. The key is derived from (mandate_id, nonce), so a retried AMP
message, a duplicated notification, or a user repeating themselves returns the original transfer
instead of moving money twice. In a voice system an utterance genuinely does get captured twice.

Shares pay.db with mandate_store, so the spend counters and the entries can never disagree.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
import datetime as dt

from .mandate import Mandate, MandateError, now_iso
from .mandate_store import conn, _lock

HOLD_TTL_SECONDS = 300          # an authorization that never captures frees its funds

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
  account_id TEXT PRIMARY KEY,
  owner TEXT NOT NULL, currency TEXT NOT NULL,
  opened TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
  entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
  txn_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  amount REAL NOT NULL,              -- signed; debit negative, credit positive
  currency TEXT NOT NULL,
  kind TEXT NOT NULL,                -- transfer | reversal
  ts TEXT NOT NULL,
  ref TEXT DEFAULT '',               -- reversals point at the txn they undo
  FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);
CREATE TABLE IF NOT EXISTS holds (
  hold_id TEXT PRIMARY KEY,
  transfer_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  amount REAL NOT NULL,
  state TEXT NOT NULL,               -- active | released
  expires TEXT NOT NULL, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_entries_account ON entries(account_id);
CREATE INDEX IF NOT EXISTS ix_entries_txn ON entries(txn_id);
CREATE INDEX IF NOT EXISTS ix_holds_account ON holds(account_id, state);
"""


class LedgerError(Exception):
    """Refusal with a reason safe to say out loud."""


def init():
    with _lock, conn() as c:
        c.executescript(SCHEMA)


def open_account(account_id: str, owner: str, currency: str, opening_balance: float = 0.0):
    with _lock, conn() as c:
        c.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?)",
                  (account_id, owner, currency.upper(), now_iso()))
        if opening_balance:
            # Demo funding, booked against an explicit source so entries still sum to zero.
            c.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?)",
                      ("acc_funding", "demo funding", currency.upper(), now_iso()))
            txn = f"txn_{uuid.uuid4().hex[:10]}"
            for acct, amt in ((account_id, opening_balance), ("acc_funding", -opening_balance)):
                c.execute("INSERT INTO entries (txn_id,account_id,amount,currency,kind,ts,ref)"
                          " VALUES (?,?,?,?,?,?,?)",
                          (txn, acct, amt, currency.upper(), "transfer", now_iso(), "opening"))


def balance(account_id: str) -> float:
    with conn() as c:
        r = c.execute("SELECT COALESCE(SUM(amount),0) b FROM entries WHERE account_id=?",
                      (account_id,)).fetchone()
    return round(r["b"], 2)


def _held(c, account_id: str) -> float:
    r = c.execute("SELECT COALESCE(SUM(amount),0) h FROM holds"
                  " WHERE account_id=? AND state='active' AND expires > ?",
                  (account_id, now_iso())).fetchone()
    return r["h"]


def available(account_id: str) -> float:
    """Balance minus live holds — what can actually be spent right now."""
    with conn() as c:
        r = c.execute("SELECT COALESCE(SUM(amount),0) b FROM entries WHERE account_id=?",
                      (account_id,)).fetchone()
        return round(r["b"] - _held(c, account_id), 2)


def idempotency_key(mandate_id: str, nonce: str) -> str:
    return f"{mandate_id}:{nonce}"


def authorize(m: Mandate, amount: float, currency: str, nonce: str,
              counters: dict, detail: dict | None = None) -> dict:
    """Validate against the mandate, reserve the funds, and record the authorization.

    Re-checks the mandate itself rather than trusting the monitor's verdict: two independent
    components reaching the same answer is the cross-check the threat model relies on.
    """
    key = idempotency_key(m.mandate_id, nonce)

    with _lock, conn() as c:
        prior = c.execute("SELECT * FROM transfers WHERE idempotency_key=?", (key,)).fetchone()
        if prior:
            return {"transfer_id": prior["transfer_id"], "state": prior["state"],
                    "replayed": True}

        tier = m.check(amount, currency, **counters)          # raises MandateError

        if m.payer.wallet == m.payee.wallet:
            raise LedgerError("payer and payee are the same wallet")
        for acct in (m.payer.wallet, m.payee.wallet):
            row = c.execute("SELECT currency FROM accounts WHERE account_id=?", (acct,)).fetchone()
            if not row:
                raise LedgerError(f"no account {acct}")
            if row["currency"] != currency.upper():
                raise LedgerError(f"{acct} is not a {currency.upper()} account")

        bal = c.execute("SELECT COALESCE(SUM(amount),0) b FROM entries WHERE account_id=?",
                        (m.payer.wallet,)).fetchone()["b"]
        if bal - _held(c, m.payer.wallet) < amount:
            raise LedgerError("there isn't enough in the wallet for that")

        transfer_id = f"tfr_{uuid.uuid4().hex[:10]}"
        expires = (dt.datetime.now(dt.timezone.utc)
                   + dt.timedelta(seconds=HOLD_TTL_SECONDS)).replace(
                       microsecond=0).isoformat().replace("+00:00", "Z")
        c.execute("INSERT INTO holds VALUES (?,?,?,?,?,?,?)",
                  (f"hld_{uuid.uuid4().hex[:10]}", transfer_id, m.payer.wallet,
                   amount, "active", expires, now_iso()))
        c.execute("INSERT INTO transfers VALUES (?,?,?,?,?,?,?,?)",
                  (transfer_id, m.mandate_id, key, amount, currency.upper(),
                   "authorized", now_iso(),
                   json.dumps({**(detail or {}), "tier": tier, "nonce": nonce})))

    return {"transfer_id": transfer_id, "state": "authorized", "tier": tier, "replayed": False}


def capture(transfer_id: str) -> dict:
    """Post the double-entry pair and release the hold. Idempotent: capturing twice is a no-op."""
    with _lock, conn() as c:
        t = c.execute("SELECT * FROM transfers WHERE transfer_id=?", (transfer_id,)).fetchone()
        if not t:
            raise LedgerError("no such transfer")
        if t["state"] == "captured":
            return {"transfer_id": transfer_id, "state": "captured", "replayed": True}
        if t["state"] != "authorized":
            raise LedgerError(f"cannot capture a {t['state']} transfer")

        md = c.execute("SELECT doc FROM mandates WHERE mandate_id=?",
                       (t["mandate_id"],)).fetchone()
        m = Mandate.from_dict(json.loads(md["doc"]))

        hold = c.execute("SELECT * FROM holds WHERE transfer_id=? AND state='active'",
                         (transfer_id,)).fetchone()
        if not hold:
            raise LedgerError("the authorization expired before it was captured")
        if hold["expires"] <= now_iso():
            c.execute("UPDATE holds SET state='released' WHERE hold_id=?", (hold["hold_id"],))
            c.execute("UPDATE transfers SET state='failed' WHERE transfer_id=?", (transfer_id,))
            raise LedgerError("the authorization expired before it was captured")

        txn = f"txn_{uuid.uuid4().hex[:10]}"
        ts = now_iso()
        for acct, amt in ((m.payer.wallet, -t["amount"]), (m.payee.wallet, t["amount"])):
            c.execute("INSERT INTO entries (txn_id,account_id,amount,currency,kind,ts,ref)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (txn, acct, amt, t["currency"], "transfer", ts, transfer_id))
        c.execute("UPDATE holds SET state='released' WHERE hold_id=?", (hold["hold_id"],))
        c.execute("UPDATE transfers SET state='captured', ts=? WHERE transfer_id=?",
                  (ts, transfer_id))

    return {"transfer_id": transfer_id, "state": "captured", "txn_id": txn, "replayed": False}


def reverse(transfer_id: str, reason: str) -> dict:
    """Compensating entries, never a deletion. The original stays in the record."""
    with _lock, conn() as c:
        t = c.execute("SELECT * FROM transfers WHERE transfer_id=?", (transfer_id,)).fetchone()
        if not t:
            raise LedgerError("no such transfer")
        if t["state"] == "reversed":
            return {"transfer_id": transfer_id, "state": "reversed", "replayed": True}
        if t["state"] != "captured":
            raise LedgerError(f"only a captured transfer can be reversed, not {t['state']}")

        md = c.execute("SELECT doc FROM mandates WHERE mandate_id=?",
                       (t["mandate_id"],)).fetchone()
        m = Mandate.from_dict(json.loads(md["doc"]))

        txn = f"txn_{uuid.uuid4().hex[:10]}"
        ts = now_iso()
        for acct, amt in ((m.payer.wallet, t["amount"]), (m.payee.wallet, -t["amount"])):
            c.execute("INSERT INTO entries (txn_id,account_id,amount,currency,kind,ts,ref)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (txn, acct, amt, t["currency"], "reversal", ts, transfer_id))
        detail = json.loads(t["detail"] or "{}")
        detail["reversal_reason"] = reason
        c.execute("UPDATE transfers SET state='reversed', detail=? WHERE transfer_id=?",
                  (json.dumps(detail), transfer_id))

    return {"transfer_id": transfer_id, "state": "reversed", "txn_id": txn, "replayed": False}


def expire_holds() -> int:
    with _lock, conn() as c:
        cur = c.execute("UPDATE holds SET state='released'"
                        " WHERE state='active' AND expires <= ?", (now_iso(),))
        n = cur.rowcount
        c.execute("UPDATE transfers SET state='failed' WHERE state='authorized'"
                  " AND transfer_id IN (SELECT transfer_id FROM holds WHERE state='released')")
    return n


def audit(txn_id: str | None = None) -> list[dict]:
    with conn() as c:
        q = "SELECT * FROM entries"
        rows = c.execute(q + (" WHERE txn_id=?" if txn_id else "") + " ORDER BY entry_id",
                         (txn_id,) if txn_id else ()).fetchall()
    return [dict(r) for r in rows]


def check_invariant() -> dict:
    """Every txn's entries must sum to zero, and so must the whole book.
    If this ever fails the ledger is lying and everything downstream is suspect."""
    with conn() as c:
        bad = c.execute(
            "SELECT txn_id, ROUND(SUM(amount),6) s FROM entries GROUP BY txn_id"
            " HAVING ABS(s) > 0.000001").fetchall()
        total = c.execute("SELECT ROUND(COALESCE(SUM(amount),0),6) s FROM entries").fetchone()["s"]
    return {"balanced": not bad and abs(total) < 1e-6,
            "unbalanced_txns": [dict(r) for r in bad], "book_total": total}
