"""AMP payment mandates — the signed, bounded agreement two agents form before any money moves.

A mandate answers one question: *is this payer allowed to pay this payee, this much, right now?*
It is agreed once, deliberately, by both humans. After that a spoken instruction can only ever
exercise it — never create authority. That separation is what makes no-touch payment defensible:
the considered moment and the casual moment are different moments.

Both parties sign. A mandate carrying only the payer's signature is a standing order the payee
never agreed to receive, so `verify()` requires both.

Canonicalisation reuses the ordering discipline of protocol.py: sorted keys, no whitespace, so the
signed bytes are unambiguous on both sides.
"""
from __future__ import annotations

import json
import time
import uuid
import datetime as dt
from dataclasses import dataclass, field, asdict

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.exceptions import InvalidSignature

VERSION = 1

# Fields covered by the signatures. Anything outside this list is metadata and may change
# without invalidating the mandate; anything that bounds authority MUST be in here.
SIGNED_FIELDS = ["mandate_id", "version", "payer", "payee", "currency",
                 "limits", "green_tier_cap", "purposes", "not_before", "not_after"]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str = "mnd") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class MandateError(Exception):
    """Raised when a mandate cannot authorize a transfer. The message is safe to speak aloud."""


@dataclass
class Party:
    agent_id: str
    wallet: str
    display_name: str = ""


@dataclass
class Limits:
    per_txn: float
    per_day: float
    per_month: float
    max_txn_per_hour: int = 10


@dataclass
class Mandate:
    payer: Party
    payee: Party
    currency: str
    limits: Limits
    green_tier_cap: float
    not_after: str
    not_before: str = field(default_factory=now_iso)
    purposes: list[str] = field(default_factory=lambda: ["peer_transfer"])
    mandate_id: str = field(default_factory=new_id)
    version: int = VERSION
    payer_sig: str = ""
    payee_sig: str = ""
    revoked_at: str = ""
    revoked_by: str = ""

    # ---- canonical form ----
    def canonical(self) -> bytes:
        """Deterministic bytes over the authority-bearing fields only."""
        d = asdict(self)
        return json.dumps({k: d[k] for k in SIGNED_FIELDS},
                          sort_keys=True, separators=(",", ":"), default=str).encode()

    # ---- signing ----
    def sign_as(self, role: str, key: Ed25519PrivateKey) -> "Mandate":
        if role not in ("payer", "payee"):
            raise ValueError("role must be payer or payee")
        sig = key.sign(self.canonical()).hex()
        setattr(self, f"{role}_sig", sig)
        return self

    def verify(self, payer_pub: Ed25519PublicKey, payee_pub: Ed25519PublicKey) -> bool:
        """Both signatures, or it is not an agreement."""
        blob = self.canonical()
        for sig_hex, pub in ((self.payer_sig, payer_pub), (self.payee_sig, payee_pub)):
            if not sig_hex:
                return False
            try:
                pub.verify(bytes.fromhex(sig_hex), blob)
            except (InvalidSignature, ValueError):
                return False
        return True

    # ---- authority ----
    def check(self, amount: float, currency: str, spent_today: float,
              spent_month: float, txns_last_hour: int, at: str | None = None) -> str:
        """Return the risk tier ('green' | 'amber') or raise MandateError.

        Raising rather than returning 'red' is deliberate: a refusal carries a reason the
        assistant can say out loud, and callers cannot accidentally treat it as a pass.
        """
        at = at or now_iso()

        if self.revoked_at:
            raise MandateError("that mandate has been revoked")
        if at < self.not_before:
            raise MandateError("that mandate is not active yet")
        if at > self.not_after:
            raise MandateError("that mandate has expired")
        if currency.upper() != self.currency.upper():
            raise MandateError(f"this mandate only covers {self.currency}")
        if amount <= 0:
            raise MandateError("the amount has to be more than zero")

        if amount > self.limits.per_txn:
            raise MandateError(
                f"that is over the per-transaction limit of {self.limits.per_txn:g} {self.currency}")
        if spent_today + amount > self.limits.per_day:
            remaining = max(0.0, self.limits.per_day - spent_today)
            raise MandateError(
                f"that would pass today's limit — {remaining:g} {self.currency} left")
        if spent_month + amount > self.limits.per_month:
            raise MandateError("that would pass this month's limit")
        if txns_last_hour >= self.limits.max_txn_per_hour:
            raise MandateError("too many transfers in the last hour")

        return "green" if amount <= self.green_tier_cap else "amber"

    def revoke(self, by: str) -> "Mandate":
        """Unilateral and immediate — either party, no counter-signature."""
        if not self.revoked_at:
            self.revoked_at = now_iso()
            self.revoked_by = by
        return self

    @property
    def active(self) -> bool:
        return not self.revoked_at and self.not_before <= now_iso() <= self.not_after

    # ---- serialisation ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Mandate":
        d = dict(d)
        return cls(payer=Party(**d.pop("payer")), payee=Party(**d.pop("payee")),
                   limits=Limits(**d.pop("limits")), **d)
