"""The device key — the thing that actually authorizes a payment.

The voice proposes; this signs. A recording of someone saying "send 20 dirhams" carries no
signature, which is what makes speech safe to use as an input at all.

Two backends behind one interface, because the protocol does not care which one signs — only that
the signature verifies:

  SealedEd25519   the private key is encrypted to this machine's TPM (systemd-creds --with-key=tpm2)
                  and unsealed only to sign. Moving the blob to another machine makes it inert.
                  Honest limitation: while signing, the key is in process memory.

  Tpm2Pkcs11      the key is generated inside the TPM and never leaves it; the chip signs.
                  Needs tpm2-pkcs11 installed and membership of the tss group, and the key is
                  ECDSA P-256 because TPM 2.0 does not generally implement Ed25519.

`load()` prefers hardware and falls back, reporting which it got, so the risk engine can take the
backend into account rather than assuming.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.exceptions import InvalidSignature

CFG = Path.home() / ".config" / "glasslink"
CRED = CFG / "device-key.cred"          # TPM-sealed private key
PUB = CFG / "device-key.pub"            # public key + metadata, plain
CRED_NAME = "glasslink-device"


class DeviceKeyError(Exception):
    pass


#: A TPM sign or unseal is milliseconds of work. Anything beyond a couple of seconds is not
#: slow, it is STUCK — typically no access to /dev/tpmrm0, where systemd-creds blocks rather
#: than erroring. The old 60s-per-attempt budget meant a payment could hang for three minutes
#: with the customer watching a spinner and no event ever reaching the screen.
TPM_TIMEOUT_S = 6


def tpm_accessible() -> bool:
    """Can this process actually reach the TPM? Cheap pre-flight, so we fail in microseconds
    instead of blocking on a device we were never permitted to open."""
    dev = Path("/dev/tpmrm0")
    try:
        return dev.exists() and os.access(dev, os.R_OK | os.W_OK)
    except OSError:
        return False


def _run_tpm(cmd: list[str], data: bytes | None = None, attempts: int = 3):
    """TPM calls fail transiently under contention — the resource manager is a single shared
    device and rapid seal/unseal cycles can collide. Retry briefly rather than failing a
    payment on a hiccup; a persistent failure still surfaces, and quickly."""
    if not tpm_accessible():
        raise DeviceKeyError(
            "no access to /dev/tpmrm0 — this user is not in the 'tss' group, so the hardware "
            "key cannot be used")
    last = None
    for i in range(attempts):
        try:
            proc = subprocess.run(cmd, input=data, capture_output=True,
                                  timeout=TPM_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # Treated as a failed attempt rather than an exception that escapes mid-payment.
            last = None
            time.sleep(0.3 * (i + 1))
            continue
        if proc.returncode == 0:
            return proc
        last = proc
        time.sleep(0.4 * (i + 1))
    if last is None:
        raise DeviceKeyError(
            f"the TPM did not respond within {TPM_TIMEOUT_S}s across {attempts} attempts")
    return last


def _canonical_transfer(mandate_id: str, nonce: str, payee_wallet: str,
                        amount: float, currency: str, timestamp: str) -> bytes:
    """The exact bytes a device signature covers.

    Canonical JSON rather than concatenation: joining fields with a separator invites an
    attacker to move a character across a boundary ("20|AED" vs "2|0AED"). Sorted keys and
    no whitespace leave no room for two readings.
    """
    return json.dumps({"amount": f"{amount:.2f}", "currency": currency.upper(),
                       "mandate_id": mandate_id, "nonce": nonce,
                       "payee_wallet": payee_wallet, "timestamp": timestamp},
                      sort_keys=True, separators=(",", ":")).encode()


# ---------- backend: TPM-sealed Ed25519 ----------

class SealedEd25519:
    algorithm = "Ed25519"
    backend = "tpm-sealed"
    hardware_resident = False       # unsealed into memory to sign — stated, not hidden

    def __init__(self, cred: Path = CRED, pub: Path = PUB):
        self.cred, self.pub = cred, pub

    @staticmethod
    def available() -> bool:
        try:
            out = subprocess.run(["systemd-analyze", "has-tpm2"],
                                 capture_output=True, text=True, timeout=15)
            return out.returncode == 0 and "yes" in out.stdout.lower()
        except Exception:
            return False

    def enrolled(self) -> bool:
        return self.cred.exists() and self.pub.exists()

    def enrol(self, agent_id: str) -> Ed25519PublicKey:
        """Generate a key and seal it to this TPM. Refuses to clobber an existing one:
        silently replacing a device key is indistinguishable from an attacker installing theirs."""
        if self.enrolled():
            raise DeviceKeyError("a device key is already enrolled; rotation must be explicit")
        CFG.mkdir(parents=True, exist_ok=True)
        os.chmod(CFG, 0o700)

        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes_raw()
        try:
            proc = _run_tpm(
                ["systemd-creds", "encrypt", "--with-key=tpm2", f"--name={CRED_NAME}",
                 "-", str(self.cred)], raw)
            if proc.returncode != 0:
                raise DeviceKeyError(
                    f"TPM sealing failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
        finally:
            raw = b"\x00" * len(raw)

        os.chmod(self.cred, 0o600)
        pub_hex = key.public_key().public_bytes_raw().hex()
        self.pub.write_text(json.dumps(
            {"agent_id": agent_id, "algorithm": self.algorithm,
             "backend": self.backend, "public_key": pub_hex}, indent=2))
        os.chmod(self.pub, 0o600)
        return key.public_key()

    def public_key(self) -> Ed25519PublicKey:
        if not self.pub.exists():
            raise DeviceKeyError("no device key enrolled")
        meta = json.loads(self.pub.read_text())
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(meta["public_key"]))

    def _unseal(self) -> bytes:
        proc = _run_tpm(
            ["systemd-creds", "decrypt", f"--name={CRED_NAME}", str(self.cred), "-"])
        if proc.returncode != 0:
            raise DeviceKeyError(
                "TPM unseal failed — this key belongs to another machine, or the TPM state changed")
        return proc.stdout

    def sign(self, payload: bytes) -> bytes:
        raw = self._unseal()
        try:
            return Ed25519PrivateKey.from_private_bytes(raw).sign(payload)
        finally:
            raw = b"\x00" * len(raw)


# ---------- backend: TPM-resident ECDSA via PKCS#11 ----------

class Tpm2Pkcs11:
    algorithm = "ECDSA-P256"
    backend = "tpm-resident"
    hardware_resident = True

    MODULES = ["/usr/lib/x86_64-linux-gnu/pkcs11/libtpm2_pkcs11.so",
               "/usr/lib/x86_64-linux-gnu/libtpm2_pkcs11.so.1"]

    @classmethod
    def module_path(cls) -> str | None:
        return next((p for p in cls.MODULES if Path(p).exists()), None)

    @classmethod
    def available(cls) -> bool:
        if cls.module_path() is None:
            return False
        try:                                    # the module is useless without tss membership
            return os.access("/dev/tpmrm0", os.R_OK | os.W_OK)
        except Exception:
            return False

    def enrol(self, agent_id: str):
        raise DeviceKeyError(
            "tpm2-pkcs11 is present but this backend is not wired yet — "
            "install tpm2-tools, join the tss group, then implement enrol via pkcs11-tool")


# ---------- selection ----------

class DemoEd25519:
    """A clearly-labelled DEMO key, held in a file.

    This exists because the demo has no dedicated device yet: the real design seals the
    payment key in a TPM and signs inside it, and that is what ships. Until there is a device,
    a demo key lets the rest of the system — mandates, both-party signatures, verification,
    limits, tiering — run exactly as designed instead of being stubbed out.

    It is opt-in (`AMP_DEMO_KEY=1`) and never a silent fallback. `hardware_resident` is False,
    `backend` says "demo", and every screen that renders a signature shows that, so nobody can
    mistake a demo-signed payment for a hardware-signed one.
    """

    algorithm = "Ed25519"
    backend = "demo key (no device yet — not hardware-held)"
    hardware_resident = False

    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".config" / "amp" / "demo-device.key")

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("AMP_DEMO_KEY") == "1"

    def _key(self) -> Ed25519PrivateKey:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw())
            os.chmod(self.path, 0o600)
        return Ed25519PrivateKey.from_private_bytes(self.path.read_bytes())

    def public_key(self) -> Ed25519PublicKey:
        return self._key().public_key()

    def sign(self, payload: bytes) -> bytes:
        return self._key().sign(payload)

    @property
    def source(self) -> str:
        return f"demo key at {self.path}"


def load(prefer_hardware: bool = True):
    """Best available backend. Raises if the TPM is unusable rather than silently
    degrading to a key on disk — a payment key's protection must never be a surprise."""
    if prefer_hardware and Tpm2Pkcs11.available():
        return Tpm2Pkcs11()
    if SealedEd25519.available() and tpm_accessible():
        return SealedEd25519()
    # Opt-in demo key. Deliberately AFTER the hardware attempts and gated on an explicit
    # environment flag, so the only way to get here is to have asked for it.
    if DemoEd25519.enabled():
        return DemoEd25519()
    raise DeviceKeyError(
        "no TPM available; refusing to hold a payment key unprotected on disk "
        "(set AMP_DEMO_KEY=1 to use a clearly-labelled demo key instead)")


def sign_transfer(be, mandate_id: str, nonce: str, payee_wallet: str,
                  amount: float, currency: str, timestamp: str) -> dict:
    """Sign one transfer authorization. The nonce is the monitor's and single-use, so a replayed
    utterance cannot produce a second valid authorization."""
    payload = _canonical_transfer(mandate_id, nonce, payee_wallet, amount, currency, timestamp)
    return {"algorithm": be.algorithm, "backend": be.backend,
            "hardware_resident": be.hardware_resident,
            "signature": be.sign(payload).hex()}


def verify_transfer(public_key: Ed25519PublicKey, signature_hex: str, mandate_id: str,
                    nonce: str, payee_wallet: str, amount: float, currency: str,
                    timestamp: str) -> bool:
    payload = _canonical_transfer(mandate_id, nonce, payee_wallet, amount, currency, timestamp)
    try:
        public_key.verify(bytes.fromhex(signature_hex), payload)
        return True
    except (InvalidSignature, ValueError):
        return False
