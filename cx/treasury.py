"""Live exchange rates from OpenWallet's treasury (TMS) — the same call the Transaction service makes
for the smart router (TreasuryServiceImpl.fetchSmartRouterOptionsFromTreasury):

    GET {treasurySystemUrl}/{transactionOptionsUrl}/{ISO3}
        -> data.fromCountries[].toDestinations[].paymentModes[].serviceProviderOffers[]
                                                              .pricingRanges[].customerOfferedRate

Only the CUSTOMER-OFFERED rate and the transfer limits leave this module. The payload also
carries costRate, margins and provider liquidity — internal treasury numbers a customer must
never see — and they are dropped here, not "not displayed" later.

Fees are NOT taken from here: the cross-border fee is the product slab (treasury is FX only),
which is what the wallet charges.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

DEFAULT_BASE = "https://api.example.com"
DEFAULT_PATH = "tms-system/api/mobile/v1/transaction/options/to"
TTL = float(os.environ.get("CX_FX_TTL", "60"))       # rates move; a minute of caching is plenty

ISO3 = {"IN": "IND", "PK": "PAK", "BD": "BGD", "PH": "PHL", "NP": "NPL", "LK": "LKA", "EG": "EGY",
        "AE": "ARE"}
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, dict]] = {}
_URL: list[str] = []


def _base_url() -> str:
    """treasury_configuration (treasurySystemUrl + transactionOptionsUrl), as the backend does."""
    if _URL:
        return _URL[0]
    base, path = os.environ.get("CX_TREASURY_URL", ""), ""
    if not base:
        try:
            import devdb
            rows = {r["config_key"]: r["config_value"] for r in devdb.q(
                "select config_key, config_value from treasury_configuration where config_key in "
                "('treasurySystemUrl','transactionOptionsUrl')")}
            base, path = rows.get("treasurySystemUrl", ""), rows.get("transactionOptionsUrl", "")
        except Exception:                                          # noqa: BLE001
            pass
    url = f"{(base or DEFAULT_BASE).rstrip('/')}/{(path or DEFAULT_PATH).strip('/')}"
    _URL.append(url)
    return url


def quote(country_iso2: str) -> dict:
    """{"ok", "currency", "modes": [{"mode", "rate", "min", "max", "available", "best"}], "message"}.

    ok=False with a customer-safe message when the corridor is not active or treasury is down.
    """
    iso3 = ISO3.get((country_iso2 or "").upper())
    if not iso3:
        return {"ok": False, "message": "no corridor"}
    with _LOCK:
        hit = _CACHE.get(iso3)
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
    try:
        with urllib.request.urlopen(f"{_base_url()}/{iso3}", timeout=6) as r:
            body = json.loads(r.read())
    except Exception as e:                                         # noqa: BLE001
        return {"ok": False, "message": f"treasury unreachable ({type(e).__name__})"}
    out = {"ok": False, "message": body.get("message") or "", "modes": [], "currency": None}
    for fc in ((body.get("data") or {}).get("fromCountries") or []):
        if fc.get("countryCodeISO3") != "ARE":
            continue
        for dest in fc.get("toDestinations") or []:
            if dest.get("countryCodeISO3") != iso3:
                continue
            out["currency"] = (dest.get("currency") or {}).get("code")
            for pm in dest.get("paymentModes") or []:
                rates = [pr.get("customerOfferedRate")
                         for o in (pm.get("serviceProviderOffers") or [])
                         for pr in (o.get("pricingRanges") or [])
                         if isinstance(pr.get("customerOfferedRate"), (int, float))]
                if not rates:
                    continue
                out["modes"].append({"mode": pm.get("paymentType"), "rate": max(rates),
                                     "min": pm.get("minTransferAmount"), "max": pm.get("maxTransferAmount"),
                                     "available": bool(pm.get("available")), "best": bool(pm.get("isBest"))})
    out["ok"] = bool(out["modes"])
    with _LOCK:
        _CACHE[iso3] = (time.time(), out)
    return out


def best(q: dict) -> dict | None:
    """The mode the app would lead with: available first, then treasury's own 'best'."""
    modes = sorted(q.get("modes") or [], key=lambda m: (not m["available"], not m["best"], -m["rate"]))
    return modes[0] if modes else None
