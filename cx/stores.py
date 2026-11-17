"""Real merchant store locations, from Mongo `store_location` / `store`.

Replaces the invented `metres` in the merchant catalogue with an actual distance, computed
from the customer's position to the merchant's nearest store.

Two things make this messier than it looks, and both are handled rather than assumed away:

* **The coordinates are not all numbers.** Some rows hold a clean float (`25.118055`), others
  hold a decorated string (`"25.10125° N (25° 6′ 4.5″ N)"`, `"55.2045° E"`). They were
  entered through different paths over time. `_coord` parses both and honours S/W as negative.
* **Most merchants have no store at all.** Of the ten in the demo, three carry store locations
  and seven do not. Those keep their placeholder distance and are marked
  `distance_source="demo"`, so nothing claims a measurement it does not have.

The customer's own position is still the static one the demo sends in `x-latitude` /
`x-longitude`. That is the next thing to become real; the merchant half already is.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.dirname(HERE)

#: Where the demo customer is standing.
#
#: Al Barsha, where the real store data actually clusters — not the transaction header's
#: Dubai-centre coordinates, which are ~12 km from every store on file.
#:
#: Worth stating plainly: the real stores are spread across Dubai, so the original "40m away"
#: placeholders were fiction. Distances are now kilometres, because that is what the data
#: says. The merchant half is real; THIS half is the next thing to replace with the device's
#: actual position. Override with CX_LAT / CX_LNG.
ME = (float(os.environ.get("CX_LAT", "25.1080")),
      float(os.environ.get("CX_LNG", "55.2025")))

_cache: dict[str, list[dict]] | None = None


def _coord(value) -> float | None:
    """A latitude or longitude from whatever shape the row happens to hold."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    # Leading decimal degrees, e.g. "25.10125° N (25° 6′ 4.5″ N)" or "-3.4" or "55.2045° E"
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    val = float(m.group(1))
    if re.search(r"\b[SW]\b|°\s*[SW]", text.upper()):
        val = -abs(val)
    return val


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine, in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _load() -> dict[str, list[dict]]:
    """Every usable store location, grouped by merchant user id. Read once per process."""
    global _cache
    if _cache is not None:
        return _cache
    out: dict[str, list[dict]] = {}
    try:
        # Share the one client. This is guarded by `_cache` so it only ran once, but a
        # second pool for a single read is still eight sockets nobody needed.
        from history import _mongo
        db, _ = _mongo()
        names = {s.get("location_id"): s for s in db["store"].find(
            {}, {"location_id": 1, "store_name": 1, "terminal_ids": 1, "status": 1})}
        for loc in db["store_location"].find({}):
            lat, lng = _coord(loc.get("latitude")), _coord(loc.get("longitude"))
            if lat is None or lng is None:
                continue
            store = names.get(loc.get("location_id")) or {}
            out.setdefault(loc.get("merchant_id"), []).append({
                "location_id": loc.get("location_id"),
                "store_name": store.get("store_name"),
                "terminals": store.get("terminal_ids") or [],
                "address": ", ".join(x for x in (loc.get("address_line1"),
                                                 loc.get("city")) if x),
                "lat": lat, "lng": lng,
            })
    except Exception:
        out = {}            # Mongo unreachable: callers fall back to placeholder distances
    _cache = out
    return _cache


def stores_for(user_id: str | None) -> list[dict]:
    return _load().get(user_id or "", [])


def nearest(user_id: str | None, me: tuple[float, float] | None = None) -> dict | None:
    """The merchant's closest store to the customer, with the distance in metres."""
    me = me or ME
    best = None
    for s in stores_for(user_id):
        d = distance_m(me, (s["lat"], s["lng"]))
        if best is None or d < best["metres"]:
            best = dict(s, metres=round(d))
    return best


def coverage() -> dict:
    data = _load()
    return {"merchants_with_stores": len(data),
            "locations": sum(len(v) for v in data.values())}


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(AMP, "openwallet", ".secrets.json")))
    print(f"  {coverage()}")
    print(f"  customer at {ME}\n")
    for m in cfg.get("merchants", []):
        n = nearest(m["userId"])
        if n:
            print(f"  {m['name'][:28]:30} {n['metres']:>6} m  {n['store_name'] or '-'} "
                  f"({n['address']})")
        else:
            print(f"  {m['name'][:28]:30}      —  no store location on file")
