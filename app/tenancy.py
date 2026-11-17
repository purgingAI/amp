"""Multi-tenant resolver for the knowledge graphs.

Each request runs under a tenant (derived from the authenticated user's company). Graph lookups
resolve to that tenant's DB at data/tenants/<tenant>/<kind>.db, falling back to the shared
data/<kind>.db when a tenant has no dedicated graph. Single-company today = everyone on the
'default' tenant => the shared DBs, so nothing changes until a customer graph is dropped in.

To onboard a customer: put their built graphs at data/tenants/<tenant>/knowledge.db and
data/tenants/<tenant>/code_knowledge.db, register the display name in TENANT_NAMES, and set their
users' `company` so tenant_for_company() maps to <tenant>.
"""
from __future__ import annotations
import os, re, contextvars
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT = os.environ.get("AMP_DEFAULT_TENANT", "default")
_current = contextvars.ContextVar("amp_tenant", default=DEFAULT)

# display names per tenant (falls back to AMP_COMPANY / "our company")
TENANT_NAMES = {"default": os.environ.get("AMP_COMPANY", "our company")}


def tenant_for_company(company: str | None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (company or "").strip().lower()).strip("-")
    return slug or DEFAULT


def set_tenant(tenant: str | None):
    _current.set(tenant or DEFAULT)


def get_tenant() -> str:
    return _current.get()


def company_name(tenant: str | None = None) -> str:
    return TENANT_NAMES.get(tenant or _current.get(), TENANT_NAMES["default"])


def db_path(kind: str) -> Path:
    """kind = 'knowledge' | 'code_knowledge'. Per-tenant DB if present, else the shared one."""
    t = _current.get()
    if t and t != DEFAULT:
        p = DATA / "tenants" / t / f"{kind}.db"
        if p.exists():
            return p
    return DATA / f"{kind}.db"


def list_tenants() -> list:
    tenants = {DEFAULT}
    td = DATA / "tenants"
    if td.exists():
        tenants |= {c.name for c in td.iterdir() if c.is_dir()}
    return sorted(tenants)
