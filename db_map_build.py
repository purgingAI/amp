#!/usr/bin/env python3
"""Build a manager-friendly TABLE → FEATURE map for APP_DB_dev.

For every table in the DB, work out which product feature (code module) uses it, and flag tables
referenced NOWHERE in code as discard candidates. Framework tables (Quartz/Liquibase/Hibernate) are
categorised as infra, not "discard". Writes data/db_map.json.

"Used" = the exact table name appears anywhere in the wallet repos (whole-word, case-insensitive) —
biased to HIGH RECALL so we never tell a manager to drop a table that's actually referenced (even in
raw SQL). Feature attribution = the code module that references it (JPA @Table entity ownership wins).

Laptop-side: needs the wallet repos on disk + the DB over VPN (creds in ~/cba/config.env). system python3.
"""
import os, re, json, sqlite3, subprocess, datetime, collections
from pathlib import Path

HOME = Path.home()
# scan ALL company repos the user has (same discovery + deny-list as the code-understanding graph)
DENY = {"flutter", ".pyenv", "team-lens", "amp", "node_modules", "go", "sdk", "build", "target",
        "trading-ai", "creator-os", "vantage", "employee-tracker", "ace_app", "amp-ios",
        "cba", "citybank", "citybank-cbt", "wallet-rmc", "core-api",
        "core-api", "core-api"}
SCAN_PARENTS = [HOME / "company-repos", HOME / "Documents", HOME / "Downloads", HOME]


def discover_company_repos():
    repos, seen = [], set()
    for parent in SCAN_PARENTS:
        if not parent.exists():
            continue
        for c in sorted(parent.iterdir()):
            if (c.is_dir() and (c / ".git").is_dir() and c.name not in DENY
                    and not c.name.startswith(".") and c.name not in seen):
                seen.add(c.name)
                repos.append(c)
    return repos


REPOS = discover_company_repos()
CODE_DB = HOME / "amp/data/code_knowledge.db"
OUT = HOME / "amp/data/db_map.json"
CFG = HOME / "cba/config.env"
INCLUDES = ["--include=*." + e for e in ("java", "kt", "sql", "xml", "yml", "yaml",
            "properties", "json", "ts", "tsx", "js", "go", "py")]
INFRA_RE = re.compile(r"^(qrtz_|databasechangelog|flyway_|hibernate_seq|shedlock|batch_|act_)", re.I)


def load_env(p):
    e = {}
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            e[k] = v.strip().strip('"').strip("'")
    return e


def db_tables(cfg):
    import pymysql
    cn = pymysql.connect(host=cfg["CBA_MYSQL_HOST"], port=int(cfg.get("CBA_MYSQL_PORT", 3306)),
                         user=cfg["CBA_MYSQL_USER"], password=cfg["CBA_MYSQL_PASS"],
                         database=cfg["CBA_MYSQL_DB"], connect_timeout=10)
    cur = cn.cursor()
    cur.execute("""SELECT table_name, table_rows, ROUND((data_length+index_length)/1048576,2)
                   FROM information_schema.tables
                   WHERE table_schema=DATABASE() AND table_type='BASE TABLE'""")
    out = {r[0].lower(): {"name": r[0], "rows": int(r[1] or 0), "size_mb": float(r[2] or 0)}
           for r in cur.fetchall()}
    cn.close()
    return out


def _grep(args, timeout=400):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        print("grep failed:", str(e)[:120]); return ""


def _module_of(path):
    parts = path.split("/")
    for i, p in enumerate(parts):
        if p == "src" and i > 0:
            return parts[i - 1]
    # fall back to the repo-relative top dir
    for r in REPOS:
        rp = str(r) + "/"
        if path.startswith(rp):
            rest = path[len(rp):].split("/")
            return rest[0] if rest else "?"
    return "?"


def code_purposes():
    out = {}
    if CODE_DB.exists():
        try:
            c = sqlite3.connect(CODE_DB)
            for mod, purpose in c.execute("SELECT module, purpose FROM modules"):
                if mod:
                    out[mod.lower()] = purpose or ""
            c.close()
        except Exception:
            pass
    return out


def entity_owner(dbset):
    """table(lower) -> module, from @Table(name="…") / @Table(name='…') JPA entities."""
    out = {}
    txt = _grep(["grep", "-rHo", "--include=*.java", r'@Table(name *= *["\x27][^"\x27]*["\x27]', *map(str, REPOS)])
    for line in txt.splitlines():
        i = line.find(":@Table")
        if i < 0:
            continue
        path, ann = line[:i], line[i + 1:]
        m = re.search(r'@Table\(name *= *["\']([^"\']+)["\']', ann)
        if m and m.group(1).lower() in dbset:
            out[m.group(1).lower()] = _module_of(path)
    return out


def references(dbset):
    """table(lower) -> Counter(module) for every whole-word appearance of the name in the repos."""
    pf = "/tmp/_dbtabs.txt"
    open(pf, "w").write("\n".join(sorted(m["name"] for m in dbset.values())))
    txt = _grep(["grep", "-rHiowF", "-f", pf, *INCLUDES, *map(str, REPOS)])
    refs = collections.defaultdict(collections.Counter)
    for line in txt.splitlines():
        i = line.rfind(":")
        if i < 0:
            continue
        path, match = line[:i], line[i + 1:].strip().lower()
        if match in dbset:
            refs[match][_module_of(path)] += 1
    return refs


FEATURE_HINTS = ("Incentive & Campaigns, Treasury, Refunds, Settlement, Cross-border Payouts, "
                 "KYC & Onboarding, Beneficiaries, Notifications, Wallet & Ledger, Merchant Management, "
                 "User & Authentication, Reporting & Analytics, AML & Risk, Cards, Bank Integrations, "
                 "Store & Terminal, QR & Payments, Configuration")
FEAT_CACHE = HOME / "amp/data/db_features.json"


def product_features(items, client):
    """Map each used table to a business PRODUCT feature (Incentive, Treasury, …) via the LLM.
    Incrementally cached so hourly rebuilds only classify NEW tables."""
    cache = {}
    if FEAT_CACHE.exists():
        try:
            cache = json.load(open(FEAT_CACHE))
        except Exception:
            cache = {}
    todo = [it for it in items if it["table"] not in cache]
    if todo and client is not None:
        for i in range(0, len(todo), 90):
            chunk = todo[i:i + 90]
            listing = "\n".join(f"- {it['table']} (code module: {it['module']})" for it in chunk)
            prompt = ("You map database tables of a fintech wallet platform to the PRODUCT FEATURE they belong to — "
                      "the business capability a manager cares about, NOT the technical microservice. Prefer these "
                      f"when they fit: {FEATURE_HINTS}. Use a concise Title-Case product name; invent a new one only "
                      "if none fit.\n\nTables:\n" + listing +
                      "\n\nReturn ONLY a JSON object mapping each table name (exactly as given) to its product feature.")
            try:
                msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=1600,
                                             messages=[{"role": "user", "content": prompt}])
                txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                txt = txt[txt.find("{"): txt.rfind("}") + 1]
                for k, v in json.loads(txt).items():
                    if isinstance(v, str):
                        cache[k.lower()] = v.strip()
            except Exception as e:
                print("feature LLM chunk failed:", str(e)[:120])
        try:
            json.dump(cache, open(FEAT_CACHE, "w"))
        except Exception:
            pass
    return cache


def main():
    cfg = load_env(CFG)
    tables = db_tables(cfg)                    # key = lower name
    owners = entity_owner(tables)              # entity ownership (precise)
    refs = references(tables)                  # any code reference (high recall)

    used = set()
    for t in tables:
        tables[t]["infra"] = bool(INFRA_RE.match(t))
        tables[t]["module"] = owners.get(t) or (refs[t].most_common(1)[0][0] if refs.get(t) else None)
        tables[t]["used"] = (t in owners) or (t in refs)
        if tables[t]["used"]:
            used.add(t)

    # cluster the used (non-infra) tables into business PRODUCT features via the LLM
    used_items = [{"table": t, "module": tables[t]["module"] or "?"}
                  for t in used if not tables[t]["infra"]]
    try:
        from app import orchestrator as _o
        client = _o._client
    except Exception:
        client = None
    prod = product_features(used_items, client)

    feat_tables = collections.defaultdict(set)
    for it in used_items:
        f = prod.get(it["table"]) or it["module"] or "Other"
        feat_tables[f].add(it["table"])
        tables[it["table"]]["feature"] = f

    unused = sorted((tables[t]["name"] for t in tables
                     if not tables[t]["used"] and not tables[t]["infra"]), key=str.lower)
    infra = sorted((tables[t]["name"] for t in tables if tables[t]["infra"]), key=str.lower)

    features = []
    for feat, ts in sorted(feat_tables.items(), key=lambda kv: -len(kv[1])):
        modules = sorted({tables[t]["module"] for t in ts if tables[t]["module"]})
        features.append({"feature": feat, "modules": modules, "table_count": len(ts),
                         "tables": sorted(tables[t]["name"] for t in ts)})

    tbl_list = [{"name": m["name"], "feature": m.get("feature"), "used": m["used"],
                 "infra": m["infra"], "rows": m["rows"], "size_mb": m["size_mb"]}
                for m in sorted(tables.values(), key=lambda x: x["name"].lower())]

    payload = {"database": cfg.get("CBA_MYSQL_DB", "APP_DB_dev"),
               "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "table_count": len(tables), "used_count": len(used),
               "unused_count": len(unused), "infra_count": len(infra),
               "feature_count": len(features), "features": features,
               "tables": tbl_list, "unused": unused, "infra": infra}
    OUT.parent.mkdir(exist_ok=True)
    json.dump(payload, open(OUT, "w"))
    print(f"db_map: {payload['table_count']} tables · {payload['used_count']} used · "
          f"{payload['feature_count']} features · {payload['unused_count']} discard-candidates · "
          f"{payload['infra_count']} infra → {OUT}")


if __name__ == "__main__":
    main()
