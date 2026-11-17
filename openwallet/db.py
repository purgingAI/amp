"""Vault -> dev MySQL helper. Never prints credentials."""
import json, re, subprocess, sys, urllib.request

def bashrc_vars():
    out = subprocess.run(["bash","-lc","set -a; source ~/.bashrc >/dev/null 2>&1; "
                          "echo \"$VAULT_URI|$VAULT_ROLE_ID|$VAULT_SECRET_ID|$VAULT_ENV\""],
                         capture_output=True, text=True).stdout.strip()
    uri, rid, sid, env = (out.split("|") + ["","","",""])[:4]
    return uri.rstrip("/"), rid, sid, (env or "dev")

def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    if token: req.add_header("X-Vault-Token", token)
    return json.load(urllib.request.urlopen(req, timeout=20))

def get(url, token):
    req = urllib.request.Request(url, headers={"X-Vault-Token": token})
    return json.load(urllib.request.urlopen(req, timeout=20))

def db_conn():
    uri, rid, sid, env = bashrc_vars()
    tok = post(f"{uri}/v1/auth/approle/login", {"role_id":rid,"secret_id":sid})["auth"]["client_token"]
    cfg = get(f"{uri}/v1/{env}/data/all-config", tok)["data"]["data"]
    url, user, pwd = cfg["spring.datasource.url"], cfg["spring.datasource.username"], cfg["spring.datasource.password"]
    m = re.match(r"jdbc:mysql://([^:/]+):?(\d+)?/([^?]+)", url)
    print(f"connecting to {m.group(1)}:{m.group(2) or 3306}/{m.group(3)} as {user[:3]}***", flush=True)
    import pymysql
    return pymysql.connect(host=m.group(1), port=int(m.group(2) or 3306), user=user, password=pwd,
                           database=m.group(3), cursorclass=pymysql.cursors.DictCursor,
                           autocommit=False, connect_timeout=20)
