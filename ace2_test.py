#!/usr/bin/env python3
"""Boot the AMP hub against the enriched TeamLens graph, open ACE 2.0 (ace2.html) headed,
log in with ACE's own login, and screenshot the dashboard (committers + code graph + globe)."""
import os, time, subprocess, signal, urllib.request
from playwright.sync_api import sync_playwright

os.environ["AMP_KNOWLEDGE_DB"] = os.path.expanduser("~/team-lens/data/teamlens.db")
os.environ["ACE_USERS_DB"] = "/tmp/ace_users_demo.db"
try: os.remove("/tmp/ace_users_demo.db")
except OSError: pass

srv = subprocess.Popen([os.path.expanduser("~/amp/.venv/bin/python"), "-m", "uvicorn",
                        "app.hub:app", "--host", "127.0.0.1", "--port", "8899"],
                       cwd=os.path.expanduser("~/amp"), env=os.environ,
                       stdout=open("/tmp/ace2hub.log", "w"), stderr=subprocess.STDOUT)
def up():
    try: urllib.request.urlopen("http://127.0.0.1:8899/ace2.html", timeout=1); return True
    except Exception: return False
for _ in range(40):
    if up(): break
    time.sleep(0.5)
print("hub up:", up())

try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, slow_mo=200, args=["--start-maximized"])
        pg = b.new_context(viewport=None).new_page()
        pg.goto("http://127.0.0.1:8899/ace2.html", wait_until="networkidle", timeout=20000)
        time.sleep(1.5); pg.screenshot(path="/tmp/ace2_login.png"); print("shot login")
        # log in: fill first email-ish input, submit
        try:
            inp = pg.query_selector("input[type=email]") or pg.query_selector("input")
            inp.fill("sagar@example.com")
            for sel in ["input[placeholder*='name' i]"]:
                n = pg.query_selector(sel)
                if n: n.fill("Sagar")
            btn = None
            for b2 in pg.query_selector_all("button"):
                t = (b2.inner_text() or "").lower()
                if any(k in t for k in ("sign in", "enter", "login", "continue", "ace")):
                    btn = b2; break
            (btn or pg.query_selector("button")).click()
        except Exception as e:
            print("login interaction err:", e)
        time.sleep(4)  # let /ace/me, /ace/overview, /ace/people, /ace/codemap load
        pg.screenshot(path="/tmp/ace2_dashboard.png"); print("shot dashboard")
        # try an ask
        try:
            ta = pg.query_selector("textarea") or pg.query_selector("input[type=text]")
            if ta:
                ta.fill("Who owns the transaction module?")
                ta.press("Enter")
                time.sleep(7)
                pg.screenshot(path="/tmp/ace2_ask.png"); print("shot ask")
        except Exception as e:
            print("ask err:", e)
        time.sleep(2); b.close()
finally:
    try: srv.send_signal(signal.SIGINT)
    except Exception: pass
    time.sleep(1); srv.terminate()
    print("hub down")
