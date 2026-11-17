#!/usr/bin/env python3
"""Local integration test for ACE 2.0: boot the hub against the fresh TeamLens graph, open ace2.html
headed, sign in with ACE's own login, and screenshot the People (committers) + code-map + ask flow."""
import os, time, subprocess, signal, urllib.request
from playwright.sync_api import sync_playwright

AMP = "/home/sagar-uniyal/amp"
PORT = 8899
env = dict(os.environ,
           AMP_KNOWLEDGE_DB="/home/sagar-uniyal/team-lens/data/teamlens.db",
           ACE_USERS_DB="/tmp/ace_users_local.db")
os.path.exists("/tmp/ace_users_local.db") and os.remove("/tmp/ace_users_local.db")

srv = subprocess.Popen([f"{AMP}/.venv/bin/python", "-m", "uvicorn", "app.hub:app",
                        "--host", "127.0.0.1", "--port", str(PORT)],
                       cwd=AMP, env=env, stdout=open("/tmp/ace_hub.log", "w"), stderr=subprocess.STDOUT)
def up():
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ace2.html", timeout=1); return True
    except Exception: return False
for _ in range(40):
    if up(): break
    time.sleep(0.5)
print("hub up:", up())
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, slow_mo=180, args=["--start-maximized"])
        pg = b.new_context(viewport=None).new_page()
        pg.goto(f"http://127.0.0.1:{PORT}/ace2.html", wait_until="networkidle", timeout=20000)
        time.sleep(2); pg.screenshot(path="/tmp/ace2_login.png"); print("shot /tmp/ace2_login.png")
        # attempt login: fill the first email-ish input and submit
        try:
            em = pg.query_selector("input[type=email]") or pg.query_selector("input[placeholder*='mail' i]") or pg.query_selector("input")
            if em:
                em.click(); em.fill("sagar@example.com")
                btn = pg.query_selector("button")
                if btn: btn.click()
                time.sleep(4)
        except Exception as e:
            print("login attempt error:", e)
        time.sleep(3); pg.screenshot(path="/tmp/ace2_dash.png", full_page=True); print("shot /tmp/ace2_dash.png")
        b.close()
finally:
    srv.send_signal(signal.SIGINT); time.sleep(1); srv.terminate()
    print("hub stopped")
