#!/usr/bin/env bash
# OpenWallet customer app demo, over HTTPS on the LAN (phones can use the microphone).
#   ./run.sh          start everything this app needs (detached)
#   ./run.sh stop     stop the app server
# Needs the openwallet server (the engine) on :8440 — start it with ~/amp/openwallet/restart.sh.
cd "$(dirname "$0")"
PORT="${PORT:-8443}"
pid() { ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$"' | grep -oP 'pid=\K[0-9]+' | head -1; }
if [ "${1:-}" = "stop" ]; then P=$(pid); [ -n "$P" ] && kill "$P" && echo "app stopped"; exit 0; fi
# a new network means a new IP; the certificate must name it or phones refuse it
for ip in $(hostname -I); do grep -q "IP:$ip" certs/san.txt 2>/dev/null || { ./certs.sh >/dev/null; break; }; done
[ -f certs/server.crt ] || ./certs.sh >/dev/null
~/amp/stt/run.sh
P=$(pid); [ -n "$P" ] && kill "$P" && sleep 1
nohup setsid python3 serve_app.py --port "$PORT" > "/tmp/openwallet-app-$PORT.log" 2>&1 < /dev/null & disown
sleep 1
curl -sk -o /dev/null -w "" "https://127.0.0.1:$PORT/" && echo "OpenWallet app running:"
for ip in $(hostname -I | tr ' ' '\n' | grep -E '^(192\.168|10\.)'); do echo "   https://$ip:$PORT/        (phone: install https://$ip:$PORT/ca.crt once to skip the warning)"; done
echo "   https://localhost:$PORT/"
curl -s -m 2 -o /dev/null http://127.0.0.1:8440/ || echo "   WARNING: the engine (:8440) is not running — ~/amp/openwallet/restart.sh"
