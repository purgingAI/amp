#!/usr/bin/env bash
# AMP hub. Open http://localhost:8090 (or http://<your-LAN-ip>:8090 on a phone).
#
# ./run.sh          plain HTTP, as before
# ./run.sh --https  HTTPS with a self-signed cert covering this machine's LAN IPs
#
# Why HTTPS is an option at all: the palm/gesture pages call getUserMedia, and browsers only
# hand over a camera in a "secure context". localhost counts as one; http://192.168.x.x does
# NOT. So the second laptop in the handoff demo cannot see a camera over plain HTTP, however
# well everything else works. The cert is self-signed, so each browser shows a warning once
# and you click through — after that the origin is secure and the camera is allowed.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${AMP_PORT:-8090}"

if [[ "${1:-}" == "--https" || "${AMP_TLS:-}" == "1" ]]; then
  CERTS=data/certs
  mkdir -p "$CERTS"
  if [[ ! -s "$CERTS/hub.pem" || ! -s "$CERTS/hub.key" ]]; then
    # SANs for every address this host answers on, so the same cert works from any laptop on
    # the LAN. IP SANs, not just CN: browsers have ignored CN for host matching for years.
    SANS="DNS:localhost,IP:127.0.0.1"
    for ip in $(hostname -I); do SANS="$SANS,IP:$ip"; done
    echo "generating a self-signed cert for $SANS"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout "$CERTS/hub.key" -out "$CERTS/hub.pem" \
      -subj "/CN=amp-hub" -addext "subjectAltName=$SANS" 2>/dev/null
    chmod 600 "$CERTS/hub.key"
  fi
  echo "https://localhost:$PORT   (and https://$(hostname -I | awk '{print $1}'):$PORT from another laptop)"
  exec ./.venv/bin/python -m uvicorn app.hub:app --host 0.0.0.0 --port "$PORT" \
    --ssl-keyfile "$CERTS/hub.key" --ssl-certfile "$CERTS/hub.pem"
fi

exec ./.venv/bin/python -m uvicorn app.hub:app --host 0.0.0.0 --port "$PORT"
