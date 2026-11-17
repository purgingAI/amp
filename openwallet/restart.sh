#!/usr/bin/env bash
# Restart the demo server using a PIDFILE.
#
# Why not pkill: the obvious `pkill -f <path>` matches the shell that is running the command,
# because the pattern appears in that shell's own command line. It kills the caller. A pidfile
# has no such ambiguity.
cd "$(dirname "$0")/.."
PORT="${1:-8440}"
PIDFILE="/tmp/openwallet-demo-$PORT.pid"
if [ -f "$PIDFILE" ]; then
  kill "$(cat "$PIDFILE")" 2>/dev/null
  sleep 1
fi
# No device yet, so the demo key is explicitly enabled here. Every screen still
# labels the signature as "not hardware-held".
# CX_REQUIRE_PIN: every money-moving action asks for the PIN and validates it against the
#   wallet's own /user/pin-validate. NOTE: five wrong attempts FREEZE the wallet server-side
#   (UserStatus.FROZEN, with email/SMS/push) and that counter is not ours to reset — the
#   assistant stops after two.
# CX_BILL_PAY_LIVE: bill payments really debit the wallet. Off, the request is built and
#   shown but not sent.
AMP_DEMO_KEY=1 CX_REQUIRE_PIN=1 CX_BILL_PAY_LIVE=1 \
  setsid ./.venv/bin/python openwallet/server.py --host 0.0.0.0 --port "$PORT" \
  > "/tmp/openwallet-demo-$PORT.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown
sleep 4
printf 'server :%s  watch=%s  matrix=%s\n' "$PORT" \
  "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/watch")" \
  "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/cx/matrix")"
