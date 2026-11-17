#!/usr/bin/env bash
# The OpenWallet assistant on a shared Linux server (no root, no systemd lingering).
#
#   ./run_server.sh           start whatever is not running (idempotent; cron calls this
#                             every minute and at reboot, so a crash is back within a minute)
#   ./run_server.sh status    what is listening
#   ./run_server.sh stop      stop everything (and take it out of cron)
#
# Same stack as the laptop's up.sh: intent model v16 (+ Laya second opinion), whisper.cpp
# speech, the engine, the HTTPS customer app. Differences, on purpose, because this box is
# SHARED:
#   * only the HTTPS app (:8443) and the signed-in, HTTPS Learn page (:8766) are reachable from the network. The engine, the model and
#     speech listen on 127.0.0.1 — the engine's console and event stream have no login of
#     their own and show every customer's conversation. Reach them with an SSH tunnel.
#   * CX_REQUIRE_LOGIN=1: every customer call needs a real OpenWallet login (mobile, OTP, PIN).
#   * CX_KEEP_STT_CLIPS=0: no voice recordings are kept on this server.
set -u
AMP="$HOME/amp"; W="$HOME/glasslink/whisper.cpp"; LOG="$AMP/logs"; mkdir -p "$LOG"
[ -f "$AMP/.env.vault" ] && set -a && . "$AMP/.env.vault" && set +a       # Vault access (chmod 600)
LIVE_MODEL="${CX_LIVE_MODEL:-openwallet-cx-intent-ml-v24}"

up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
start() {  # name port logfile command...
  local name=$1 port=$2 log=$3; shift 3
  if up "$port"; then return; fi
  # still loading (the intent model takes ~30 s before it opens its port): don't start twice
  if pgrep -u "$(id -u)" -f -- "--port $port" >/dev/null; then return; fi
  ( cd "$AMP" && nohup setsid "$@" >> "$LOG/$log" 2>&1 < /dev/null & ) ; echo "  $name: starting on :$port"
}

case "${1:-up}" in
  status)
    for p in 8450:intent 8178:speech 8440:engine 8443:app 8766:learn 8467:studio; do
      printf '  %-7s :%s  ' "${p#*:}" "${p%%:*}"; up "${p%%:*}" && echo up || echo DOWN; done; exit 0;;
  stop)
    crontab -l 2>/dev/null | grep -v 'amp/deploy/run_server.sh' | crontab -
    for p in 8467 8766 8443 8440 8178 8450; do
      PID=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1)
      [ -n "$PID" ] && kill "$PID" && echo "  stopped :$p"; done; exit 0;;
esac

# keep it running: cron re-runs this every minute and at boot
if ! crontab -l 2>/dev/null | grep -q 'amp/deploy/run_server.sh'; then
  ( crontab -l 2>/dev/null; echo "* * * * * $AMP/deploy/run_server.sh >/dev/null 2>&1";
    echo "@reboot $AMP/deploy/run_server.sh >/dev/null 2>&1" ) | crontab -
fi

start intent 8450 intent.log env "CX_MODEL_DIR=$AMP/cx/model/$LIVE_MODEL" \
  "$AMP/laya/.venv/bin/python" "$AMP/cx/model/serve.py" --host 127.0.0.1 --port 8450
start speech 8178 stt.log "$W/build/bin/whisper-server" -m "$W/models/ggml-small-q5_1.bin" \
  --host 127.0.0.1 --port 8178 -t 6 -l auto --vad -vm "$W/models/ggml-silero-v6.2.0.bin" --suppress-nst
start engine 8440 engine.log env AMP_DEMO_KEY=1 CX_REQUIRE_PIN=1 CX_BILL_PAY_LIVE=1 \
  CX_REQUIRE_LOGIN=1 CX_KEEP_STT_CLIPS=0 CX_ENV=demo CX_STT_DETECT=0 CX_STT_PARTIALS=0 \
  "$AMP/.venv/bin/python" "$AMP/openwallet/server.py" --host 127.0.0.1 --port 8440
[ -f "$AMP/cx/app/certs/server.crt" ] || "$AMP/cx/app/certs.sh" >/dev/null
start app 8443 app.log python3 "$AMP/cx/app/serve_app.py" --host 0.0.0.0 --port 8443
# The Learn page (model explainer + Ten's tour), behind its own sign-in: @example.com email codes
# or the admin password. HTTPS with the app's certificate; it refuses plain HTTP on a network
# address. Its live demos call the engine on 127.0.0.1 only.
start learn 8766 learn.log "$AMP/.venv/bin/python" "$AMP/cx/docs/model-brief/serve.py" --host 0.0.0.0 --port 8766 \
  --engine http://127.0.0.1:8440 --cert "$AMP/cx/app/certs/server.crt" --key "$AMP/cx/app/certs/server.key"
# Agent Studio (agent graph, Laya routing, document collections, training jobs) — 127.0.0.1 only;
# the Learn page above is its front door (sign-in, admin check). Training runs here, not on laptops.
start studio 8467 studio.log env STUDIO_THREADS=4 "$AMP/laya/.venv/bin/python" "$AMP/cx/studio/app.py" --port 8467
