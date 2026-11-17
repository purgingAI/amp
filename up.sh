#!/usr/bin/env bash
# Start the whole OpenWallet assistant stack as USER SYSTEMD services.
#
# Why not nohup/setsid: those processes stay in the launching session's cgroup and were
# killed whenever a Claude Code session ended — v14's training died at step 3,000 of 3,516
# that way. A transient user unit lives in its own cgroup and survives.
#
#   ./up.sh            start anything that is not running (idempotent)
#   ./up.sh status     list the units
#   ./up.sh stop       stop them all
#   restart one:       systemctl --user restart openwallet-engine
#   logs:              journalctl --user -u openwallet-engine -f
set -u
AMP=~/amp
W=~/glasslink/whisper.cpp
UNITS=(openwallet-intent openwallet-stt openwallet-engine openwallet-app openwallet-brief)
# Vault access for the engine (PII decryption, merchants, MySQL). A service has no login shell,
# and ~/.bashrc returns early when not interactive, so read the four values once from an
# interactive shell and hand them to the units. Visible only to this user.
VAULT_ENV_ARGS=()
while IFS='=' read -r k v; do [ -n "$v" ] && VAULT_ENV_ARGS+=(-E "$k=$v"); done < <(
  bash -ic 'for k in VAULT_URI VAULT_ROLE_ID VAULT_SECRET_ID VAULT_ENV; do echo "$k=${!k}"; done' 2>/dev/null)
unit() {  # name, workdir, command...
  local name=$1 dir=$2; shift 2
  if systemctl --user is-active --quiet "$name"; then echo "  $name: already running"; return; fi
  # a unit stuck in a restart loop still holds its name: clear it before starting afresh
  systemctl --user stop "$name" 2>/dev/null; systemctl --user reset-failed "$name" 2>/dev/null
  systemd-run --user --unit="$name" --collect -p WorkingDirectory="$dir" \
    -p Restart=on-failure -p RestartSec=3 -p LimitNOFILE=65536 "${VAULT_ENV_ARGS[@]}" "$@" >/dev/null && echo "  $name: started"
}
case "${1:-up}" in
  status) systemctl --user list-units 'openwallet-*' --no-legend; exit 0;;
  stop)   for u in "${UNITS[@]}"; do systemctl --user stop "$u" 2>/dev/null && echo "  $u: stopped"; done; exit 0;;
esac
# The LIVE intent model. Promoted to v16 on 2026-10-14 (BRD 92.6% end to end, leaks = v9).
# Roll back: systemctl --user stop openwallet-intent; CX_LIVE_MODEL=openwallet-cx-intent-ml-v23 ~/amp/up.sh   (v24 live 2026-10-16)
LIVE_MODEL="${CX_LIVE_MODEL:-openwallet-cx-intent-ml-v24}"
unit openwallet-intent "$AMP/cx/model" /usr/bin/env "CX_MODEL_DIR=$AMP/cx/model/$LIVE_MODEL" \
  "$AMP/laya/.venv/bin/python" "$AMP/cx/model/serve.py" --host 127.0.0.1 --port 8450
# Whisper on the Intel Arc GPU (Vulkan) when that build exists: 0.3-0.4 s per utterance vs
# 1.7 s on the CPU, identical text. The CPU build stays as the fallback.
WBIN="$W/build/bin/whisper-server"; [ -x "$W/build-vk/bin/whisper-server" ] && WBIN="$W/build-vk/bin/whisper-server"
unit openwallet-stt    "$W" "$WBIN" -m "$W/models/ggml-small-q5_1.bin" \
                  --host 127.0.0.1 --port 8178 -t 4 -l auto \
                  --vad -vm "$W/models/ggml-silero-v6.2.0.bin" --suppress-nst
# warm the GPU once: the first inference compiles shaders (~1 s); do it now, not on a customer
( for _ in $(seq 30); do curl -s -m 1 -o /dev/null http://127.0.0.1:8178/ && break; sleep 1; done
  python3 -c "import wave,sys; w=wave.open('/tmp/openwallet-warm.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\\0\\0'*16000); w.close()"
  curl -s -m 30 -o /dev/null http://127.0.0.1:8178/inference -F file=@/tmp/openwallet-warm.wav -F language=en ) &
unit openwallet-engine "$AMP" env AMP_DEMO_KEY=1 CX_REQUIRE_PIN=1 CX_BILL_PAY_LIVE=1 \
                  "$AMP/.venv/bin/python" "$AMP/openwallet/server.py" --host 0.0.0.0 --port 8440
unit openwallet-app    "$AMP/cx/app" /usr/bin/python3 "$AMP/cx/app/serve_app.py" --port 8443
# Learn page on the Wi-Fi too (HTTPS with the app's certificate; serve.py refuses plain HTTP off localhost)
unit openwallet-brief  "$AMP/cx/docs/model-brief" /usr/bin/python3 "$AMP/cx/docs/model-brief/serve.py" --host 0.0.0.0 --port 8766 \
                  --cert "$AMP/cx/app/certs/server.crt" --key "$AMP/cx/app/certs/server.key"
