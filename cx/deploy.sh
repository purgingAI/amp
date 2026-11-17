#!/usr/bin/env bash
# Deploy the OpenWallet CX assistant onto a host INSIDE the dev VPC.
#
# Why a VPC host rather than our EC2 box: the assistant reads real bills, campaigns, ledger
# rows and balances from dev MySQL (127.0.0.1), Mongo (127.0.0.1) and Vault
# (127.0.0.1). Those sit in ap-south-1 and are not reachable from the eu-north-1 EC2
# instance, so that box can only ever proxy to a laptop that must stay awake. A host already
# inside the VPC reaches them directly and needs neither a VPN nor a tunnel.
#
#   ./deploy.sh                      deploy and start
#   ./deploy.sh status               what is running
#   ./deploy.sh stop                 stop the services
#   ./deploy.sh remove               stop AND delete the tree (do this when the demo is done)
#
# TARGET IS SOMEBODY ELSE'S SERVER. It is a temporary demo host. `remove` exists because the
# deployed tree contains .secrets.json — a JWT secret, the Mongo URI and per-user encryption
# keys — and leaving that on a shared box is the part that actually matters.
set -euo pipefail

HOST="${CX_DEPLOY_HOST:-riley@127.0.0.1}"
DIR="${CX_DEPLOY_DIR:-/home/riley/openwallet-cx}"
PORT="${CX_DEPLOY_PORT:-8440}"
MODEL_PORT="${CX_DEPLOY_MODEL_PORT:-8450}"
SIM_PORT="${CX_DEPLOY_SIM_PORT:-8480}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # ~/amp

ssh_() { ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
             -o ConnectTimeout=15 "$HOST" "$@"; }

# Start a long-running process on the far side and RETURN. Getting this wrong is why the
# first deploy hung: the service came up fine, but ssh sat waiting for the daemon's stdout
# to close and the script never reached the next step. nohup + setsid + every descriptor
# redirected + an explicit exit is what actually detaches it.
start_remote() {   # start_remote <logfile> <port> <command...>
  # FIRE AND FORGET, then verify the PORT.
  #
  # ssh does not return from this even with -n, every descriptor redirected, setsid, nohup
  # and an explicit exit — it holds the channel open for the daemon. The daemon itself starts
  # and survives perfectly well (verified: process alive, port listening, outlives the
  # session). So waiting on ssh was measuring the wrong thing, and under `set -e` the
  # timeout killed the deploy at the first service every time.
  #
  # What matters is whether the port answers, so that is what we check.
  local log="$1" port="$2"; shift 2
  timeout 10 ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=15 "$HOST" \
      "cd / && nohup setsid bash -c '$*' > '$log' 2>&1 < /dev/null & exit 0" >/dev/null 2>&1 || true
  for _ in $(seq 1 "${3:-20}"); do
    ssh_ "ss -ltn 2>/dev/null | grep -q ':$port '" 2>/dev/null && return 0
    sleep 2
  done
  echo "  WARNING: nothing listening on :$port — log follows"
  ssh_ "tail -6 '$log'" 2>/dev/null || true
  return 1
}

case "${1:-deploy}" in
status)
  ssh_ "for p in $PORT $MODEL_PORT $SIM_PORT; do printf '  :%s  ' \$p;
        ss -ltn 2>/dev/null | grep -q \":\$p \" && echo listening || echo down; done;
        echo '  disk:' \$(df -h $DIR 2>/dev/null | tail -1 | awk '{print \$4}') free"
  exit 0;;
stop)
  ssh_ "for p in $PORT $MODEL_PORT $SIM_PORT; do
          PID=\$(ss -ltnp 2>/dev/null | grep \":\$p \" | grep -oP 'pid=\\K[0-9]+' | head -1);
          [ -n \"\$PID\" ] && kill \$PID && echo \"  stopped :\$p (\$PID)\" || echo \"  :\$p not running\";
        done"
  exit 0;;
remove)
  "$0" stop || true
  # The tree holds .secrets.json. Deleting it is the point of this command.
  ssh_ "rm -rf '$DIR' && echo '  removed $DIR'"
  echo "Remember: nothing of ours should remain on $HOST now. Verify with: $0 status"
  exit 0;;
esac

echo "→ checking the target"
ssh_ "echo '  host:' \$(hostname); echo '  cores:' \$(nproc); \
      echo '  ram:' \$(free -m | awk '/Mem:/{print \$2\"MB total, \"\$7\"MB available\"}'); \
      echo '  python:' \$(python3 -V 2>&1)"

echo "→ can it reach dev directly?"
ssh_ "for hp in 127.0.0.1:3306 127.0.0.1:27017 127.0.0.1:8200; do
        h=\${hp%:*}; p=\${hp#*:}; printf '  %-22s ' \$hp;
        timeout 5 bash -c \"cat < /dev/null > /dev/tcp/\$h/\$p\" 2>/dev/null && echo open || echo UNREACHABLE;
      done"

echo "→ copying"
ssh_ "mkdir -p '$DIR'"
# --delete so a redeploy cannot leave a stale module shadowing a new one. Excludes keep the
# model weights (470MB) and caches out of the copy; the model is fetched on the far side.
rsync -az --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
  --exclude 'model/openwallet-cx-intent-*' --exclude 'data/schedules.sqlite3' \
  --exclude 'laya' \
  "$HERE/cx" "$HERE/openwallet" "$HERE/app" \
  -e "ssh -o StrictHostKeyChecking=accept-new" "$HOST:$DIR/"

echo "→ python environment"
# Two venvs, same split as the laptop and for the same reason: the model needs torch, the
# payment path does not, and keeping them apart means the sidecar can be restarted or
# switched off without touching anything that moves money.
ssh_ "cd '$DIR' && python3 -m venv .venv 2>/dev/null || true; \
      ./.venv/bin/pip -q install --upgrade pip >/dev/null && \
      ./.venv/bin/pip -q install pymysql pymongo cryptography requests 2>&1 | tail -2; \
      echo '  app venv ready'"

if [ "${CX_WITH_MODEL:-1}" = "1" ]; then
  echo "→ model environment (torch, CPU wheels only)"
  # --index-url cpu: the default wheel pulls ~2.5GB of CUDA libraries onto a box with no GPU.
  ssh_ "cd '$DIR' && python3 -m venv .venv-model 2>/dev/null || true; \
        ./.venv-model/bin/pip -q install --upgrade pip >/dev/null && \
        ./.venv-model/bin/pip -q install --index-url https://download.pytorch.org/whl/cpu \
            torch 2>&1 | tail -1 && \
        ./.venv-model/bin/pip -q install transformers 2>&1 | tail -1; \
        echo '  model venv ready:' \$(./.venv-model/bin/python -c 'import torch;print(torch.__version__)' 2>&1)"
fi

echo "→ starting the DTOne stand-in (the one piece that is not real)"
ssh_ "pkill -f 'sim.py --port $SIM_PORT' 2>/dev/null; exit 0"
start_remote /tmp/cx-dtone.log "$SIM_PORT" \
  "cd '$DIR/cx/bills' && ../../.venv/bin/python sim.py --port $SIM_PORT" \
  && echo "  sim up on :$SIM_PORT"

if [ "${CX_WITH_MODEL:-1}" = "1" ]; then
  echo "→ copying the model ($(du -sh "$HERE/cx/model/openwallet-cx-intent-ml-v5" 2>/dev/null | cut -f1)) and starting the sidecar"
  rsync -az --info=progress2 "$HERE/cx/model/openwallet-cx-intent-ml-v5" \
    -e "ssh -o StrictHostKeyChecking=accept-new" "$HOST:$DIR/cx/model/" 2>&1 | tail -1
  # Bound to localhost. The sidecar is an opinion service with no authentication of its own
  # and this is a shared host; nothing outside the box needs to reach it.
  ssh_ "pkill -f 'serve.py --host 127.0.0.1 --port $MODEL_PORT' 2>/dev/null; exit 0"
  echo "  starting the sidecar (the model takes ~30s to load)"
  start_remote /tmp/cx-model.log "$MODEL_PORT" \
    "cd '$DIR/cx/model' && ../../.venv-model/bin/python serve.py --host 127.0.0.1 --port $MODEL_PORT" \
    || true
  ssh_ "for i in \$(seq 1 120); do \
          curl -s -m 2 http://127.0.0.1:$MODEL_PORT/health 2>/dev/null | grep -q 'true' && break; \
          sleep 2; done; \
        curl -s -m 5 http://127.0.0.1:$MODEL_PORT/health | head -c 200; echo"
fi

echo "→ starting the assistant"
ssh_ "pkill -f 'openwallet/server.py --host 0.0.0.0 --port $PORT' 2>/dev/null; exit 0"
start_remote /tmp/cx-assist.log "$PORT" \
  "cd '$DIR' && AMP_DEMO_KEY=1 CX_REQUIRE_PIN=1 CX_BILL_PAY_LIVE=0 CX_DTONE=sim CX_DTONE_SIM=http://127.0.0.1:$SIM_PORT ./.venv/bin/python openwallet/server.py --host 0.0.0.0 --port $PORT" \
  && echo "  assistant up on :$PORT"
ssh_ "tail -3 /tmp/cx-assist.log" 2>/dev/null || true

echo
echo "→ verifying from here"
for path in assist watch; do
  printf '  %-8s ' "/$path"
  curl -s -o /dev/null -m 10 -w '%{http_code}\n' "http://${HOST#*@}:$PORT/$path" || echo unreachable
done
echo
echo "  open  http://${HOST#*@}:$PORT/assist"
echo
echo "NOTE: CX_BILL_PAY_LIVE is 0 on this host — live bill payments are OFF by default on a"
echo "      shared server. Set it to 1 deliberately if you want them there."
echo "REMEMBER: ./deploy.sh remove  when the demo is finished."
