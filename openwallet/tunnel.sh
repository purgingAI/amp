#!/usr/bin/env bash
# Expose the laptop's PayI->OpenWallet demo at https://localhost/payi/
#
# The demo CANNOT run on the EC2 box: the TPM device key is sealed to this laptop and
# the dev MySQL/Vault hosts are only reachable from here. So the laptop stays the
# server and Caddy proxies /payi/* down this reverse tunnel, exactly as /glass does.
#   ./openwallet/tunnel.sh         open (or re-open) the tunnel
#   ./openwallet/tunnel.sh stop    close it
HOST=ubuntu@16.171.46.219
FWD=8098:127.0.0.1:${OPENWALLET_PORT:-8440}
pkill -f "$FWD" 2>/dev/null
[ "$1" = "stop" ] && { echo "tunnel closed"; exit 0; }

# A dropped laptop connection leaves sshd STILL LISTENING on the remote port while
# forwarding nowhere, so every request times out and the local process looks healthy.
# ExitOnForwardFailure then refuses to rebind. Clear the stale forwarder first.
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$HOST" \
  'PID=$(sudo ss -ltnp 2>/dev/null | grep "127.0.0.1:8098" | grep -oP "pid=\K[0-9]+" | head -1);
   [ -n "$PID" ] && [ "$PID" != "$PPID" ] && sudo kill "$PID"' 2>/dev/null
sleep 1
setsid ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -R "$FWD" "$HOST"
sleep 3
if pgrep -f "$FWD" >/dev/null; then
  echo "tunnel up"
  echo "  send   https://localhost/payi/send     (localhost for the mic)"
  echo "  watch  https://localhost/payi/watch"
  echo "  wallet https://localhost/payi/wallet"
else
  echo "tunnel FAILED to start"; exit 1
fi
