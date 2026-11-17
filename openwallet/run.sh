#!/usr/bin/env bash
# PayI -> OpenWallet two-screen demo (real dev transaction).
#   ./openwallet/run.sh     screen 1 http://localhost:8440/send , screen 2 .../watch
cd "$(dirname "$0")/.."
exec ./.venv/bin/python openwallet/server.py --host 0.0.0.0 --port "${OPENWALLET_PORT:-8440}"
