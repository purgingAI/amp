#!/usr/bin/env bash
# PayI three-screen demo. Run on Sagar's laptop (it holds the TPM device key and the webcam).
#   ./payi/run.sh            → screen 1 http://localhost:8420/payer (Chrome, for mic + camera)
#   other laptops open the /watch and /wallet URLs it prints (same Wi-Fi)
cd "$(dirname "$0")/.."
exec ./.venv/bin/python payi/server.py --host 0.0.0.0 --port "${PAYI_PORT:-8420}"
