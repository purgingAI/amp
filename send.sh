#!/usr/bin/env bash
# Laptop 1 — say it, send it.
#   ./send.sh                              sends 20 AED to Alex
#   ./send.sh "send 50 dirhams to alex morgan"
#   ./send.sh "send 150 dirhams to alex" --step-up sup_xxx   (after confirming)
cd "$(dirname "$0")"
SENTENCE="${1:-send 20 dirhams to alex morgan}"
shift || true
PYTHONPATH="$PWD" exec ./.venv/bin/python pay_client.py "$SENTENCE" \
  --hub "${GLASSLINK_HUB:-https://localhost/pay}" "$@"
