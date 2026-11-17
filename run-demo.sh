#!/usr/bin/env bash
# Bring up the payee wallet and the observer for the three-laptop demo.
#   ./run-demo.sh            bind to localhost (one machine)
#   ./run-demo.sh --lan      bind to 0.0.0.0 so laptops 2 and 3 can reach them
cd "$(dirname "$0")"
HOST=127.0.0.1
[[ "${1:-}" == "--lan" ]] && HOST=0.0.0.0
./.venv/bin/python payee_agent.py    --agent agt_alex --port 8100 --host "$HOST" &
./.venv/bin/python observer_agent.py --port 8200 --host "$HOST" &
IP=$(hostname -I | awk '{print $1}')
echo "  payee wallet : http://${IP}:8100"
echo "  flow monitor : http://${IP}:8200"
wait
