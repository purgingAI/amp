#!/usr/bin/env bash
# Speech-to-text for the customer app: whisper.cpp's server, model kept warm in memory.
#
# Multilingual small model, 5-bit (190 MB): English, Arabic and Hindi, language auto-detect.
# The openwallet server's /api/stt talks to this on 127.0.0.1:8178 and falls back to
# whisper-cli (slower: it reloads the model every call) when it is not running.
#   ./run.sh          start (detached)        ./run.sh stop
W=~/glasslink/whisper.cpp
PORT="${PORT:-8178}"
pid() { ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$"' | grep -oP 'pid=\K[0-9]+' | head -1; }
if [ "${1:-}" = "stop" ]; then P=$(pid); [ -n "$P" ] && kill "$P" && echo "stopped"; exit 0; fi
[ -n "$(pid)" ] && { echo "already running on :$PORT"; exit 0; }
nohup setsid "$W/build/bin/whisper-server" -m "$W/models/ggml-small-q5_1.bin" \
  --host 127.0.0.1 --port "$PORT" -t "${THREADS:-6}" -l auto \
  > "/tmp/whisper-server-$PORT.log" 2>&1 < /dev/null & disown
for _ in $(seq 40); do curl -s -m 1 -o /dev/null "http://127.0.0.1:$PORT/" && { echo "whisper-server ready on :$PORT"; exit 0; }; sleep 1; done
echo "whisper-server did not start — see /tmp/whisper-server-$PORT.log"; exit 1
