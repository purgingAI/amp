#!/usr/bin/env bash
# Near-live "today" for the daily briefing: fetch newest commits across the local repos, rebuild the
# lightweight commits.json snapshot, and push it to the EC2 hub — WITHOUT the heavy hourly code-graph
# rebuild (see sync_knowledge.sh for that). Runs every few minutes via cron.
#
# Why laptop-side: the EC2 box cannot reach git.example.com (internal/VPN-only), so the laptop stays
# the git source. This just keeps the briefing's commit feed fresh between the hourly graph syncs.
set -uo pipefail
HOME_DIR="${HOME}"
KEY="${HOME_DIR}/.ssh/id_rsa"
# Box address: shared config, env override wins. See ~/.ace-ec2.env (no Elastic IP on the
# instance, so the public IP moves on every stop/start). The env value is captured BEFORE the
# file is sourced, otherwise sourcing would overwrite the override it is meant to respect.
_ace_ec2_override="${ACE_EC2:-}"
[ -f "${HOME}/.ace-ec2.env" ] && . "${HOME}/.ace-ec2.env"
ACE_EC2="${_ace_ec2_override:-${ACE_EC2:-16.171.46.219}}"
EC2="ubuntu@${ACE_EC2}"
CJSON="/home/ubuntu/amp/data/commits.json"
cd "${HOME_DIR}/amp" || exit 1

# 1) fetch latest commits across all known repos, in parallel, each bounded (best-effort; if the VPN
#    is down a fetch just fails and we snapshot whatever is local).
while IFS= read -r d; do
  [ -d "${d}/.git" ] && ( GIT_TERMINAL_PROMPT=0 timeout 15 git -C "${d}" fetch --all -q 2>/dev/null ) &
done < <(.venv/bin/python -c "from app import orchestrator as o; print('\n'.join(str(p) for p in o.REPOS.values()))")
wait

# 2) rebuild commits.json (fast: git log --all across repos).
.venv/bin/python -c "from app import orchestrator as o; o.write_snapshot()" >/dev/null 2>&1

# 2b) guard: never push a degenerate snapshot (would empty the company briefing). Require >=1 company commit.
CO=$(.venv/bin/python -c "import json;d=json.load(open('data/commits.json'));deny={'trading-ai','creator-os','vantage','employee-tracker','ace_app','amp','amp-ios','team-lens','cba','citybank','citybank-cbt','wallet-rmc'};print(sum(1 for c in d if (c.get('repo') or '') not in deny))" 2>/dev/null || echo 0)
if [ "${CO:-0}" -lt 1 ]; then
  echo "$(date -Iseconds) SKIP push: snapshot had 0 company commits" >> "${HOME_DIR}/amp/data/sync.log"; exit 0
fi

# 3) ship it atomically (.tmp then mv).
# Log what ACTUALLY happened. The success line used to sit outside the && chain, so the
# log recorded "pushed commits.json" every run — including the hours after the EC2 box
# went down, when nothing had been pushed at all. A log that reports success regardless
# is worse than no log: it was read as evidence the host was alive.
if scp -q -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "${KEY}" \
       "${HOME_DIR}/amp/data/commits.json" "${EC2}:${CJSON}.tmp" \
   && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "${KEY}" "${EC2}" \
       "mv -f ${CJSON}.tmp ${CJSON}"; then
  echo "$(date -Iseconds) pushed commits.json -> ${EC2}" >> "${HOME_DIR}/amp/data/sync.log"
else
  echo "$(date -Iseconds) FAILED to push commits.json -> ${EC2} (host unreachable?)" \
    >> "${HOME_DIR}/amp/data/sync.log"
fi
