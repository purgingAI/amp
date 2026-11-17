#!/usr/bin/env bash
# Refresh the company knowledge graph from LOCAL repos (via TeamLens's ingester, which we never
# modify) and ship the built graph to the AMP hub on EC2. Keeps agents <1h stale.
#
# Atomic: we scp to a .tmp then `mv` on the server (same filesystem) so read-only SQLite readers
# on the hub never observe a half-written file. No hub restart needed — knowledge.py opens the DB
# fresh (read-only) on every query.
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
REMOTE="/home/ubuntu/amp/data/knowledge.db"

# 1) rebuild the graph from local git repos + notes (TeamLens ingester, read-only to us)
cd "${HOME_DIR}/team-lens" || exit 1
.venv/bin/python -c "from app.ingest import run_ingest; run_ingest()" >/dev/null 2>&1

# 2) stage a copy for AMP
cp "${HOME_DIR}/team-lens/data/teamlens.db" "${HOME_DIR}/amp/data/knowledge.db" || exit 1

# 2a-0) auto-discover + clone any NEW company repos added on GitLab (needs ~/.amp_gitlab creds).
cd "${HOME_DIR}/amp" || exit 1
.venv/bin/python discover_repos.py >> "${HOME_DIR}/amp/data/codegraph.log" 2>&1 || true

# 2a) pull the extra company repos cloned from GitLab (merchant-portal, mobile-app plugins, …) so
#     their latest code is re-analyzed too. Uses the stored git credential; never blocks on prompt.
for d in "${HOME_DIR}"/company-repos/*/; do
    [ -d "$d/.git" ] && GIT_TERMINAL_PROMPT=0 git -C "$d" pull -q --ff-only 2>/dev/null || true
done

# 2b) re-tune the CODE-UNDERSTANDING graph (incremental: only modules whose code changed get
#     re-analyzed by the LLM, so this is cheap hourly). This is the "auto-tune with new changes".
cd "${HOME_DIR}/amp" || exit 1
CODE_REMOTE="/home/ubuntu/amp/data/code_knowledge.db"
.venv/bin/python codegraph_build.py >> "${HOME_DIR}/amp/data/codegraph.log" 2>&1

# 2c) refresh the commit snapshot (feeds the spoken DAILY BRIEFING: who shipped what today).
CJSON="/home/ubuntu/amp/data/commits.json"
.venv/bin/python -c "from app import orchestrator as o; o.write_snapshot()" >> "${HOME_DIR}/amp/data/codegraph.log" 2>&1 || true

# 3) ship graphs + commit snapshot atomically to the hub (scp to .tmp then mv on same fs = no torn reads)
scp -q -o StrictHostKeyChecking=no -i "${KEY}" \
    "${HOME_DIR}/amp/data/knowledge.db" "${EC2}:${REMOTE}.tmp" \
  && ssh -o StrictHostKeyChecking=no -i "${KEY}" "${EC2}" "mv -f ${REMOTE}.tmp ${REMOTE}"
scp -q -o StrictHostKeyChecking=no -i "${KEY}" \
    "${HOME_DIR}/amp/data/code_knowledge.db" "${EC2}:${CODE_REMOTE}.tmp" \
  && ssh -o StrictHostKeyChecking=no -i "${KEY}" "${EC2}" "mv -f ${CODE_REMOTE}.tmp ${CODE_REMOTE}"
scp -q -o StrictHostKeyChecking=no -i "${KEY}" \
    "${HOME_DIR}/amp/data/commits.json" "${EC2}:${CJSON}.tmp" \
  && ssh -o StrictHostKeyChecking=no -i "${KEY}" "${EC2}" "mv -f ${CJSON}.tmp ${CJSON}"

# 4) refresh the DB→feature map (needs APP_DB_dev over VPN). Best-effort: only ship on success,
#    so a VPN/DB outage keeps the last good map instead of blanking it.
DBMAP="/home/ubuntu/amp/data/db_map.json"
if .venv/bin/python db_map_build.py >> "${HOME_DIR}/amp/data/codegraph.log" 2>&1; then
  scp -q -o StrictHostKeyChecking=no -i "${KEY}" \
      "${HOME_DIR}/amp/data/db_map.json" "${EC2}:${DBMAP}.tmp" \
    && ssh -o StrictHostKeyChecking=no -i "${KEY}" "${EC2}" "mv -f ${DBMAP}.tmp ${DBMAP}"
fi

echo "$(date -Iseconds) synced knowledge.db + code_knowledge.db + commits.json + db_map.json -> ${EC2}" >> "${HOME_DIR}/amp/data/sync.log"
