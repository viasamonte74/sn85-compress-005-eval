#!/bin/bash
# second sweep sample: clips that score but below 0.75 (user 2026-11-10: relabel those too); runs after the first sample
cd /workspace/sn85-compression-005
. /opt/nvm/nvm.sh >/dev/null 2>&1
while pm2 jlist | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-hard-sweep' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 60; done
exec scripts/run_hard_rescue_sweep.sh --clips "$(cat data/hard_rescue_mid_clips.txt)"
