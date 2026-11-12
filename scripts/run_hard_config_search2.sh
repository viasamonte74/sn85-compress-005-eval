#!/bin/bash
# ink-cloud clip where tune 0 wins (user 2026-11-10): search from that reference, after the first search
cd /workspace/sn85-compression-005
. /opt/nvm/nvm.sh >/dev/null 2>&1
while pm2 jlist | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-config-search' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 60; done
exec scripts/run_hard_config_search.sh aig3_hard_005_1080,aig3_hard_005
