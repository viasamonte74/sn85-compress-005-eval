#!/bin/bash
# waits for the hard-arm e2e sweep (timed runs must be alone on the CPU), then runs q0708_round5.py
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
. /opt/nvm/nvm.sh >/dev/null 2>&1
while pm2 jlist | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-hardarm' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 30; done
python scripts/q0708_round5.py
echo "round5 done"
