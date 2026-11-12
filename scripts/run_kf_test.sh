#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH TMPDIR=/dev/shm/sn85tmp
. /opt/nvm/nvm.sh >/dev/null 2>&1
while pm2 jlist | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-segfinal-e2e' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 30; done
exec python scripts/kf_interval_test.py
