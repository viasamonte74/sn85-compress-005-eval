#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
. /opt/nvm/nvm.sh >/dev/null 2>&1
while pm2 jlist | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-user-t5' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 20; done
exec python scripts/check_colour_tags.py
