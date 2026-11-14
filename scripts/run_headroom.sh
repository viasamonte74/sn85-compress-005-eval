#!/bin/bash
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
python scripts/preset_headroom.py --list data/headroom_clips.txt --workers 8 > results/headroom/run.log 2>&1
echo "CHAIN DONE rc=$?" >> results/headroom/run.log
