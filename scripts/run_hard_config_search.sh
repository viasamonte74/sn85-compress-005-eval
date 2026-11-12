#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
exec nice -n 19 python scripts/hard_config_search.py --clips "${1:-aig3_hard_002_1080,ai_hard_20_1080,aix_hard_031,ai_hard_04,rnd_hard_03}"
