#!/bin/bash
# nice 19: the round-v12 e2e runs (wall-clock speed is scored) and labelling keep priority
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
nice -n 19 python scripts/hard_rescue_sweep.py "$@"
