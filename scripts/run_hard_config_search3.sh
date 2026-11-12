#!/bin/bash
# wide search 2 (2026-11-11): one zero-score AI hard clip, extended tools + NVENC / other encoders
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
exec nice -n 19 python scripts/hard_config_search.py --wide2 --max-add 8 --clips "${1:-ai_hard_12_1080}"
