#!/bin/bash
# 2026-11-12 q08 lab (user: raise q08 to VMAF >= 75 with untried methods; q08 only first). Alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done
python scripts/q08_lab.py "$@"
echo "q08 lab batch done: $*"
