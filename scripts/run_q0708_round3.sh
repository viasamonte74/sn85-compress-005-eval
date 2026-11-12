#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH TMPDIR=/dev/shm/sn85tmp; mkdir -p $TMPDIR
python scripts/q08_frames.py > results/user_tests/q08_frames.log 2>&1
exec python scripts/q0708_round3.py
