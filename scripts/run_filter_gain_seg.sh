#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
python scripts/filter_gain_seg.py
echo "filter gain seg done"
