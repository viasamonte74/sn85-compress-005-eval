#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
FGAIN_ARMS=svt_crf_p8_t1 python scripts/filter_gain_proxy.py
echo "filter gain proxy done"
