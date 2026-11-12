#!/bin/bash
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_QUICK_VMAF_PAIRING=rate STREAMENC_QUICK_VMAF_THREADS=8,8 PREDFIX_SUB=1 PREDFIX_NOFILTER=1 PREDFIX_TAG=exp2b
python scripts/predfix_exp2.py
echo "predfix exp2b done"
