#!/bin/bash
# Relabel the two VFR open clips (keyframe gate failed with a frame-count GOP) with the timestamp-aware GOP.
set -u
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LABEL_ARM_WORKERS=${LABEL_ARM_WORKERS:-9}
exec python scripts/rerun_label_trials.py --labels data/encoder_rank_v8/merged/labels.jsonl \
  --trials data/encoder_rank_v8/merged/trials.jsonl --full wm_f754444e_00,wm_f754444e_02 \
  --out data/encoder_rank_v8/fix_vfr --work-dir work/label_fix_vfr --shard 0 --nshards 1
