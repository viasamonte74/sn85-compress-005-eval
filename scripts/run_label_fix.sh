#!/bin/bash
# Re-run label trials affected by the v7 rate-search issues (policy 8). Usage: run_label_fix.sh <shard> <nshards>
set -u
i=$1; n=$2
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LABEL_ARM_WORKERS=${LABEL_ARM_WORKERS:-6}
exec python scripts/rerun_label_trials.py --labels data/encoder_rank_v7/merged/labels.jsonl \
  --trials data/encoder_rank_v7/merged/trials.jsonl --pairs data/encoder_rank_v7/fix/pairs.json \
  --full ai_easy_05,ai_easy_13,ai_hard_02,ai_easy_01,ai_easy_08,ai_easy_14,ai_easy_04,ai_easy_12,ai_hard_01 \
  --out data/encoder_rank_v7/fix --shard $i --nshards $n
