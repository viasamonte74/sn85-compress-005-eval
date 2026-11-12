#!/bin/bash
# Full labelling over data/full_label_clips.txt split into N shards. Usage: run_label_full.sh <shard> <nshards> [LABEL_VER]
set -u
i=$1; n=$2; ver=${3:-v7}
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LABEL_ARM_WORKERS=${LABEL_ARM_WORKERS:-6}
mkdir -p data/encoder_rank_$ver
awk -v i="$i" -v n="$n" 'NR % n == i' data/full_label_clips.txt > data/encoder_rank_$ver/shard$i.clips
exec python scripts/build_encoder_rank_dataset.py --resume --clip-list data/encoder_rank_$ver/shard$i.clips \
  --out-dir data/encoder_rank_$ver/shard$i --work-dir work/encoder_rank_$ver/shard$i
