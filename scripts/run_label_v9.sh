#!/bin/bash
# Full labelling over ${CLIPS:-data/label_v9_clips.txt} split into N shards. Usage: run_label_full.sh <shard> <nshards> [LABEL_VER]
set -u
i=$1; n=$2; ver=${3:-v9}
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LABEL_ARM_WORKERS=${LABEL_ARM_WORKERS:-6}
# temp files (canonical y4m ~1.8 GB per 1080p60 trial, raw cache) in RAM: the root overlay is nearly full
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
mkdir -p data/encoder_rank_$ver
awk -v i="$i" -v n="$n" 'NR % n == i' ${CLIPS:-data/label_v9_clips.txt} > data/encoder_rank_$ver/shard$i.clips
exec python scripts/build_encoder_rank_dataset.py --resume --clip-list data/encoder_rank_$ver/shard$i.clips \
  --out-dir data/encoder_rank_$ver/shard$i --work-dir work/encoder_rank_$ver/shard$i
