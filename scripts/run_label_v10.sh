#!/bin/bash
# v10 labelling: the 1080p anim clips not yet in encoder_rank_all (data/label_v10_clips.txt).
# CPU-capped (default CPUs 0-11, 4 arm workers/shard) to leave the 23-CPU container quota room for
# other jobs. Usage: run_label_v10.sh <shard> <nshards>
export CLIPS=data/label_v10_clips.txt
export LABEL_ARM_WORKERS=${LABEL_ARM_WORKERS:-4}
exec taskset -c "${LABEL_CPUS:-0-11}" /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v10
