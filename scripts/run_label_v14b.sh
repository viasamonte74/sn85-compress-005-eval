#!/bin/bash
# v14b labelling: gen3 Wan clips (native 1080p + downscaled 720p twins), data/label_v14b_clips.txt. Usage: run_label_v14b.sh <shard> <nshards>
export CLIPS=data/label_v14b_clips.txt
# nice 19: never take CPU from the e2e runs (their wall-clock speed is scored)
exec nice -n 19 /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v14b
