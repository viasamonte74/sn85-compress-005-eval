#!/bin/bash
# v13b labelling: the hfr_wan / render2 clips finished after the v13 list was built (data/label_v13b_clips.txt).
# Usage: run_label_v13b.sh <shard> <nshards>
export CLIPS=data/label_v13b_clips.txt
exec /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v13b
