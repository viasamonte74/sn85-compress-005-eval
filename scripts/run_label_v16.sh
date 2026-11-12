#!/bin/bash
# v16 labelling: the 8 official practice-pack clips (CC0), all 9 label configs. Usage: run_label_v16.sh <shard> <nshards>
export CLIPS=data/label_v16_clips.txt
exec /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v16
