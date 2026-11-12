#!/bin/bash
# v15 labelling: 100 real screen recordings from Hugging Face nickossaf/Screen-Content-Video (CC BY 2.0),
# /workspace/dataset/screen_real, data/label_v15_clips.txt. Usage: run_label_v15.sh <shard> <nshards>
export CLIPS=data/label_v15_clips.txt
# nice 19: yield the CPU to the training experiments and e2e runs
exec nice -n 19 /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v15
