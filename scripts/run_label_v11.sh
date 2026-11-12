#!/bin/bash
# v11 labelling: animh_hard_009-020 _1080 + hfr_screen + hfr_open (data/label_v11_clips.txt, largest first).
# Full CPU (no ChipForge job running). Usage: run_label_v11.sh <shard> <nshards>
export CLIPS=data/label_v11_clips.txt
exec /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v11
