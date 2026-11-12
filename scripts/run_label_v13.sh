#!/bin/bash
# v13 labelling: hfr_render + hfr_wan + render2 + render_film clips complete at list time, not in v12
# (data/label_v13_clips.txt). Usage: run_label_v13.sh <shard> <nshards>
export CLIPS=data/label_v13_clips.txt
exec /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v13
