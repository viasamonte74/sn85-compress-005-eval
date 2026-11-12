#!/bin/bash
# v12 labelling: hfr_ai (RIFE) + hfr_render (Blender) clips (data/label_v12_clips.txt). Usage: run_label_v12.sh <shard> <nshards>
export CLIPS=data/label_v12_clips.txt
exec /workspace/sn85-compression-005/scripts/run_label_v9.sh "$1" "$2" v12
