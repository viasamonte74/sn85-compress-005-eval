#!/bin/bash
# Generate the animated-movie clip sets with Wan2.2 after the base-set generation (no --set flag)
# has finished: one Wan process at a time on the GPU. Resumable: existing clips are skipped.
cd /workspace/sn85-compression-005
while ps -eo args | grep "[g]enerate_ai_clips.py" | grep -qv -- "--set"; do sleep 60; done
PY=/workspace/dataset-src/wan-env/bin/python
$PY scripts/generate_ai_clips.py --set anim  --tiers hard,medium
$PY scripts/generate_ai_clips.py --set animh --tiers hard
echo "anim generation done $(date -u)"
