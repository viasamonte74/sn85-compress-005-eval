#!/bin/bash
# 1080p versions of the animated-movie sets (same prompts, Wan2.2-TI2V-5B rendered natively at
# 1920x1088, centre-cropped to 1920x1080). Starts once the 720p anim job (run_anim_gen.sh) has exited
# so only one Wan process uses the GPU. Resumable; on failure (e.g. CUDA OOM) retries with T5 on CPU.
cd /workspace/sn85-compression-005
while pgrep -f "run_anim_gen.sh" >/dev/null || ps -eo args | grep -q "[g]enerate_ai_clips.py"; do sleep 60; done
echo "1080p generation start $(date -u)"
PY=/workspace/dataset-src/wan-env/bin/python
run_set() {
  local set=$1 tiers=$2
  for attempt in 1 2 3; do
    # T5 on CPU (--offload): at 1080p the DiT alone peaks ~30 GiB. Tiled VAE decode + per-clip OOM
    # retries live in generate_ai_clips.py; this loop only restarts after a crash.
    $PY scripts/generate_ai_clips.py --set "$set" --tiers "$tiers" --res 1080p --offload && return 0
    echo "set $set attempt $attempt failed (exit $?) $(date -u); restarting"
    sleep 60
  done
  return 1
}
run_set anim hard,medium
run_set animh hard
echo "1080p generation done $(date -u)"
