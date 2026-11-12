#!/bin/bash
# gen3 (2026-11-10): 24 new Wan2.2-TI2V-5B prompts generated natively at 1920x1088 (crop 1920x1080) only, 720p twin
# = Lanczos downscale of the same frames -> /workspace/dataset/ai_gen3. T5 on CPU (--offload; the 1080p DiT peaks
# ~30 GiB). Resumable; restarts after a crash (per-clip OOM retries live in generate_ai_clips.py).
cd /workspace/sn85-compression-005
echo "gen3 start $(date -u)"
PY=/workspace/dataset-src/wan-env/bin/python
for attempt in 1 2 3; do
  $PY scripts/generate_ai_clips.py --set gen3 --res 1080p --offload --out-dir /workspace/dataset/ai_gen3 && { echo "gen3 done $(date -u)"; exit 0; }
  echo "gen3 attempt $attempt failed (exit $?) $(date -u); restarting"
  sleep 60
done
exit 1
