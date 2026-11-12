#!/bin/bash
# Native HFR AI clips with Wan2.2-TI2V-5B (slow-motion prompts, stored at 50/60 fps, 121 frames):
# 10 prompts x {720p, 1080p} -> /workspace/dataset/hfr_wan. Runs beside the Blender renders on the GPU.
cd /workspace/sn85-compression-005
PY=/workspace/dataset-src/wan-env/bin/python
echo "HFR Wan start $(date -u)"
for res in 720p 1080p; do
  for attempt in 1 2 3; do
    $PY scripts/generate_ai_clips.py --set hfrwan --res $res --offload --out-dir /workspace/dataset/hfr_wan && break
    echo "$res attempt $attempt failed (exit $?) $(date -u); restarting"; sleep 60
  done
done
echo "hfr_wan: $(ls /workspace/dataset/hfr_wan/*.mkv 2>/dev/null | grep -v tmp | wc -l) clips"
echo "HFR Wan done $(date -u)"
