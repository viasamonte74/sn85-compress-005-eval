#!/bin/bash
# High-frame-rate clip set (720p + 1080p pairs). Starts after the Wan 1080p job (sn85-gen-1080) has
# finished so the GPU is free. GPU stages run one at a time; CPU/network stages run alongside.
#   GPU: RIFE interpolation of Wan clips (hfr_ai), then Blender Cycles renders (hfr_render)
#   CPU: synthetic screen content (hfr_screen), real 50/60 fps footage re-cut at native fps (hfr_open)
cd /workspace/sn85-compression-005
while pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-gen-1080' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)" \
      || ps -eo args | grep -q "[g]enerate_ai_clips.py"; do sleep 120; done
echo "HFR generation start $(date -u)"
PY=/venv/main/bin/python
WAN=/workspace/dataset-src/wan-env/bin/python
( $PY scripts/hfr_screen_gen.py > logs/hfr_screen.log 2>&1; echo "screen done $(date -u) rc=$?" ) &
( $PY scripts/hfr_open_clips.py > logs/hfr_open.log 2>&1; echo "open done $(date -u) rc=$?" ) &
$PY scripts/hfr_build_interp_plan.py data/hfr_interp_plan.jsonl
for attempt in 1 2; do
  $WAN scripts/hfr_interpolate.py --plan data/hfr_interp_plan.jsonl > logs/hfr_interp.log 2>&1 && break
  echo "interp attempt $attempt failed $(date -u)"; sleep 60
done
echo "interp done $(date -u)"
$PY scripts/hfr_render_all.py > logs/hfr_render.log 2>&1
echo "render done $(date -u) rc=$?"
wait
echo "HFR generation done $(date -u)"
for d in hfr_ai hfr_render hfr_screen hfr_open; do echo "$d: $(ls /workspace/dataset/$d/*.mkv 2>/dev/null | grep -v tmp | wc -l) clips"; done
