#!/bin/bash
# Re-run the two HFR parts that failed on 2026-11-10 06:00 (RIFE fp16 dtype bug, OptiX denoiser),
# in parallel on the GPU. Both scripts skip finished outputs.
cd /workspace/sn85-compression-005
PY=/venv/main/bin/python
WAN=/workspace/dataset-src/wan-env/bin/python
echo "HFR fix start $(date -u)"
( $WAN scripts/hfr_interpolate.py --plan data/hfr_interp_plan.jsonl > logs/hfr_interp.log 2>&1; echo "interp done $(date -u) rc=$?" ) &
( $PY scripts/hfr_render_all.py > logs/hfr_render.log 2>&1; echo "render done $(date -u) rc=$?" ) &
wait
for d in hfr_ai hfr_render; do echo "$d: $(ls /workspace/dataset/$d/*.mkv 2>/dev/null | grep -v tmp | wc -l) clips"; done
echo "HFR fix done $(date -u)"
