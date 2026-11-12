#!/bin/bash
# 2026-11-12 round 6c: SC (slack 0.008 + cblur16_ds85) + NEG prefilter fallback (cblur4_ds85) on the HFR list, the only
# list with a NEG fallback in round 6 (hfr_rnd_hard_03). Other lists have no NEG fallback -> identical outputs.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
export STREAMENC_RATE_SLACK=0.008 STREAMENC_HARD_PREFILTER=cblur16_ds85 STREAMENC_NEG_PREFILTER_FALLBACK=cblur4_ds85
python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips data/e2e_hfr_nonat.txt --model models/crf_predictor_eval_c_abs.pkl \
  --out results/e2e/r6_hfr22_SCF > results/e2e_r6_hfr22_SCF.log 2>&1
echo "$(date -u +%H:%M) r6_hfr22_SCF: $(tr -d '\n ' < results/e2e/r6_hfr22_SCF/summary.json | cut -c1-300)"
