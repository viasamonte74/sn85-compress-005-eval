#!/bin/bash
# 2026-11-11: hard path (cblur4_ds85 below predicted VMAF 80) also forces a slower preset (STREAMENC_HARD_ARM).
# On top of the speed setup (OMP_NUM_THREADS=1, 2 pilots) + all three candidate features, alone on the CPU.
# Baseline: p4_{practice,ai1080,test20,hfr22}_pmax2 (before the segfinal bump fix); p5_*_bump (with it).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export OMP_NUM_THREADS=1 STREAMENC_PILOT_MAX=2 STREAMENC_SEGFINAL=1 STREAMENC_SEGFILL=1 STREAMENC_HARD_PREFILTER=cblur4_ds85
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
round() { tag=$1
  e2e data/e2e_practice.txt     models/crf_predictor_c0.pkl         p5_practice_$tag
  e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl p5_ai1080_$tag
  e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl p5_test20_$tag
  e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl p5_hfr22_$tag; }
round bump
STREAMENC_HARD_ARM=svt_crf_p2_qm_la round hp2
STREAMENC_HARD_ARM=svt_crf_p1_qm_la round hp1
log "hard-arm sweep done"
