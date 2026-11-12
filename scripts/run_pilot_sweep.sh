#!/bin/bash
# 2026-11-11: speed work. Prediction on one thread (OMP_NUM_THREADS=1) + fewer / looser pilot encodes, all three
# candidate features on, alone on the CPU. Baseline for practice: practice_c0_timing_omp1 (118.4 s, 0.7611).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export OMP_NUM_THREADS=1 STREAMENC_SEGFINAL=1 STREAMENC_SEGFILL=1 STREAMENC_HARD_PREFILTER=cblur4_ds85
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
round() { tag=$1
  [ "$tag" != omp1 ] && e2e data/e2e_practice.txt models/crf_predictor_c0.pkl p4_practice_$tag
  e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl p4_ai1080_$tag
  e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl p4_test20_$tag
  e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl p4_hfr22_$tag; }
round omp1
STREAMENC_PILOT_MAX=3 round pmax3
STREAMENC_PILOT_MAX=2 round pmax2
STREAMENC_PILOT_TOL=0.08 STREAMENC_PILOT_BRACKET=0.25 round loose
log "pilot sweep done"
