#!/bin/bash
# 2026-11-11: with the new pipeline (segmented final + segment fill + hard-AI prefilter) re-tune the two knobs that
# interact with it: time-cost weight (lam) and the hard-prefilter threshold. Base = practice_c0_segfinal2 /
# v19_*_segfinal2 (lam 0.012, threshold 80). Alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_SEGFINAL=1 STREAMENC_SEGFILL=1 STREAMENC_HARD_PREFILTER=cblur4_ds85
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
set4() { e2e data/e2e_practice.txt models/crf_predictor_c0.pkl t3_practice_$1
         e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl t3_ai1080_$1
         e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl t3_test20_$1
         e2e data/e2e_hfr_nonat.txt models/crf_predictor_eval_c_abs.pkl t3_hfr22_$1; }
STREAMENC_LAM=0.008 set4 lam008
STREAMENC_LAM=0.018 set4 lam018
STREAMENC_HARD_VMAF=90 set4 hard90
STREAMENC_HARD_VMAF=101 set4 hardall
log "tune3 done"
