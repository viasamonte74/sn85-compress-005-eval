#!/bin/bash
# 2026-11-12 q08 round 7b e2e (fill-down AFTER segfill): option A defaults + probe fill-down (STREAMENC_SEGFINAL_FILLDOWN=1), alone on the CPU.
# Compare with stA_*_r1 (option A). Also the sandbox-like 4-clip batch timing is in simtest (108.0 s for A).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
export STREAMENC_SEGFINAL_FILLDOWN=1
e2e data/e2e_practice.txt     models/crf_predictor_c0.pkl         r7b_practice_FD
e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl r7b_ai1080_FD
e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl r7b_test20_FD
e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl r7b_hfr22_FD
log "round7b e2e done"
