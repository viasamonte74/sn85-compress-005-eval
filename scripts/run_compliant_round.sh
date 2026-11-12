#!/bin/bash
# Compliant round (2026-11-11): eval models trained WITHOUT natural clips (MINING.md training-data rule; user:
# exclude all kind "other", keep files), absolute vs difference target, then clean e2e on test lists without
# natural clips. Nothing else may run on the CPU during the e2e (wall time is scored).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
log() { echo "$(date -u +%H:%M) $*"; }
T="python scripts/train_crf_predictor.py --data data/encoder_rank_all --exclude-split test --exclude-kinds other"
$T --out models/crf_predictor_eval_c_abs.pkl > results/train_eval_c_abs.log 2>&1 || { log "train abs FAILED"; exit 1; }
log "trained abs: $(grep -E 'excluded|clips,' results/train_eval_c_abs.log | cut -c1-120 | tr '\n' ' ')"
$T --target delta --ref svt_crf_p4_qm_la --out models/crf_predictor_eval_c_delta.pkl > results/train_eval_c_delta.log 2>&1 || { log "train delta FAILED"; exit 1; }
log "trained delta: $(grep -E 'delta models' results/train_eval_c_delta.log | tr '\n' ' ')"
E="python scripts/e2e_eval.py --cpus 0-31 --workers 4"
run() { $E --clips $1 --model $2 --out results/e2e/v13_$3 > results/e2e_v13_$3.log 2>&1
        log "$3: $(cat results/e2e/v13_$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-330)"; }
run data/e2e_test26_nonat.txt  models/crf_predictor_eval_c_abs.pkl   test20_abs
run data/e2e_test26_nonat.txt  models/crf_predictor_eval_c_delta.pkl test20_delta
run data/e2e_ai1080_nonat.txt  models/crf_predictor_eval_c_delta.pkl ai1080_delta
run data/e2e_ai1080_nonat.txt  models/crf_predictor_eval_c_abs.pkl   ai1080_abs
run data/e2e_hfr_nonat.txt     models/crf_predictor_eval_c_abs.pkl   hfr22_abs
run data/e2e_hfr_nonat.txt     models/crf_predictor_eval_c_delta.pkl hfr22_delta
log "compliant round done"
