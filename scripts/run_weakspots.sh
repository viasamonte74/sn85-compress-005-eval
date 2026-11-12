#!/bin/bash
# 2026-11-12 weak-spot round (user: "try fix weak spots"), rc4 defaults + one switch each, validator-style 4-clip
# batches, alone on the CPU. Baselines: p0_practice_pair (same settings as rc4) and rc4_heldout.
#  W1 early bump (segfinal), W2 measured prefilter decision, W3 stage 3 for hopeless clips only (gate < 75).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
log "tests: $(python -m pytest -q tests/ -p no:cacheprovider 2>&1 | tail -1)"
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
pr() { python scripts/e2e_batched.py --clips data/e2e_practice.txt --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
(export STREAMENC_SEGFINAL_EARLY_BUMP=1
 pr w1_practice_ebump models/crf_predictor.pkl; ho w1_heldout_ebump models/crf_predictor_eval_c_abs.pkl)
(export STREAMENC_FILTER_PROBE=1
 pr w2_practice_fprobe models/crf_predictor.pkl; ho w2_heldout_fprobe models/crf_predictor_eval_c_abs.pkl)
(export STREAMENC_STAGE3=1 STREAMENC_STAGE3_MODE=hopeless STREAMENC_STAGE3_MAX_PRED=75
 pr w3_practice_hopeless models/crf_predictor_c0_s3.pkl; ho w3_heldout_hopeless models/crf_predictor_eval_c_abs_s3.pkl)
log "weakspots done"
