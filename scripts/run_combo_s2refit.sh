#!/bin/bash
# 2026-11-13 combined candidate config: stage-2 prefilter threshold 90 + batch-aware segmented-final refit, five sets.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/combo_s2refit.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 4 --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
export STREAMENC_HARD_VMAF_STAGE2=90 STREAMENC_SEGFINAL_REFIT=1
log "start"
run cb_practice data/e2e_practice.txt $M
run cb_val16 data/e2e_val16.txt $M
run cb_test16 data/e2e_test16_hidden_like.txt $M
run cb_test16hard data/e2e_test16_hard.txt $M
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 4 \
    --work work/e2e_cb_heldout --out results/e2e_batched/cb_heldout > results/e2e_batched/cb_heldout.log 2>&1
log "cb_heldout: $(summ cb_heldout)"; rm -rf work/e2e_cb_heldout
log "done"
