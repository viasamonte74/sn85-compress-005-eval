#!/bin/bash
# 2026-11-13 slow-preset cap test (user "yes"): rc9 + STREAMENC_SLOW_CAP_VMAF=98.5 on all five sets, 4-clip batches.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
export STREAMENC_SLOW_CAP_VMAF=${1:-98.5}
T=sc${STREAMENC_SLOW_CAP_VMAF/./}
L=results/slowcap.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 8 --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
log "start cap=$STREAMENC_SLOW_CAP_VMAF"
run ${T}_val16 data/e2e_val16.txt models/crf_predictor.pkl
run ${T}_practice data/e2e_practice.txt models/crf_predictor.pkl
run ${T}_test16 data/e2e_test16_hidden_like.txt models/crf_predictor.pkl
run ${T}_test16hard data/e2e_test16_hard.txt models/crf_predictor.pkl
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --score-procs 8 --work work/e2e_${T}_heldout --out results/e2e_batched/${T}_heldout > results/e2e_batched/${T}_heldout.log 2>&1
log "${T}_heldout: $(summ ${T}_heldout)"; rm -rf work/e2e_${T}_heldout
log "done"
