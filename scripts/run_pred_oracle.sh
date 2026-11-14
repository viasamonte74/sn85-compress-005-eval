#!/bin/bash
# 2026-11-13 "perfect prediction" oracle (user: "model prediction VMAF is higher than real; did you train it well?"):
# same runs as rc8_test16hard / rc7_heldout, but every arm's stage-2 VMAF is shifted by (real - predicted) of that clip.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/predfix/oracle.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
log "start"
STREAMENC_PRED_SHIFT_FILE=$PWD/data/oracle_shift_test16hard.json python scripts/e2e_batched.py --clips data/e2e_test16_hard.txt \
    --model models/crf_predictor.pkl --score-procs 8 --out results/e2e_batched/oracle_test16hard > results/e2e_batched/oracle_test16hard.log 2>&1
log "oracle_test16hard: $(summ oracle_test16hard)"
STREAMENC_PRED_SHIFT_FILE=$PWD/data/oracle_shift_heldout.json python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --score-procs 8 --out results/e2e_batched/oracle_heldout > results/e2e_batched/oracle_heldout.log 2>&1
log "oracle_heldout: $(summ oracle_heldout)"
log "done"
