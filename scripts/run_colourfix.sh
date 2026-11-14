#!/bin/bash
# 2026-11-13 colour fix verification (user: "this is serious, focus on this issue, start upgrading"). All scoring uses
# the corrected validator-exact canonicalization. old = STREAMENC_COLOUR_RELABEL=0 (rc10 encode behaviour), new = fix.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/colourfix.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs ${4:-8} --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
log "start"
run cf_new_practice data/e2e_practice.txt $M
run cf_new_val16 data/e2e_val16.txt $M
run cf_new_test16 data/e2e_test16_hidden_like.txt $M
STREAMENC_COLOUR_RELABEL=0 run cf_old_test16 data/e2e_test16_hidden_like.txt $M
run cf_new_test16hard data/e2e_test16_hard.txt $M
STREAMENC_COLOUR_RELABEL=0 run cf_old_test16hard data/e2e_test16_hard.txt $M
for T in new old; do
  [ $T = old ] && export STREAMENC_COLOUR_RELABEL=0 || unset STREAMENC_COLOUR_RELABEL
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 6 \
      --work work/e2e_cf_${T}_heldout --out results/e2e_batched/cf_${T}_heldout > results/e2e_batched/cf_${T}_heldout.log 2>&1
  log "cf_${T}_heldout: $(summ cf_${T}_heldout)"; rm -rf work/e2e_cf_${T}_heldout
done
log "done"
