#!/bin/bash
# 2026-11-13 prefilter threshold re-test after the colour fix (user "yes"): STREAMENC_HARD_VMAF 85 / 90 / 95 vs 80
# (rc11 = cf_new_* / rc11c_* runs), all five sets, validator-exact scoring.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/hardvmaf_sweep.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs ${4:-8} --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
log "start"
for T in 90 85 95; do
  export STREAMENC_HARD_VMAF=$T
  run hv${T}_practice data/e2e_practice.txt $M
  run hv${T}_val16 data/e2e_val16.txt $M
  run hv${T}_test16 data/e2e_test16_hidden_like.txt $M
  run hv${T}_test16hard data/e2e_test16_hard.txt $M
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 6 \
      --work work/e2e_hv${T}_heldout --out results/e2e_batched/hv${T}_heldout > results/e2e_batched/hv${T}_heldout.log 2>&1
  log "hv${T}_heldout: $(summ hv${T}_heldout)"; rm -rf work/e2e_hv${T}_heldout
done
log "done"
