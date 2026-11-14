#!/bin/bash
# 2026-11-13 segmented-final refit + stage-2 prefilter threshold (after the colour fix), all five sets, 4 scorers.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/refit_sweep.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 4 --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
export STREAMENC_SEGFINAL_REFIT=1
log "start"
for T in 80 90 95; do
  export STREAMENC_HARD_VMAF_STAGE2=$T
  run rf${T}_practice data/e2e_practice.txt $M
  run rf${T}_val16 data/e2e_val16.txt $M
  run rf${T}_test16 data/e2e_test16_hidden_like.txt $M
  run rf${T}_test16hard data/e2e_test16_hard.txt $M
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 4 \
      --work work/e2e_rf${T}_heldout --out results/e2e_batched/rf${T}_heldout > results/e2e_batched/rf${T}_heldout.log 2>&1
  log "rf${T}_heldout: $(summ rf${T}_heldout)"; rm -rf work/e2e_rf${T}_heldout
done
log "done"
