#!/bin/bash
# 2026-11-13 stage-2-only prefilter threshold (stage 1 stays 80): STREAMENC_HARD_VMAF_STAGE2 90 / 95 / 85 vs 80
# (rc11 = cf_new_* / rc11c_* runs), all five sets, validator-exact scoring.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/hardvmaf_s2_sweep.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs ${4:-4} --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; rm -rf work/e2e_$1; }
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
log "start"
for T in 90 95 85; do
  export STREAMENC_HARD_VMAF_STAGE2=$T
  run hs${T}_practice data/e2e_practice.txt $M
  run hs${T}_val16 data/e2e_val16.txt $M
  run hs${T}_test16 data/e2e_test16_hidden_like.txt $M
  run hs${T}_test16hard data/e2e_test16_hard.txt $M
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 4 \
      --work work/e2e_hs${T}_heldout --out results/e2e_batched/hs${T}_heldout > results/e2e_batched/hs${T}_heldout.log 2>&1
  log "hs${T}_heldout: $(summ hs${T}_heldout)"; rm -rf work/e2e_hs${T}_heldout
done
log "done"
