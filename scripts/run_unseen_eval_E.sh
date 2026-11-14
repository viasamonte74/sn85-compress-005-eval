#!/bin/bash
# 2026-11-13 variant E (per-clip filter probe in the 80-90 band) on the unseen sets, then the existing five.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
L=results/unseen_eval.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
until grep -q " done$" $L 2>/dev/null; do sleep 30; done
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
runE() {
  for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
  STREAMENC_FILTER_PROBE=1 STREAMENC_FILTER_PROBE_MAX_PRED=90 STREAMENC_FILTER_PROBE_EARLY_MAX=92 STREAMENC_HARD_VMAF_STAGE2=90 STREAMENC_SEGFINAL_REFIT=1 \
    python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 4 --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1
  log "$1: $(summ $1)"; rm -rf work/e2e_$1
}
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
log "E start"
runE un_E_val16b data/e2e_val16b.txt $M
runE un_E_val16c data/e2e_val16c.txt $M
runE ex_E_practice data/e2e_practice.txt $M
runE ex_E_val16 data/e2e_val16.txt $M
runE ex_E_test16 data/e2e_test16_hidden_like.txt $M
runE ex_E_test16hard data/e2e_test16_hard.txt $M
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done
STREAMENC_FILTER_PROBE=1 STREAMENC_FILTER_PROBE_MAX_PRED=90 STREAMENC_FILTER_PROBE_EARLY_MAX=92 STREAMENC_HARD_VMAF_STAGE2=90 STREAMENC_SEGFINAL_REFIT=1 \
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model $ME --score-procs 4 \
  --work work/e2e_ex_E_heldout --out results/e2e_batched/ex_E_heldout > results/e2e_batched/ex_E_heldout.log 2>&1
log "ex_E_heldout: $(summ ex_E_heldout)"; rm -rf work/e2e_ex_E_heldout
log "E done"
