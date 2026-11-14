#!/bin/bash
# 2026-11-12 rc8 compliance (licence review): re-check the speed trade-off lam (VMAF per CPU-second, model lam and
# DEFAULT_LAM) on compliant data only: practice pack (CC0) + held-out non-natural clips, 4-clip batches, alone.
# lam 0.012 = rc7 (results rc7_heldout / h2c_practice); here 0.008 and 0.016 via STREAMENC_LAM.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
for L in 0.008 0.016; do
  export STREAMENC_LAM=$L
  python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor.pkl --out results/e2e_batched/lam${L}_practice > results/e2e_batched/lam${L}_practice.log 2>&1
  log "lam $L practice: $(summ lam${L}_practice)"
  python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
      --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/lam${L}_heldout > results/e2e_batched/lam${L}_heldout.log 2>&1
  log "lam $L heldout: $(summ lam${L}_heldout)"
done
log "lam sweep done"
