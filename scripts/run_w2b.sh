#!/bin/bash
# 2026-11-12 W2b (user: "start single-segment prefilter"): rc5 defaults + STREAMENC_FILTER_PROBE=1 (mode segment:
# one 2 s segment pair on a background thread after the first pilot, used where the fixed rule would filter).
# Validator-style 4-clip batches, alone on the CPU. Baselines: w1_practice_ebump (rc5 code), rc5_heldout.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
log "tests: $(python -m pytest -q tests/ -p no:cacheprovider 2>&1 | tail -1)"
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
export STREAMENC_FILTER_PROBE=1
python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor.pkl --out results/e2e_batched/w2b_practice_seg > results/e2e_batched/w2b_practice_seg.log 2>&1
log "w2b_practice_seg: $(summ w2b_practice_seg)"
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/w2b_heldout_seg > results/e2e_batched/w2b_heldout_seg.log 2>&1
log "w2b_heldout_seg: $(summ w2b_heldout_seg)"
log "w2b done"
