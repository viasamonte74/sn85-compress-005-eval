#!/bin/bash
# 2026-11-12 W4 (user: "Go ahead"): rc5 defaults + STREAMENC_FILTER_MODEL=1 (zero-cost prefilter decision model,
# models *_fg.pkl = same stage 1/2 + .fgain_model). 4-clip batches, alone on the CPU. Baselines: w1_practice_ebump, rc5_heldout.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
export STREAMENC_FILTER_MODEL=1
python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor_c0_fg.pkl --out results/e2e_batched/w4_practice_fmodel > results/e2e_batched/w4_practice_fmodel.log 2>&1
log "w4_practice_fmodel: $(summ w4_practice_fmodel)"
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs_fg.pkl --out results/e2e_batched/w4_heldout_fmodel > results/e2e_batched/w4_heldout_fmodel.log 2>&1
log "w4_heldout_fmodel: $(summ w4_heldout_fmodel)"
log "w4 done"
