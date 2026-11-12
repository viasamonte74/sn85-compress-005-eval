#!/bin/bash
# 2026-11-12 H2c (user: H2 for q08-like clips only): rc5 + preset-2 hard arm + cblur16, only if stage-2 best < 70,
# height <= 720 and mean pilot CPU <= 0.08 s per megapixel-frame. 4-clip batches, alone. Baselines: w1_practice_ebump, rc5_heldout.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
export STREAMENC_HARD_ARM=svt_crf_p2_qm_la_lqb_h3 STREAMENC_HARD_ARM_PREFILTER=cblur16_ds85 STREAMENC_HARD_ARM_MAX_PRED=70 \
       STREAMENC_HARD_ARM_MAX_HEIGHT=720 STREAMENC_HARD_ARM_MAX_PILOT_CPU=0.08
python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor.pkl --out results/e2e_batched/h2c_practice > results/e2e_batched/h2c_practice.log 2>&1
log "h2c_practice: $(summ h2c_practice)"
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/h2c_heldout > results/e2e_batched/h2c_heldout.log 2>&1
log "h2c_heldout: $(summ h2c_heldout)"
log "h2c done"
