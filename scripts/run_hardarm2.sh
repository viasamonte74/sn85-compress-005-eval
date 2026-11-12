#!/bin/bash
# 2026-11-12 q08 follow-up (user "yes"): rc5 + preset-2 hard arm (svt_crf_p2_qm_la_lqb_h3) + cblur16_ds85 with it.
# H2 = only clips with stage-2 predicted best < 70 (q08-like); H1 = every hard-path clip (< 80). 4-clip batches, alone.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
pr() { python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor.pkl --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
export STREAMENC_HARD_ARM=svt_crf_p2_qm_la_lqb_h3 STREAMENC_HARD_ARM_PREFILTER=cblur16_ds85
(export STREAMENC_HARD_ARM_MAX_PRED=70; pr h2_practice_p2lt70; ho h2_heldout_p2lt70)
(pr h1_practice_p2all; ho h1_heldout_p2all)
log "hardarm2 done"
