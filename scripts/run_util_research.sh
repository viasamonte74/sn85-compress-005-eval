#!/bin/bash
# 2026-11-12 utilisation research (user yes after the external review): validator-style 4-clip batches, alone on the CPU.
# Baselines: rc3 practice (here) and rc3 held-out (results/e2e_batched/rc3_heldout).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
pr() { python scripts/e2e_batched.py --clips data/e2e_practice.txt --model models/crf_predictor.pkl \
         --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
pr rc3_practice
export STREAMENC_QUICK_VMAF_THREADS=8,4
pr u1_practice_negthr
ho u1_heldout_negthr
export STREAMENC_SEGFINAL_PAR=4
pr u2_practice_par4
ho u2_heldout_par4
log "util research done"
