#!/bin/bash
# 2026-11-12 prediction fix round (user: "solve this model prediction issue"): retrain with stage 3 (new files), check
# stage 1/2 unchanged, then validator-style 4-clip batched e2e: P0 = exact VMAF-check pairing only, P1 = P0 + stage 3.
# Both with the VMAF-check threads (8,4). Baseline = u1_*_negthr. Alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
T="python scripts/train_crf_predictor.py --data data/encoder_rank_all --exclude-kinds other"
$T --exclude-split test --out models/crf_predictor_eval_c_abs_s3.pkl > results/predfix/train_eval_s3.log 2>&1 || { log "train eval FAILED"; exit 1; }
$T --out models/crf_predictor_c0_s3.pkl > results/predfix/train_c0_s3.log 2>&1 || { log "train c0 FAILED"; exit 1; }
log "trained: $(grep -c '^stage3' results/predfix/train_eval_s3.log) + $(grep -c '^stage3' results/predfix/train_c0_s3.log) stage-3 arms"
python scripts/predfix_check_models.py models/crf_predictor_eval_c_abs.pkl models/crf_predictor_eval_c_abs_s3.pkl
python scripts/predfix_check_models.py models/crf_predictor_c0.pkl models/crf_predictor_c0_s3.pkl
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
pr() { python scripts/e2e_batched.py --clips data/e2e_practice.txt --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
export STREAMENC_QUICK_VMAF_THREADS=8,4 STREAMENC_QUICK_VMAF_PAIRING=rate
pr p0_practice_pair models/crf_predictor_c0.pkl
ho p0_heldout_pair models/crf_predictor_eval_c_abs.pkl
export STREAMENC_STAGE3=1
pr p1_practice_s3 models/crf_predictor_c0_s3.pkl
ho p1_heldout_s3 models/crf_predictor_eval_c_abs_s3.pkl
log "predfix round done"
