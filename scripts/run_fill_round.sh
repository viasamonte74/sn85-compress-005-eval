#!/bin/bash
# 2026-11-11 (user: "start"): (1) train the shipping candidate (absolute target, compliant data = no natural
# clips, all splits, + 100 real screen recordings) -> models/crf_predictor_c.pkl (NOT yet the shipped file);
# (2) budget-fill e2e with the compliant eval model: final aim 0.965/0.92 (shipped) vs A 0.985/0.95 vs B 0.99/0.97.
# Nothing else may run on the CPU (wall time is scored).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
log() { echo "$(date -u +%H:%M) $*"; }
python scripts/train_crf_predictor.py --data data/encoder_rank_c --exclude-kinds other \
  --out models/crf_predictor_c.pkl > results/train_final_c.log 2>&1 || { log "train FAILED"; exit 1; }
log "trained shipping candidate: $(grep -E 'excluded|clips,' results/train_final_c.log | cut -c1-90 | tr '\n' ' ')"
M=models/crf_predictor_eval_c_abs.pkl
run() { local set=$1 tag=$2 aim=$3 easy=$4 low=$5
  STREAMENC_FINAL_AIM=$aim STREAMENC_FINAL_AIM_EASY=$easy STREAMENC_ACCEPT_LOW=$low \
    python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips data/e2e_${set}_nonat.txt --model $M \
    --out results/e2e/v14_${set}_${tag} > results/e2e_v14_${set}_${tag}.log 2>&1
  log "${set}_${tag} (aim $aim easy $easy low $low): $(cat results/e2e/v14_${set}_${tag}/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
run ai1080 fillA 0.985 0.95 0.955
run ai1080 fillB 0.99  0.97 0.97
run ai1080 base  0.965 0.92 0.93
run test26 fillB 0.99  0.97 0.97
run test26 fillA 0.985 0.95 0.955
run hfr    fillA 0.985 0.95 0.955
run hfr    fillB 0.99  0.97 0.97
log "fill round done"
