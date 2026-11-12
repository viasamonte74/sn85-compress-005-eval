#!/bin/bash
# 2026-11-11: time-cost weight (VMAF per CPU-second) re-tuned on the no-natural test sets (compliance: the shipped
# 0.012 was calibrated with natural clips in the e2e sets), with budget-fill setting A applied.
# Base point lam 0.012 + fill A = results/e2e/v14_*_fillA. Nothing else may run on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.95 STREAMENC_ACCEPT_LOW=0.955
log() { echo "$(date -u +%H:%M) $*"; }
run() { local set=$1 lam=$2
  STREAMENC_LAM=$lam python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips data/e2e_${set}_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e/v15_${set}_lam${lam} > results/e2e_v15_${set}_lam${lam}.log 2>&1
  log "${set} lam ${lam}: $(cat results/e2e/v15_${set}_lam${lam}/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
run ai1080 0.008
run ai1080 0.018
run test26 0.018
run test26 0.008
run hfr 0.008
run hfr 0.018
log "lam round done"
