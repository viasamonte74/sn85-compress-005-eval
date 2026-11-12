#!/bin/bash
# 2026-11-11 final check of the candidate version, held out: the shipping model's recipe (absolute target,
# data/encoder_rank_c without natural clips, incl. real screen recordings) but with the test split excluded,
# + budget fill A (env) + compliant NEG shifts (code) + time-cost weight 0.012 (model default).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.95 STREAMENC_ACCEPT_LOW=0.955
log() { echo "$(date -u +%H:%M) $*"; }
python scripts/train_crf_predictor.py --data data/encoder_rank_c --exclude-split test --exclude-kinds other \
  --out models/crf_predictor_eval_c_final.pkl > results/train_eval_c_final.log 2>&1 || { log "train FAILED"; exit 1; }
log "trained: $(grep -E 'held out|excluded|clips,' results/train_eval_c_final.log | cut -c1-80 | tr '\n' ' ')"
run() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model models/crf_predictor_eval_c_final.pkl \
    --out results/e2e/v16_$2 > results/e2e_v16_$2.log 2>&1
  log "$2: $(cat results/e2e/v16_$2/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
run data/e2e_ai1080_nonat.txt   ai1080
run data/e2e_test26_nonat.txt   test20
run data/e2e_hfr_nonat.txt      hfr22
run data/e2e_screenreal_test.txt screenreal15
log "final check done"
