#!/bin/bash
# 2026-11-11: "fill lite" = raise only the normal final aim (0.985); easy aim 0.92 and fill-up threshold 0.93 as
# shipped (fill A's extra fill-up re-encodes on slow presets cost 15 % time on the practice pack). Clean CPU.
# Practice pack with the compliant shipping model (never saw it); held-out sets with its held-out twin (c_abs).
# Then resume the BT.709 screen chain (scripts/run_round_c2b.sh).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.92 STREAMENC_ACCEPT_LOW=0.93
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
e2e data/e2e_practice.txt       models/crf_predictor_c0.pkl       practice_c0_lite
e2e data/e2e_ai1080_nonat.txt   models/crf_predictor_eval_c_abs.pkl v18_ai1080_lite
e2e data/e2e_test26_nonat.txt   models/crf_predictor_eval_c_abs.pkl v18_test20_lite
e2e data/e2e_hfr_nonat.txt      models/crf_predictor_eval_c_abs.pkl v18_hfr22_lite
log "fill lite done"
unset STREAMENC_FINAL_AIM STREAMENC_FINAL_AIM_EASY STREAMENC_ACCEPT_LOW
exec bash scripts/run_round_c2b.sh
