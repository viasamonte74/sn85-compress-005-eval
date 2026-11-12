#!/bin/bash
# 2026-11-11: practice pack, separate the effect of the compliant model (c0) from budget fill A. Clean CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips data/e2e_practice.txt --model $1 --out results/e2e/$2 > results/e2e_$2.log 2>&1
        log "$2: $(cat results/e2e/$2/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
e2e models/crf_predictor_c0.pkl practice_c0_default
STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.95 STREAMENC_ACCEPT_LOW=0.955 e2e models/crf_predictor.pkl practice_shipped_fillA
e2e models/crf_predictor.pkl practice_shipped_rep
log "practice split done"
