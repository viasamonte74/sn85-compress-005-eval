#!/bin/bash
# 2026-11-12 q08 round 6 e2e (user: "solve the q08 issue first"), alone on the CPU, against the stability baseline
# (st_*_r1..3: practice 0.7628, held-out pooled 0.6300). S = use 0.8 % of the validator's documented 1 % average-bitrate
# slack (STREAMENC_RATE_SLACK=0.008); SC = S + hard-path prefilter cblur16_ds85 instead of cblur4_ds85.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
run_set() {
  e2e data/e2e_practice.txt     models/crf_predictor_c0.pkl         r6_practice_$1
  e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl r6_ai1080_$1
  e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl r6_test20_$1
  e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl r6_hfr22_$1
}
export STREAMENC_RATE_SLACK=0.008
export STREAMENC_HARD_PREFILTER=${R6_PREFILTER:-cblur16_ds85}; run_set SC
export STREAMENC_HARD_PREFILTER=cblur4_ds85;                    run_set S
log "round6 e2e done"
