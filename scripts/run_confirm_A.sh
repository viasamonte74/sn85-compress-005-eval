#!/bin/bash
# 2026-11-12 option A confirmation (user chose A: rate slack 0.008 default on), defaults only, no overrides.
# segmented final + two-try fix, segment fill, cblur4_ds85 hard path) run 3x on every list, alone on the CPU, no
# overrides. Compare per-clip sha256 / CRF / wall across repeats. Practice: c0 model; held-out: its twin c_abs.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
for r in ${REPEATS:-1 2 3}; do
  e2e data/e2e_practice.txt     models/crf_predictor_c0.pkl         stA_practice_r$r
  e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl stA_ai1080_r$r
  e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl stA_test20_r$r
  e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl stA_hfr22_r$r
done
log "option A confirmation done"
