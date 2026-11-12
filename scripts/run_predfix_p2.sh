#!/bin/bash
# 2026-11-12 prediction fix P2 (user: "try" the gate): stage 3 only when stage 2 predicts best < 94. Waits for the
# ungated round (sn85-predfix-round) to finish so the e2e timings never overlap. Alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
until grep -q "predfix round done" results/predfix/round.log 2>/dev/null || ! pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='sn85-predfix-round' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 10; done
log "tests: $(python -m pytest -q tests/ -p no:cacheprovider 2>&1 | tail -1)"
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
pr() { python scripts/e2e_batched.py --clips data/e2e_practice.txt --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model $2 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
export STREAMENC_QUICK_VMAF_THREADS=8,4 STREAMENC_QUICK_VMAF_PAIRING=rate STREAMENC_STAGE3=1 STREAMENC_STAGE3_MAX_PRED=94
pr p2_practice_s3g94 models/crf_predictor_c0_s3.pkl
ho p2_heldout_s3g94 models/crf_predictor_eval_c_abs_s3.pkl
log "predfix p2 done"
