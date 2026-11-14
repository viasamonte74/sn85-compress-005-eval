#!/bin/bash
# 2026-11-12: the items-1-3 chain's held-out scoring ran out of disk (20 parallel canonical y4m at 1080p60 > 26 GB free:
# 20 clips VMAF_UNAVAILABLE). Stop the chain before its held-out step and redo both held-out runs with 8 scorers.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
until grep -q "nf_test16hard:" results/predfix/items123.log 2>/dev/null; do sleep 5; done
PG=$(ps -eo pgid,cmd | awk '/run_items123.sh/ && !/awk/ {print $1; exit}'); [ -n "$PG" ] && kill -9 -$PG 2>/dev/null; sleep 2
log "stopped items123 chain (pgid $PG) before its held-out step"
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
ho() { python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
         --model models/crf_predictor_eval_c_abs.pkl --score-procs 8 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
(export STREAMENC_FREE_FILL=1; ho ff_heldout)
(export STREAMENC_NEG_FULL_FPS=50; ho nf_heldout)
log "heldout rescore done"
