#!/bin/bash
# 2026-11-13 rc9 recheck (user: "recheck carefully ... find any issue"): grayscale / chroma gate stress on desaturated
# clips (reference colour energy just above the 0.03 exemption). rc9 defaults vs rc8 behaviour (no chroma QP, bump 2).
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=results/stress_lowsat.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
python scripts/e2e_batched.py --clips data/e2e_stress_lowsat.txt --model models/crf_predictor.pkl --score-procs 8 \
    --work work/e2e_stress_rc9 --out results/e2e_batched/stress_rc9 > results/e2e_batched/stress_rc9.log 2>&1
log "stress_rc9: $(summ stress_rc9)"
STREAMENC_SVT_EXTRA= STREAMENC_EARLY_BUMP_MAX_STEPS=2 python scripts/e2e_batched.py --clips data/e2e_stress_lowsat.txt --model models/crf_predictor.pkl --score-procs 8 \
    --work work/e2e_stress_rc8 --out results/e2e_batched/stress_rc8 > results/e2e_batched/stress_rc8.log 2>&1
log "stress_rc8: $(summ stress_rc8)"
log "done"
