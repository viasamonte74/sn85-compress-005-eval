#!/bin/bash
# 2026-11-12 rc6 on held-out AI clips by difficulty (user request). rc6 defaults, eval model (trained without the test
# split), validator-style 4-clip batches per tier, alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
python scripts/e2e_batched.py --clips data/e2e_ai_test_easy.txt data/e2e_ai_test_medium.txt data/e2e_ai_test_hard.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/rc6_ai_tiers > results/e2e_batched/rc6_ai_tiers.log 2>&1
log "rc6_ai_tiers: $(python3 -c "import json; s=json.load(open('results/e2e_batched/rc6_ai_tiers/summary.json')); print({k: s[k] for k in s if k not in ('batches','model','cpus','batch')})")"
log "rc6 ai tiers done"
