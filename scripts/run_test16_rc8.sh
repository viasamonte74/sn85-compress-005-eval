#!/bin/bash
# 2026-11-12 test16 (user: "prepare test data like the hidden set and test rc8"): 16 clips no model has seen = the 8
# practice-pack clips + 8 new ones generated for this test (/workspace/dataset/test16: 3 Wan 2.2 AI, 3 Blender, 2
# screen; evaluation only, never train on them). rc8 as shipped (models/crf_predictor.pkl), no overrides, validator-
# style 4-clip batches in a fixed mixed order (data/e2e_test16_hidden_like.txt), alone on the CPU.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
python scripts/e2e_batched.py --clips data/e2e_test16_hidden_like.txt --model models/crf_predictor.pkl \
    --out results/e2e_batched/rc8_test16 > results/e2e_batched/rc8_test16.log 2>&1
log "rc8_test16: $(python3 -c "import json; print(json.load(open('results/e2e_batched/rc8_test16/summary.json'))['all'])")"
log "rc8 test16 done"
