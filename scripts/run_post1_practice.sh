#!/bin/bash
# 2026-11-14 rc13.post1 on the practice pack (q01-q08): candidate code + defaults, validator-style 4-clip batches, exact scoring.
R=/workspace/sn85-compression-005
cd $R
L=$R/results/post1_practice.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
C=/workspace/sn85-candidates/sn85-005-1.0.0rc13.post1
ln -sfn $R/third_party $C/third_party
cd $C
log "start"
python scripts/e2e_batched.py --clips $R/data/e2e_practice.txt --model $C/models/crf_predictor.pkl --score-procs 2 \
    --work $R/work/e2e_post1_practice --out $R/results/e2e_batched/post1_practice > $R/results/e2e_batched/post1_practice.log 2>&1
log "post1_practice: $(python3 -c "import json; print(json.load(open('$R/results/e2e_batched/post1_practice/summary.json'))['all'])")"
rm -rf $R/work/e2e_post1_practice
rm -f $C/third_party; find $C -name __pycache__ -type d -prune -exec rm -rf {} +
log "done"
