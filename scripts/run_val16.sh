#!/bin/bash
# 2026-11-13 val16 evaluation (user "yes"): rc9 and rc8 candidate code (each from its own folder, shipped model, no env),
# validator-style 4-clip batches in val16 batch order.
R=/workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=$R/results/val16.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
for V in rc9 rc8; do
  C=/workspace/sn85-candidates/sn85-005-1.0.0$V
  ln -sfn $R/third_party $C/third_party   # pinned ffmpeg, as /usr/local/bin in the image
  cd $C
  python scripts/e2e_batched.py --clips $R/data/e2e_val16.txt --model $C/models/crf_predictor.pkl --score-procs 8 \
      --work $R/work/e2e_val16_$V --out $R/results/e2e_batched/val16_$V > $R/results/e2e_batched/val16_$V.log 2>&1
  rm -f $C/third_party; find $C -name __pycache__ -type d -prune -exec rm -rf {} +
  log "val16_$V: $(summ val16_$V)"
done
log "done"
