#!/bin/bash
# 2026-11-13 development check of F1 (NEG check when unknown / filtered) + F2 (pilot CPU calibration) on rc12 code.
# Shares the machine with val16d/e generation: judge decisions / quality, not timing.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
L=results/f1f2_dev.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() {
  for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
  STREAMENC_NEG_CHECK_UNKNOWN=1 STREAMENC_CPU_CALIB=1 python scripts/e2e_batched.py --clips $2 --model models/crf_predictor.pkl --score-procs 4 \
      --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1
  log "$1: $(summ $1)"; rm -rf work/e2e_$1
}
log "start"
run f12dev_val16b data/e2e_val16b.txt
run f12dev_practice data/e2e_practice.txt
log "done"
