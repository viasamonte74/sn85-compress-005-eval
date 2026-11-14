#!/bin/bash
# 2026-11-13 base-risk fixes: dev check (seen data) then pre-registered evaluation on unseen val16d / val16e
# (results/decision_rule_f1f2.md incl. amendment). Clean machine: starts after generation.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
L=results/f13_eval.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() {  # tag clips F1 F3
  for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
  STREAMENC_NEG_CHECK_UNKNOWN=$3 STREAMENC_WALL_GUARD=$4 python scripts/e2e_batched.py --clips $2 --model models/crf_predictor.pkl --score-procs 4 \
      --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1
  log "$1: $(summ $1)"; rm -rf work/e2e_$1
}
log "start (restart 2: guard slack 1.5, no p8 fallback; packaging already done)"
run f13dev_val16b data/e2e_val16b.txt 1 1
run f13dev_practice data/e2e_practice.txt 1 1
log "dev done"
for S in val16d val16e; do
  run fx_C_$S data/e2e_$S.txt 0 0
  run fx_G_$S data/e2e_$S.txt 1 0
  run fx_I_$S data/e2e_$S.txt 0 1
  run fx_J_$S data/e2e_$S.txt 1 1
done
log "done"
