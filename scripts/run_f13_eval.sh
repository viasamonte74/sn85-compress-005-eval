#!/bin/bash
# 2026-11-13 base-risk fixes: dev check (seen data) then pre-registered evaluation on unseen val16d / val16e
# (results/decision_rule_f1f2.md incl. amendment). Clean machine: starts after generation.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
L=results/f13_eval.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
G=/tmp/claude-0/-workspace/d67412ae-2e2e-445a-8c46-716946569785/scratchpad/val16de/gen.log
until grep -q "val16de generation done" $G 2>/dev/null; do sleep 20; done
for S in val16d val16e; do
  python scripts/package_val16.py --name $S --prefix ${S/val16/v16} --seed $(( 20261113 + $([ $S = val16e ] && echo 17 || echo 11) )) > results/package_$S.log 2>&1
  log "package $S rc=$? $(tail -1 results/package_$S.log)"
done
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() {  # tag clips F1 F3
  for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
  STREAMENC_NEG_CHECK_UNKNOWN=$3 STREAMENC_WALL_GUARD=$4 python scripts/e2e_batched.py --clips $2 --model models/crf_predictor.pkl --score-procs 4 \
      --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1
  log "$1: $(summ $1)"; rm -rf work/e2e_$1
}
log "start"
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
