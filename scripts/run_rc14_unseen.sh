#!/bin/bash
# 2026-11-14 rc14 decision on UNSEEN val16f / val16g (results/decision_rule_rc14_segspec.md): repo code, R = SEGSPEC off,
# S = on, balanced order R_f S_f S_g R_g, nothing else running. Starts after generation + packaging and the dev chain.
R=/workspace/sn85-compression-005
cd $R
L=$R/results/rc14_unseen.log
G=/tmp/claude-0/-workspace/d67412ae-2e2e-445a-8c46-716946569785/scratchpad/val16fg/gen.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
until grep -q "val16fg generation done" $G 2>/dev/null && grep -qE "^[0-9-]+ [0-9:]+ done$" $R/results/rc14_dev2.log 2>/dev/null; do sleep 20; done
log "inputs: $(grep -E 'package val16[fg]' $G | tr '\n' ' ')"
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
log "start"
for J in R:val16f S:val16f S:val16g R:val16g; do
  V=${J%%:*}; T=${J#*:}
  [ $V = S ] && export STREAMENC_SEGSPEC=1 || export STREAMENC_SEGSPEC=0
  python scripts/e2e_batched.py --clips data/e2e_$T.txt --model models/crf_predictor.pkl --score-procs 4 \
      --work work/e2e_u14_${V}_$T --out results/e2e_batched/u14_${V}_$T > results/e2e_batched/u14_${V}_$T.log 2>&1
  log "u14_${V}_$T: $(summ u14_${V}_$T)"; rm -rf work/e2e_u14_${V}_$T
done
unset STREAMENC_SEGSPEC
log "done"
