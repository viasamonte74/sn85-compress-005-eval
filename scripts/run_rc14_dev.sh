#!/bin/bash
# 2026-11-14 rc14 dev (seen data): repo code (rc13 + SEGSPEC switch), R = SEGSPEC off, S = on, interleaved per set so
# both see the same machine load; slow sets first. results/decision_rule_rc14_segspec.md
R=/workspace/sn85-compression-005
cd $R
L=$R/results/rc14_dev.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
until grep -qE "^[0-9-]+ [0-9:]+ done$" $R/results/rc13_verify.log 2>/dev/null; do sleep 20; done
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
log "start"
for S in val16e:e2e_val16e val16d:e2e_val16d val16b:e2e_val16b practice:e2e_practice; do
  T=${S%%:*}; F=${S#*:}
  for V in R S; do
    [ $V = S ] && export STREAMENC_SEGSPEC=1 || export STREAMENC_SEGSPEC=0
    python scripts/e2e_batched.py --clips data/$F.txt --model models/crf_predictor.pkl --score-procs 4 \
        --work work/e2e_d14_${V}_$T --out results/e2e_batched/d14_${V}_$T > results/e2e_batched/d14_${V}_$T.log 2>&1
    log "d14_${V}_$T: $(summ d14_${V}_$T)"; rm -rf work/e2e_d14_${V}_$T
  done
done
unset STREAMENC_SEGSPEC
log "done"
