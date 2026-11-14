#!/bin/bash
# 2026-11-14 rc16 pre-registered test (results/decision_rule_rc16.md): R = rc13.post3, N = rc16, own folders + defaults,
# validator-style 4-clip batches, exact scoring (2 scorers: disk). Balanced interleaving on 4 timed sets, then N on the rest.
R=/workspace/sn85-compression-005
cd $R
L=$R/results/rc16_test.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
declare -A DIR=([R]=/workspace/sn85-candidates/sn85-005-1.0.0rc13.post3 [N]=/workspace/sn85-candidates/sn85-005-1.0.0rc16)
for V in R N; do ln -sfn $R/third_party ${DIR[$V]}/third_party; done
one() {  # one <R|N> <tag> <list...>
  V=$1; T=$2; shift 2; C=${DIR[$V]}; M=$C/models/crf_predictor.pkl; [ $T = heldout ] && M=$R/models/crf_predictor_eval_c_abs.pkl
  cl=(); for f in "$@"; do cl+=($R/data/$f.txt); done
  ( cd $C && python scripts/e2e_batched.py --clips "${cl[@]}" --model $M --score-procs 2 --work $R/work/e2e_t16_${V}_$T \
      --out $R/results/e2e_batched/t16_${V}_$T > $R/results/e2e_batched/t16_${V}_$T.log 2>&1 )
  log "t16_${V}_$T: $(summ t16_${V}_$T)"; rm -rf $R/work/e2e_t16_${V}_$T
}
log "start"
one R practice e2e_practice; one N practice e2e_practice
one N val16f e2e_val16f;     one R val16f e2e_val16f
one R val16g e2e_val16g;     one N val16g e2e_val16g
one N val16d e2e_val16d;     one R val16d e2e_val16d
for S in val16:e2e_val16 val16b:e2e_val16b val16c:e2e_val16c val16e:e2e_val16e test16:e2e_test16_hidden_like \
         hard:e2e_test16_hard lowsat:e2e_stress_lowsat; do one N ${S%%:*} ${S#*:}; done
one N heldout e2e_test26_nonat e2e_ai1080_nonat e2e_hfr_nonat
for V in R N; do rm -f ${DIR[$V]}/third_party; find ${DIR[$V]} -name __pycache__ -type d -prune -exec rm -rf {} +; done
log "done"
