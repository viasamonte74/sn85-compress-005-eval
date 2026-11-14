#!/bin/bash
# 2026-11-13 unseen-data verification of rc12's options (pre-registered: results/decision_rule_rc12_threshold.md).
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
L=results/unseen_eval.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
G=/tmp/claude-0/-workspace/d67412ae-2e2e-445a-8c46-716946569785/scratchpad/val16bc/gen.log
until grep -q "val16bc generation done" $G 2>/dev/null; do sleep 20; done
for S in val16b val16c; do
  python scripts/package_val16.py --name $S --prefix ${S/val16/v16} --seed $(( 20261113 + ${#S} + $([ $S = val16c ] && echo 7 || echo 3) )) > results/package_$S.log 2>&1
  log "package $S rc=$? $(tail -1 results/package_$S.log)"
done
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() {  # tag clips model stage2 refit
  for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
  STREAMENC_HARD_VMAF_STAGE2=$4 STREAMENC_SEGFINAL_REFIT=$5 python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 4 \
      --work work/e2e_$1 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1
  log "$1: $(summ $1)"; rm -rf work/e2e_$1
}
M=models/crf_predictor.pkl; ME=models/crf_predictor_eval_c_abs.pkl
log "start"
for S in val16b val16c; do
  run un_A_$S data/e2e_$S.txt $M 80 0
  run un_B_$S data/e2e_$S.txt $M 80 1
  run un_C_$S data/e2e_$S.txt $M 90 1
  run un_D_$S data/e2e_$S.txt $M 85 1
done
# variant B on the five existing sets (A = rc11c_* / cf_new_practice, C = rc12c_* / cb_practice exist) for the cross-check
run ex_B_practice data/e2e_practice.txt $M 80 1
run ex_B_val16 data/e2e_val16.txt $M 80 1
run ex_B_test16 data/e2e_test16_hidden_like.txt $M 80 1
run ex_B_test16hard data/e2e_test16_hard.txt $M 80 1
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done
STREAMENC_HARD_VMAF_STAGE2=80 STREAMENC_SEGFINAL_REFIT=1 python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model $ME --score-procs 4 --work work/e2e_ex_B_heldout --out results/e2e_batched/ex_B_heldout > results/e2e_batched/ex_B_heldout.log 2>&1
log "ex_B_heldout: $(summ ex_B_heldout)"; rm -rf work/e2e_ex_B_heldout
log "done"
