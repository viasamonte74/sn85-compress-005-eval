#!/bin/bash
# 2026-11-13 rc9 development e2e (user: "build rc9 and apply them, test like validators, fix problems, test again"):
# chroma QP on every final encode, start-up QP offset + screen-content mode on clips the classifier calls render/screen.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
TAG=${1:-v1}
EXTRA=${2-chq63}; CLEANT=${3-sqo8,scm1}; MP=${4-models/crf_predictor_kind.pkl}; ME=${5-models/crf_predictor_eval_c_abs_kind.pkl}
export STREAMENC_SVT_EXTRA=$EXTRA STREAMENC_SVT_EXTRA_CLEAN=$CLEANT
export STREAMENC_EARLY_BUMP_MAX_STEPS=${6-2}
L=results/rc9dev.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 8 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
log "start $TAG extra=$STREAMENC_SVT_EXTRA clean=$STREAMENC_SVT_EXTRA_CLEAN models $MP $ME bump_max $STREAMENC_EARLY_BUMP_MAX_STEPS"
run rc9${TAG}_practice data/e2e_practice.txt $MP
run rc9${TAG}_test16 data/e2e_test16_hidden_like.txt $MP
run rc9${TAG}_test16hard data/e2e_test16_hard.txt $MP
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model $ME --score-procs 8 --out results/e2e_batched/rc9${TAG}_heldout > results/e2e_batched/rc9${TAG}_heldout.log 2>&1
log "rc9${TAG}_heldout: $(summ rc9${TAG}_heldout)"
log "done $TAG"
