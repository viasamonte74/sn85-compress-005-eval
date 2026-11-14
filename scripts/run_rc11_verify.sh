#!/bin/bash
# 2026-11-13 rc11 verification: the candidate folder's own code and defaults (no STREAMENC_* env), validator-style 4-clip batches.
C=/workspace/sn85-candidates/sn85-005-1.0.0rc11
R=/workspace/sn85-compression-005
cd $C
# the candidate has no third_party (the image installs ffmpeg in /usr/local/bin); link the pinned build for this run only
ln -sfn $R/third_party $C/third_party
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
L=$R/results/rc11_verify.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --score-procs 8 --work $R/work/e2e_rc11verify --out $R/results/e2e_batched/$1 > $R/results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
log "start (candidate code, defaults)"
run rc11c_test16 $R/data/e2e_test16_hidden_like.txt $C/models/crf_predictor.pkl
run rc11c_test16hard $R/data/e2e_test16_hard.txt $C/models/crf_predictor.pkl
python scripts/e2e_batched.py --clips $R/data/e2e_test26_nonat.txt $R/data/e2e_ai1080_nonat.txt $R/data/e2e_hfr_nonat.txt \
    --model $R/models/crf_predictor_eval_c_abs.pkl --score-procs 8 --work $R/work/e2e_rc11verify --out $R/results/e2e_batched/rc11c_heldout > $R/results/e2e_batched/rc11c_heldout.log 2>&1
log "rc11c_heldout: $(summ rc11c_heldout)"
run rc11c_val16 $R/data/e2e_val16.txt $C/models/crf_predictor.pkl
find $C -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf $R/work/e2e_rc11verify
rm -f $C/third_party
log "done"
