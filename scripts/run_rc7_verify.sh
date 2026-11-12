#!/bin/bash
# 2026-11-12 rc7 verification (user: "Yes build rc7"): (1) sandbox-like practice sim of the rc7 tree; (2) held-out
# 65 clips and (3) the 53 test-split AI clips by tier, 4-clip batches, repo code (identical to rc7), no overrides.
log() { echo "$(date -u +%H:%M) $*"; }
/workspace/sn85-candidates/simtest/run_sim_rc7.sh
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/rc7_heldout > results/e2e_batched/rc7_heldout.log 2>&1
log "rc7_heldout: $(python3 -c "import json; print(json.load(open('results/e2e_batched/rc7_heldout/summary.json'))['all'])")"
python scripts/e2e_batched.py --clips data/e2e_ai_test_easy.txt data/e2e_ai_test_medium.txt data/e2e_ai_test_hard.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/rc7_ai_tiers > results/e2e_batched/rc7_ai_tiers.log 2>&1
log "rc7_ai_tiers: $(python3 -c "import json; s=json.load(open('results/e2e_batched/rc7_ai_tiers/summary.json')); print({k: s[k] for k in s if k not in ('batches','model','cpus','batch')})")"
log "rc7 verify done"
