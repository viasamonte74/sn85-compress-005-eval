#!/bin/bash
# 2026-11-12 rc4 verification: (1) sandbox-like practice sim of the rc4 tree (nobody, env -i, 2 x 4 clips);
# (2) held-out 4-clip batched e2e with the repo code (identical to rc4), no STREAMENC_* overrides. Alone on the CPU.
log() { echo "$(date -u +%H:%M) $*"; }
/workspace/sn85-candidates/simtest/run_sim_rc4.sh
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/rc4_heldout > results/e2e_batched/rc4_heldout.log 2>&1
log "rc4_heldout: $(python3 -c "import json; print(json.load(open('results/e2e_batched/rc4_heldout/summary.json'))['all'])")"
log "rc4 verify done"
