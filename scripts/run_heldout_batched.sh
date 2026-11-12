#!/bin/bash
# 2026-11-12 rc3: held-out non-natural clips (65) in validator-style 4-clip batches, alone on the CPU.
# Same lists / eval model as round 7b (r7b_*_FD, single run.sh call per list); compare outputs + time.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt \
    --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/rc3_heldout
echo "heldout batched done"
