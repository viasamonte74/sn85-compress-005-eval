#!/bin/bash
# Clean e2e (nothing else running, 2026-11-11): current eval model (absolute VMAF) vs difference-target eval model,
# both trained on the same 1,079 labels with the test split held out. Sets alternate model order to spread drift.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
E="python scripts/e2e_eval.py --cpus 0-31 --workers 4"
run() { $E --clips $1 --model $2 --out results/e2e/v12_$3 > results/e2e_v12_$3.log 2>&1
        echo "$(date -u +%H:%M) $3: $(cat results/e2e/v12_$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-330)"; }
run data/e2e_test26.txt models/crf_predictor_eval.pkl test26_abs
run data/e2e_test26.txt models/crf_predictor_eval_delta.pkl test26_delta
run data/e2e_ai1080_test.txt models/crf_predictor_eval_delta.pkl ai1080_delta
run data/e2e_ai1080_test.txt models/crf_predictor_eval.pkl ai1080_abs
run data/e2e_hfr_test.txt models/crf_predictor_eval.pkl hfr_abs
run data/e2e_hfr_test.txt models/crf_predictor_eval_delta.pkl hfr_delta
echo "e2e delta done"
