#!/bin/bash
# Train the eval (test split held out) and final CRF predictors on data/encoder_rank_all.
set -eu
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
python scripts/train_crf_predictor.py --data data/encoder_rank_all --exclude-split test --out models/crf_predictor_eval.pkl > results/train_eval_all.log 2>&1
python scripts/train_crf_predictor.py --data data/encoder_rank_all --out models/crf_predictor.pkl > results/train_final_all.log 2>&1
echo trained
