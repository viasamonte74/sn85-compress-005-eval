#!/bin/bash
# Strict label policy v3 over the whole dataset; AI hard (720p + 1080p) and Wan hard clips first.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
exec python scripts/build_encoder_rank_dataset.py --resume --per-tier 0 --with-1080 \
  --extra-dirs ai_gen2,open_clips --include '^(ai_hard_|aix_hard_)'
