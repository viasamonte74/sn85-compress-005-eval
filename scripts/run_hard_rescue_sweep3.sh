#!/bin/bash
# round 2 on two over-budget clips: superres + QP compression, frame hold, temporal blend, combinations
exec /workspace/sn85-compression-005/scripts/run_hard_rescue_sweep.sh --workers 8 \
  --clips rnd_hard_03,gen_hard_19_edge_boost_ai_hard_19 \
  --configs "svt_crf_p4_qm_sr16_qpc,svt_crf_p4_qm_la+hold2,svt_crf_p4_qm_la+hold3,svt_crf_p4_qm_la+tmix3,svt_crf_p4_qm_la+hold2_ds50,svt_crf_p4_qm_sr16+hold2,svt_crf_p4_qm_sr16_qpc+hold2,svt_crf_p4_qm_sr16_qpc+hold3"
