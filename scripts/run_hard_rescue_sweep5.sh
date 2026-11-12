#!/bin/bash
# round 3 around the reference sr16 + hold3 (rnd_hard_03 valid at VMAF 56.8): slower preset, hold2 + low-pass, averaged hold
exec /workspace/sn85-compression-005/scripts/run_hard_rescue_sweep.sh --workers 8 \
  --clips rnd_hard_03,rnd_hard_04_1080,gen_hard_19_edge_boost_ai_hard_19 \
  --configs "svt_crf_p4_qm_sr16+hold3,svt_crf_p2_qm_sr16+hold3,svt_crf_p2_qm_sr16+hold2,svt_crf_p4_qm_sr16+hold2_ds85,svt_crf_p4_qm_sr16+hold2_ds67,svt_crf_p2_qm_sr16+hold2_ds85,svt_crf_p4_qm_sr16+avg2hold,svt_crf_p4_qm_sr16+avg3hold,svt_crf_p2_qm_sr16+avg3hold"
