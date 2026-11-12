#!/bin/bash
# gen3 rally-car pair (user 2026-11-10: aig3_hard_002 scores low): quality tools on a clip that fits the budget
exec /workspace/sn85-compression-005/scripts/run_hard_rescue_sweep.sh --workers 8 \
  --clips aig3_hard_002_1080,aig3_hard_002 \
  --configs "svt_crf_p4_qm_la,svt_crf_p4_qm_sr12,svt_crf_p4_qm_la_vb,svt_crf_p4_qm_la_tf3,svt_crf_p4_qm_la_fg,svt_crf_p4_qm_la_qpc,svt_crf_p4_qm_la+hqdn3d_light,svt_crf_p4_qm_la+ds85"
