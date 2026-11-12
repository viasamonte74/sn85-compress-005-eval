#!/bin/bash
# round 4: (a) slower presets on rnd_hard_03 (p2 sr16+hold3 = VMAF 59.57, 0.43 below the scoring line);
# (b) the reference sr16 + hold3 / hold2 on all 13 clips that had no valid encode
S=/workspace/sn85-compression-005/scripts/run_hard_rescue_sweep.sh
bash $S --workers 4 --clips rnd_hard_03 --configs "svt_crf_p1_qm_sr16+hold3,svt_crf_p0_qm_sr16+hold3,svt_crf_p1_qm_sr16+hold2_ds85,svt_crf_p0_qm_sr16+hold2"
exec $S --workers 6 --clips "rnd_hard_03,rnd_hard_11,gen_hard_19_edge_boost_ai_hard_19,gen_hard_22_edge_boost_ai_hard_20,gen_hard_25_edge_boost_ai_medium_04,rnd_hard_04,rnd_hard_11_1080,gen_hard_04_sharp_grain_ai_hard_19,gen_hard_20_edge_boost_ai_hard_23,rnd_hard_04_1080,gen_hard_02_sharp_grain_ai_hard_13,gen_hard_18_edge_boost_ai_hard_13,gen_hard_21_edge_boost_ai_hard_25" --configs "svt_crf_p2_qm_sr16+hold3,svt_crf_p4_qm_sr16+hold3,svt_crf_p2_qm_sr16+hold2,svt_crf_p4_qm_sr16"
