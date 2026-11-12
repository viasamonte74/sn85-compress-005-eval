#!/bin/bash
# Round 2 of the hard-clip encoder exploration: SVT combos/knobs, SVT 1.7 arms, prefilters.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
CL=ai_hard_18,ai_hard_05,ai_hard_09,ai_hard_01,ai_hard_17,ai_hard_02_1080
python scripts/hard_encoder_explore.py --clips $CL --workers 6 \
  --arms svt_p3_combo,svt_p4_combo,svt_p4_la120_qm,svt_p4_la120_tfs3,svt_p4_la120_open,svt_p4_la120_10b,svt_p4_la120_t0,svt17_p5_t1,svt17_p5_t0,svt17_p4_t0
python scripts/hard_encoder_explore.py --clips $CL --workers 6 \
  --arms svt_p4_t1_la120 --prefilters blur05,ds75,ds85,hqdn_mild,hard_denoise,cas03
