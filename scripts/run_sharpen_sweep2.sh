#!/bin/bash
# sharpening sweep stage 2 (2026-11-11): per clip, the best sharpening settings with NEG gap <= 2.7 at p4, on p0 and p1
R=/workspace/sn85-compression-005/scripts/run_hard_rescue_sweep.sh
c() { for p in p0 p1; do for f in "$@"; do printf 'svt_crf_%s_qm_la+%s,' $p $f; done; done; }
$R --workers 6 --clips ai_hard_12_1080 --configs "$(c ds85_usm30 ds67_usm30)"
$R --workers 6 --clips ai_hard_20_1080 --configs "$(c ds67_usm10 ds67_usm20)"
$R --workers 6 --clips ai_hard_04_1080 --configs "$(c ds67_usm40 usm7_20)svt_crf_p0_qm_la"
