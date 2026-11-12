#!/bin/bash
# 2026-11-11 (user: yes): candidate features hard-AI prefilter (cblur4_ds85 below predicted VMAF 80) + segment fill,
# end to end, alone on the CPU. Practice pack with the compliant shipping model (c0); held-out sets with its
# held-out twin (c_abs). Baselines (features off): practice_c0_default, v13_{ai1080,test20,hfr22}_abs.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
export STREAMENC_SEGFINAL=1 STREAMENC_SEGFILL=1 STREAMENC_HARD_PREFILTER=cblur4_ds85
log() { echo "$(date -u +%H:%M) $*"; }
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
e2e data/e2e_practice.txt     models/crf_predictor_c0.pkl         practice_c0_segfinal
e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c_abs.pkl v19_ai1080_segfinal
e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c_abs.pkl v19_test20_segfinal
e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c_abs.pkl v19_hfr22_segfinal
log "segfinal e2e done"
