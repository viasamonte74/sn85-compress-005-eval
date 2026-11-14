#!/bin/bash
# 2026-11-13: after the clean search, check the cheap winners at the runtime presets on 10 clean + 4 AI training clips.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
while ! grep -q "^DONE" results/clean_search/search.log; do sleep 20; done
rm -rf work/clean_search
CLIPS=hfr_rnd_medium_04_1080,film_cosmos_24_02,rnd_easy_06,film_bbb_60_02_1080,ai_easy_14_i60,ai_medium_05_i48,aix_hard_028,anim_hard_006
python scripts/clean_combo_check.py --clips $CLIPS --workers 8 >> results/clean_search/combo.log 2>&1
echo "CHAIN DONE rc=$?" >> results/clean_search/combo.log
