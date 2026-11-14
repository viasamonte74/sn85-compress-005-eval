#!/bin/bash
# 2026-11-13 clean-content (render / screen) encoder-config search, training / val clips only.
cd /workspace/sn85-compression-005
source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
CLIPS=rnd_hard_06,scr2_medium_03,film_bbb_30_01,scr_hard_07,rnd2_medium_06,film_caminandes2_24_00,scr_easy_05,hfr_rnd_medium_04_1080,film_cosmos_24_02,scr_medium_06,rnd_easy_06,film_bbb_60_02_1080
python scripts/clean_config_search.py --clips $CLIPS --workers 8 > results/clean_search/search.log 2>&1
echo "DONE rc=$?" >> results/clean_search/search.log
