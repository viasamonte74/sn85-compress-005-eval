#!/bin/bash
# 2026-11-11 chain: (1) candidate without the BT.601-tagged screen recordings -> models/crf_predictor_c0.pkl;
# (2) official practice pack: shipped version vs candidate (fill A) - timed, nothing else running;
# (3) screen recordings converted to BT.709 (screen_real709) + relabelled (v15b);
# (4) held-out model with the fixed screen data + e2e (fill A) to decide whether they help.
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
log() { echo "$(date -u +%H:%M) $*"; }
FILL="STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.95 STREAMENC_ACCEPT_LOW=0.955"
python scripts/train_crf_predictor.py --data data/encoder_rank_all --exclude-kinds other \
  --out models/crf_predictor_c0.pkl > results/train_final_c0.log 2>&1 || { log "train c0 FAILED"; exit 1; }
log "trained c0: $(grep -E 'clips,' results/train_final_c0.log | cut -c1-40)"
e2e() { python scripts/e2e_eval.py --cpus 0-31 --workers 4 --clips $1 --model $2 --out results/e2e/$3 > results/e2e_$3.log 2>&1
        log "$3: $(cat results/e2e/$3/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"; }
e2e data/e2e_practice.txt models/crf_predictor.pkl practice_shipped
env $FILL bash -c "$(declare -f e2e log); e2e data/e2e_practice.txt models/crf_predictor_c0.pkl practice_cand"
python scripts/import_hf_clips.py > results/import_screen709.log 2>&1; log "import 709: $(tail -1 results/import_screen709.log)"
ls /workspace/dataset/screen_real709/scrr_*.mkv > data/label_v15b_clips.txt
for i in 0 1 2; do CLIPS=data/label_v15b_clips.txt scripts/run_label_v9.sh $i 3 v15b > results/label_v15b_$i.log 2>&1 & done; wait
python scripts/merge_label_shards.py v15b > /dev/null 2>&1
log "v15b labels: $(wc -l < data/encoder_rank_v15b/merged/labels.jsonl)"
python scripts/combine_label_dirs.py --out data/encoder_rank_c2 data/encoder_rank_all data/encoder_rank_v15b/merged > /dev/null 2>&1
python scripts/train_crf_predictor.py --data data/encoder_rank_c2 --exclude-split test --exclude-kinds other \
  --out models/crf_predictor_eval_c2.pkl > results/train_eval_c2.log 2>&1 || { log "train eval c2 FAILED"; exit 1; }
log "trained eval c2: $(grep -E 'clips,' results/train_eval_c2.log | cut -c1-40)"
python - <<'P'
import json
L=[json.loads(l) for l in open('data/encoder_rank_v15b/merged/labels.jsonl')]
open('data/e2e_screenreal709_test.txt','w').write('# screen recordings converted to BT.709, test split\n'+'\n'.join(sorted(r['path'] for r in L if r['split']=='test'))+'\n')
P
export STREAMENC_FINAL_AIM=0.985 STREAMENC_FINAL_AIM_EASY=0.95 STREAMENC_ACCEPT_LOW=0.955
e2e data/e2e_ai1080_nonat.txt models/crf_predictor_eval_c2.pkl v17_ai1080
e2e data/e2e_test26_nonat.txt models/crf_predictor_eval_c2.pkl v17_test20
e2e data/e2e_hfr_nonat.txt    models/crf_predictor_eval_c2.pkl v17_hfr22
e2e data/e2e_screenreal709_test.txt models/crf_predictor_eval_c2.pkl v17_screen709
e2e data/e2e_screenreal709_test.txt models/crf_predictor_eval_c_abs.pkl v17_screen709_c0
log "round c2 done"
