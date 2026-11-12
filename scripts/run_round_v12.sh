#!/bin/bash
# Round after v10/v11 labelling + HFR fix (2026-11-10):
#  1. wait for sn85-label-v11-*; 2. label hfr_ai + finished hfr_render clips (v12, 3 shards);
#  2b. wait for sn85-hfr-fix + sn85-hfr-wan + sn85-render2 + sn85-render-film, label hfr_render + hfr_wan + render2 +
#      render_film clips not in v12 (v13); 3. merge v11/v12/v13, rebuild data/encoder_rank_all = old all (kept as encoder_rank_all_v9) + v10 + v11 + v12;
#  4. retrain eval + final models (v9 models kept as *_v9.*); 5. e2e: test26 with the new eval model
#     (vs v9c), and HFR test-split clips with the old and the new eval model.
set -u
# temp files (VMAF canonical y4m, raw cache) in RAM: the root overlay is nearly full (2026-11-10)
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
cd /workspace/sn85-compression-005
. /opt/nvm/nvm.sh >/dev/null 2>&1
PY=/venv/main/bin/python
log() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
online() { pm2 jlist 2>/dev/null | python3 -c "import json,sys; n=sys.argv[1:]; sys.exit(0 if any(p['name'] in n and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)" "$@"; }
# shard <i> of label set <v> is complete when its labels.jsonl has a row for every clip of its shard list
shard_done() { local f=data/encoder_rank_$1/shard$2; [ -s $f.clips ] && [ -f $f/labels.jsonl ] && [ "$(wc -l < $f/labels.jsonl)" -ge "$(wc -l < $f.clips)" ]; }
# start the shards of data/label_<v>_clips.txt that are neither running nor complete (safe across chain restarts)
label_set() {
  local v=$1 i
  for i in 0 1 2; do
    online sn85-label-$v-$i && continue
    shard_done $v $i && continue
    pm2 delete sn85-label-$v-$i >/dev/null 2>&1
    pm2 start scripts/run_label_$v.sh --name sn85-label-$v-$i --interpreter bash --no-autorestart -- $i 3 >/dev/null
  done
}

log "waiting for sn85-label-v11-*"
while online sn85-label-v11-0 sn85-label-v11-1 sn85-label-v11-2; do sleep 120; done

# v12 starts without waiting for the last Blender render (user request 2026-11-10 12:10 UTC): hfr_ai + the
# hfr_render clips finished so far (renders are written to .tmp.mkv and renamed, so only complete files match);
# hfr_wan joins only if its generation has already finished. The list is kept across chain restarts so that
# --resume sees the same shard assignment.
if [ ! -s data/label_v12_clips.txt ]; then
  wan_glob=""; online sn85-hfr-wan || wan_glob="/workspace/dataset/hfr_wan/*.mkv"
  ls -S /workspace/dataset/hfr_ai/*.mkv /workspace/dataset/hfr_render/*.mkv $wan_glob 2>/dev/null | grep -v '\.tmp\.' > data/label_v12_clips.txt
fi
log "v12 clips: $(wc -l < data/label_v12_clips.txt)"
if [ -s data/label_v12_clips.txt ]; then
  label_set v12
  sleep 60
  while online sn85-label-v12-0 sn85-label-v12-1 sn85-label-v12-2; do sleep 120; done
  log "v12 labelling finished"
fi

# v13 (started without waiting for the generators, user request 2026-11-10 16:55 UTC): every clip of hfr_render +
# hfr_wan + render2 + render_film complete at list time and not in v12 (generators write .tmp.mkv and rename, so only
# complete files match). v13b: the clips the generators finish after that, labelled as soon as all generators stop.
# Lists are kept across chain restarts so that --resume sees the same shard assignment.
# screen2 (screen set 2, 24/25/30 fps, added 2026-11-10 17:05) is listed after the v13 list was built -> v13b
GEN_DIRS="/workspace/dataset/hfr_render /workspace/dataset/hfr_wan /workspace/dataset/render2 /workspace/dataset/render_film /workspace/dataset/screen2"
gen_clips() { local d; for d in $GEN_DIRS; do ls $d/*.mkv 2>/dev/null; done | grep -v '\.tmp\.' | xargs -r ls -S; }
if [ ! -s data/label_v13_clips.txt ]; then
  gen_clips | grep -vxF -f data/label_v12_clips.txt > data/label_v13_clips.txt
fi
log "v13 clips (hfr_render + hfr_wan + render2 + render_film not in v12): $(wc -l < data/label_v13_clips.txt)"
[ -s data/label_v13_clips.txt ] && label_set v13
log "waiting for sn85-hfr-fix + sn85-hfr-wan + sn85-render2 + sn85-render-film + sn85-screen2"
while online sn85-hfr-fix sn85-hfr-wan sn85-render2 sn85-render-film sn85-screen2; do sleep 120; done
log "HFR fix: $(tail -n 3 ~/.pm2/logs/sn85-hfr-fix-out.log | tr '\n' ' ')"
if [ ! -s data/label_v13b_clips.txt ]; then
  gen_clips | grep -vxF -f data/label_v12_clips.txt | grep -vxF -f data/label_v13_clips.txt > data/label_v13b_clips.txt
fi
log "v13b clips (finished after the v13 list): $(wc -l < data/label_v13b_clips.txt)"
[ -s data/label_v13b_clips.txt ] && label_set v13b
sleep 60
while online sn85-label-v13-0 sn85-label-v13-1 sn85-label-v13-2 sn85-label-v13b-0 sn85-label-v13b-1 sn85-label-v13b-2; do sleep 120; done
log "v13 + v13b labelling finished"

for v in v11 v12 v13 v13b; do
  [ -d data/encoder_rank_$v/shard0 ] || continue
  $PY scripts/merge_label_shards.py $v
  log "$v: clips $(cat data/encoder_rank_$v/shard*.clips | wc -l), labels $(wc -l < data/encoder_rank_$v/merged/labels.jsonl)"
done

[ -d data/encoder_rank_all_v9 ] || mv data/encoder_rank_all data/encoder_rank_all_v9
dirs="data/encoder_rank_all_v9 data/encoder_rank_v10/merged"
for v in v11 v12 v13 v13b; do [ -f data/encoder_rank_$v/merged/labels.jsonl ] && dirs="$dirs data/encoder_rank_$v/merged"; done
$PY scripts/combine_label_dirs.py --out data/encoder_rank_all $dirs || { log "combine FAILED"; exit 1; }
log "encoder_rank_all: $(wc -l < data/encoder_rank_all/labels.jsonl) labels from $dirs"

for f in crf_predictor.pkl crf_predictor.json crf_predictor_eval.pkl crf_predictor_eval.json; do
  [ -f models/${f%.*}_v9.${f##*.} ] || cp models/$f models/${f%.*}_v9.${f##*.}
done
log "training"
bash scripts/run_train_all.sh || { log "training FAILED"; exit 1; }
log "trained: $(tail -n 2 results/train_eval_all.log | tr '\n' ' ')"

# HFR test-split clips (families never seen by either eval model)
$PY - <<'PY'
import json
rows=[json.loads(l) for l in open('data/encoder_rank_all/labels.jsonl')]
hfr=sorted(r['path'] for r in rows if r.get('split')=='test' and '/hfr_' in r['path'])
open('data/e2e_hfr_test.txt','w').write('# HFR (>=50 fps) test-split clips: hfr_ai/render/screen/open, 720p + 1080p\n'+'\n'.join(hfr)+'\n')
print(len(hfr))
PY
log "HFR test clips: $(grep -vc '^#' data/e2e_hfr_test.txt)"

E="$PY scripts/e2e_eval.py --cpus 0-31 --workers 4"
$E --clips data/e2e_test26.txt --model models/crf_predictor_eval.pkl --out results/e2e/v10_test26 > results/e2e_v10_test26.log 2>&1
log "e2e test26 new: $(cat results/e2e/v10_test26/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
if grep -vq '^#' data/e2e_hfr_test.txt; then
  $E --clips data/e2e_hfr_test.txt --model models/crf_predictor_eval_v9.pkl --out results/e2e/v10_hfr_oldmodel > results/e2e_v10_hfr_old.log 2>&1
  log "e2e HFR old model: $(cat results/e2e/v10_hfr_oldmodel/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
  $E --clips data/e2e_hfr_test.txt --model models/crf_predictor_eval.pkl --out results/e2e/v10_hfr_newmodel > results/e2e_v10_hfr_new.log 2>&1
  log "e2e HFR new model: $(cat results/e2e/v10_hfr_newmodel/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
fi
log "round v12 done"
