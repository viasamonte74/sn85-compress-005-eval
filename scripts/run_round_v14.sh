#!/bin/bash
# Round v14 (2026-11-10, user: "AI native clips for 1080"): label the gen3 Wan clips (/workspace/dataset/ai_gen3:
# native 1920x1080 + Lanczos-downscaled 720p twins) and retrain.
#  1. v14 = gen3 clips complete now, labelled at once (user request 2026-11-10 19:50 UTC) at nice 19, so the round-v12
#     e2e runs (wall-clock speed is part of their score) keep the CPUs they need; 2. once sn85-gen3 stops, v14b = the rest;
#  3. wait for sn85-round-v12 to end cleanly (its e2e uses models/crf_predictor_eval.pkl, which retraining replaces);
#  4. merge, rebuild data/encoder_rank_all = round-v12 sources + v14 + v14b;
#  5. keep the round-v12 models as *_v10.*, retrain; 6. e2e: test26 + HFR test + 1080p AI test-split clips
#     (round-v12 model vs new model) -> results/e2e/v11_*.
set -u
export TMPDIR=/dev/shm/sn85tmp; mkdir -p "$TMPDIR"
cd /workspace/sn85-compression-005
. /opt/nvm/nvm.sh >/dev/null 2>&1
PY=/venv/main/bin/python
log() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
online() { pm2 jlist 2>/dev/null | python3 -c "import json,sys; n=sys.argv[1:]; sys.exit(0 if any(p['name'] in n and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)" "$@"; }
shard_done() { local f=data/encoder_rank_$1/shard$2; [ -s $f.clips ] && [ -f $f/labels.jsonl ] && [ "$(wc -l < $f/labels.jsonl)" -ge "$(wc -l < $f.clips)" ]; }
label_set() {
  local v=$1 i
  for i in 0 1 2; do
    online sn85-label-$v-$i && continue
    shard_done $v $i && continue
    pm2 delete sn85-label-$v-$i >/dev/null 2>&1
    pm2 start scripts/run_label_$v.sh --name sn85-label-$v-$i --interpreter bash --no-autorestart -- $i 3 >/dev/null
  done
}
gen3_clips() { ls /workspace/dataset/ai_gen3/*.mkv 2>/dev/null | grep -v '\.tmp\.' | xargs -r ls -S; }

[ -s data/label_v14_clips.txt ] || gen3_clips > data/label_v14_clips.txt
log "v14 clips (gen3 complete now): $(wc -l < data/label_v14_clips.txt)"
[ -s data/label_v14_clips.txt ] && label_set v14
log "waiting for sn85-gen3"
while online sn85-gen3; do sleep 120; done
log "gen3: $(tail -n 2 ~/.pm2/logs/sn85-gen3-out.log | tr '\n' ' ')"
[ -s data/label_v14b_clips.txt ] || gen3_clips | grep -vxF -f data/label_v14_clips.txt > data/label_v14b_clips.txt
log "v14b clips (gen3 finished later): $(wc -l < data/label_v14b_clips.txt)"
[ -s data/label_v14b_clips.txt ] && label_set v14b
sleep 60
while online sn85-label-v14-0 sn85-label-v14-1 sn85-label-v14-2 sn85-label-v14b-0 sn85-label-v14b-1 sn85-label-v14b-2; do sleep 120; done
log "v14 + v14b labelling finished"
log "waiting for sn85-round-v12"
while online sn85-round-v12; do sleep 120; done
grep -q "round v12 done" ~/.pm2/logs/sn85-round-v12-out.log || { log "round v12 did not finish cleanly, stopping"; exit 1; }

for v in v14 v14b; do
  [ -d data/encoder_rank_$v/shard0 ] || continue
  $PY scripts/merge_label_shards.py $v
  log "$v: clips $(cat data/encoder_rank_$v/shard*.clips | wc -l), labels $(wc -l < data/encoder_rank_$v/merged/labels.jsonl)"
done
dirs="data/encoder_rank_all_v9 data/encoder_rank_v10/merged"
for v in v11 v12 v13 v13b v14 v14b; do [ -f data/encoder_rank_$v/merged/labels.jsonl ] && dirs="$dirs data/encoder_rank_$v/merged"; done
$PY scripts/combine_label_dirs.py --out data/encoder_rank_all $dirs || { log "combine FAILED"; exit 1; }
log "encoder_rank_all: $(wc -l < data/encoder_rank_all/labels.jsonl) labels from $dirs"

for f in crf_predictor.pkl crf_predictor.json crf_predictor_eval.pkl crf_predictor_eval.json; do
  [ -f models/${f%.*}_v10.${f##*.} ] || cp models/$f models/${f%.*}_v10.${f##*.}
done
log "training"
bash scripts/run_train_all.sh || { log "training FAILED"; exit 1; }
log "trained: $(tail -n 2 results/train_eval_all.log | tr '\n' ' ')"

# 1080p AI test-split clips (families never seen by either eval model)
$PY - <<'PY'
import json, subprocess
rows = [json.loads(l) for l in open('data/encoder_rank_all/labels.jsonl')]
def height(p):
    o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=height', '-of', 'csv=p=0', p],
                       capture_output=True, text=True).stdout.strip()
    return int(o or 0)
ai = sorted(r['path'] for r in rows if r.get('split') == 'test' and r.get('kind') == 'ai')
sel = [p for p in ai if height(p) == 1080]
open('data/e2e_ai1080_test.txt', 'w').write('# 1080p AI test-split clips (all AI sets incl. gen3)\n' + '\n'.join(sel) + '\n')
print(len(sel))
PY
log "1080p AI test clips: $(grep -vc '^#' data/e2e_ai1080_test.txt)"

E="$PY scripts/e2e_eval.py --cpus 0-31 --workers 4"
$E --clips data/e2e_test26.txt --model models/crf_predictor_eval.pkl --out results/e2e/v11_test26 > results/e2e_v11_test26.log 2>&1
log "e2e test26 new: $(cat results/e2e/v11_test26/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
for m in v10 new; do
  mp=models/crf_predictor_eval.pkl; [ $m = v10 ] && mp=models/crf_predictor_eval_v10.pkl
  $E --clips data/e2e_ai1080_test.txt --model $mp --out results/e2e/v11_ai1080_$m > results/e2e_v11_ai1080_$m.log 2>&1
  log "e2e 1080p AI ($m model): $(cat results/e2e/v11_ai1080_$m/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
done
$E --clips data/e2e_hfr_test.txt --model models/crf_predictor_eval.pkl --out results/e2e/v11_hfr > results/e2e_v11_hfr.log 2>&1
log "e2e HFR new: $(cat results/e2e/v11_hfr/summary.json 2>/dev/null | tr -d '\n' | cut -c1-400)"
log "round v14 done"
