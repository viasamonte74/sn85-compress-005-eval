#!/bin/bash
# 2026-11-12 (user: "start from 1 to 3"): (1) harder hidden-like test set + rc8 baseline, (2) free-time refill,
# (3) full-frame VMAF-NEG check at >= 50 fps. Validator-style 4-clip batches, alone on the CPU (waits for Wan).
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
log() { echo "$(date -u +%H:%M) $*"; }
G=/tmp/claude-0/-workspace/d67412ae-2e2e-445a-8c46-716946569785/scratchpad/test16/gen_b.log
until grep -q "test16b generation done" $G 2>/dev/null; do sleep 20; done
summ() { python3 -c "import json; s=json.load(open('results/e2e_batched/$1/summary.json'))['all']; print(s)"; }
run() { python scripts/e2e_batched.py --clips $2 --model $3 --out results/e2e_batched/$1 > results/e2e_batched/$1.log 2>&1; log "$1: $(summ $1)"; }
ls /workspace/dataset/test16/t16bai*.mkv | grep -v "\.tmp" > data/e2e_t16b_new_ai.txt
run rc8_t16b_difficulty data/e2e_t16b_new_ai.txt models/crf_predictor.pkl
python3 - <<'PY'
import json, random
rows = list(map(json.loads, open("results/e2e_batched/rc8_t16b_difficulty/per_clip.jsonl")))
v = {r["ref"]: (r.get("vmaf") or 0) for r in rows}
hard = sorted([p for p in v if v[p] < 80], key=lambda p: v[p])[:4]
medium = sorted([p for p in v if p not in hard], key=lambda p: v[p])[:1]
pp = lambda q: f"/workspace/dataset-src/practice_pack_005/{q}/clip.mkv"
t = lambda n: f"/workspace/dataset/test16/{n}.mkv"
clips = [pp("q07"), pp("q08")] + hard + medium + [pp("q01"), pp("q02"), pp("q03"), pp("q04"), t("t16rnd_medium_01"), t("t16rnd_hard_01"),
         pp("q05"), pp("q06"), t("t16scr_hard_01")]
clips += [t("t16ai_hard_001")] if len(clips) < 16 else []
clips = clips[:16]
random.Random(1616).shuffle(clips)
open("data/e2e_test16_hard.txt", "w").write("# harder hidden-like 16 clips (q07/q08 + new Wan clips measured < 80 VMAF), 2026-11-12\n" + "\n".join(clips) + "\n")
print("hard set:", [(c.split("/")[-2] if "practice" in c else c.split("/")[-1][:-4], round(v.get(c, -1), 1)) for c in clips])
PY
run rc8_test16hard data/e2e_test16_hard.txt models/crf_predictor.pkl
(export STREAMENC_FREE_FILL=1
 run ff_practice data/e2e_practice.txt models/crf_predictor.pkl
 run ff_test16 data/e2e_test16_hidden_like.txt models/crf_predictor.pkl
 run ff_test16hard data/e2e_test16_hard.txt models/crf_predictor.pkl
 python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/ff_heldout > results/e2e_batched/ff_heldout.log 2>&1; log "ff_heldout: $(summ ff_heldout)")
(export STREAMENC_NEG_FULL_FPS=50
 run nf_practice data/e2e_practice.txt models/crf_predictor.pkl
 run nf_test16 data/e2e_test16_hidden_like.txt models/crf_predictor.pkl
 run nf_test16hard data/e2e_test16_hard.txt models/crf_predictor.pkl
 python scripts/e2e_batched.py --clips data/e2e_test26_nonat.txt data/e2e_ai1080_nonat.txt data/e2e_hfr_nonat.txt --model models/crf_predictor_eval_c_abs.pkl --out results/e2e_batched/nf_heldout > results/e2e_batched/nf_heldout.log 2>&1; log "nf_heldout: $(summ nf_heldout)")
log "items123 done"
