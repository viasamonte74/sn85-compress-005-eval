#!/bin/bash
# 2026-11-14 rc15 (= rc13.post1 + safety nets) verification: candidate code + defaults, 12 sets, validator-style 4-clip
# batches, exact scoring (2 scorers: disk). Compare with sf_rc13_* / u14_R_val16f/g: identical outputs expected unless a
# safety net deliberately changes one; no gate failure; <= +4 s per 3428 frames.
R=/workspace/sn85-compression-005
cd $R
L=$R/results/rc15_verify.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
until [ "$(grep -c ' done$' $R/results/post1_practice.log 2>/dev/null)" -ge 2 ]; do sleep 10; done
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
ME=$R/models/crf_predictor_eval_c_abs.pkl
V=rc15; C=/workspace/sn85-candidates/sn85-005-1.0.0$V
log "start"
ln -sfn $R/third_party $C/third_party
cd $C
for S in practice:e2e_practice val16f:e2e_val16f val16g:e2e_val16g val16b:e2e_val16b val16:e2e_val16 val16c:e2e_val16c \
         val16d:e2e_val16d val16e:e2e_val16e test16:e2e_test16_hidden_like hard:e2e_test16_hard lowsat:e2e_stress_lowsat; do
  T=${S%%:*}; F=${S#*:}
  python scripts/e2e_batched.py --clips $R/data/$F.txt --model $C/models/crf_predictor.pkl --score-procs 2 \
      --work $R/work/e2e_sf_${V}_$T --out $R/results/e2e_batched/sf_${V}_$T > $R/results/e2e_batched/sf_${V}_$T.log 2>&1
  log "sf_${V}_$T: $(summ sf_${V}_$T)"; rm -rf $R/work/e2e_sf_${V}_$T
done
python scripts/e2e_batched.py --clips $R/data/e2e_test26_nonat.txt $R/data/e2e_ai1080_nonat.txt $R/data/e2e_hfr_nonat.txt --model $ME \
    --score-procs 2 --work $R/work/e2e_sf_${V}_heldout --out $R/results/e2e_batched/sf_${V}_heldout > $R/results/e2e_batched/sf_${V}_heldout.log 2>&1
log "sf_${V}_heldout: $(summ sf_${V}_heldout)"; rm -rf $R/work/e2e_sf_${V}_heldout
rm -f $C/third_party; find $C -name __pycache__ -type d -prune -exec rm -rf {} +
log "done"
