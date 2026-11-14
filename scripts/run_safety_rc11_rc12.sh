#!/bin/bash
# 2026-11-13 safety comparison rc11 vs rc12 (user: "after done, test rc12 again compared with rc11 focusing on the
# safety"): each candidate's own code + defaults, ten sets, validator-style 4-clip batches, validator-exact scoring.
R=/workspace/sn85-compression-005
cd $R
L=$R/results/safety_rc11_rc12.log
log() { echo "$(date -u +%F' '%T) $*" >> $L; }
until grep -qE "^[0-9-]+ [0-9:]+ done$" $R/results/f13_eval.log 2>/dev/null; do sleep 30; done
source /venv/main/bin/activate
export PATH=$R/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH
for v in $(env | grep -o '^STREAMENC_[A-Z_]*'); do unset "$v"; done; unset OMP_NUM_THREADS
summ() { python3 -c "import json; print(json.load(open('$R/results/e2e_batched/$1/summary.json'))['all'])"; }
M=$R/models/crf_predictor.pkl; ME=$R/models/crf_predictor_eval_c_abs.pkl
log "start"
for V in rc11 rc12; do
  C=/workspace/sn85-candidates/sn85-005-1.0.0$V
  ln -sfn $R/third_party $C/third_party
  cd $C
  for S in practice:e2e_practice val16:e2e_val16 val16b:e2e_val16b val16c:e2e_val16c val16d:e2e_val16d val16e:e2e_val16e \
           test16:e2e_test16_hidden_like hard:e2e_test16_hard lowsat:e2e_stress_lowsat; do
    T=${S%%:*}; F=${S#*:}
    python scripts/e2e_batched.py --clips $R/data/$F.txt --model $C/models/crf_predictor.pkl --score-procs 4 \
        --work $R/work/e2e_sf_${V}_$T --out $R/results/e2e_batched/sf_${V}_$T > $R/results/e2e_batched/sf_${V}_$T.log 2>&1
    log "sf_${V}_$T: $(summ sf_${V}_$T)"; rm -rf $R/work/e2e_sf_${V}_$T
  done
  python scripts/e2e_batched.py --clips $R/data/e2e_test26_nonat.txt $R/data/e2e_ai1080_nonat.txt $R/data/e2e_hfr_nonat.txt --model $ME \
      --score-procs 4 --work $R/work/e2e_sf_${V}_heldout --out $R/results/e2e_batched/sf_${V}_heldout > $R/results/e2e_batched/sf_${V}_heldout.log 2>&1
  log "sf_${V}_heldout: $(summ sf_${V}_heldout)"; rm -rf $R/work/e2e_sf_${V}_heldout
  rm -f $C/third_party; find $C -name __pycache__ -type d -prune -exec rm -rf {} +
  cd $R
done
log "done"
