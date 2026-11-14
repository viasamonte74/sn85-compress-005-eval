# Safety comparison rc11 vs rc12 (2026-11-14 01:35 UTC)

Each candidate's own code and defaults, 4-clip validator-style batches, validator-exact scoring, run alone on a clean machine.

| Set | rc11 final | rc12 final | rc11 s/3428 | rc12 s/3428 | slowest batch rc11/rc12 | gate fails / missing (rc11, rc12) | clips changed | worst rc12 loss | best rc12 gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| practice | 0.7673 | 0.7673 | 110.0 | 110.1 | 29.9 / 30.2 | none | 0 | none | none |
| val16 | 0.8922 | 0.8946 | 102.3 | 103.0 | 30.7 / 31.6 | none | 1 | none | +1.50 v16ai_hard_001 |
| val16b | 0.8479 | 0.8483 | 133.6 | 133.6 | 61.8 / 61.6 | v16bai_medium_001:VMAF_MODEL_DELTA_EXCEEDED | 1 | none | +0.22 v16bai_hard_002 |
| val16c | 0.8719 | 0.8721 | 111.4 | 111.8 | 37.5 / 37.3 | none | 1 | none | +0.09 v16cai_hard_002 |
| val16d | 0.8812 | 0.8811 | 149.3 | 149.6 | 54.0 / 54.3 | none | 0 | none | none |
| val16e | 0.8356 | 0.8339 | 123.1 | 122.9 | 47.9 / 47.8 | none | 1 | -1.11 v16eai_medium_001 | none |
| test16 | 0.8481 | 0.8481 | 94.3 | 94.6 | 32.1 / 32.0 | none | 0 | none | none |
| hard | 0.6943 | 0.6940 | 109.2 | 110.2 | 34.3 / 34.5 | none | 1 | -0.17 t16bai_hard_005 | none |
| lowsat | 0.4632 | 0.4641 | 194.9 | 193.7 | 43.9 / 43.6 | none | 1 | none | +0.23 t16bai_hard_006_lowsat |
| heldout | 0.6380 | 0.6394 | 156.0 | 155.2 | 71.6 / 71.3 | rnd_hard_03:MISSING | 9 | -0.57 aix_hard_033 | +1.23 anim_hard_002_1080 |

Repeatability:
  rc11 val16 vs rc11c_val16: final 0.8922 -> 0.8922, time 102.4 -> 102.3, 0 of 16 outputs differ
  rc12 val16 vs rc12c_val16: final 0.8946 -> 0.8946, time 102.7 -> 103.0, 0 of 16 outputs differ
  rc11 val16b vs un_A_val16b: final 0.8480 -> 0.8479, time 133.4 -> 133.6, 0 of 16 outputs differ
  rc12 val16b vs un_C_val16b: final 0.8487 -> 0.8483, time 132.7 -> 133.6, 0 of 16 outputs differ
  rc11 val16c vs un_A_val16c: final 0.8719 -> 0.8719, time 110.9 -> 111.4, 0 of 16 outputs differ
  rc12 val16c vs un_C_val16c: final 0.8721 -> 0.8721, time 111.0 -> 111.8, 0 of 16 outputs differ
  rc12 val16d vs fx_C_val16d: final 0.8810 -> 0.8811, time 149.8 -> 149.6, 0 of 16 outputs differ
  rc12 val16e vs fx_C_val16e: final 0.8339 -> 0.8339, time 123.0 -> 122.9, 0 of 16 outputs differ
  rc11 test16 vs rc11c_test16: final 0.8481 -> 0.8481, time 93.7 -> 94.3, 0 of 16 outputs differ
  rc12 test16 vs rc12c_test16: final 0.8481 -> 0.8481, time 93.9 -> 94.6, 0 of 16 outputs differ
  rc11 hard vs rc11c_test16hard: final 0.6943 -> 0.6943, time 108.0 -> 109.2, 0 of 16 outputs differ
  rc12 hard vs rc12c_test16hard: final 0.6940 -> 0.6940, time 108.9 -> 110.2, 0 of 16 outputs differ
  rc11 heldout vs rc11c_heldout: final 0.6380 -> 0.6380, time 155.8 -> 156.0, 0 of 56 outputs differ
  rc12 heldout vs rc12c_heldout: final 0.6397 -> 0.6394, time 154.2 -> 155.2, 0 of 56 outputs differ

Notes
- Gate failures are identical in both versions (same output files): val16b v16bai_medium_001 VMAF - VMAF-NEG 3.25 (q08 route,
  gap not measured; F1 would measure it), held-out rnd_hard_03 missing (over budget at CRF 63, scores 0 for that clip only).
- held-out has 65 entries over 56 distinct clips (lists overlap); the per-clip checks count distinct clips.
- rc12's changes were tuned on practice, val16 and held-out. On the sets not used for tuning (val16b-e, test16, hard,
  lowsat) the per-set finals sum to -0.0006 for rc12 vs rc11; worst clip -1.11 VMAF (val16e v16eai_medium_001: stage-2
  prediction 89.95 < 90 -> chroma blur + 85 % low-pass; 93.48 unfiltered -> 92.36).
- Time: both versions are over 114.3 s per 3428 frames on val16b, val16d, val16e, held-out and lowsat; the slowest batch
  each time holds one heavy 1080p clip at preset 2-3. rc12 does not change time (-1.2 to +1.0 s per set).
