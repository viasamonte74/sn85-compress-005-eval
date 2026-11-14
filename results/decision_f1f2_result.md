# Result of the pre-registered F1 / F3 decision (results/decision_rule_f1f2.md), 2026-11-14 00:05 UTC

Unseen sets val16d / val16e, used once. rc12 code, validator-style 4-clip batches, validator-exact scoring.

| Variant | val16d final | val16d s/3428 | val16e final | val16e s/3428 | passed |
| --- | --- | --- | --- | --- | --- |
| C rc12            | 0.8810 | 149.8 | 0.8339 | 123.0 | 16/16, 16/16 |
| G C + F1          | 0.8811 | 149.5 | 0.8342 | 122.5 | 16/16, 16/16 |
| I C + F3          | 0.8842 | 111.4 | 0.8338 | 123.2 | 16/16, 16/16 |
| J C + F1 + F3     | 0.8842 | 111.3 | 0.8342 | 122.4 | 16/16, 16/16 |

Outputs: G == C byte-identical on both sets (F1 never fired); J == I on val16d, J == C on val16e.

F1: rule met on both sets (no clip loses anything, time not higher). It never fired on unseen data; its only measured
effect is the val16b q08-route clip (gap 3.25 unmeasured -> measured, fallback, 0 -> 0.35). Adoptable as a safety net.

F3: REJECTED.
  val16d: final 0.8810 -> 0.8842, but v16drnd_hard_03_1080 -5.91 VMAF (p3 -> p6), v16dai_hard_001 -2.31 (p2 -> p4),
          v16dscr_hard_01_1080 -1.16 -> violates "no clip loses > 1.5 VMAF".
  val16e: never fired; outputs identical; final 0.8338 < C 0.8339 (timing noise). The 47.9 s batch (target 37.1 s)
          came from v16eai_hard_001 at p3_lqb: pilots 2.15 / 1.56 s -> predicted final 16.8 s, real segmented final
          33.9 s (2 tries, CPU contention). The pilot-seconds x ratio time model is wrong by 2x in both directions.
