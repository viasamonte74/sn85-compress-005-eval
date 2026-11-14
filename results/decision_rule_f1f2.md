# Pre-registered decision rule for the base-risk fixes (written 2026-11-13 ~21:15 UTC, before val16d / val16e exist)

Fixes (found on val16b, which is therefore "seen" for them):
  F1  STREAMENC_NEG_CHECK_UNKNOWN=1  measure VMAF - VMAF-NEG whenever the gap model has no prediction for the shipped
      arm or a prefilter is on it (q08-route clip shipped gap 3.25 unmeasured -> 0).
  F2  STREAMENC_CPU_CALIB=1          scale every arm's predicted CPU up by measured / predicted pilot CPU (never down)
      before the stage-2 arm choice (1080p50 swarm: p3_t0 641 CPU-s, 61 s batch, set over the speed limit).

Unseen data: val16d, val16e (16 new clips each, generated after the fixes were written; same slots and format as
val16..val16c). Used once, for this decision only.

Variants (rc12 code, clean machine, validator-style 4-clip batches, validator-exact scoring):
  C  rc12 defaults            G  C + F1            H  C + F2            F  C + F1 + F2
Decision:
  F1: adopt if, on both sets, no clip loses > 0.5 VMAF because of F1 and the set time rises by <= 2 s per 3428
      frames (a safety net: adopt unless clearly harmful). Report every gate failure with and without F1.
  F2: adopt if final(H) >= final(C) on both sets and no clip loses > 1.0 VMAF because of F2.
  If both are adopted, F must not be worse than the better of G / H on either set by more than 0.002.

Amendment (2026-11-13 ~21:35 UTC, val16d / val16e still being generated, no result exists):
  F2 is WITHDRAWN before evaluation: development runs showed the measured/predicted pilot CPU ratio is 1.5-1.9 for
  most clips (systematic model offset, already absorbed by lam), so F2 re-weights the speed trade for all clips
  (practice q03 -0.71 VMAF) and still did not move the swarm clip off preset 3 (real final/pilot CPU 11x vs model ~3x).
  F3 replaces it: STREAMENC_WALL_GUARD=1. Per batch: target = 114.3 s / 3428 frames x the batch's frames; before the
  final encode, predicted final wall = this clip's measured pilot seconds x the arm's final/pilot wall ratio (medians
  from clean runs x 1.15); if elapsed + predicted > target - 3 s, the highest-VMAF arm that fits is used instead.
Variants on val16d / val16e: C rc12, G C+F1, I C+F3, J C+F1+F3.
  F3: adopt if final(I) >= final(C) on both sets and no clip loses > 1.5 VMAF because of F3.
  F1 rule unchanged. If both are adopted, J must not be worse than the better of G / I by more than 0.002 on either set.

Amendment 2 (2026-11-13 ~23:30 UTC, still no val16d / val16e result exists): the first working F3 run on val16b (seen)
was too strict (exact per-batch share; fell back to the p8 speed arm: swarm -10 VMAF). The validator scores TOTAL time,
batches vary 0.7-1.25x their share, the outlier was 1.75x. F3 parameters fixed now: STREAMENC_WALL_GUARD_SLACK=1.5 (target
= 1.5 x the batch's share) and fallback arms limited to presets <= 6 (never p8). Rules for F3 unchanged otherwise.
