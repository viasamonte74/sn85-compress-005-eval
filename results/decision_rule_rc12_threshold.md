# Pre-registered decision rule (written 2026-11-13 17:41 UTC, before any val16b / val16c result exists)

Question: is rc12's constant stage-2 prefilter threshold (90) safe on unseen clips, and is the batch-aware refit?

Unseen data: val16b and val16c (16 new clips each, generated after rc12 was frozen; 8 renders / 4 screen / 4 AI,
720p/1080p, 24-60 fps, 3428 frames per set, practice-pack format). Used ONCE, for this decision only; never for tuning.

Variants (same code = rc12; switches via environment):
  A  rc11 behaviour        STREAMENC_HARD_VMAF_STAGE2=80  STREAMENC_SEGFINAL_REFIT=0
  B  rc11 + refit          STREAMENC_HARD_VMAF_STAGE2=80  STREAMENC_SEGFINAL_REFIT=1
  C  rc12                  defaults (stage-2 threshold 90, refit on)
  D  sensitivity           STREAMENC_HARD_VMAF_STAGE2=85  STREAMENC_SEGFINAL_REFIT=1

Decision (per validator-style 4-clip batches, validator-exact scoring):
  1. Refit: keep (B over A) only if final(B) >= final(A) on both sets and no clip loses > 0.5 VMAF from the refit.
  2. Threshold 90: keep (C over B) only if final(C) >= final(B) on both sets AND no clip loses > 1.0 VMAF from the
     stage-2 prefilter AND the per-clip outcomes in the 80-90 band are not a coin flip (more gains than losses).
  3. If 2 fails but 1 passes -> ship rc11 + refit (no stage-2 threshold). If both fail -> stay on rc11.
  4. D only informs whether the effect is monotone (robust) or zigzags (fragile); it is not a candidate by itself.
Also reported: largest single-clip loss of each variant vs A, time per 3428 frames, gate failures.

Addendum (written 2026-11-13 ~17:50 UTC, still before any val16b / val16c result exists):
  E  per-clip probe        STREAMENC_FILTER_PROBE=1 (single-segment, with / without the prefilter, exact VMAF)
                           STREAMENC_FILTER_PROBE_MAX_PRED=90 STREAMENC_FILTER_PROBE_EARLY_MAX=92 + refit (+ stage-2 90 fallback)
  E replaces the constant in the 80-90 band by a measurement. E is preferred over C only if, on both unseen sets,
  final(E) >= final(C) and final(E) >= final(B), and its time per 3428 frames stays <= 114.3 s on val16b / val16c.
  E is also run on the five existing sets (cost / benefit), but those were seen during development.
