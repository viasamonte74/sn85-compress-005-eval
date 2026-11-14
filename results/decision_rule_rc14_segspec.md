# Pre-registered decision rule for rc14 = rc13 + speculative probe trims (written 2026-11-14 ~02:25 UTC, before any
# SEGSPEC code or result exists; user: "start rc14 after rc13 verification done")

Change (STREAMENC_SEGSPEC): in the segmented final, while the other segments of a clip are encoding, the probe
segment is also encoded at the most likely trim CRFs (predicted from the probe's measured size), only when other
clips of the batch leave cores free. When the trim CRF the unchanged algorithm asks for has a finished speculative
encode, that file is used (same command, same output path -> byte-identical); otherwise the encode runs as today.
Unused speculative encodes are cancelled; their CPU never enters the CPU figures the fill decisions use.

Dev (seen data, may be used to fix bugs / the core budget rule): val16b, val16d, val16e (the slow sets), practice.
  Go on to unseen only if dev outputs are identical to rc13 and the slow sets get faster.
Unseen data: val16f, val16g (16 new clips each, generation started 2026-11-14 01:52 UTC, never used for anything).
Variants: R = rc13 defaults, S = rc13 + SEGSPEC, both from the same code, validator-style 4-clip batches, exact scoring.
Adopt (rc14) only if, on BOTH unseen sets:
  1. no new gate failure or missing output, and no clip loses > 0.5 VMAF vs R (expected: outputs byte-identical);
  2. set time S <= set time R + 1.0 s per 3428 frames;
  3. final(S) >= final(R) - 0.0005.
Then the rc14 candidate runs the ten regression sets (practice, val16, val16b-e, test16, hard, lowsat, held-out) vs
rc13: no gate change, no clip loses > 0.5 VMAF, no set slower by > 2 s per 3428 frames. Otherwise rc13 stays.
