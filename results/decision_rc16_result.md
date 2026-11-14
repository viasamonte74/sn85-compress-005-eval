# Result of the pre-registered rc16 rule (results/decision_rule_rc16.md), 2026-11-14 08:25 UTC

rc16 = rc13.post3 + raw-cache prefetch + P2 exact pick + P3 /tmp reservation. PASSES all three conditions.
1. Outputs byte-identical to rc13 on all 12 sets (practice, val16, val16b-g, test16, hard, lowsat, held-out); no new gate
   failure (held-out rnd_hard_03 missing in both: over budget at CRF 63). P2 fired once (val16f), picked the same file.
2. Paired timing (quiet machine, R N / N R / R N / N R): practice 114.0 -> 111.4, val16f 126.2 -> 125.1,
   val16g 178.0 -> 174.5, val16d 156.8 -> 155.7 s per 3428; mean 143.8 -> 141.7 (-2.1 s), every set faster.
   Other sets vs this morning's rc13 runs: -0.1 .. -5.4 s. prefetch_wait 0.0 s (the raw cache is fully hidden).
3. Build emulation of local commit 670f527fbe4af9e1a2bb64a6a48414cc001d8525 (tree 5b5aa337c31ba78998c21cf9798cb41680bbaece):
   real base image, every Dockerfile step: BUILD OK (offline; ffmpeg sha OK; 8 wheels; pip check; model loads);
   run.sh as uid 65534: practice exit 0 / 0, 8/8 outputs identical to the tested ones, /tmp clean.
Gain: same quality, ~1-4 s faster per set -> final +0.000..+0.002 on sets over the 114.3 s limit, nothing elsewhere.
Not pushed, not enrolled (rc13.post3 is enrolled as contender 145; switching needs the organiser's swap procedure).
