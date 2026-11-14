# Result of the pre-registered rc14 decision (results/decision_rule_rc14_segspec.md), 2026-11-14 04:25 UTC

Unseen val16f / val16g (generated 01:51-04:08 UTC, used once), repo code, balanced order R_f S_f S_g R_g, quiet machine.

| Set | R (SEGSPEC off = rc13) | S (SEGSPEC on) | outputs | rule |
| --- | --- | --- | --- | --- |
| val16f | 0.8891, 124.0 s, 16/16 | 0.8889, 124.5 s, 16/16 | identical | pass (+0.5 s <= 1.0) |
| val16g | 0.8290, 172.2 s, 16/16 | 0.8328, 161.1 s, 16/16 | 1 differs: v16gai_hard_001 -0.23 VMAF | pass |

PASSES the unseen rule. Caveats (honest reading):
- val16f: speculation allowed in 2 of 16 segmented finals (core budget), 1 hit -> no gain.
- val16g -11.1 s = ~2.3 s from a speculative hit + ~9 s because fill-down was skipped on that batch's critical clip:
  the speculative probe encode ran concurrently with the bump wave, its CPU-s (19.1 vs 14.9) entered the fill-down cost
  (0.229 vs gain 0.217) -> skip -> -0.23 VMAF, -11.6 s on the critical path. Outputs are therefore not always
  byte-identical (CPU-measured decisions can flip), as the rule allowed (<= 0.5 VMAF).
- Dev (seen): val16d -5.7 s (hit on the batch-2 critical clip), val16e -1.0 s, practice +1.3 s / batch (render load).
Separate finding (val16g): the 80.7 s batch is one 1080p50 render shipped at preset 2 (1014 CPU-s): stage 1 (features)
predicted an easy clip (p6, 37 CPU-s); after the pilots (3.5x over budget) stage 2 chose p2 with stage-1-sized CPU costs.
