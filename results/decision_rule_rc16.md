# Pre-registered rule for rc16 = rc13.post3 + raw-cache prefetch + P2 exact pick + P3 /tmp reservation
# (written 2026-11-14 ~06:35 UTC, before any rc16 code or result exists; user: "make one better or equal than
# rc13.post3, review carefully and upgrade a little"; only ONE entry will be enrolled)

Changes (all intended to leave every output byte-identical):
  prefetch  probe + packet timestamps + raw cache of the source start in a background thread at the start of each
            clip, overlapping the feature extraction (sequential today: 1.5-2.7 s + 0.4-2.8 s on the critical path).
  P2        when no VMAF-NEG fallback reaches 2.85 on the 1-in-5 estimate, pick by every-frame measurement (rc15 code).
  P3        raw caches being written are reserved under a lock (rc15 code).
  (rc15's P1 final check is NOT included: +4.5..+7.1 s per set.)
Test: rc13.post3 (R) and rc16 (N) from their own folders, quiet machine, interleaved R N N R on practice, val16f,
  val16g, val16d; then rc16 on the remaining regression sets (val16, val16b, val16c, val16e, test16, hard, lowsat,
  held-out) against the rc13 outputs of today.
Adopt rc16 only if ALL hold:
  1. every output byte-identical to rc13 on all 12 sets (a CPU-measured fill decision may flip: allowed only if no clip
     loses > 0.3 VMAF), no new gate failure or missing output;
  2. interleaved timing: mean of N <= mean of R over the 4 timed sets, and no timed set slower by > 1.0 s per 3428;
  3. build emulation of the final commit (real base image, every Dockerfile step, run.sh as uid 65534) OK, practice
     outputs identical.
Otherwise rc13.post3 stays the entry.
