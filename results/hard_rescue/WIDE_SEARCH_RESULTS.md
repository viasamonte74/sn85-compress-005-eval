# Wide encoder-config search on hard clips — results (recorded 2026-11-11)

Method: `scripts/hard_config_search.py`. Stage A = base `svt_crf_p4_qm_la` + one change (62 configs); stage B =
forward selection from the best single. Every trial is a full label trial (rate fit at 0.05 bpp + validation).
Raw data: `search_trials.jsonl` (490 trials), `search_best.jsonl`; sweeps: `trials.jsonl`.

## Wide search 1 (62 configs, CPU SVT-AV1 + libaom / p8 references)

| Clip | Base p4 | Label best | Best config | VMAF | Clip score |
|---|---|---|---|---|---|
| aig3_hard_002_1080 (30 fps) | 71.60 | 72.63 | p0 | 73.73 | 0.343 |
| ai_hard_20_1080 | 57.74 | 59.93 | p0 + tune2 + tf-strength 2 + ds67 | 63.32 | 0.083 |
| aix_hard_031 | 82.28 | 82.93 | p0 + 10-bit | 83.79 | 0.595 |
| ai_hard_04 | 61.92 | 62.70 | p0 + hierarchical-levels 3 | 64.31 | 0.108 |
| rnd_hard_03 (1080p60) | 1.93x budget | none valid | tmix3 + sr16 fits (0.98x) but fails TONE_MANIPULATION | 45.2 | 0 (invalid) |
| aig3_hard_005_1080 (ink) | 86.95 | 88.94 | tune0 + p0 + qm-min 4 | 89.44 | 0.736 |
| aig3_hard_005 (ink, 720p) | 86.68 | 88.87 | tune0 + 10-bit + p0 + ac-bias 1 + chroma-qm-min 0 | 89.94 | 0.749 (stage B stopped before the last step) |

Single-change gains over base (VMAF):
- aig3_hard_002_1080: p0 +2.12, p1 +1.84, p2 +1.03, hl3 +0.57, p3 +0.28, hl4 +0.28, acb3 +0.15, ds67 +0.09
- ai_hard_20_1080: ds67 +3.14, p0 +2.98, p1 +2.37, p2 +2.19, ds85 +1.92, p3 +0.84, tune2 +0.57, tf2 +0.40, vb3 +0.34
- aix_hard_031: p0 +0.82, p1 +0.60, p2 +0.41, b10 +0.27, mtx32 +0.25
- ai_hard_04: p0 +2.25, p1 +1.61, p2 +0.78, hl3 +0.45, tf2 +0.24, p3 +0.23, lqb10 +0.20
- aig3_hard_005_1080: tune0 +1.94, b10 +0.91, p0 +0.73, p1 +0.56, rest0 +0.31
- aig3_hard_005: tune0 +1.84, b10 +1.29, p0 +1.23, p1 +1.11, p2 +0.60

Findings: slower presets are the only consistent gain; ds67 matters on one very hard 1080p AI clip; tune 0 on the
low-CRF ink clip; 10-bit +0.3..+1.3 (validator acceptance of 10-bit output NOT verified); all other tools < +-0.6.

## Clips with no valid encode (13) — super-resolution + frame hold sweeps

| Clip | Best valid config | VMAF | Clip score |
|---|---|---|---|
| rnd_hard_04_1080 | svt_crf_p2_qm_sr16+hold3 (p4: 72.38) | 72.74 | 0.319 |
| rnd_hard_11_1080 | svt_crf_p2_qm_sr16+hold3 (p4: 69.83) | 70.53 | 0.263 |
| rnd_hard_03 | svt_crf_p0_qm_sr16+hold3 (p1 60.12, p2 59.57, p0 sr16+hold2 60.53) | 60.62 | 0.015 |
| rnd_hard_04, rnd_hard_11 (720p) | none (over budget) | - | 0 |
| gen_hard 04/20/21/22/25 | fit with sr16 (+hold2/3) | 28-43 | 0 |
| gen_hard 02/18/19 | none (over budget) | - | 0 |

rnd_hard_03 bitrate/budget at CRF 63: base 1.93, sr16 1.54, qpc 1.70, ds50 1.82, hold2 1.63, hold3 1.47,
sr16+hold2 1.08, NVENC AV1 2.68 (ignores the target). Frame blends (tmix3, avgNhold) fit but fail TONE_MANIPULATION.
Frame hold is NOT yet checked against the official validator for a frozen/duplicate-frame rule.

## Wide search 2 (2026-11-11, `--wide2`, PM2 sn85-config-search3) — ai_hard_12_1080
Clip: 1080p24 photoreal AI, label best 56.37 (svt_crf_p3_qm_la), score 0. Stage A: 146 configs (62 old + 49 new SVT
tools / prefilters + 32 other encoders incl. 18 NVENC on the GPU), 7 failed.

| Config | VMAF | vs p4 base 55.00 | CPU-s / encode |
|---|---|---|---|
| p0 + scm1 + tf2 + ds85 (stage B) | 58.43 | +3.43 | 5467 |
| p0 + ds85 | 58.18 | +3.18 | 2717 |
| p0 | 58.10 | +3.10 | 3643 |
| aom_c3_psnr + ds67 (best non-SVT) | 57.40 | +2.41 | 1047 |
| p1 / p3 / p2 | 57.11 / 56.37 / 56.08 | +2.11 / +1.37 / +1.08 | 1117 / 308 / 520 |
| best NVENC: nv_av1_p7_uhq + ds67 | 48.56 | -6.44 | GPU |
| NVENC HEVC (best) | 42.79 | -12.2 | GPU |
| ds67 + unsharp | 58.90 | +3.91 | INVALID: VMAF-NEG gap 3.18 > 3.0 |

- No config reaches VMAF 60: clip score stays 0 with every config -> for such clips the label stays the speed arm.
- Stage B additions after p0 (+0.08 ds85, +0.12 scm1, +0.14 tf2) are below the CRF quarter-step landing noise; the
  search's keep margin (0.05 VMAF) is too small to call them gains.
- New SVT tools: only scm1 (+0.60) and ydc4 (+0.13) helped; tune4 -4.3, vb curve 1 -5.6, ydc-4 -3.1, sharp7 -3.0,
  fixed-qindex mode -1.65; startup-qp-offset, low-delay, octile-2 boost, max-qp 58 broke the budget.
- 11 configs were NO-OPS (byte-identical encode to the base): chroma-u/v dc/ac qindex offsets (all 5 values),
  key-frame-qindex-offset (both), scm0, enable-dg 0, lookahead 32/60, scd 1, recode-loop 4. Checked on a test clip:
  the qindex offsets only act with use-fixed-qindex-offsets=1, and the chroma offsets change nothing even then in
  this SVT-AV1 v4.3.0 build. So the chroma-to-luma bit shift idea was not actually tested.
- New prefilters: ds85 +0.64, nlm +0.41, ds67 lanczos +0.33, horizontal-only ds67 +0.28; blurs, ds50, area
  downscale, frame hold and blends all lose.

## Sharpening sweep (2026-11-11, PM2 sn85-sharpen-sweep) — p4 base, luma unsharp / cas, alone and after ds67 / ds85
Validator rule: |VMAF - VMAF-NEG| > 3.0 -> clip score 0 (runtime guard keeps 2.85). Sharpening raises VMAF more than NEG.

| Clip | p4 base | best gap <= 2.7 | best gap <= 3.0 | first failing |
|---|---|---|---|---|
| ai_hard_12_1080 | 55.00 | ds85+usm0.3 57.81 (2.65) | ds67+usm0.4 58.54 (2.88) | ds67+usm0.5 58.90 (3.18) |
| ai_hard_20_1080 | 57.74 (ds67: 60.88) | ds67+usm0.2 62.04 (2.63) | same | ds67+usm0.3 63.06 (3.05) |
| ai_hard_04_1080 | 56.44 | ds67+usm0.4 58.79 (2.61) | ds67+usm0.5 59.12 (2.83) | none failed |

- Sharpening + low-pass adds +2.3..+3.5 VMAF at p4 (CPU unchanged); usm 5x5 better than 3x3, cas weakest.
- The NEG gap grows ~0.25-0.4 per 0.1 of unsharp strength and differs per clip (0.3 fails on ai_hard_20_1080, 0.5 passes
  on ai_hard_04_1080) -> any runtime use must measure the gap per clip.
- Stage 2 (sn85-sharpen-sweep2): the gap <= 2.7 settings on p0 and p1 (+ p0 alone on ai_hard_04_1080).

## Sharpening sweep (2026-11-11, PM2 sn85-sharpen-sweep / sn85-sharpen-sweep2)
Luma unsharp (5x5, strength 0.1-0.5), alone or after a 67 % / 85 % bicubic low-pass; NEG gap limit 3.0.
Stage 1 at p4 (17 settings): best passing per clip — ai_hard_12_1080 ds67_usm40 58.54 (gap 2.88, p4 55.00);
ai_hard_20_1080 ds67_usm20 62.04 (gap 2.63, p4 57.74); ai_hard_04_1080 ds67_usm 59.12 (gap 2.83, p4 56.44).
Gap varies by clip: ds67_usm (0.5) = 3.18 / 3.95 / 2.83 on the three clips.
Stage 2 at p0 / p1 (settings with gap <= 2.7 at p4):

| Clip | p2 label | best p1 + sharpen | best p0 + sharpen | gap | CPU-s p1 / p0 |
|---|---|---|---|---|---|
| ai_hard_12_1080 | 56.08 | ds85_usm30 60.12 (0.003) | ds85_usm30 61.06 (0.026) | 2.80 / 2.84 | 1015 / 2252 |
| ai_hard_20_1080 | 59.93 | ds67_usm20 64.96 (0.124) | ds67_usm20 64.81 (0.120) | 2.67 / 2.68 | 776 / 2047 |
| ai_hard_04_1080 | 58.72 | ds67_usm40 61.62 (0.040) | ds67_usm40 62.50 (0.063) | 2.70 / 2.75 | 982 / 2543 |

All three cross 60 with p0/p1 + sharpening, but the scores are small (0.003-0.124) and the slow presets cost
5-20x the CPU of p4 (p4 126-181 CPU-s); at the runtime trade-off (0.012 VMAF per CPU-s) p0/p1 do not pay.
Slower presets raise the gap slightly (+0.1..+0.2 vs p4 for the same filter).

### Stage 2: sharpening on p0 / p1 (all three clips cross VMAF 60)

| Clip | Old label best | p0 alone | Best p1 + sharpening | Best p0 + sharpening |
|---|---|---|---|---|
| ai_hard_12_1080 | 56.37 | 58.10 | ds85+usm0.3: 60.12 (gap 2.80, score 0.003, 1015 CPU-s) | ds85+usm0.3: 61.06 (gap 2.84, score 0.026, 2252 CPU-s) |
| ai_hard_20_1080 | 59.93 | 60.72 | ds67+usm0.2: 64.96 (gap 2.67, score 0.124, 776 CPU-s) | ds67+usm0.2: 64.81 (gap 2.68, score 0.120, 2047 CPU-s) |
| ai_hard_04_1080 | 58.72 | 59.43 | ds67+usm0.4: 61.62 (gap 2.70, score 0.040, 982 CPU-s) | ds67+usm0.4: 62.50 (gap 2.75, score 0.063, 2543 CPU-s) |

- Sharpening adds +3.1 VMAF on top of p0 (ai_hard_04_1080: 59.43 -> 62.50); preset and sharpening gains stack.
- The NEG gap rises 0.1-0.2 from p4 to p0/p1 for the same filter (ai_hard_12_1080 2.65 -> 2.84): settle the strength at
  the final preset, not at p4.
