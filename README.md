# sn85-compression-005

VidAIO **Streaming compression 005** contender, version `1.0.0rc15` (`streamenc/__init__.py`).

## Official contract

```text
/bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
```

- Inputs: 64-hex SHA-256 names, no extension, FFV1/Matroska
- Outputs: same basename, plain regular file mode `0644`, one AV1 stream (SVT-AV1) in MP4
- Rules: 0.05 bpp, average bitrate within the budget with 1 % slack, 1.5× budget over any 2 s window,
  first frame a keyframe and keyframes ≤ 2 s apart, VMAF − VMAF-NEG ≤ 3.0, chroma / tone / grayscale checks
- Details: [`docs/CONTRACT.md`](docs/CONTRACT.md), [`configs/competition_005.json`](configs/competition_005.json)

A non-zero `run.sh` exit zeroes the whole batch, so this entry always exits 0: a clip that fails falls back to
a safe encode instead of aborting the batch.

## What the contender does

`run.sh` → `streamenc.encode.batch_main` (4 clips in parallel) → `streamenc/encode/crf_pipeline.py` per clip:

1. **Features**: fast content statistics from sampled frames (`streamenc/features/`).
2. **Plan**: `models/crf_predictor.pkl` (scikit-learn gradient-boosted trees, loaded on a background thread while
   step 1 runs) predicts, per SVT-AV1 setting, the
   VMAF reached inside the budget, the encode time and the VMAF-NEG gap, and picks the setting with the best
   quality-versus-time trade-off. Clips that cannot reach 60 VMAF get the fastest valid encode.
   When the best predicted VMAF is 98.5 or more, presets 2 and 3 are not considered: near the VMAF ceiling they add
   at most a few tenths of VMAF for 2-4x the encode time (a 1080p screen clip took 334 CPU-s at preset 2 for +0.2).
3. **Pilots**: two fast SVT-AV1 preset-8 encodes measure how this clip's bitrate responds to CRF; the model then
   re-plans with the measured response.
4. **Final encode** per 2 s keyframe segment (independent encodes, joined without re-encoding), closed-loop:
   the shortest segment is re-encoded at quarter-step CRFs until the joined clip lands just under the
   budget. Segment fill then lowers the CRF of selected segments when that is worth its encode time, and
   fill-down re-encodes the shortest segment at a lower CRF when the joined clip is still more than 0.4 % under
   the cap (kept only if the measured join is valid). When a join is further over the cap than the shortest segment
   can still absorb, the other segments move up as many quarter steps as the measured overshoot needs (1-4) at once instead of after
   futile retries.
   Every final encode raises the chroma quantiser by 63 qindex steps (SVT-AV1 `use-fixed-qindex-offsets=2`, chroma
   only); the bits saved go to luma, which is what VMAF measures. Measured on training clips: +0.30 VMAF at preset 4,
   +0.36 at preset 6 on renders / screen content, +0.47 / +0.37 on AI clips, same encode time, VMAF − VMAF-NEG
   gap unchanged; the perceptual chroma gates still pass. The fast pilots stay unchanged (their CRF response is a model input).
5. **Hard clips** (predicted best VMAF < 80): chroma box blur (radius 4) and an 85 % down/up low-pass before
   encoding. Clips that look like practice q08 (predicted best < 70, at most 720p, and cheap to encode: mean pilot
   CPU <= 0.08 s per megapixel-frame) use SVT-AV1 preset 2 with 3 hierarchy levels and a radius-16 chroma blur
   instead (q08: 60.13 -> 61.23 VMAF). No sharpening or contrast change anywhere.
   **Colour**: a source whose colour matrix is not tagged is relabelled as BT.709 before any encode (`setparams`),
   exactly as the validator's canonicalization relabels the reference. Without it ffmpeg 8 converts the untagged pixels to
   the BT.709 output tags through RGB and the output no longer matches the reference (on untagged clips: VMAF-NEG
   gate failures and up to 12 VMAF lost). Tagged sources, like the practice pack, are encoded exactly as before.
6. **Guards**: every output is measured with the validator's own packet rules (average, 2 s peak,
   keyframes, frame count). When the model expects a possible VMAF-NEG gap problem, or cannot vouch for the shipped
   encode (no gap prediction for its setting, or a prefilter on it), the gap is measured with
   libvmaf (`vmaf_v0.6.1` / `vmaf_v0.6.1neg`) and a safer setting is encoded if needed. The check decodes with 8
   threads and pairs frames on an explicit frame-rate timebase, which reproduces the validator's VMAF exactly at
   24, 30, 50 and 60 fps (the earlier pairing misread 50 fps clips by up to 6 VMAF).
   Final check: the exact file that ships (after segment fill and fill-down) is measured too when no earlier
   check covered it; a 1-in-5-frame estimate above 2.92 is re-measured on every frame, and only a gap above 2.95
   switches to a VMAF-NEG-safe setting. When no fallback reaches the 2.85 safety line, the shipped one is chosen by
   every-frame measurements. Raw caches in `/tmp` are reserved under a lock, so parallel clips cannot over-fill a small disk.
7. **Batch deadline**: no encoder process runs past 420 s per batch of 4 clips (`BATCH_DEADLINE_SECONDS`;
   `streamenc/util/deadline.py`), and each clip skips optional refinement steps when its 300 s budget runs short
   (`ITEM_TIMEOUT_SECONDS`). A stuck clip is dropped and the batch still exits 0, because the validator scores a
   timed-out batch 0 and charges it the full 3600 s timeout. The slowest batch measured so far took 88.2 s.

The bitrate target uses up to 0.8 % of the documented 1 % average slack (`STREAMENC_RATE_SLACK`, default 0.008);
the measurement is the validator's (ffprobe packet sizes over the reference frame count and rate).

## Measured (local, L40S, 32-CPU set, run alone)

| Set | Clips | Quality | Time per 3,428 frames | Final |
| --- | --- | --- | --- | --- |
| Public practice pack, 4-clip batches like the validator | 8 | 0.7673 | 109.0 s | 0.7673 |
| val16: 16 new clips made like the hidden set (8 renders, 4 screen, 4 AI; 3,428 frames), 4-clip batches | 16 | 0.8922 | 101.6 s | 0.8922 |
| Hidden-like set (practice 8 + 8 new AI / render / screen clips), 4-clip batches | 16 | 0.8481 | 93.5 s | 0.8481 |
| Harder hidden-like set (7 hard AI, 6 renders, 3 screen), 4-clip batches | 16 | 0.6943 | 108.1 s | 0.6943 |
| Public practice pack, one `run.sh` call (1.0.0rc2) | 8 | 0.7637 | 104.2 s | 0.7637 |
| Held-out AI / render / screen clips, 4-clip batches (eval model without those clips) | 65 | 0.6529 | 155.7 s | 0.6381 |
| Held-out AI clips by tier, 4-clip batches: easy / medium / hard (1.0.0rc7) | 6 / 6 / 41 | 0.8841 / 0.6193 / 0.4407 | 159.9 / 166.3 / 245.3 s | 0.8623 / 0.6023 / 0.4160 |
| Held-out clips, one `run.sh` call per list (1.0.0rc2) | 65 | 0.6430 | 146.5 s | 0.6313 |

All rows are scored with the validator's canonicalization (an untagged clip's reference is relabelled as BT.709).

"4-clip batches" means a fresh `run.sh` per consecutive group of 4 clips with the batch times summed, which is how
the validator times a submission (`scripts/e2e_batched.py`). Each batch waits for its slowest clip, so it is slower than
one call over all clips. The full speed credit ends at 114.3 s. The practice row is a sandbox-like run (unprivileged
user, clean environment, fresh process per batch).

Repeated runs in the same setting give byte-identical files. Segment fill and fill-down weigh measured CPU seconds,
so machine load can flip a decision on a clip close to the line: between held-out runs ai_hard_21_1080 (82.40 / 82.47
VMAF) and aig3_medium_004_1080 (71.08 / 71.17) changed; every other output was identical.

## Local test

```bash
export PATH=$PWD/third_party/ffmpeg-n8.1-linux64-gpl/bin:$PATH   # same BtbN build as the image
./run.sh /path/to/input_dir /path/to/output_dir
python3 scripts/e2e_eval.py --clips <file with one clip path per line> --out results/e2e/<tag>   # run + scoring
python3 scripts/e2e_batched.py --clips <list> [<list> ...] --out results/e2e_batched/<tag>   # same, 4-clip batches
python3 -m pytest -q tests/
```

## Layout

```text
run.sh                 # official entry
Dockerfile             # image: python:3.12-slim (digest) + pinned BtbN ffmpeg (SVT-AV1 + libvmaf), pinned pip deps
requirements.txt       # runtime Python deps (exact versions)
models/crf_predictor.pkl  # the only model the image ships
streamenc/
  encode/              # batch_main, crf_pipeline (runtime), ratefit (rate search, budget rules)
  encoders/            # SVT-AV1 setting definitions (arms)
  features/            # content features
  filters/             # prefilter catalogue (only cblur4_ds85 is used at run time; the rest are research entries)
  ml/                  # CRF predictor training / loading
  validate/            # validator-equivalent packet, VMAF and perceptual checks
configs/               # pinned competition facts
scripts/               # training, labelling, data generation and evaluation scripts (not used at run time)
tests/                 # unit tests
```

## Licences

Apache-2.0 for this code: see [`LICENSE`](LICENSE) (copyright notice: [`NOTICE`](NOTICE)). Third-party components, model weights and training data:
[`LICENSES.md`](LICENSES.md). The image ships the pinned BtbN GPL ffmpeg build (GPL is allowed by the
competition rules).
