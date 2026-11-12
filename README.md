# sn85-compression-005

VidAIO **Streaming compression 005** contender, version `1.0.0rc5` (`streamenc/__init__.py`).

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
3. **Pilots**: two fast SVT-AV1 preset-8 encodes measure how this clip's bitrate responds to CRF; the model then
   re-plans with the measured response.
4. **Final encode** per 2 s keyframe segment (independent encodes, joined without re-encoding), closed-loop:
   the shortest segment is re-encoded at quarter-step CRFs until the joined clip lands just under the
   budget. Segment fill then lowers the CRF of selected segments when that is worth its encode time, and
   fill-down re-encodes the shortest segment at a lower CRF when the joined clip is still more than 0.4 % under
   the cap (kept only if the measured join is valid). When a join is further over the cap than the shortest segment
   can still absorb, the other segments move up 1-2 quarter steps at once instead of after two futile retries.
5. **Hard clips** (predicted best VMAF < 80): chroma box blur (radius 4) and an 85 % down/up low-pass before
   encoding. No sharpening or contrast change anywhere.
6. **Guards**: every output is measured with the validator's own packet rules (average, 2 s peak,
   keyframes, frame count). When the model expects a possible VMAF-NEG gap problem, the gap is measured with
   libvmaf (`vmaf_v0.6.1` / `vmaf_v0.6.1neg`) and a safer setting is encoded if needed. The check decodes with 8
   threads and pairs frames on an explicit frame-rate timebase, which reproduces the validator's VMAF exactly at
   24, 30, 50 and 60 fps (the earlier pairing misread 50 fps clips by up to 6 VMAF).
7. **Batch deadline**: no encoder process runs past 420 s per batch of 4 clips (`BATCH_DEADLINE_SECONDS`;
   `streamenc/util/deadline.py`), and each clip skips optional refinement steps when its 300 s budget runs short
   (`ITEM_TIMEOUT_SECONDS`). A stuck clip is dropped and the batch still exits 0, because the validator scores a
   timed-out batch 0 and charges it the full 3600 s timeout. The slowest batch measured so far took 88.2 s.

The bitrate target uses up to 0.8 % of the documented 1 % average slack (`STREAMENC_RATE_SLACK`, default 0.008);
the measurement is the validator's (ffprobe packet sizes over the reference frame count and rate).

## Measured (local, L40S, 32-CPU set, run alone)

| Set | Clips | Quality | Time per 3,428 frames | Final |
| --- | --- | --- | --- | --- |
| Public practice pack, 4-clip batches like the validator | 8 | 0.7637 | 105.3 s | 0.7637 |
| Public practice pack, one `run.sh` call (1.0.0rc2) | 8 | 0.7637 | 104.2 s | 0.7637 |
| Held-out AI / render / screen clips, 4-clip batches (eval model without those clips) | 65 | 0.6430 | 163.6 s | 0.6260 |
| Held-out clips, one `run.sh` call per list (1.0.0rc2) | 65 | 0.6430 | 146.5 s | 0.6313 |

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

Apache-2.0 for this code: see [`LICENSE`](LICENSE). Third-party components, model weights and training data:
[`LICENSES.md`](LICENSES.md). The image ships the pinned BtbN GPL ffmpeg build (GPL is allowed by the
competition rules).
