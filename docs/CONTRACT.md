# Official run contract (sn85-compression-005)

Sources (not invented):

1. Hub API manifest: `https://vidaio.io/api/hub/v1/competitions/sn85-compression-005/manifest`
2. Official example: `vidAio-subnet/miner-vidaio` → `examples/competition_contenders/streaming_example/`
3. Mining rules: `docs/MINING.md` in miner-vidaio (streaming compression section)

## Invocation

```text
/bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
```

Evaluator facts from the official example README:

- Sandbox has **no network**
- Root filesystem is **read-only**; writable: `OUTPUT_DIR` and `/tmp`
- **No injected environment variables**
- `ENTRYPOINT`/`CMD` are ignored; evaluator calls `/bin/sh /app/run.sh …`
- A **non-zero exit zeroes the whole batch**; failing items should be logged/skipped and the script should still exit 0
- Outputs must be plain regular files (mode 0644), not symlinks
- Caps: 512 MiB per file, 2 GiB per batch at any moment

## Inputs

- Every regular, non-hidden file in `INPUT_DIR` is one item
- Filename is the clip's **64-hex SHA-256** with **no extension**
- Container/codec: lossless **FFV1 in Matroska**
- Live path mentioned by the example: `/evaluation-inputs/clips`

## Outputs

- Exactly one encoded file per input, **same basename**
- One H.264, HEVC, VP9, or AV1 video stream
- Preserve width, height, frame count, and frame rate
- Decodable; colour fidelity gates apply

## Streaming rules (manifest `streaming_rules`)

| Field | Value |
| --- | --- |
| `bits_per_pixel` | `0.05` |
| `max_keyframe_interval_seconds` | `2` |
| `peak_ratio` | `1.5` |
| `peak_window_seconds` | `2` |

Budget:

```text
target_kbps = bits_per_pixel * width * height * fps / 1000
target_bps  = bits_per_pixel * width * height * fps
```

Measured on **video packet payloads** (container overhead excluded).

Hard gates that zero a clip:

- Average packet bitrate > budget + 1%
- Any measured peak window > `peak_ratio * budget`
- First frame not a keyframe
- Keyframe gap > 2 seconds
- Wrong frame count / resolution / frame rate
- Undecodable
- VMAF − VMAF-NEG > 3.0 (anti-sharpening)
- Tone / colour / chroma gate failures (official formulas live in validator scorer)

## Scoring

```text
clip_score = clamp((VMAF - 60) / 40, 0, 1)
quality_score = mean(clip_scores)
spf = total_runtime_seconds / 3428
speed_score = clamp(log(2 / spf) / log(2 / (1/30)), 0, 1)
final_score = quality_score * (0.7 + 0.3 * speed_score)
```

All `run.sh` wall time counts (analysis, encode, validation, retries, model load).
Container start is not counted.

## Sandbox

- GPU: L40S
- CPU: 32
- RAM: 64 GiB
- Batch size: 4
- Batch timeout: 3600 s
- Image limit: 20 GB

## Phase 1 note on peak-window semantics

The official example's local packet helper uses a frame-count window
`round(peak_window_seconds * fps)` over packet sizes. The validator's
authoritative implementation is in `vidaio/scoring/streaming.py` inside
validator-vidaio. Phase 3 must match that implementation; until then treat
any local peak check as a proxy unless cross-checked against the pinned scorer.
