# Third-party components

This entry's own code is Apache-2.0 (see `LICENSE`). Everything the image ships allows commercial
use; the competition rule (miner-vidaio `docs/MINING.md`, "Licences: commercial use only") lists
GPL/LGPL such as ffmpeg among the allowed licences. Every download in the `Dockerfile` is pinned by
content (base image by digest, ffmpeg tarball by sha256, pip packages by exact `==` version).

## Shipped components

| Component | Version / pin | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| Python, official `python:3.12-slim-bookworm` image | pinned by index digest `935c3c5b…` | https://hub.docker.com/_/python | PSF-2.0 (Python); Debian 12 packages under their own free licences | base image |
| FFmpeg + ffprobe, BtbN static build `ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1` | release `autobuild-2026-11-06-12-50`, tarball sha256 `2d17c1f5…` verified at build | https://github.com/BtbN/FFmpeg-Builds | GPL-3.0-or-later (`--enable-gpl --enable-version3`), run as a separate program | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe`; build licence text at `/usr/local/share/doc/ffmpeg/LICENSE.txt` |
| — bundled in that ffmpeg: SVT-AV1 4.3 (the only encoder `run.sh` uses) | static | https://gitlab.com/AOMediaCodec/SVT-AV1 | BSD-3-Clause-Clear + AOM patent licence | inside ffmpeg |
| — bundled: libvmaf + `vmaf_v0.6.1` / `vmaf_v0.6.1neg` models (used at run time to measure the VMAF / VMAF-NEG gap of an encode before shipping it) | static | https://github.com/Netflix/vmaf | BSD-2-Clause-Patent | inside ffmpeg |
| — bundled, not used by `run.sh`: libaom, libvpx, libx265, libx264, NVENC wrappers | static | — | BSD-2 (aom/vpx), GPL-2.0-or-later (x264/x265) | inside ffmpeg |
| numpy | 2.5.4 | https://pypi.org/project/numpy | BSD-3-Clause (with bundled 0BSD, MIT, Zlib, CC0-1.0 parts) | pip |
| opencv-python-headless | 5.0.0.93 | https://pypi.org/project/opencv-python-headless | Apache-2.0 | pip |
| scikit-learn | 1.9.1 | https://pypi.org/project/scikit-learn | BSD-3-Clause | pip |
| scipy | 1.18.1 | https://pypi.org/project/scipy | BSD-3-Clause | pip |
| joblib | 1.6.0 | https://pypi.org/project/joblib | BSD-3-Clause | pip |
| threadpoolctl | 3.8.0 | https://pypi.org/project/threadpoolctl | BSD-3-Clause | pip |
| Model weights `models/crf_predictor.pkl` (the only model file the image ships) | trained by this entry | this repository (`scripts/train_crf_predictor.py`) | Apache-2.0, as the entry's own code | `/app/models/crf_predictor.pkl` |

## Model weights and training data

`models/crf_predictor.pkl` is a set of scikit-learn gradient-boosted regression trees trained by this
entry. Per clip it predicts, from measured video features, which SVT-AV1 setting to use, the VMAF each
setting reaches inside the bitrate budget, its encode time and its VMAF / VMAF-NEG gap. No pretrained
weights are used.

It was trained only on material that allows training a model for commercial use:

| Training data | Licence | Source / attribution |
| --- | --- | --- |
| AI-generated clips made by this entry with Wan 2.2 TI2V-5B | model Apache-2.0; the outputs are ours | https://github.com/Wan-Video/Wan2.2 |
| Frame-interpolated versions of those clips (Practical-RIFE v4.26) | MIT (tool); outputs are ours | https://github.com/hzwer/Practical-RIFE |
| 3D renders made by this entry (procedural Blender 4.5 Cycles scenes, no outside assets) | ours (Blender is GPL, its outputs are not) | `scripts/hfr_blender_scene.py` (the earlier 2026-11-05 set came from a generator script that is not kept in this repository) |
| Segments of Blender Foundation open movies: Big Buck Bunny, Caminandes 1–3, Cosmos Laundromat | CC BY 3.0 / CC BY 4.0 | Blender Foundation; Caminandes 1–2 by Pablo Vazquez; https://studio.blender.org/films/ |
| Screen-content clips rendered by this entry (text from the CPython standard library, DejaVu fonts) | ours (PSF-2.0 text, DejaVu font licence) | `scripts/hfr_screen_gen.py` (the earlier 2026-11-05 set came from a generator script that is not kept in this repository) |
| Degraded copies (noise, grain, edge boost) of the entry's own AI clips | ours | `data/encoder_rank*/` labels, `generated_hard` set |

The shipped weights were trained on the 669 labelled clips of the kinds above (AI-generated, renders, screen content,
degraded copies); natural (camera) footage, including the CC BY-SA clips in the local research set, was
excluded (`scripts/train_crf_predictor.py --exclude-kinds other`).

Runtime settings (thresholds, filter strengths, search limits in `streamenc/encode/crf_pipeline.py` and
`streamenc/encode/ratefit.py`) were chosen by measuring end-to-end score on the public practice pack
(`sn85-compression-005` practice clips, CC0) and on held-out clips of the kinds above; no natural footage.
