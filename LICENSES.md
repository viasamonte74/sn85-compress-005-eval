# Third-party components

This entry's own code is Apache-2.0 (see `LICENSE`). Everything the image ships allows commercial
use; the competition rule (miner-vidaio `docs/MINING.md`, "Licences: commercial use only") lists
GPL/LGPL such as ffmpeg among the allowed licences. Every download in the `Dockerfile` is pinned by
content (base image by digest, ffmpeg tarball by sha256, pip packages by exact `==` version).

## Shipped components

| Component | Version / pin | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| Python, official `python:3.12-slim-bookworm` image | pinned by index digest `935c3c5b…` | https://hub.docker.com/_/python | PSF-2.0 (Python); Debian 12 packages under their own free licences | base image |
| FFmpeg + ffprobe, BtbN static build `ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1` | release `autobuild-2026-11-06-12-50`, tarball sha256 `2d17c1f5…` verified at build | https://github.com/BtbN/FFmpeg-Builds (fallback: unmodified mirror of the same archive, https://github.com/viasamonte74/ffmpeg-btbn-mirror, same sha256) | GPL-3.0-or-later (`--enable-gpl --enable-version3`), run as a separate program | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe`; build licence text at `/usr/local/share/doc/ffmpeg/LICENSE.txt` |
| — bundled in that ffmpeg: SVT-AV1 4.3 (the only encoder `run.sh` uses) | static | https://gitlab.com/AOMediaCodec/SVT-AV1 | BSD-3-Clause-Clear + AOM patent licence | inside ffmpeg |
| — bundled: libvmaf + `vmaf_v0.6.1` / `vmaf_v0.6.1neg` models (used at run time to measure the VMAF / VMAF-NEG gap of an encode before shipping it) | static | https://github.com/Netflix/vmaf | BSD-2-Clause-Patent | inside ffmpeg |
| — bundled: libdav1d (the AV1 decoder ffmpeg uses when the VMAF check decodes an encode) | static | https://code.videolan.org/videolan/dav1d | BSD-2-Clause | inside ffmpeg |
| — bundled, not used by `run.sh`: libaom, libvpx, libx265, libx264, NVENC wrappers and the other libraries of the BtbN GPL variant (full list: `ffmpeg -buildconf`; BtbN's `scripts.d/` at that release names each library's source and licence) | static | https://github.com/BtbN/FFmpeg-Builds | free-software licences (BSD, MIT, LGPL, GPL, Apache-2.0, MPL-2.0); built without `--enable-nonfree` | inside ffmpeg |
| numpy | 2.5.4 | https://pypi.org/project/numpy | BSD-3-Clause (with bundled 0BSD, MIT, Zlib, CC0-1.0 parts) | pip |
| opencv-python-headless | 5.0.0.93 | https://pypi.org/project/opencv-python-headless | Apache-2.0 | pip |
| scikit-learn | 1.9.1 | https://pypi.org/project/scikit-learn | BSD-3-Clause | pip |
| scipy | 1.18.1 | https://pypi.org/project/scipy | BSD-3-Clause | pip |
| joblib | 1.6.0 | https://pypi.org/project/joblib | BSD-3-Clause | pip |
| threadpoolctl | 3.8.0 | https://pypi.org/project/threadpoolctl | BSD-3-Clause | pip |
| cloudpickle (dependency of joblib) | 3.1.2 | https://pypi.org/project/cloudpickle | BSD-3-Clause | pip |
| narwhals (dependency of scikit-learn) | 2.28.0 | https://pypi.org/project/narwhals | MIT | pip |
| Model weights `models/crf_predictor.pkl` (the only model file the image ships) | trained by this entry | this repository (`scripts/train_crf_predictor.py`) | Apache-2.0, as the entry's own code | `/app/models/crf_predictor.pkl` |
| Scoring / measurement logic derived from VidAIO's miner and validator code: packet rules (`vidaio/scoring/streaming.py`, `vidaio/scoring_worker/streaming_backend.py`, the `streaming_example` helper), CPU perceptual (chroma / tone / grayscale) decision formulas (`vidaio/scoring/perceptual_cpu.py`), VMAF canonicalization and libvmaf options (`vidaio/scoring/canonicalize.py`, `scoring_worker/service.py`, `backends_real.py`) | scorer `vidaio-scorer/1+8f10f37c629e` | https://github.com/vidAio-subnet/miner-vidaio, https://github.com/vidAio-subnet/validator-vidaio | MIT, Copyright (c) 2026 VidAIO (notice below) | `streamenc/validate/official_streaming.py`, `streamenc/validate/perceptual.py`, `streamenc/validate/vmaf.py`, `streamenc/encode/packets.py` |

The image installs no apt packages: the ffmpeg tarball is fetched with the base image's Python (`docker/fetch_ffmpeg.py`:
the BtbN release, else the mirror, which BtbN's 14-day pruning of daily builds makes necessary) and checked by sha256 in
the same `RUN` step. The pip wheels bundle their own native libraries under their licences (numpy / scipy:
OpenBLAS BSD-3-Clause and the GCC runtime under GPL-3.0 with the GCC Runtime Library Exception; scikit-learn: the GCC
OpenMP runtime libgomp under the same exception; opencv-python-headless:
FFmpeg LGPL-2.1 and image codecs under BSD / zlib-style licences, listed in each wheel's own licence files).

Development-only tools (`requirements-dev.txt`, never installed in the image): pytest 9.2.0 (MIT),
pandas 3.1.1 (BSD-3-Clause), pyarrow 26.0.1 (Apache-2.0). The copyright notice of this entry is in `NOTICE`.

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
(`sn85-compression-005` practice clips, CC0, whose README allows training and tuning on it; q07 / q08 were made with
Wan 2.2) and on held-out clips of the kinds above; no natural footage. Two settings first derived when natural clips
were still in the label set were re-derived on the compliant set on 2026-11-12: the VMAF-NEG fallback order (656
compliant clips) and the speed trade-off `lam` = 0.012 (compliant practice + held-out sweep); the VMAF-NEG CRF
shifts were recomputed on 756 compliant clips on 2026-11-11. The 1.0.0rc9 chroma quantiser offset and early-bump
step limit (2026-11-13) were chosen on own renders, Blender Foundation open-movie segments (CC BY), own screen clips
and own Wan 2.2 clips from the training split (`scripts/clean_config_search.py`, `scripts/clean_combo_check.py`), then
checked end to end on the practice pack and held-out clips of the same kinds. The 1.0.0rc10 slow-preset cap (98.5 predicted VMAF) was set on
val16, 16 evaluation clips made by this entry the same way (own Blender renders, own screen clips, own Wan 2.2 clips). The 1.0.0rc13 VMAF-NEG check of
encodes the gap model cannot vouch for was found on the same kind of evaluation clips (own Wan 2.2 clip). The 1.0.0rc15
final-check thresholds (2.92 / 2.95) follow from the measured error of the 1-in-5-frame gap estimate (<= 0.07, own
720p / 1080p clips) and the validator's 3.0 limit.

## Licence notice for the code derived from VidAIO's repositories

```text
MIT License

Copyright (c) 2026 VidAIO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
