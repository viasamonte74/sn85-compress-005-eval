# sn85-compression-005 contender image.
#
# Sandbox builder rules (miner-vidaio docs/MINING.md): plain FROM/COPY/RUN/ENV/WORKDIR only,
# image references carry a digest OR a tag (never both), builds have network, the run has none.
# The evaluator ignores ENTRYPOINT/CMD and calls:  /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
#
# python:3.12-slim-bookworm (Debian 12), pinned by its multi-arch index digest.
FROM python@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTENDER_APP_DIR=/app \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ffmpeg + ffprobe: BtbN static GPL build n8.1.3-34-ga35157abb9 (release autobuild-2026-11-06-12-50),
# the exact binary the entry was developed and measured with. run.sh uses only its libsvtav1 4.3 encoder
# and libvmaf (vmaf_v0.6.1 + neg models built in); the other encoders it carries are not used.
# GPL-3.0 is explicitly allowed by the competition licence rule ("GPL/LGPL such as ffmpeg").
# No apt packages (every download stays pinned by content): fetched with the base image's Python
# (docker/fetch_ffmpeg.py: the BtbN release, else an unmodified mirror of the same archive, since BtbN prunes daily
# builds after 14 days; both sha256-verified), checked again here, and unpacked with Python's tarfile (xz).
COPY docker/fetch_ffmpeg.py /tmp/fetch_ffmpeg.py
RUN set -eu; \
    cd /tmp; \
    python3 -I /tmp/fetch_ffmpeg.py /tmp/ff.tar.xz; \
    echo "2d17c1f53aa6e15882ff76d2a0089d63c3553a401bf2bdc723d798beb5eb4f1b  /tmp/ff.tar.xz" | sha256sum -c -; \
    python3 -c "import tarfile; d = 'ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1'; t = tarfile.open('/tmp/ff.tar.xz', 'r:xz'); [t.extract(d + '/' + n, '/tmp', filter='data') for n in ('bin/ffmpeg', 'bin/ffprobe', 'LICENSE.txt')]"; \
    install -m 0755 /tmp/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/bin/ffmpeg  /usr/local/bin/ffmpeg; \
    install -m 0755 /tmp/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/bin/ffprobe /usr/local/bin/ffprobe; \
    mkdir -p /usr/local/share/doc/ffmpeg; \
    cp /tmp/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/LICENSE.txt /usr/local/share/doc/ffmpeg/LICENSE.txt; \
    rm -rf /tmp/ff.tar.xz /tmp/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1 /tmp/fetch_ffmpeg.py

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Exactly the pinned packages (--no-deps); pip check fails the build if one of their dependencies is missing.
RUN pip3 install --no-cache-dir --no-deps -r /app/requirements.txt && pip3 check

COPY run.sh /app/run.sh
COPY streamenc /app/streamenc
COPY configs /app/configs
# Only the model run.sh loads (batch_main --model default); research / eval models stay out of the image.
COPY models/crf_predictor.pkl /app/models/crf_predictor.pkl
COPY LICENSE /app/LICENSE
COPY NOTICE /app/NOTICE
COPY LICENSES.md /app/LICENSES.md
COPY README.md /app/README.md

RUN find /app -type d -exec chmod 0755 {} \; \
 && find /app -type f -exec chmod 0644 {} \; \
 && chmod 0755 /app/run.sh

# Fail the BUILD (not the evaluation) when something the run relies on is missing.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libsvtav1 \
 && ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf \
 && ffprobe -version >/dev/null \
 && python3 -c "import numpy, cv2, streamenc.encode.batch_main, streamenc.pipeline.item" \
 && python3 -c "from streamenc.ml.crf_predictor import load_model; load_model('/app/models/crf_predictor.pkl')"

CMD ["/bin/sh", "/app/run.sh"]
