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
# Downloaded at build time and sha256-verified in the same RUN step.
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl xz-utils; \
    rm -rf /var/lib/apt/lists/*; \
    cd /tmp; \
    curl -fsSL -o ff.tar.xz \
      https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-06-12-50/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1.tar.xz; \
    echo "2d17c1f53aa6e15882ff76d2a0089d63c3553a401bf2bdc723d798beb5eb4f1b  ff.tar.xz" | sha256sum -c -; \
    tar -xJf ff.tar.xz; \
    install -m 0755 ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/bin/ffmpeg  /usr/local/bin/ffmpeg; \
    install -m 0755 ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/bin/ffprobe /usr/local/bin/ffprobe; \
    mkdir -p /usr/local/share/doc/ffmpeg; \
    cp ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1/LICENSE.txt /usr/local/share/doc/ffmpeg/LICENSE.txt; \
    rm -rf /tmp/ff.tar.xz /tmp/ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY run.sh /app/run.sh
COPY streamenc /app/streamenc
COPY configs /app/configs
# Only the model run.sh loads (batch_main --model default); research / eval models stay out of the image.
COPY models/crf_predictor.pkl /app/models/crf_predictor.pkl
COPY LICENSE /app/LICENSE
COPY LICENSES.md /app/LICENSES.md
COPY README.md /app/README.md

RUN find /app -type d -exec chmod 0755 {} \; \
 && find /app -type f -exec chmod 0644 {} \; \
 && chmod 0755 /app/run.sh

# Fail the BUILD (not the evaluation) when something the run relies on is missing.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libsvtav1 \
 && ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf \
 && ffprobe -version >/dev/null \
 && python3 -c "import numpy, cv2, streamenc.encode.batch_main, streamenc.pipeline.item"

CMD ["/bin/sh", "/app/run.sh"]
