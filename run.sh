#!/bin/sh
# Competition contract: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
# sn85-compression-005 streaming compression contender (Phases 1–10).
#
BITS_PER_PIXEL=${BITS_PER_PIXEL:-0.05}
MAX_KEYFRAME_INTERVAL=${MAX_KEYFRAME_INTERVAL:-2.0}
PEAK_WINDOW=${PEAK_WINDOW:-2.0}
PEAK_RATIO=${PEAK_RATIO:-1.5}

# No `set -e`: batch driver always returns 0; one bad item must not abort others.
set -u
umask 022

[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
input_dir=$1
output_dir=$2
app_dir=${CONTENDER_APP_DIR:-$(CDPATH= cd -- "$(dirname "$0")" && pwd)}

if [ -n "${PYTHON:-}" ]; then
    python=$PYTHON
elif [ -x /venv/main/bin/python ]; then
    python=/venv/main/bin/python
elif [ -x /venv/main/bin/python3 ]; then
    python=/venv/main/bin/python3
else
    python=python3
fi

PATH=/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}
PYTHONDONTWRITEBYTECODE=1
if [ -d /opt/nvidia-drivers/lib64 ]; then
    LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH
fi
export PATH PYTHONDONTWRITEBYTECODE
# Model prediction on one thread: sklearn's thread pool fought the encoders for cores (e2e 2026-11-11:
# per-clip prediction 1.2-2.7 s -> 0.02-0.26 s, practice 123.1 -> 118.4 s, outputs identical)
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS
PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$app_dir
export PYTHONPATH

# Per-clip soft budget (optional refinement steps are skipped when it runs short) and the hard batch deadline
# per round of BATCH_WORKERS clips: no encoder runs past it, the batch exits 0 with the outputs that are done.
# Slowest clip measured so far 124 s; the validator's batch timeout is 3600 s and a timed-out batch scores 0.
ITEM_TIMEOUT_SECONDS=${ITEM_TIMEOUT_SECONDS:-300}
BATCH_DEADLINE_SECONDS=${BATCH_DEADLINE_SECONDS:-420}
BATCH_BUDGET_SECONDS=${BATCH_BUDGET_SECONDS:-3300}
BATCH_MODE=${BATCH_MODE:-parallel}
BATCH_WORKERS=${BATCH_WORKERS:-4}
CONTENDER_FLAGS=${CONTENDER_FLAGS:-}

mkdir -p -- "$output_dir" 2>/dev/null || true

# shellcheck disable=SC2086
"$python" -m streamenc.encode.batch_main \
    --mode "$BATCH_MODE" \
    --workers "$BATCH_WORKERS" \
    --bits-per-pixel "$BITS_PER_PIXEL" \
    --max-keyframe-interval "$MAX_KEYFRAME_INTERVAL" \
    --peak-window "$PEAK_WINDOW" \
    --peak-ratio "$PEAK_RATIO" \
    --batch-budget "$BATCH_BUDGET_SECONDS" \
    --item-timeout "$ITEM_TIMEOUT_SECONDS" \
    --batch-deadline "$BATCH_DEADLINE_SECONDS" \
    $CONTENDER_FLAGS \
    "$input_dir" "$output_dir"
rc=$?

# Final scrub of temp artifacts in the writable output directory.
rm -f -- "$output_dir"/.streaming-partial-* \
         "$output_dir"/.corr-partial-* \
         "$output_dir"/.corr-valid-*.mp4 \
         "$output_dir"/.bench-partial-* 2>/dev/null || true

# Map internal 66 (no inputs) through; otherwise always 0 for competition safety.
if [ "$rc" -eq 66 ]; then
    exit 66
fi
exit 0
