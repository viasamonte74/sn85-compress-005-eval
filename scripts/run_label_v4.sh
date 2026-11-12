#!/bin/bash
# Re-label the 35 official-scored clips with the FIXED (canonicalizing) VMAF. Catalog = current LABEL_CATALOG_IDS; LABEL_VER selects data/work dir (default v5).
# Shard i of 3; each shard has its own out/work dir, merged afterwards. Usage: run_label_v4.sh <shard>
set -u
i=$1
cd /workspace/sn85-compression-005 && source /venv/main/bin/activate
export LD_LIBRARY_PATH=/opt/nvidia-drivers/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LABEL_ARM_WORKERS=6
clips=$(paste -sd'|' data/label_v4_shard$i.txt)
n=$(wc -l < data/label_v4_shard$i.txt)
exec python scripts/build_encoder_rank_dataset.py --per-tier 0 --no-generated --limit "$n" \
  --include "^($clips)$" \
  --out-dir data/encoder_rank_${LABEL_VER:-v5}/shard$i --work-dir work/encoder_rank_${LABEL_VER:-v5}/shard$i
