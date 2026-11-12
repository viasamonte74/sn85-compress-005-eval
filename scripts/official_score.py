"""Score (reference, candidate) with validator-vidaio's own scoring worker code, streaming rules of comp-005."""
# Usage: /workspace/validator-vidaio/.venv/bin/python -I scripts/official_score.py REF.mkv CANDIDATE [WORK_DIR]
# Needs validator-vidaio synced (uv sync --frozen --no-dev) plus numpy + opencv-python-headless in its .venv.
import sys, hashlib, json, tempfile
from pathlib import Path
from vidaio.scoring_worker.config import ScoringWorkerConfig
from vidaio.scoring_worker.service import real_backends, _score_sync
from vidaio.scoring.config import ScoringConfig
from vidaio.services.protocol import ScoreRequest

FF = "/workspace/sn85-compression-005/third_party/ffmpeg-n8.1-linux64-gpl/bin"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main(ref, cand, work_root):
    work = Path(tempfile.mkdtemp(prefix="vscore-", dir=work_root))
    cfg = ScoringWorkerConfig(work_dir=work, ffmpeg_path=f"{FF}/ffmpeg", ffprobe_path=f"{FF}/ffprobe",
                              subprocess_timeout=1800.0, request_timeout=3600.0)
    be = real_backends(cfg)
    r = sha(ref)
    req = ScoreRequest(track="compression", challenge_id="local-check", item_id=Path(cand).name[:64],
                       reference_path=ref, reference_digest=r, miner_input_path=ref, miner_input_digest=r,
                       output_path=cand, output_digest=sha(cand),
                       params={"scoring": "streaming", "bits_per_pixel": 0.05, "max_keyframe_interval_s": 2.0,
                               "peak_ratio": 1.5, "peak_window_s": 2.0})
    item = _score_sync(req, cfg, ScoringConfig(), be, scorer_version=cfg.scorer_version)
    out = {"score": item.score, "gate_passed": item.gate_passed, "violations": [v.code for v in item.violations],
           "metrics": {k: v for k, v in item.metrics.items() if any(t in k for t in ("vmaf", "kbps", "stream", "keyframe", "frame"))},
           "breakdown": item.breakdown.model_dump() if item.breakdown else None,
           "backend_versions": item.backend_versions}
    print(json.dumps(out, indent=1, default=str))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "/tmp")
