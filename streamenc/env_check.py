"""Phase 1 environment capability probe.

Reports what is available on this host for sn85-compression-005 development.
Does not invent competition API behaviour; contract facts come from
configs/competition_005.json (sourced from the official manifest + streaming_example).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from streamenc.util.paths import project_root
from streamenc.util.subprocess_run import run_command


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str
    required_for_phase1: bool = True


@dataclass
class EnvReport:
    checks: list[CheckItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required_for_phase1)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
        }


def _ffmpeg_has_encoder(name: str) -> bool:
    r = run_command(["ffmpeg", "-hide_banner", "-encoders"], timeout=30)
    return name in r.stdout


def _ffmpeg_has_filter(name: str) -> bool:
    r = run_command(["ffmpeg", "-hide_banner", "-filters"], timeout=30)
    return name in r.stdout


def _try_nvenc(encoder: str, work: Path) -> tuple[bool, str]:
    sample = Path("/workspace/dataset/ai/ai_easy_01.mkv")
    if not sample.exists():
        return False, f"sample missing: {sample}"
    out = work / f"smoke_{encoder}.mkv"
    argv = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(sample),
        "-an",
        "-c:v",
        encoder,
        "-preset",
        "p6",
        "-rc",
        "vbr",
        "-b:v",
        "1.1M",
        "-g",
        "48",
        "-forced-idr",
        "1",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        "24",
        str(out),
    ]
    r = run_command(argv, timeout=60)
    if not r.ok or not out.exists() or out.stat().st_size == 0:
        return False, (r.stderr or r.stdout or "encode failed")[:300]
    probe = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        timeout=30,
    )
    return True, probe.stdout.strip()


def _try_svt(work: Path) -> tuple[bool, str]:
    sample = Path("/workspace/dataset/ai/ai_easy_01.mkv")
    if not sample.exists():
        return False, f"sample missing: {sample}"
    out = work / "smoke_libsvtav1.mkv"
    argv = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(sample),
        "-an",
        "-c:v",
        "libsvtav1",
        "-preset",
        "8",
        "-b:v",
        "1.1M",
        "-g",
        "48",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        "24",
        str(out),
    ]
    r = run_command(argv, timeout=120)
    if not r.ok or not out.exists() or out.stat().st_size == 0:
        return False, (r.stderr or "svt encode failed")[:300]
    return True, f"bytes={out.stat().st_size}"


def collect_env_report(work_dir: Path | None = None) -> EnvReport:
    root = project_root()
    work = work_dir or (root / "work" / "env_check")
    work.mkdir(parents=True, exist_ok=True)
    report = EnvReport()

    report.checks.append(
        CheckItem(
            "python",
            sys.version_info >= (3, 10),
            f"{platform.python_version()} ({sys.executable})",
        )
    )
    report.checks.append(
        CheckItem("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "missing")
    )
    report.checks.append(
        CheckItem("ffprobe", shutil.which("ffprobe") is not None, shutil.which("ffprobe") or "missing")
    )

    if shutil.which("ffmpeg"):
        ver = run_command(["ffmpeg", "-version"], timeout=15).stdout.splitlines()[0]
        report.checks.append(CheckItem("ffmpeg_version", True, ver, required_for_phase1=False))
        report.checks.append(
            CheckItem(
                "encoder_av1_nvenc_listed",
                _ffmpeg_has_encoder("av1_nvenc"),
                "av1_nvenc in ffmpeg -encoders",
            )
        )
        report.checks.append(
            CheckItem(
                "encoder_hevc_nvenc_listed",
                _ffmpeg_has_encoder("hevc_nvenc"),
                "hevc_nvenc in ffmpeg -encoders",
            )
        )
        report.checks.append(
            CheckItem(
                "encoder_libsvtav1_listed",
                _ffmpeg_has_encoder("libsvtav1"),
                "libsvtav1 in ffmpeg -encoders",
            )
        )
        report.checks.append(
            CheckItem(
                "filter_libvmaf",
                _ffmpeg_has_filter("libvmaf"),
                "libvmaf present" if _ffmpeg_has_filter("libvmaf") else "MISSING — needed in Phase 3",
                required_for_phase1=False,
            )
        )

    encode_lib = Path("/opt/nvidia-drivers/lib64/libnvidia-encode.so.1")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    encode_ok = encode_lib.exists() or any(
        (Path(p) / "libnvidia-encode.so.1").exists() for p in ld_path.split(":") if p
    )
    report.checks.append(
        CheckItem(
            "libnvidia_encode",
            encode_ok,
            str(encode_lib) if encode_lib.exists() else f"missing; LD_LIBRARY_PATH={ld_path!r}",
        )
    )

    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        gpu = run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            timeout=30,
        )
        report.checks.append(
            CheckItem("nvidia_smi", gpu.ok, gpu.stdout.strip().replace("\n", " | ") or gpu.stderr[:200])
        )
    else:
        report.checks.append(CheckItem("nvidia_smi", False, "nvidia-smi missing"))

    # Live encode smokes (Phase 1 must prove encoders actually run).
    if encode_ok and _ffmpeg_has_encoder("av1_nvenc"):
        ok, detail = _try_nvenc("av1_nvenc", work)
        report.checks.append(CheckItem("smoke_av1_nvenc", ok, detail))
    else:
        report.checks.append(CheckItem("smoke_av1_nvenc", False, "skipped: encoder/lib unavailable"))

    if encode_ok and _ffmpeg_has_encoder("hevc_nvenc"):
        ok, detail = _try_nvenc("hevc_nvenc", work)
        report.checks.append(CheckItem("smoke_hevc_nvenc", ok, detail, required_for_phase1=False))

    if _ffmpeg_has_encoder("libsvtav1"):
        ok, detail = _try_svt(work)
        report.checks.append(CheckItem("smoke_libsvtav1", ok, detail))

    dataset_root = Path("/workspace/dataset")
    mkv_count = len(list(dataset_root.rglob("*.mkv"))) if dataset_root.exists() else 0
    report.checks.append(
        CheckItem(
            "dataset_mkv_files",
            mkv_count > 0,
            f"{mkv_count} mkv under {dataset_root}",
        )
    )

    cfg = root / "configs" / "competition_005.json"
    report.checks.append(
        CheckItem("competition_config", cfg.is_file(), str(cfg) if cfg.is_file() else "missing")
    )
    if cfg.is_file():
        data = json.loads(cfg.read_text())
        rules = data.get("streaming_rules", {})
        expected = {
            "bits_per_pixel": 0.05,
            "max_keyframe_interval_seconds": 2,
            "peak_ratio": 1.5,
            "peak_window_seconds": 2,
        }
        ok = all(rules.get(k) == v or float(rules.get(k, -1)) == float(v) for k, v in expected.items())
        report.checks.append(
            CheckItem("streaming_rules_match_manifest", ok, json.dumps(rules, sort_keys=True))
        )

    return report


def main() -> int:
    # Ensure encode libs are discoverable on this Vast image.
    enc = "/opt/nvidia-drivers/lib64"
    if Path(enc).is_dir():
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if enc not in cur.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{enc}:{cur}" if cur else enc

    report = collect_env_report()
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
