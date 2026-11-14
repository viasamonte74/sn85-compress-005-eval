"""Image build step: fetch the pinned BtbN ffmpeg archive, sha256-verified.

Sources, in order (both must give the same bytes; anything else fails the build):
  1. the BtbN release (BtbN keeps only its last 14 daily builds, so this one is pruned around 2026-11-20);
  2. an unmodified mirror of the same archive, split in two parts below GitHub's 100 MB file limit, at a fixed commit.
"""
import hashlib
import sys
import time
import urllib.request

SHA256 = "2d17c1f53aa6e15882ff76d2a0089d63c3553a401bf2bdc723d798beb5eb4f1b"
NAME = "ffmpeg-n8.1.3-34-ga35157abb9-linux64-gpl-8.1.tar.xz"
TAG = "autobuild-2026-11-06-12-50"
MIRROR = "https://raw.githubusercontent.com/viasamonte74/ffmpeg-btbn-mirror/9a31fc03f5da677c98da1d71ad584015860b94d8"
SOURCES = (
    ("BtbN release", [f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{TAG}/{NAME}"]),
    ("mirror", [f"{MIRROR}/{TAG}/{NAME}.part00", f"{MIRROR}/{TAG}/{NAME}.part01"]),
)


def _get(url: str, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - network error: retry, then the next source
            print(f"fetch {url} (try {i + 1}/{tries}): {exc}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    return None


def main(out: str) -> int:
    for name, urls in SOURCES:
        parts = []
        for u in urls:
            p = _get(u)
            if p is None:
                break
            parts.append(p)
        if len(parts) != len(urls):
            print(f"{name}: not reachable", file=sys.stderr)
            continue
        data = b"".join(parts)
        if hashlib.sha256(data).hexdigest() != SHA256:
            print(f"{name}: sha256 mismatch, not used", file=sys.stderr)
            continue
        with open(out, "wb") as f:
            f.write(data)
        print(f"ffmpeg archive from {name}: {len(data)} bytes, sha256 {SHA256}")
        return 0
    print("ffmpeg archive: no source gave the pinned sha256", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ff.tar.xz"))
