#!/usr/bin/env python3
"""Native high-frame-rate screen-content clips (50/60 fps), each scene rendered natively at
1280x720 and 1920x1080 (UI scaled by height/720, so text stays crisp at both sizes).

Scene types: code editor scroll, document/web scroll, terminal output, live dashboard, window drag on
a desktop, slide transitions, spreadsheet scroll, map pan/zoom. Motion is pixel-exact where real UIs
are (scrolling by whole pixels) and resampled where real UIs resample (zoom, window open animation).

Output: /workspace/dataset/hfr_screen/<id>[_1080].mkv, FFV1 yuv420p, 5 s, plus JSON sidecar and a
manifest line. Text: CPython stdlib sources/docstrings (PSF-2.0); font DejaVu. Resumable.

  /venv/main/bin/python scripts/hfr_screen_gen.py [--only id1,id2] [--res 720p,1080p] [--seconds 5]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FF = "/workspace/sn85-compression-005/third_party/ffmpeg-n8.1-linux64-gpl/bin/ffmpeg"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
STDLIB = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
LICENCE = "own render; text CPython stdlib PSF-2.0; font DejaVu"

# id, scene, tier, fps, params
SCENES = [
    ("hfr_scr_easy_01", "doc", "easy", 60, {"speed": 0.6, "pause": 0.5, "theme": "light"}),
    ("hfr_scr_easy_02", "slides", "easy", 50, {"hold": 1.1}),
    ("hfr_scr_easy_03", "code", "easy", 60, {"speed": 0.5, "pause": 0.6, "theme": "light"}),
    ("hfr_scr_medium_01", "code", "medium", 60, {"speed": 1.3, "pause": 0.3, "theme": "dark"}),
    ("hfr_scr_medium_02", "sheet", "medium", 50, {"speed": 1.0}),
    ("hfr_scr_medium_03", "windows", "medium", 60, {"speed": 1.0}),
    ("hfr_scr_medium_04", "dashboard", "medium", 60, {"panels": 6, "rate": 1.0}),
    ("hfr_scr_hard_01", "terminal", "hard", 60, {"burst": 1.0}),
    ("hfr_scr_hard_02", "code", "hard", 60, {"speed": 3.2, "pause": 0.1, "theme": "dark", "flick": True}),
    ("hfr_scr_hard_03", "map", "hard", 60, {"speed": 1.0}),
    ("hfr_scr_hard_04", "doc", "hard", 50, {"speed": 2.6, "pause": 0.1, "theme": "dark", "flick": True}),
    ("hfr_scr_hard_05", "dashboard", "hard", 50, {"panels": 9, "rate": 2.5}),
]
# screen set 2 (2026-11-10, user: "generate 1080p clips more"): the same 8 scene types with new seeds/params at the
# standard rates (24/25/30 fps), 3 per type (easy/medium/hard), native 720p + 1080p -> /workspace/dataset/screen2
SCENES_STD2 = [
    ("scr2_easy_01", "code", "easy", 30, {"speed": 0.4, "pause": 0.7, "theme": "dark"}),
    ("scr2_medium_01", "code", "medium", 24, {"speed": 1.1, "pause": 0.3, "theme": "light"}),
    ("scr2_hard_01", "code", "hard", 30, {"speed": 2.8, "pause": 0.1, "theme": "light", "flick": True}),
    ("scr2_easy_02", "doc", "easy", 24, {"speed": 0.5, "pause": 0.6, "theme": "dark"}),
    ("scr2_medium_02", "doc", "medium", 30, {"speed": 1.4, "pause": 0.3, "theme": "light"}),
    ("scr2_hard_02", "doc", "hard", 25, {"speed": 3.0, "pause": 0.1, "theme": "light", "flick": True}),
    ("scr2_easy_03", "terminal", "easy", 30, {"burst": 0.4}),
    ("scr2_medium_03", "terminal", "medium", 25, {"burst": 1.0}),
    ("scr2_hard_03", "terminal", "hard", 30, {"burst": 2.2}),
    ("scr2_easy_04", "dashboard", "easy", 25, {"panels": 4, "rate": 0.5}),
    ("scr2_medium_04", "dashboard", "medium", 30, {"panels": 6, "rate": 1.5}),
    ("scr2_hard_04", "dashboard", "hard", 24, {"panels": 12, "rate": 3.0}),
    ("scr2_easy_05", "windows", "easy", 30, {"speed": 0.5}),
    ("scr2_medium_05", "windows", "medium", 24, {"speed": 1.2}),
    ("scr2_hard_05", "windows", "hard", 30, {"speed": 2.2}),
    ("scr2_easy_06", "slides", "easy", 24, {"hold": 1.6}),
    ("scr2_medium_06", "slides", "medium", 30, {"hold": 0.9}),
    ("scr2_hard_06", "slides", "hard", 25, {"hold": 0.45}),
    ("scr2_easy_07", "sheet", "easy", 30, {"speed": 0.5}),
    ("scr2_medium_07", "sheet", "medium", 25, {"speed": 1.3}),
    ("scr2_hard_07", "sheet", "hard", 30, {"speed": 2.6}),
    ("scr2_easy_08", "map", "easy", 24, {"speed": 0.4}),
    ("scr2_medium_08", "map", "medium", 30, {"speed": 1.0}),
    ("scr2_hard_08", "map", "hard", 30, {"speed": 2.0}),
]
SETS = {"hfr": (SCENES, "/workspace/dataset/hfr_screen"), "std2": (SCENES_STD2, "/workspace/dataset/screen2")}
RES = {"720p": (1280, 720, ""), "1080p": (1920, 1080, "_1080")}


def font(size: float, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    if bold:
        name += "-Bold"
    return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), max(6, int(round(size))))


def stdlib_text(rng: random.Random, n_files: int = 6) -> list[str]:
    files = sorted(p for p in STDLIB.glob("*.py") if p.stat().st_size > 20000)
    lines: list[str] = []
    for p in rng.sample(files, min(n_files, len(files))):
        lines += p.read_text(errors="replace").expandtabs(4).splitlines()[:400]
    return lines


def prose(rng: random.Random, n: int) -> list[str]:
    """Sentences from stdlib docstrings."""
    words = []
    for line in stdlib_text(rng, 4):
        s = line.strip().strip('"#')
        if s and not any(c in s for c in "=():[]{}") and len(s.split()) > 4:
            words += s.split()
    out, i = [], 0
    while len(out) < n and words:
        k = rng.randint(8, 22)
        out.append(" ".join(words[i:i + k]).capitalize() + ".")
        i = (i + k) % max(1, len(words) - 30)
    return out


def scroll_profile(rng: random.Random, n: int, fps: int, speed: float, pause: float, flick: bool,
                   max_pos: float, s: float) -> np.ndarray:
    """Integer pixel scroll offsets: alternating pauses and eased scrolls (or flicks with decay)."""
    pos, out, v = 0.0, [], 0.0
    i = 0
    while i < n:
        if rng.random() < pause:
            hold = rng.randint(int(0.15 * fps), int(0.6 * fps))
            out += [pos] * hold
            i += hold
            continue
        if flick:
            v = rng.uniform(25, 60) * speed * s * (1 if rng.random() > 0.15 else -0.6)
            while abs(v) > 0.5 and i < n:
                pos = min(max(pos + v * 60 / fps, 0.0), max_pos)
                out.append(pos)
                v *= 0.93 ** (60 / fps)
                i += 1
        else:
            dur = rng.randint(int(0.3 * fps), int(1.0 * fps))
            dist = rng.uniform(120, 420) * speed * s
            start = pos
            for k in range(dur):
                e = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / dur)
                pos = min(start + dist * e, max_pos)
                out.append(pos)
            i += dur
    return np.round(np.array(out[:n])).astype(int)


def token_colors(theme: str):
    from pygments.styles import get_style_by_name

    st = get_style_by_name("monokai" if theme == "dark" else "default")
    bg = st.background_color or ("#272822" if theme == "dark" else "#ffffff")
    def color(tt):
        while tt is not None:
            c = st.style_for_token(tt).get("color")
            if c:
                return "#" + c
            tt = tt.parent
        return "#f8f8f2" if theme == "dark" else "#000000"
    return bg, color


def render_code_canvas(rng, w: int, s: float, theme: str):
    from pygments.lexers import PythonLexer

    lines = stdlib_text(rng)
    bg, color = token_colors(theme)
    fs = 14 * s
    f = font(fs, mono=True)
    lh = int(round(fs * 1.45))
    gutter = int(56 * s)
    img = Image.new("RGB", (w, lh * len(lines) + 10), bg)
    d = ImageDraw.Draw(img)
    cw = f.getlength("M")
    num_col = "#90908a" if theme == "dark" else "#999999"
    for i, line in enumerate(lines):
        y = i * lh + 4
        d.text((gutter - 10 * s - f.getlength(str(i + 1)), y), str(i + 1), font=f, fill=num_col)
        x = gutter + 6 * s
        for tt, val in PythonLexer().get_tokens(line):
            val = val.rstrip("\n")
            if val:
                d.text((x, y), val, font=f, fill=color(tt))
                x += cw * len(val)
                if x > w:
                    break
    return img, bg, lh


def chrome(d: ImageDraw.ImageDraw, w: int, h: int, s: float, theme: str, title: str) -> tuple[int, int]:
    """Window title/tab bar + status bar; returns (top, bottom) of the content area."""
    dark = theme == "dark"
    bar, txt = ("#1e1f1c", "#d0d0d0") if dark else ("#e8e8e8", "#333333")
    top, sb = int(34 * s), int(24 * s)
    d.rectangle([0, 0, w, top], fill=bar)
    for k, name in enumerate([title, "utils.py", "README.md"]):
        x0 = int(10 * s + k * 170 * s)
        d.rectangle([x0, int(6 * s), x0 + int(160 * s), top], fill=("#272822" if dark else "#ffffff") if k == 0 else bar)
        d.text((x0 + 10 * s, 11 * s), name, font=font(12 * s), fill=txt)
    d.rectangle([0, h - sb, w, h], fill="#007acc")
    d.text((10 * s, h - sb + 5 * s), "Ln 1, Col 1   Spaces: 4   UTF-8   Python", font=font(11 * s), fill="#ffffff")
    return top, h - sb


def scene_code(rng, w, h, fps, n, p, s):
    theme = p.get("theme", "dark")
    sidebar = int(220 * s)
    canvas, bg, lh = render_code_canvas(rng, w - sidebar, s, theme)
    base = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(base)
    top, bot = chrome(d, w, h, s, theme, "module.py")
    side_bg = "#252526" if theme == "dark" else "#f3f3f3"
    d.rectangle([0, top, sidebar, bot], fill=side_bg)
    for k, name in enumerate(sorted(x.name for x in STDLIB.glob("*.py"))[rng.randint(0, 50):][:int((bot - top) / (22 * s))]):
        d.text((14 * s, top + 8 * s + k * 22 * s), name, font=font(12 * s), fill="#c5c5c5" if theme == "dark" else "#444444")
    view_h = bot - top
    offs = scroll_profile(rng, n, fps, p["speed"], p["pause"], p.get("flick", False), canvas.height - view_h, s)
    for k in range(n):
        frame = base.copy()
        frame.paste(canvas.crop((0, offs[k], w - sidebar, offs[k] + view_h)), (sidebar, top))
        if (k // max(1, fps // 2)) % 2 == 0:
            cy = top + (lh * 7 - offs[k] % lh)
            ImageDraw.Draw(frame).rectangle([sidebar + 80 * s, cy, sidebar + 82 * s, cy + lh - 2], fill="#f8f8f0" if theme == "dark" else "#000000")
        yield frame


def procedural_image(rng, w: int, h: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c0, c1 = np.array([rng.randint(0, 255) for _ in range(3)]), np.array([rng.randint(0, 255) for _ in range(3)])
    t = (xx / w * rng.uniform(0.3, 1) + yy / h * rng.uniform(0.3, 1)) / 2
    img = (c0 * (1 - t[..., None]) + c1 * t[..., None])
    for _ in range(rng.randint(3, 7)):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.1, 0.4) * min(w, h)
        m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
        img = img * (1 - m) + np.array([rng.randint(0, 255) for _ in range(3)]) * m
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 6, (h, w, 1))
    return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))


def scene_doc(rng, w, h, fps, n, p, s):
    dark = p.get("theme") == "dark"
    bg, fg, head = ("#1b1d23", "#d7dae0", "#ffffff") if dark else ("#ffffff", "#222222", "#111111")
    col_w = int(min(w * 0.62, 900 * s))
    x0 = (w - col_w) // 2
    paras = prose(rng, 80)
    fb, fh = font(16 * s), font(26 * s, bold=True)
    lh = int(26 * s)
    blocks, y = [], int(30 * s)
    canvas = Image.new("RGB", (w, int(9000 * s)), bg)
    d = ImageDraw.Draw(canvas)
    for i, para in enumerate(paras):
        if i % 5 == 0:
            d.text((x0, y), para[:48].rstrip(".,"), font=fh, fill=head)
            y += int(48 * s)
        if i % 7 == 3:
            ih = int(rng.uniform(220, 360) * s)
            canvas.paste(procedural_image(rng, col_w, ih), (x0, y))
            y += ih + int(20 * s)
        words, line = para.split(), ""
        for wd in words:
            if fb.getlength(line + " " + wd) > col_w:
                d.text((x0, y), line, font=fb, fill=fg)
                y += lh
                line = wd
            else:
                line = (line + " " + wd).strip()
        d.text((x0, y), line, font=fb, fill=fg)
        y += lh + int(14 * s)
        if y > canvas.height - 600 * s:
            break
    canvas = canvas.crop((0, 0, w, y))
    nav = int(56 * s)
    offs = scroll_profile(rng, n, fps, p["speed"], p["pause"], p.get("flick", False), canvas.height - (h - nav), s)
    for k in range(n):
        frame = Image.new("RGB", (w, h), bg)
        frame.paste(canvas.crop((0, offs[k], w, offs[k] + h - nav)), (0, nav))
        dd = ImageDraw.Draw(frame)
        dd.rectangle([0, 0, w, nav], fill="#24292f" if dark else "#f6f8fa")
        dd.text((24 * s, 16 * s), "docs.example / library reference", font=font(15 * s, bold=True), fill="#ffffff" if dark else "#24292f")
        frac = offs[k] / max(1, canvas.height - (h - nav))
        dd.rectangle([w - 8 * s, nav + frac * (h - nav - 80 * s), w - 2 * s, nav + frac * (h - nav - 80 * s) + 80 * s], fill="#888888")
        yield frame


def scene_terminal(rng, w, h, fps, n, p, s):
    f = font(13 * s, mono=True)
    lh = int(17 * s)
    rows = (h - int(30 * s)) // lh
    src = stdlib_text(rng, 8)
    cols = ["#d0d0d0", "#8ae234", "#fce94f", "#729fcf", "#ef2929", "#ad7fa8"]
    buf: list[tuple[str, str]] = []
    si = 0
    typing, typed = "", 0
    for k in range(n):
        phase = (k / fps) % 2.5
        if phase < 0.6:  # typing a command
            if not typing:
                typing, typed = "$ python -m " + rng.choice(["pytest -x -q", "build --release", "pip install -r req.txt", "compileall src"]), 0
            typed = min(len(typing), typed + max(1, int(round(40 / fps))))
            cur = (typing[:typed], "#ffffff")
        else:
            if typing:
                buf.append((typing, "#ffffff"))
                typing = ""
            burst = int(rng.expovariate(1 / (6 * p["burst"])) * 60 / fps) + 1
            for _ in range(burst):
                line = src[si % len(src)][: int(w / (7.8 * s))]
                si += 1
                tag = rng.random()
                prefix = "[ OK ] " if tag < 0.5 else ("[WARN] " if tag < 0.6 else ("[FAIL] " if tag < 0.63 else ""))
                buf.append((prefix + line, cols[0 if not prefix else (1 if "OK" in prefix else (2 if "WARN" in prefix else 4))]))
            cur = ("", "#ffffff")
        buf = buf[-rows:]
        frame = Image.new("RGB", (w, h), "#300a24")
        d = ImageDraw.Draw(frame)
        d.rectangle([0, 0, w, int(26 * s)], fill="#3c3b37")
        d.text((12 * s, 6 * s), "user@build-host: ~/project", font=font(12 * s, bold=True), fill="#dfdbd2")
        y = int(30 * s)
        for text, c in buf + [cur]:
            d.text((8 * s, y), text, font=f, fill=c)
            y += lh
        yield frame


def scene_dashboard(rng, w, h, fps, n, p, s):
    npan = p["panels"]
    cols = 3
    rows_ = math.ceil(npan / cols)
    pad, top = int(12 * s), int(48 * s)
    pw, ph = (w - pad * (cols + 1)) // cols, (h - top - pad * (rows_ + 1)) // rows_
    series = [np.cumsum(np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 1, 2000)) for _ in range(npan)]
    kinds = [rng.choice(["line", "bars", "gauge", "number", "area"]) for _ in range(npan)]
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]
    fl, fn = font(12 * s), font(34 * s, bold=True)
    for k in range(n):
        frame = Image.new("RGB", (w, h), "#111217")
        d = ImageDraw.Draw(frame)
        d.text((pad, 14 * s), "Service overview  •  live", font=font(16 * s, bold=True), fill="#e0e0e0")
        d.text((w - 200 * s, 16 * s), time.strftime("%H:%M:", time.gmtime(0)) + f"{(k / fps) % 60:05.2f}", font=font(14 * s, mono=True), fill="#9aa0a6")
        off = int(k * p["rate"] * 60 / fps)
        for i in range(npan):
            x0 = pad + (i % cols) * (pw + pad)
            y0 = top + (i // cols) * (ph + pad)
            d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=int(6 * s), fill="#181b1f", outline="#2c3235")
            d.text((x0 + 10 * s, y0 + 8 * s), f"metric_{i:02d}  p95", font=fl, fill="#c7d0d9")
            seg = series[i][off:off + 120]
            seg = (seg - seg.min()) / (np.ptp(seg) + 1e-6)
            cx0, cy0, cw_, ch_ = x0 + 12 * s, y0 + 32 * s, pw - 24 * s, ph - 44 * s
            c = palette[i % len(palette)]
            if kinds[i] in ("line", "area"):
                pts = [(cx0 + j * cw_ / (len(seg) - 1), cy0 + ch_ * (1 - v)) for j, v in enumerate(seg)]
                if kinds[i] == "area":
                    d.polygon(pts + [(cx0 + cw_, cy0 + ch_), (cx0, cy0 + ch_)], fill="#22303f")
                d.line(pts, fill=c, width=max(1, int(2 * s)))
            elif kinds[i] == "bars":
                nb = 24
                for j in range(nb):
                    v = seg[int(j * (len(seg) - 1) / (nb - 1))]
                    bx = cx0 + j * cw_ / nb
                    d.rectangle([bx + 1, cy0 + ch_ * (1 - v), bx + cw_ / nb - 2, cy0 + ch_], fill=c)
            elif kinds[i] == "gauge":
                v = seg[-1]
                r = min(cw_, ch_ * 2) / 2.2
                gx, gy = cx0 + cw_ / 2, cy0 + ch_ - 4 * s
                d.arc([gx - r, gy - r, gx + r, gy + r], 180, 360, fill="#2c3235", width=int(14 * s))
                d.arc([gx - r, gy - r, gx + r, gy + r], 180, 180 + 180 * v, fill=c, width=int(14 * s))
                d.text((gx - 30 * s, gy - 40 * s), f"{v * 100:4.1f}%", font=font(20 * s, bold=True), fill="#ffffff")
            else:
                d.text((cx0, cy0 + ch_ / 3), f"{1000 + series[i][off + 119] * 37:,.0f}", font=fn, fill=c)
        yield frame


def scene_windows(rng, w, h, fps, n, p, s):
    wall = procedural_image(rng, w // 4, h // 4).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3 * s))
    code_w, code_h = int(w * 0.5), int(h * 0.55)
    code_gen = scene_code(random.Random(rng.randint(0, 1 << 30)), code_w, code_h, fps, 1, {"speed": 1, "pause": 0, "theme": "dark"}, s * 0.8)
    win1 = next(code_gen)
    doc_gen = scene_doc(random.Random(rng.randint(0, 1 << 30)), int(w * 0.42), int(h * 0.6), fps, 1, {"speed": 1, "pause": 0}, s * 0.8)
    win2 = next(doc_gen)
    shadow = Image.new("L", (code_w + 40, code_h + 40), 0)
    ImageDraw.Draw(shadow).rectangle([20, 20, code_w + 20, code_h + 20], fill=140)
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    pts = [(rng.uniform(0, w - code_w), rng.uniform(30 * s, h - code_h)) for _ in range(4)]
    for k in range(n):
        t = k / max(1, n - 1)
        u = (t * 3 * p["speed"]) % 1.0
        seg = int(t * 3 * p["speed"]) % 3
        a, b = pts[seg], pts[seg + 1]
        e = 0.5 - 0.5 * math.cos(math.pi * u)
        x, y = int(a[0] + (b[0] - a[0]) * e), int(a[1] + (b[1] - a[1]) * e)
        frame = wall.copy()
        d = ImageDraw.Draw(frame)
        d.rectangle([0, h - int(40 * s), w, h], fill="#202020")
        for i in range(8):
            d.rounded_rectangle([20 * s + i * 52 * s, h - 34 * s, 56 * s + i * 52 * s, h - 6 * s], radius=int(6 * s), fill=["#3a7bd5", "#e94e77", "#f0a830", "#35b779", "#8e44ad", "#16a085", "#c0392b", "#7f8c8d"][i])
        tw = min(1.0, max(0.0, (t - 0.45) / 0.15))  # second window opens with zoom+fade
        if tw > 0:
            sc = 0.7 + 0.3 * tw
            w2 = win2.resize((max(1, int(win2.width * sc)), max(1, int(win2.height * sc))), Image.BILINEAR)
            px, py = int(w * 0.52 - w2.width / 2 + win2.width / 2), int(h * 0.2)
            frame.paste(Image.blend(frame.crop((px, py, px + w2.width, py + w2.height)), w2, tw), (px, py))
        frame.paste((0, 0, 0), (x - 20 + 6, y - 20 + 10), shadow)
        frame.paste(win1, (x, y))
        cx, cy = x + int(120 * s), y + int(14 * s)
        d.polygon([(cx, cy), (cx, cy + 18 * s), (cx + 5 * s, cy + 13 * s), (cx + 12 * s, cy + 13 * s)], fill="#ffffff", outline="#000000")
        yield frame


def scene_slides(rng, w, h, fps, n, p, s):
    titles = prose(rng, 12)
    slides = []
    for i in range(5):
        im = Image.new("RGB", (w, h), ["#ffffff", "#0b3d91", "#fdf6e3", "#1e1e1e", "#f4f4f4"][i])
        d = ImageDraw.Draw(im)
        dark = i in (1, 3)
        fg = "#ffffff" if dark else "#1a1a1a"
        d.text((80 * s, 70 * s), titles[i][:40], font=font(40 * s, bold=True), fill=fg)
        if i % 2 == 0:
            for j in range(5):
                d.text((110 * s, (190 + j * 60) * s), "•  " + titles[5 + (i + j) % 7][:60], font=font(22 * s), fill=fg)
        else:
            im.paste(procedural_image(rng, int(w * 0.5), int(h * 0.5)), (int(w * 0.4), int(h * 0.35)))
            vals = [rng.uniform(0.2, 1) for _ in range(6)]
            for j, v in enumerate(vals):
                d.rectangle([90 * s + j * 60 * s, h * 0.85 - v * h * 0.4, 130 * s + j * 60 * s, h * 0.85], fill="#f28e2b")
        slides.append(im)
    hold, trans = int(p["hold"] * fps), int(0.45 * fps)
    per = hold + trans
    kinds = ["push", "fade", "zoom", "push"]
    for k in range(n):
        i, r = divmod(k, per)
        a, b = slides[i % 5], slides[(i + 1) % 5]
        if r < hold:
            yield a
            continue
        t = (r - hold + 1) / trans
        e = 0.5 - 0.5 * math.cos(math.pi * t)
        kind = kinds[i % 4]
        if kind == "push":
            dx = int(w * e)
            fr = Image.new("RGB", (w, h))
            fr.paste(a.crop((dx, 0, w, h)), (0, 0))
            fr.paste(b.crop((0, 0, dx, h)), (w - dx, 0)) if dx > 0 else None
            yield fr
        elif kind == "fade":
            yield Image.blend(a, b, e)
        else:
            sc = 1 + 0.25 * e
            za = a.resize((int(w * sc), int(h * sc)), Image.BILINEAR).crop((int(w * (sc - 1) / 2), int(h * (sc - 1) / 2), int(w * (sc - 1) / 2) + w, int(h * (sc - 1) / 2) + h))
            yield Image.blend(za, b, e)


def scene_sheet(rng, w, h, fps, n, p, s):
    cw_, rh = int(96 * s), int(24 * s)
    ncol, nrow = 60, 400
    f = font(12 * s, mono=True)
    canvas = Image.new("RGB", (ncol * cw_, nrow * rh), "#ffffff")
    d = ImageDraw.Draw(canvas)
    for r in range(nrow):
        for c in range(ncol):
            v = rng.lognormvariate(4, 1.5) * (1 if rng.random() > 0.1 else -1)
            txt = f"{v:,.2f}" if c % 4 else rng.choice(["North", "South", "East", "West", "Total", "Q1", "Q2", "Q3"])
            d.text((c * cw_ + cw_ - 6 * s - f.getlength(txt), r * rh + 5 * s), txt, font=f, fill="#c00000" if v < 0 and c % 4 else "#202020")
        d.line([(0, r * rh), (ncol * cw_, r * rh)], fill="#e1e1e1")
    for c in range(ncol):
        d.line([(c * cw_, 0), (c * cw_, nrow * rh)], fill="#e1e1e1")
    hdr, rowhdr, tb = int(24 * s), int(44 * s), int(70 * s)
    vy = scroll_profile(rng, n, fps, p["speed"], 0.35, False, canvas.height - h, s)
    vx = scroll_profile(random.Random(rng.random()), n, fps, p["speed"] * 0.6, 0.6, False, canvas.width - w, s)
    for k in range(n):
        frame = Image.new("RGB", (w, h), "#ffffff")
        frame.paste(canvas.crop((vx[k], vy[k], vx[k] + w - rowhdr, vy[k] + h - tb - hdr)), (rowhdr, tb + hdr))
        dd = ImageDraw.Draw(frame)
        dd.rectangle([0, 0, w, tb], fill="#217346")
        dd.text((16 * s, 10 * s), "Book1 - Sheet", font=font(14 * s, bold=True), fill="#ffffff")
        dd.rectangle([0, tb, w, tb + hdr], fill="#f3f3f3")
        dd.rectangle([0, tb, rowhdr, h], fill="#f3f3f3")
        c0, r0 = vx[k] // cw_, vy[k] // rh
        for c in range(c0, c0 + w // cw_ + 2):
            x = rowhdr + c * cw_ - vx[k]
            dd.text((x + cw_ / 2 - 5 * s, tb + 5 * s), chr(65 + c % 26), font=f, fill="#444444")
        for r in range(r0, r0 + h // rh + 2):
            y = tb + hdr + r * rh - vy[k]
            if y > tb + hdr - rh:
                dd.text((6 * s, y + 5 * s), str(r + 1), font=f, fill="#444444")
        sel = (k // max(1, fps // 6)) % 12
        sx, sy = rowhdr + (2 + sel % 4) * cw_ - vx[k] % cw_, tb + hdr + (3 + sel // 4) * rh - vy[k] % rh
        dd.rectangle([sx, sy, sx + cw_, sy + rh], outline="#217346", width=max(2, int(2 * s)))
        yield frame


def scene_map(rng, w, h, fps, n, p, s):
    big = 4
    W, H = w * big, h * big
    img = Image.new("RGB", (W, H), "#f2efe9")
    d = ImageDraw.Draw(img)
    for _ in range(6):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.05, 0.15) * W
        d.ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], fill="#aad3df")
    for _ in range(40):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        d.rectangle([x0, y0, x0 + rng.uniform(60, 300) * s, y0 + rng.uniform(60, 300) * s], fill="#c8facc")
    blk = int(70 * s)
    for gx in range(0, W, blk):
        for gy in range(0, H, blk):
            if rng.random() < 0.7:
                d.rectangle([gx + 6 * s, gy + 6 * s, gx + blk - 6 * s, gy + blk - 6 * s], fill=rng.choice(["#e0dfdf", "#d9d0c9", "#f2dad9"]))
    for gx in range(0, W, blk):
        d.line([(gx, 0), (gx, H)], fill="#ffffff", width=max(2, int(4 * s)))
    for gy in range(0, H, blk):
        d.line([(0, gy), (W, gy)], fill="#ffffff", width=max(2, int(4 * s)))
    for _ in range(8):
        pts = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(5)]
        d.line(pts, fill="#f9b29c", width=int(10 * s))
        d.line(pts, fill="#fcd6a4", width=int(6 * s))
    lf = font(13 * s, bold=True)
    for _ in range(120):
        d.text((rng.uniform(0, W), rng.uniform(0, H)), rng.choice(["Main St", "Park Ave", "Elm Rd", "Harbor", "Station", "Market", "Oak Ln", "Museum"]), font=lf, fill="#555555")
    cx, cy = W / 2, H / 2
    for k in range(n):
        t = k / fps
        zoom = 1.0 + 0.6 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 5.0)) * p["speed"]
        cx += math.cos(t * 0.7) * 6 * s * p["speed"] * 60 / fps
        cy += math.sin(t * 0.5) * 4 * s * p["speed"] * 60 / fps
        vw, vh = w / zoom, h / zoom
        crop = img.crop((int(cx - vw / 2), int(cy - vh / 2), int(cx + vw / 2), int(cy + vh / 2)))
        frame = crop.resize((w, h), Image.BILINEAR)
        dd = ImageDraw.Draw(frame)
        dd.rounded_rectangle([16 * s, 16 * s, 360 * s, 56 * s], radius=int(8 * s), fill="#ffffff", outline="#cccccc")
        dd.text((30 * s, 27 * s), "Search places", font=font(15 * s), fill="#777777")
        yield frame


GEN = {"code": scene_code, "doc": scene_doc, "terminal": scene_terminal, "dashboard": scene_dashboard,
       "windows": scene_windows, "slides": scene_slides, "sheet": scene_sheet, "map": scene_map}


def write_clip(frames, out: Path, w: int, h: int, fps: int) -> int:
    tmp = out.with_suffix(".tmp.mkv")
    enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                            "-i", "-", "-vf", "format=yuv420p", "-c:v", "ffv1", "-level", "3", "-threads", "4", str(tmp)],
                           stdin=subprocess.PIPE)
    n = 0
    for fr in frames:
        enc.stdin.write(np.asarray(fr.convert("RGB"), dtype=np.uint8).tobytes())
        n += 1
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {out}")
    tmp.rename(out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hfr", choices=sorted(SETS))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--res", default="720p,1080p")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    scenes, default_dir = SETS[args.set]
    out_dir = Path(args.out_dir or default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    for sid, scene, tier, fps, params in scenes:
        if only and sid not in only:
            continue
        for res in args.res.split(","):
            w, h, suffix = RES[res]
            cid = sid + suffix
            out = out_dir / f"{cid}.mkv"
            if out.exists():
                continue
            t0 = time.time()
            seed = int.from_bytes(sid.encode(), "little") % (1 << 31)  # same content at both resolutions
            n = int(round(args.seconds * fps))
            frames = GEN[scene](random.Random(seed), w, h, fps, n, params, h / 720.0)
            n = write_clip(frames, out, w, h, fps)
            meta = {"id": cid, "kind": "screen", "intended_tier": tier, "scene": scene, "width": w, "height": h,
                    "fps": float(fps), "frames": n, "params": params, "seed": seed, "native_hfr": fps > 30,
                    "source": "scripts/hfr_screen_gen.py", "licence": LICENCE, "path": str(out),
                    "seconds": round(time.time() - t0, 1)}
            (out_dir / f"{cid}.json").write_text(json.dumps(meta, indent=1))
            with open(out_dir / "manifest.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            print(f"{cid}: {scene} {w}x{h}@{fps} {n} frames, {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
