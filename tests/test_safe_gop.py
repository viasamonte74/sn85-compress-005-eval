from streamenc.probe.ffprobe_meta import gop_for_timestamps


def _spans(pts, fps, n):
    keys = pts[::n]
    end = pts[-1] + 1.0 / fps
    return [b - a for a, b in zip(keys, keys[1:])] + [end - keys[-1]]


def test_cfr_keeps_nominal_gop():
    for fps in (23.976, 24, 25, 29.97, 30, 50, 59.94, 60):
        pts = [round(i / fps, 3) for i in range(int(5 * fps))]  # 1 ms timebase rounding
        assert gop_for_timestamps(pts, fps) == int(2 * fps + 1e-9)


def test_vfr_gaps_shrink_gop_within_official_limit():
    fps = 30000 / 1001
    pts, t = [], 0.0
    for i in range(121):  # 121 frames over ~5 s: every 5th frame doubled in duration
        pts.append(round(t, 3))
        t += (2.0 if i % 5 == 0 else 1.0) / fps * 1.25
    n = gop_for_timestamps(pts, fps)
    assert n < int(2 * fps)
    assert max(_spans(pts, fps, n)) <= 2.0 + 0.5 / fps
