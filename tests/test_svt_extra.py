"""rc9 clean-content SVT tools: merged into final arms only, never into the p8 pilot / p12 speed arms."""
from streamenc.encode import crf_pipeline as cp
from streamenc.encoders.arms import build_commands


def test_default_is_chroma_qp_only():
    assert cp.SVT_EXTRA == ["chq63"] and cp.SVT_EXTRA_CLEAN == []


def test_final_arm_gets_chroma_qp_and_pilot_does_not():
    fin = cp._arm("svt_crf_p4_qm_la", ["chq63"])["svt"]
    assert fin["use-fixed-qindex-offsets"] == 2 and fin["chroma-qindex-offsets"] == "63,63,63,63,63,63"
    assert fin["key-frame-chroma-qindex-offset"] == 63 and fin["lookahead"] == 120
    for a in ("svt_crf_p8_t1", "svt_crf_p12_t1"):
        assert "use-fixed-qindex-offsets" not in cp._arm(a, ["chq63"])["svt"]
    assert "use-fixed-qindex-offsets" not in cp._arm("svt_crf_p4_qm_la", [])["svt"]


def test_command_line_and_registry_untouched():
    from streamenc.encoders.arms import get_arm
    before = dict(get_arm("svt_crf_p4_qm_la")["svt"])
    cmd = build_commands(cp._arm("svt_crf_p4_qm_la", ["chq63"]), "in.mkv", "out.mp4", 1000.0, 48, "/tmp/x", None, crf=40)[0]
    params = cmd[cmd.index("-svtav1-params") + 1]
    assert "use-fixed-qindex-offsets=2" in params and "chroma-qindex-offsets=63,63,63,63,63,63" in params
    assert get_arm("svt_crf_p4_qm_la")["svt"] == before
