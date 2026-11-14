"""1.0.0rc13: the VMAF-NEG gap is measured when the gap model cannot vouch for the shipped encode."""
import importlib
import inspect


def test_neg_check_unknown_default_on(monkeypatch):
    monkeypatch.delenv("STREAMENC_NEG_CHECK_UNKNOWN", raising=False)
    import streamenc.encode.crf_pipeline as cp
    cp = importlib.reload(cp)
    assert cp.NEG_CHECK_UNKNOWN is True


def test_neg_check_unknown_switch_off(monkeypatch):
    monkeypatch.setenv("STREAMENC_NEG_CHECK_UNKNOWN", "0")
    import streamenc.encode.crf_pipeline as cp
    cp = importlib.reload(cp)
    assert cp.NEG_CHECK_UNKNOWN is False
    monkeypatch.delenv("STREAMENC_NEG_CHECK_UNKNOWN")
    importlib.reload(cp)


def test_guard_measures_unknown_or_filtered():
    import streamenc.encode.crf_pipeline as cp
    src = inspect.getsource(cp)
    assert "unknown_ = NEG_CHECK_UNKNOWN and (p_fail is None or bool(hard_vf))" in src
    assert "if unknown_ or (p_fail is not None and p_fail >= NEG_CHECK_P):" in src
