"""rc10 slow-preset cap: near the predicted VMAF ceiling only presets >= 4 compete (both model stages)."""
from streamenc.ml import crf_predictor as cp


def test_default_on_at_98_5():
    assert cp.SLOW_CAP_VMAF == 98.5 and cp.SLOW_CAP_MIN_PRESET == 4


def test_preset_parse():
    assert [cp._arm_preset(a) for a in ("svt_crf_p2_qm_la", "svt_crf_p3_t0_qm_la", "svt_crf_p2_qm_la_lqb_h3",
                                        "svt_crf_p4_qm_la_lqb", "svt_crf_p12_t1")] == [2, 3, 2, 4, 12]


def test_cap_drops_slow_arms_only_near_the_ceiling():
    u = {"svt_crf_p2_qm_la": 99.0, "svt_crf_p3_qm_la": 98.9, "svt_crf_p4_qm_la": 98.8, "svt_crf_p6_qm_la": 98.5}
    assert set(cp._slow_cap(u, 98.5)) == {"svt_crf_p4_qm_la", "svt_crf_p6_qm_la"}
    assert cp._slow_cap(u, 98.49) == u                       # below the cap: unchanged
    only_slow = {"svt_crf_p2_qm_la": 99.0}
    assert cp._slow_cap(only_slow, 99.5) == only_slow        # never leaves the table empty
