import pytest

from nanovllm.config import Config
from nanovllm.mock.timing import build_timing_backend
from nanovllm.mock.timing.ac_model import cs4_offload as CS4M
from nanovllm.mock.timing.gptoss_roofline import HELIOS, GPTOSS, gpu_only_point, hybrid_point


def test_parametric_timing_backend_preserves_existing_formulas():
    config = Config(
        "__mock__",
        mock_backend=True,
        prefill_base_ms=1.5,
        prefill_ms_per_token=0.25,
        decode_base_ms=0.7,
        decode_ms_per_token=0.05,
        attention_ms_base=0.4,
        attention_ms_per_token=0.02,
        attention_ms_per_isl_token=0.001,
        cs_rest_ms_base=0.6,
        cs_rest_ms_per_token=0.03,
        link_ms_one_way=0.2,
    )
    backend = build_timing_backend(config)

    assert backend.prefill_ms(batch_size=3, isl=10) == pytest.approx(9.0)
    assert backend.colocated_decode_ms(batch_size=4, context_len=128) == pytest.approx(0.9)

    stages = backend.afd_decode_stages_ms(microbatch_size=2, context_len=100)
    assert stages.attention_ms == pytest.approx(0.64)
    assert stages.gpu_to_cs_link_ms == pytest.approx(0.2)
    assert stages.cs_rest_ms == pytest.approx(0.66)
    assert stages.cs_to_gpu_link_ms == pytest.approx(0.2)


def test_gptoss_roofline_backend_attention_is_monotonic_with_context():
    config = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="afd",
        timing_backend="gptoss_roofline",
        roofline_gpu_backend="measured",
        roofline_tp_g=1,
    )
    backend = build_timing_backend(config)

    short = backend.afd_decode_stages_ms(microbatch_size=4, context_len=8192)
    long = backend.afd_decode_stages_ms(microbatch_size=4, context_len=131072)

    assert long.attention_ms > short.attention_ms
    assert "timing_backend=gptoss_roofline" in short.notes


def test_gptoss_rawdata_regression_points_match_section5_8k():
    gpu = gpu_only_point(HELIOS, GPTOSS, B=256, isl=8192, tp_g=1, backend="measured")
    assert gpu["x"] == pytest_approx_pct(163.7, rel=0.005)
    assert gpu["y"] == pytest_approx_pct(41896.2, rel=0.005)

    old_link = CS4M.CLOS_LAT_US
    CS4M.CLOS_LAT_US = 12.0
    try:
        hybrid = hybrid_point(HELIOS, GPTOSS, gb=256, isl=8192, tp_g=1, a_g=1, ck=128, backend="measured")
    finally:
        CS4M.CLOS_LAT_US = old_link
    assert hybrid["x"] == pytest_approx_pct(327.8, rel=0.005)
    assert hybrid["y"] == pytest_approx_pct(83920.8, rel=0.005)


def test_gptoss_link_latency_changes_interactivity_not_pipeline_filled_throughput():
    old_link = CS4M.CLOS_LAT_US
    try:
        CS4M.CLOS_LAT_US = 4.0
        fast = hybrid_point(HELIOS, GPTOSS, gb=256, isl=8192, tp_g=1, a_g=1, ck=64, backend="measured")
        CS4M.CLOS_LAT_US = 36.0
        slow = hybrid_point(HELIOS, GPTOSS, gb=256, isl=8192, tp_g=1, a_g=1, ck=64, backend="measured")
    finally:
        CS4M.CLOS_LAT_US = old_link

    assert fast["x"] > slow["x"]
    assert fast["y"] == pytest_approx_pct(slow["y"], rel=0.0001)


def pytest_approx_pct(expected, rel):
    return pytest.approx(expected, rel=rel)
