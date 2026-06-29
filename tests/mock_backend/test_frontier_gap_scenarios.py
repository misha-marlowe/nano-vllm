import pytest

from nanovllm.mock.frontier_gap_scenarios import ParetoPoint, baseline_global_des


def small_point() -> ParetoPoint:
    return ParetoPoint(link_us=12.0, tp_g=1, a_g=1, ck=4, gb=16, isl=8192)


def test_baseline_global_des_reports_positive_curve_point():
    result = baseline_global_des(small_point(), batches=4)

    assert result.name == "afd_global_des"
    assert result.interactivity > 0
    assert result.tok_s_per_gpu > 0
    assert result.first_microbatch_ms > 0
    assert result.effective_batch_ms > 0


def test_baseline_global_des_amortizes_more_batches():
    one = baseline_global_des(small_point(), batches=1)
    many = baseline_global_des(small_point(), batches=8)

    assert many.tok_s_per_gpu >= one.tok_s_per_gpu
    assert many.first_microbatch_ms == pytest.approx(one.first_microbatch_ms)
