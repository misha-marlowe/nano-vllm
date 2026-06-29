import pytest

from nanovllm.mock.frontier_gap_scenarios import (
    ParetoPoint,
    baseline_global_des,
    context_growth_global_des,
    timed_scenario,
)


def small_point() -> ParetoPoint:
    return ParetoPoint(link_us=12.0, tp_g=1, a_g=1, ck=4, gb=16, isl=8192)


def test_baseline_global_des_reports_positive_curve_point():
    result = baseline_global_des(small_point(), 4, "measured")

    assert result.name == "afd_global_des"
    assert result.interactivity > 0
    assert result.tok_s_per_gpu > 0
    assert result.first_microbatch_ms > 0
    assert result.effective_batch_ms > 0


def test_baseline_global_des_amortizes_more_batches():
    one = baseline_global_des(small_point(), 1, "measured")
    many = baseline_global_des(small_point(), 8, "measured")

    assert many.tok_s_per_gpu >= one.tok_s_per_gpu
    assert many.first_microbatch_ms == pytest.approx(one.first_microbatch_ms)


def test_timed_scenario_records_wall_time():
    result = timed_scenario(baseline_global_des, small_point(), batches=2)

    assert result.wall_time_s >= 0
    assert result.tok_s_per_gpu > 0


def test_context_growth_never_improves_over_constant_context():
    baseline = baseline_global_des(small_point(), 8, "measured")
    grown = context_growth_global_des(small_point(), 8, "measured")

    assert grown.tok_s_per_gpu <= baseline.tok_s_per_gpu
    assert grown.first_microbatch_ms == pytest.approx(baseline.first_microbatch_ms)
    assert "context_growth_per_batch=1" in grown.notes
