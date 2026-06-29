import pytest

from nanovllm.mock.frontier_gap_scenarios import (
    ParetoPoint,
    baseline_global_des,
    collective_contention_global_des,
    context_growth_global_des,
    kv_transfer_global_des,
    operator_overheads_global_des,
    prefill_interference_global_des,
    replica_imbalance_global_des,
    roofline_backend_global_des,
    runtime_optimizations_global_des,
    sparse_arrivals_global_des,
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


def test_prefill_interference_reduces_throughput_and_interactivity():
    baseline = baseline_global_des(small_point(), 8, "measured")
    interfered = prefill_interference_global_des(small_point(), 8, "measured")

    assert interfered.tok_s_per_gpu < baseline.tok_s_per_gpu
    assert interfered.interactivity < baseline.interactivity
    assert "attention_reservation_ms" in interfered.notes


@pytest.mark.parametrize(
    "scenario",
    [
        sparse_arrivals_global_des,
        operator_overheads_global_des,
        collective_contention_global_des,
        replica_imbalance_global_des,
        kv_transfer_global_des,
    ],
)
def test_degradation_scenarios_do_not_improve_throughput(scenario):
    baseline = baseline_global_des(small_point(), 8, "measured")
    result = scenario(small_point(), 8, "measured")

    assert result.tok_s_per_gpu <= baseline.tok_s_per_gpu


def test_runtime_optimization_scenario_improves_throughput():
    baseline = baseline_global_des(small_point(), 8, "measured")
    optimized = runtime_optimizations_global_des(small_point(), 8, "measured")

    assert optimized.tok_s_per_gpu >= baseline.tok_s_per_gpu
    assert optimized.interactivity >= baseline.interactivity


def test_roofline_backend_scenario_runs_and_is_named():
    result = roofline_backend_global_des(small_point(), 4, "measured")

    assert result.name == "roofline_backend"
    assert result.tok_s_per_gpu > 0
    assert "gpu_backend=roofline" in result.notes
