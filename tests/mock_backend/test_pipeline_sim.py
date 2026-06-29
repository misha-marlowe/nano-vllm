import pytest

from nanovllm.engine.pipeline_sim import (
    PipelineStage,
    simulate_discrete_pipeline,
    split_into_microbatches,
    uniform_steady_state_formula_ms,
)


def constant_stage(name, cost_ms):
    return PipelineStage(name, lambda _microbatch_size: cost_ms)


def test_discrete_pipeline_small_m_event_schedule():
    stages = [
        constant_stage("attention", 1.0),
        constant_stage("link", 0.5),
        constant_stage("cs_rest", 2.0),
        constant_stage("return_link", 0.5),
    ]

    result = simulate_discrete_pipeline(stages, [8, 8, 8, 8])

    assert result.total_ms == pytest.approx(10.0)
    assert len(result.events) == 16

    first_mb = [event for event in result.events if event.microbatch_id == 0]
    assert [(event.stage, event.start_ms, event.end_ms) for event in first_mb] == [
        ("attention", 0.0, 1.0),
        ("link", 1.0, 1.5),
        ("cs_rest", 1.5, 3.5),
        ("return_link", 3.5, 4.0),
    ]

    cs_events = [event for event in result.events if event.stage == "cs_rest"]
    assert [(event.start_ms, event.end_ms) for event in cs_events] == [
        (1.5, 3.5),
        (3.5, 5.5),
        (5.5, 7.5),
        (7.5, 9.5),
    ]


def test_variable_microbatch_sizes_change_stage_costs():
    stages = [
        PipelineStage("attention", lambda mb: 0.2 + 0.1 * mb),
        PipelineStage("cs_rest", lambda mb: 0.5 + 0.05 * mb),
    ]

    result = simulate_discrete_pipeline(stages, [8, 8, 2])
    events = result.events

    assert split_into_microbatches(18, 8) == [8, 8, 2]
    assert events[0].microbatch_size == 8
    assert events[-1].microbatch_size == 2
    assert events[0].end_ms - events[0].start_ms == pytest.approx(1.0)
    assert events[-2].end_ms - events[-2].start_ms == pytest.approx(0.4)
    assert result.total_ms == pytest.approx(3.5)


def test_steady_state_formula_is_lower_than_small_m_discrete_pipeline():
    stages = [
        constant_stage("attention", 1.0),
        constant_stage("link", 0.5),
        constant_stage("cs_rest", 2.0),
        constant_stage("return_link", 0.5),
    ]

    explicit = simulate_discrete_pipeline(stages, [8] * 4).total_ms
    ideal = uniform_steady_state_formula_ms(
        stages,
        microbatch_size=8,
        num_microbatches=4,
        num_layers=32,
    )

    assert explicit == pytest.approx(10.0)
    assert ideal == pytest.approx(8.0625)
    assert explicit > ideal


def test_discrete_pipeline_converges_toward_bottleneck_throughput():
    stages = [
        constant_stage("attention", 1.0),
        constant_stage("link", 0.5),
        constant_stage("cs_rest", 2.0),
        constant_stage("return_link", 0.5),
    ]

    small = simulate_discrete_pipeline(stages, [8] * 4).total_ms / 4
    large = simulate_discrete_pipeline(stages, [8] * 128).total_ms / 128

    assert small == pytest.approx(2.5)
    assert large == pytest.approx((4.0 + 127 * 2.0) / 128)
    assert large < small
    assert large == pytest.approx(2.0, abs=0.02)


def test_multi_resource_attention_feeds_shared_cs():
    stages = [
        PipelineStage("attention", lambda _mb: 4.0, resources=2, routing="round_robin"),
        PipelineStage("cs_rest", lambda _mb: 1.0, resources=1),
    ]

    result = simulate_discrete_pipeline(stages, [1, 1, 1, 1])
    attention_events = [event for event in result.events if event.stage == "attention"]
    cs_events = [event for event in result.events if event.stage == "cs_rest"]

    assert [(event.resource_id, event.start_ms, event.end_ms) for event in attention_events] == [
        (0, 0.0, 4.0),
        (1, 0.0, 4.0),
        (0, 4.0, 8.0),
        (1, 4.0, 8.0),
    ]
    assert [(event.start_ms, event.end_ms) for event in cs_events] == [
        (4.0, 5.0),
        (5.0, 6.0),
        (8.0, 9.0),
        (9.0, 10.0),
    ]
    assert result.total_ms == pytest.approx(10.0)
