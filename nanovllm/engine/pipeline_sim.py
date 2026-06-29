from dataclasses import dataclass
from typing import Callable


StageCostFn = Callable[[int], float]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    cost_ms: StageCostFn


@dataclass(frozen=True)
class PipelineEvent:
    microbatch_id: int
    microbatch_size: int
    stage: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class PipelineSimulationResult:
    total_ms: float
    events: list[PipelineEvent]


def simulate_discrete_pipeline(
    stages: list[PipelineStage],
    microbatch_sizes: list[int],
) -> PipelineSimulationResult:
    """Simulate an in-order resource pipeline for small microbatch counts.

    Each stage has one resource. Microbatch j can enter stage i only after:
    - microbatch j has completed stage i - 1, and
    - stage i has finished microbatch j - 1.

    This is intentionally a conservative event-level reference for small M. It
    does not assume cross-layer steady state or amortized fill/drain bubbles.
    """

    if not stages:
        raise ValueError("at least one pipeline stage is required")
    if not microbatch_sizes:
        return PipelineSimulationResult(total_ms=0.0, events=[])
    if any(size <= 0 for size in microbatch_sizes):
        raise ValueError("microbatch sizes must be positive")

    stage_available_ms = [0.0] * len(stages)
    events: list[PipelineEvent] = []

    for microbatch_id, microbatch_size in enumerate(microbatch_sizes):
        ready_ms = 0.0
        for stage_idx, stage in enumerate(stages):
            cost_ms = stage.cost_ms(microbatch_size)
            if cost_ms < 0:
                raise ValueError(f"stage {stage.name} returned negative cost")
            start_ms = max(ready_ms, stage_available_ms[stage_idx])
            end_ms = start_ms + cost_ms
            events.append(PipelineEvent(
                microbatch_id=microbatch_id,
                microbatch_size=microbatch_size,
                stage=stage.name,
                start_ms=start_ms,
                end_ms=end_ms,
            ))
            ready_ms = end_ms
            stage_available_ms[stage_idx] = end_ms

    return PipelineSimulationResult(
        total_ms=max(stage_available_ms),
        events=events,
    )


def uniform_steady_state_formula_ms(
    stages: list[PipelineStage],
    microbatch_size: int,
    num_microbatches: int,
    num_layers: int,
) -> float:
    """Compute the steady-state formula for uniform microbatches.

    t_pipe = M * s_max + sum_{i != argmax} s_i / L

    This helper is for validating the formula path. For small M or non-uniform
    microbatch sizes, prefer `simulate_discrete_pipeline`.
    """

    if not stages:
        raise ValueError("at least one pipeline stage is required")
    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    if num_microbatches < 0:
        raise ValueError("num_microbatches must be non-negative")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_microbatches == 0:
        return 0.0

    costs = [stage.cost_ms(microbatch_size) for stage in stages]
    if any(cost < 0 for cost in costs):
        raise ValueError("stage costs must be non-negative")
    max_idx = max(range(len(costs)), key=lambda idx: costs[idx])
    s_max = costs[max_idx]
    bubble_ms = sum(cost for idx, cost in enumerate(costs) if idx != max_idx) / num_layers
    return num_microbatches * s_max + bubble_ms


def split_into_microbatches(batch_size: int, microbatch_size: int) -> list[int]:
    if batch_size < 0:
        raise ValueError("batch size must be non-negative")
    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    full, remainder = divmod(batch_size, microbatch_size)
    sizes = [microbatch_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes
