from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from nanovllm.mock.des_engine import DESConfig
from nanovllm.mock.global_pipeline import ResourceReservation, simulate_global_afd_batches


@dataclass(frozen=True)
class ParetoPoint:
    link_us: float
    tp_g: int
    a_g: int
    ck: int
    gb: int
    isl: int


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    interactivity: float
    tok_s_per_gpu: float
    first_microbatch_ms: float
    effective_batch_ms: float
    wall_time_s: float = 0.0
    notes: str = ""


ScenarioFn = Callable[[ParetoPoint, int, str], ScenarioResult]


def timed_scenario(fn: ScenarioFn, point: ParetoPoint, *, batches: int, gpu_backend: str = "measured") -> ScenarioResult:
    start = time.perf_counter()
    result = fn(point, batches, gpu_backend)
    elapsed = time.perf_counter() - start
    return ScenarioResult(
        name=result.name,
        interactivity=result.interactivity,
        tok_s_per_gpu=result.tok_s_per_gpu,
        first_microbatch_ms=result.first_microbatch_ms,
        effective_batch_ms=result.effective_batch_ms,
        wall_time_s=elapsed,
        notes=result.notes,
    )


def baseline_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Evaluate the saturated global AFD replay baseline."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
    )
    first_ms = replay.first_microbatch_ms
    effective_batch_ms = replay.total_ms / replay.batches
    return ScenarioResult(
        name="afd_global_des",
        interactivity=1000.0 / (first_ms * 36),
        tok_s_per_gpu=replay.tokens * 1000.0 / (replay.total_ms * 36 * point.tp_g),
        first_microbatch_ms=first_ms,
        effective_batch_ms=effective_batch_ms,
        notes=f"batches={batches}",
    )


def context_growth_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Replay decode batches with KV context growing by one token per step."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        context_growth_per_batch=1,
    )
    first_ms = replay.first_microbatch_ms
    effective_batch_ms = replay.total_ms / replay.batches
    return ScenarioResult(
        name="context_growth",
        interactivity=1000.0 / (first_ms * 36),
        tok_s_per_gpu=replay.tokens * 1000.0 / (replay.total_ms * 36 * point.tp_g),
        first_microbatch_ms=first_ms,
        effective_batch_ms=effective_batch_ms,
        notes=f"batches={batches};context_growth_per_batch=1",
    )


def prefill_interference_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Reserve attention resources up front to model decode sharing GPU with prefill."""

    base = baseline_global_des(point, batches, gpu_backend)
    reservation = ResourceReservation(
        stage="decode_attention",
        start_ms=0.0,
        duration_ms=0.10 * base.effective_batch_ms,
    )
    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        reservations=(reservation,),
    )
    first_ms = replay.first_microbatch_ms
    effective_batch_ms = replay.total_ms / replay.batches
    return ScenarioResult(
        name="prefill_interference",
        interactivity=1000.0 / (first_ms * 36),
        tok_s_per_gpu=replay.tokens * 1000.0 / (replay.total_ms * 36 * point.tp_g),
        first_microbatch_ms=first_ms,
        effective_batch_ms=effective_batch_ms,
        notes=f"batches={batches};attention_reservation_ms={reservation.duration_ms:.6f}",
    )


def sparse_arrivals_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Model workload gaps by spacing ready decode batches apart."""

    base = baseline_global_des(point, batches, gpu_backend)
    gap_ms = 0.25 * base.effective_batch_ms
    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        inter_batch_gap_ms=gap_ms,
    )
    return _scenario_from_replay("sparse_arrivals", replay, point, f"inter_batch_gap_ms={gap_ms:.6f}")


def operator_overheads_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Add small per-stage overheads that represent non-aggregated operators."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        stage_add_ms={
            "decode_attention": 0.002,
            "cs_rest": 0.004,
        },
    )
    return _scenario_from_replay("operator_overheads", replay, point, "attention_add=2us;cs_add=4us")


def roofline_backend_global_des(
    point: ParetoPoint,
    batches: int,
    _gpu_backend: str,
) -> ScenarioResult:
    """Use the analytical roofline backend instead of measured GPU fit."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, "roofline"),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
    )
    return _scenario_from_replay("roofline_backend", replay, point, "gpu_backend=roofline")


def collective_contention_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Increase both AFD link stages to mimic collective/link contention."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        stage_scales={
            "gpu_to_cs_link": 1.35,
            "cs_to_gpu_link": 1.35,
        },
    )
    return _scenario_from_replay("collective_contention", replay, point, "link_stage_scale=1.35")


def replica_imbalance_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Model imperfect attention-replica routing/load balance."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        stage_scales={"decode_attention": 1.10},
    )
    return _scenario_from_replay("replica_imbalance", replay, point, "attention_stage_scale=1.10")


def kv_transfer_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Add KV/cache transfer overhead proportional to context length."""

    kv_add_ms = min(0.20, point.isl * 1e-7)
    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        stage_add_ms={
            "gpu_to_cs_link": kv_add_ms,
            "cs_to_gpu_link": kv_add_ms,
        },
    )
    return _scenario_from_replay("kv_transfer", replay, point, f"kv_link_add_ms={kv_add_ms:.6f}")


def runtime_optimizations_global_des(
    point: ParetoPoint,
    batches: int,
    gpu_backend: str,
) -> ScenarioResult:
    """Approximate CUDA-graph/fusion wins with reduced attention and CS stages."""

    replay = simulate_global_afd_batches(
        _base_des_config(point, gpu_backend),
        batch_size=point.gb,
        context_len=point.isl,
        microbatch_size=point.ck,
        batches=batches,
        stage_scales={
            "decode_attention": 0.97,
            "cs_rest": 0.95,
        },
    )
    return _scenario_from_replay("runtime_optimizations", replay, point, "attention_scale=0.97;cs_scale=0.95")


SCENARIOS: tuple[ScenarioFn, ...] = (
    baseline_global_des,
    context_growth_global_des,
    prefill_interference_global_des,
    sparse_arrivals_global_des,
    operator_overheads_global_des,
    roofline_backend_global_des,
    collective_contention_global_des,
    replica_imbalance_global_des,
    kv_transfer_global_des,
    runtime_optimizations_global_des,
)


def _scenario_from_replay(name, replay, point: ParetoPoint, notes: str) -> ScenarioResult:
    first_ms = replay.first_microbatch_ms
    effective_batch_ms = replay.total_ms / replay.batches
    return ScenarioResult(
        name=name,
        interactivity=1000.0 / (first_ms * 36),
        tok_s_per_gpu=replay.tokens * 1000.0 / (replay.total_ms * 36 * point.tp_g),
        first_microbatch_ms=first_ms,
        effective_batch_ms=effective_batch_ms,
        notes=f"batches={replay.batches};{notes}",
    )


def _base_des_config(point: ParetoPoint, gpu_backend: str) -> DESConfig:
    return DESConfig(
        mode="afd",
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        attention_replicas=point.a_g,
        gpu_to_cs_link_resources=1,
        cs_rest_resources=1,
        cs_to_gpu_link_resources=1,
        timing_backend="gptoss_roofline",
        roofline_gpu_backend=gpu_backend,
        roofline_tp_g=point.tp_g,
        gpu_cs_link_us=point.link_us,
        chunk_batch=point.ck,
        des_batch_decode=True,
        des_max_batch_size=point.gb,
    )
