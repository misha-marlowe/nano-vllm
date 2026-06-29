from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from nanovllm.mock.des_engine import DESConfig
from nanovllm.mock.global_pipeline import simulate_global_afd_batches


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
