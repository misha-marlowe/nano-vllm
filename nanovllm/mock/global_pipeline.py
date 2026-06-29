from dataclasses import dataclass

from nanovllm.engine.pipeline_sim import split_into_microbatches
from nanovllm.mock.des_engine import DESConfig
from nanovllm.mock.timing import build_timing_backend


@dataclass(frozen=True)
class GlobalAFDReplayResult:
    """Summary for saturated cross-batch AFD pipeline replay.

    `first_microbatch_ms` is the round-trip latency of the first microbatch in
    the first batch. `total_ms` is the time to drain all replayed batches.
    """

    first_microbatch_ms: float
    total_ms: float
    batches: int
    tokens: int
    microbatches: int


@dataclass(frozen=True)
class ResourceReservation:
    stage: str
    start_ms: float
    duration_ms: float
    resource_id: int = 0


def simulate_global_afd_batches(
    config: DESConfig,
    *,
    batch_size: int,
    context_len: int,
    microbatch_size: int,
    batches: int,
    context_growth_per_batch: int = 0,
    reservations: tuple[ResourceReservation, ...] = (),
    stage_scales: dict[str, float] | None = None,
    stage_add_ms: dict[str, float] | None = None,
    inter_batch_gap_ms: float = 0.0,
) -> GlobalAFDReplayResult:
    """Replay many AFD decode batches against persistent stage resources.

    The batch-scoped DES path resets the local pipeline for each scheduled
    batch and returns only after that batch drains. This helper instead keeps
    attention, link, and CS resources live across a sequence of ready batches.
    It models the saturated case where the next batch can feed GPU attention
    while earlier microbatches are still draining through link/CS resources.
    """

    if config.mode != "afd":
        raise ValueError("global AFD replay requires DESConfig(mode='afd')")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    if batches <= 0:
        raise ValueError("batches must be positive")
    if context_growth_per_batch < 0:
        raise ValueError("context_growth_per_batch must be non-negative")
    if inter_batch_gap_ms < 0:
        raise ValueError("inter_batch_gap_ms must be non-negative")

    timing = build_timing_backend(config)
    stage_scales = stage_scales or {}
    stage_add_ms = stage_add_ms or {}
    for stage, scale in stage_scales.items():
        if scale < 0:
            raise ValueError(f"stage scale for {stage} must be non-negative")
    for stage, add_ms in stage_add_ms.items():
        if add_ms < 0:
            raise ValueError(f"stage additive time for {stage} must be non-negative")
    attention_available = [0.0] * config.attention_replicas
    gpu_to_cs_available = [0.0] * config.gpu_to_cs_link_resources
    cs_rest_available = [0.0] * config.cs_rest_resources
    cs_to_gpu_available = [0.0] * config.cs_to_gpu_link_resources
    _apply_reservations(
        reservations,
        attention_available=attention_available,
        gpu_to_cs_available=gpu_to_cs_available,
        cs_rest_available=cs_rest_available,
        cs_to_gpu_available=cs_to_gpu_available,
    )
    microbatch_sizes = split_into_microbatches(batch_size, microbatch_size)

    first_microbatch_ms: float | None = None
    total_ms = 0.0
    microbatch_count = 0

    for batch_id in range(batches):
        batch_context_len = context_len + batch_id * context_growth_per_batch
        batch_ready_ms = batch_id * inter_batch_gap_ms
        for microbatch_id, size in enumerate(microbatch_sizes):
            timings = timing.afd_decode_stages_ms(size, batch_context_len)

            attn_resource = microbatch_id % len(attention_available)
            attn_start = max(batch_ready_ms, attention_available[attn_resource])
            attn_end = attn_start + _duration("decode_attention", timings.attention_ms, stage_scales, stage_add_ms)
            attention_available[attn_resource] = attn_end

            out_link_resource = _earliest(gpu_to_cs_available)
            out_start = max(attn_end, gpu_to_cs_available[out_link_resource])
            out_end = out_start + _duration("gpu_to_cs_link", timings.gpu_to_cs_link_ms, stage_scales, stage_add_ms)
            gpu_to_cs_available[out_link_resource] = out_end

            cs_resource = _earliest(cs_rest_available)
            cs_start = max(out_end, cs_rest_available[cs_resource])
            cs_end = cs_start + _duration("cs_rest", timings.cs_rest_ms, stage_scales, stage_add_ms)
            cs_rest_available[cs_resource] = cs_end

            ret_link_resource = _earliest(cs_to_gpu_available)
            ret_start = max(cs_end, cs_to_gpu_available[ret_link_resource])
            ret_end = ret_start + _duration("cs_to_gpu_link", timings.cs_to_gpu_link_ms, stage_scales, stage_add_ms)
            cs_to_gpu_available[ret_link_resource] = ret_end

            if first_microbatch_ms is None:
                first_microbatch_ms = ret_end
            total_ms = max(total_ms, ret_end)
            microbatch_count += 1

    return GlobalAFDReplayResult(
        first_microbatch_ms=first_microbatch_ms or 0.0,
        total_ms=total_ms,
        batches=batches,
        tokens=batch_size * batches,
        microbatches=microbatch_count,
    )


def _earliest(available_ms: list[float]) -> int:
    return min(range(len(available_ms)), key=lambda idx: (available_ms[idx], idx))


def _duration(stage: str, base_ms: float, scales: dict[str, float], adds: dict[str, float]) -> float:
    return base_ms * scales.get(stage, 1.0) + adds.get(stage, 0.0)


def _apply_reservations(
    reservations: tuple[ResourceReservation, ...],
    *,
    attention_available: list[float],
    gpu_to_cs_available: list[float],
    cs_rest_available: list[float],
    cs_to_gpu_available: list[float],
):
    pools = {
        "decode_attention": attention_available,
        "gpu_to_cs_link": gpu_to_cs_available,
        "cs_rest": cs_rest_available,
        "cs_to_gpu_link": cs_to_gpu_available,
    }
    for reservation in reservations:
        pool = pools.get(reservation.stage)
        if pool is None:
            raise ValueError(f"unknown reservation stage {reservation.stage}")
        if reservation.duration_ms < 0:
            raise ValueError("reservation duration must be non-negative")
        resource_id = reservation.resource_id % len(pool)
        pool[resource_id] = max(pool[resource_id], reservation.start_ms + reservation.duration_ms)
