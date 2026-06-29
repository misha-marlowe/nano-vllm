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


def simulate_global_afd_batches(
    config: DESConfig,
    *,
    batch_size: int,
    context_len: int,
    microbatch_size: int,
    batches: int,
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

    timing = build_timing_backend(config)
    attention_available = [0.0] * config.attention_replicas
    gpu_to_cs_available = [0.0] * config.gpu_to_cs_link_resources
    cs_rest_available = [0.0] * config.cs_rest_resources
    cs_to_gpu_available = [0.0] * config.cs_to_gpu_link_resources
    microbatch_sizes = split_into_microbatches(batch_size, microbatch_size)

    first_microbatch_ms: float | None = None
    total_ms = 0.0
    microbatch_count = 0

    for _batch_id in range(batches):
        for microbatch_id, size in enumerate(microbatch_sizes):
            timings = timing.afd_decode_stages_ms(size, context_len)

            attn_resource = microbatch_id % len(attention_available)
            attn_start = attention_available[attn_resource]
            attn_end = attn_start + timings.attention_ms
            attention_available[attn_resource] = attn_end

            out_link_resource = _earliest(gpu_to_cs_available)
            out_start = max(attn_end, gpu_to_cs_available[out_link_resource])
            out_end = out_start + timings.gpu_to_cs_link_ms
            gpu_to_cs_available[out_link_resource] = out_end

            cs_resource = _earliest(cs_rest_available)
            cs_start = max(out_end, cs_rest_available[cs_resource])
            cs_end = cs_start + timings.cs_rest_ms
            cs_rest_available[cs_resource] = cs_end

            ret_link_resource = _earliest(cs_to_gpu_available)
            ret_start = max(cs_end, cs_to_gpu_available[ret_link_resource])
            ret_end = ret_start + timings.cs_to_gpu_link_ms
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
