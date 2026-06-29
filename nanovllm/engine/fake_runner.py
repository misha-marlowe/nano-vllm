from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.pipeline_sim import (
    PipelineStage,
    simulate_discrete_pipeline,
    split_into_microbatches,
)


class RunnerStageEvent:

    def __init__(self, stage: str, start_ms: float, end_ms: float, notes: str = ""):
        self.stage = stage
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.notes = notes


class FakeColocatedRunner:
    """CPU-only runner that preserves the engine/scheduler contract.

    The real runner returns one sampled token per scheduled sequence. This
    runner does the same without tensors, CUDA state, model weights, or sleeps.
    The engine consumes `last_latency_ms` to advance virtual time and write
    traces.
    """

    def __init__(self, config: Config):
        self.config = config
        self.last_latency_ms = 0.0
        self.last_stage_events: list[RunnerStageEvent] = []

    def call(self, method_name, *args):
        method = getattr(self, method_name)
        return method(*args)

    def exit(self):
        return None

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.last_stage_events = []
        batch_size = len(seqs)
        if is_prefill:
            isl = max(seq.num_scheduled_tokens for seq in seqs)
            self.last_latency_ms = (
                self.config.prefill_base_ms
                + isl * self.config.prefill_ms_per_token * batch_size
            )
        else:
            self.last_latency_ms = (
                self.config.decode_base_ms
                + self.config.decode_ms_per_token * batch_size
            )
        return [self._next_token_id(seq) for seq in seqs]

    def _next_token_id(self, seq: Sequence) -> int:
        return self.config.mock_token_base + seq.seq_id * 100000 + seq.num_completion_tokens + 1


class FakeAFDRunner(FakeColocatedRunner):
    """Fake AFD runner with GPU-attention/link/CS/link decode timing."""

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.last_stage_events = []
        if is_prefill:
            return super().run(seqs, is_prefill)
        if self.config.pipeline_mode == "sequential":
            self._run_sequential_decode(seqs)
        elif self.config.pipeline_mode == "ideal_pipeline":
            self._run_ideal_pipeline_decode(seqs)
        else:
            self._run_discrete_pipeline_decode(seqs)
        return [self._next_token_id(seq) for seq in seqs]

    def _attention_ms(self, microbatch_size: int, context_len: int) -> float:
        return (
            self.config.attention_ms_base
            + self.config.attention_ms_per_token * microbatch_size
            + self.config.attention_ms_per_isl_token * context_len * microbatch_size
        )

    def _cs_rest_ms(self, microbatch_size: int) -> float:
        return (
            self.config.cs_rest_ms_base
            + self.config.cs_rest_ms_per_token * microbatch_size
        )

    def _stage_defs(self, context_len: int) -> list[PipelineStage]:
        return [
            PipelineStage(
                "decode_attention",
                lambda mb: self._attention_ms(mb, context_len),
                resources=self.config.attention_replicas,
                routing="round_robin",
            ),
            PipelineStage(
                "gpu_to_cs_link",
                lambda _mb: self.config.link_ms_one_way,
                resources=self.config.gpu_to_cs_link_resources,
            ),
            PipelineStage(
                "cs_rest",
                self._cs_rest_ms,
                resources=self.config.cs_rest_resources,
            ),
            PipelineStage(
                "cs_to_gpu_link",
                lambda _mb: self.config.link_ms_one_way,
                resources=self.config.cs_to_gpu_link_resources,
            ),
        ]

    def _run_sequential_decode(self, seqs: list[Sequence]):
        batch_size = len(seqs)
        context_len = max(len(seq) for seq in seqs)
        stage_durations = [
            ("decode_attention", self._attention_ms(batch_size, context_len)),
            ("gpu_to_cs_link", self.config.link_ms_one_way),
            ("cs_rest", self._cs_rest_ms(batch_size)),
            ("cs_to_gpu_link", self.config.link_ms_one_way),
        ]
        cursor = 0.0
        for stage, duration in stage_durations:
            self.last_stage_events.append(RunnerStageEvent(f"{stage}_start", cursor, cursor))
            self.last_stage_events.append(RunnerStageEvent(f"{stage}_end", cursor + duration, cursor + duration))
            cursor += duration
        self.last_latency_ms = cursor

    def _run_discrete_pipeline_decode(self, seqs: list[Sequence]):
        context_len = max(len(seq) for seq in seqs)
        microbatch_sizes = split_into_microbatches(len(seqs), self.config.microbatch_size)
        result = simulate_discrete_pipeline(self._stage_defs(context_len), microbatch_sizes)
        for event in result.events:
            notes = (
                f"microbatch={event.microbatch_id};"
                f"microbatch_size={event.microbatch_size};"
                f"resource={event.stage}_{event.resource_id}"
            )
            self.last_stage_events.append(RunnerStageEvent(
                f"{event.stage}_start",
                event.start_ms,
                event.start_ms,
                notes,
            ))
            self.last_stage_events.append(RunnerStageEvent(
                f"{event.stage}_end",
                event.end_ms,
                event.end_ms,
                notes,
            ))
        self.last_latency_ms = result.total_ms

    def _run_ideal_pipeline_decode(self, seqs: list[Sequence]):
        context_len = max(len(seq) for seq in seqs)
        microbatch_sizes = split_into_microbatches(len(seqs), self.config.microbatch_size)
        stage_defs = self._stage_defs(context_len)
        per_mb_costs = [
            [stage.cost_ms(size) for stage in stage_defs]
            for size in microbatch_sizes
        ]
        if len(microbatch_sizes) == 1:
            cursor = 0.0
            for stage, duration in zip(stage_defs, per_mb_costs[0]):
                self.last_stage_events.append(RunnerStageEvent(
                    f"{stage.name}_start",
                    cursor,
                    cursor,
                    "mode=ideal_pipeline;single_microbatch_round_trip",
                ))
                cursor += duration
                self.last_stage_events.append(RunnerStageEvent(
                    f"{stage.name}_end",
                    cursor,
                    cursor,
                    "mode=ideal_pipeline;single_microbatch_round_trip",
                ))
            self.last_latency_ms = cursor
            return
        bottleneck_ms = sum(max(costs) for costs in per_mb_costs)
        max_stage_costs = [
            max(costs[stage_idx] for costs in per_mb_costs)
            for stage_idx in range(len(stage_defs))
        ]
        bottleneck_stage_idx = max(range(len(max_stage_costs)), key=lambda idx: max_stage_costs[idx])
        bubble_ms = (
            sum(cost for idx, cost in enumerate(max_stage_costs) if idx != bottleneck_stage_idx)
            / self.config.num_layers
        )
        self.last_latency_ms = bottleneck_ms + bubble_ms

        cursor = 0.0
        self.last_stage_events.append(RunnerStageEvent(
            "pipeline_fill_start",
            0.0,
            0.0,
            f"microbatches={len(microbatch_sizes)};mode=ideal_pipeline",
        ))
        self.last_stage_events.append(RunnerStageEvent(
            "pipeline_steady_state_end",
            bottleneck_ms,
            bottleneck_ms,
            f"bottleneck_stage={stage_defs[bottleneck_stage_idx].name}",
        ))
        cursor = bottleneck_ms
        if bubble_ms:
            self.last_stage_events.append(RunnerStageEvent("pipeline_drain_start", cursor, cursor))
            self.last_stage_events.append(RunnerStageEvent("pipeline_drain_end", self.last_latency_ms, self.last_latency_ms))
