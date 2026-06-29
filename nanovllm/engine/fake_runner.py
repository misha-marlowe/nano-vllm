from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.pipeline_sim import (
    PipelineStage,
    simulate_discrete_pipeline,
    split_into_microbatches,
)
from nanovllm.mock import DESConfig, DESEngine, DESRequest
from nanovllm.mock.timing import build_timing_backend


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
        self.timing = build_timing_backend(config)
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
            self.last_latency_ms = self.timing.prefill_ms(batch_size, isl)
        else:
            context_len = max(len(seq) for seq in seqs)
            self.last_latency_ms = self.timing.colocated_decode_ms(batch_size, context_len)
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
        return self.timing.afd_decode_stages_ms(microbatch_size, context_len).attention_ms

    def _cs_rest_ms(self, microbatch_size: int) -> float:
        return self.timing.afd_decode_stages_ms(microbatch_size, 1).cs_rest_ms

    def _stage_durations(self, microbatch_size: int, context_len: int):
        return self.timing.afd_decode_stages_ms(microbatch_size, context_len)

    def _stage_defs(self, context_len: int) -> list[PipelineStage]:
        return [
            PipelineStage(
                "decode_attention",
                lambda mb: self._stage_durations(mb, context_len).attention_ms,
                resources=self.config.attention_replicas,
                routing="round_robin",
            ),
            PipelineStage(
                "gpu_to_cs_link",
                lambda mb: self._stage_durations(mb, context_len).gpu_to_cs_link_ms,
                resources=self.config.gpu_to_cs_link_resources,
            ),
            PipelineStage(
                "cs_rest",
                lambda mb: self._stage_durations(mb, context_len).cs_rest_ms,
                resources=self.config.cs_rest_resources,
            ),
            PipelineStage(
                "cs_to_gpu_link",
                lambda mb: self._stage_durations(mb, context_len).cs_to_gpu_link_ms,
                resources=self.config.cs_to_gpu_link_resources,
            ),
        ]

    def _run_sequential_decode(self, seqs: list[Sequence]):
        batch_size = len(seqs)
        context_len = max(len(seq) for seq in seqs)
        timings = self._stage_durations(batch_size, context_len)
        stage_durations = [
            ("decode_attention", timings.attention_ms),
            ("gpu_to_cs_link", timings.gpu_to_cs_link_ms),
            ("cs_rest", timings.cs_rest_ms),
            ("cs_to_gpu_link", timings.cs_to_gpu_link_ms),
        ]
        cursor = 0.0
        for stage, duration in stage_durations:
            self.last_stage_events.append(RunnerStageEvent(f"{stage}_start", cursor, cursor, timings.notes))
            self.last_stage_events.append(RunnerStageEvent(f"{stage}_end", cursor + duration, cursor + duration, timings.notes))
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


class FakeDESRunner(FakeColocatedRunner):
    """nano-vLLM runner that delegates scheduled decode batches to DESEngine.

    This keeps the normal mock ``LLMEngine.step() -> Scheduler.schedule()``
    front-end, then uses the DES resource model for decode timing and traceable
    stage events.
    """

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.last_stage_events = []
        if is_prefill:
            return super().run(seqs, is_prefill)
        rows = self._run_des_decode(seqs)
        start_ms = self._decode_start_ms(rows)
        end_ms = max(float(row["virtual_time_ms"]) for row in rows if row["stage"] == "token_emit")
        self.last_latency_ms = end_ms - start_ms
        self._record_stage_events(rows, start_ms)
        return [self._next_token_id(seq) for seq in seqs]

    def _run_des_decode(self, seqs: list[Sequence]) -> list[dict[str, str]]:
        config = DESConfig(
            mode=self.config.mock_mode,
            prefill_base_ms=0.0,
            prefill_ms_per_token=0.0,
            decode_base_ms=self.config.decode_base_ms,
            decode_ms_per_token=self.config.decode_ms_per_token,
            attention_ms_base=self.config.attention_ms_base,
            attention_ms_per_token=self.config.attention_ms_per_token,
            attention_ms_per_isl_token=self.config.attention_ms_per_isl_token,
            cs_rest_ms_base=self.config.cs_rest_ms_base,
            cs_rest_ms_per_token=self.config.cs_rest_ms_per_token,
            link_ms_one_way=self.config.link_ms_one_way,
            attention_replicas=self.config.attention_replicas,
            gpu_to_cs_link_resources=self.config.gpu_to_cs_link_resources,
            cs_rest_resources=self.config.cs_rest_resources,
            cs_to_gpu_link_resources=self.config.cs_to_gpu_link_resources,
            mock_block_size=self.config.kvcache_block_size,
            mock_kv_capacity_tokens=self.config.num_kvcache_blocks * self.config.kvcache_block_size,
            mock_token_base=self.config.mock_token_base,
            timing_backend=self.config.timing_backend,
            roofline_gpu_arch=self.config.roofline_gpu_arch,
            roofline_gpu_backend=self.config.roofline_gpu_backend,
            roofline_tp_g=self.config.roofline_tp_g,
            attention_groups=self.config.attention_groups,
            chunk_batch=self.config.microbatch_size,
            gpu_cs_link_us=self.config.gpu_cs_link_us,
            des_batch_decode=True,
            des_max_batch_size=len(seqs),
        )
        engine = DESEngine(config)
        for idx, seq in enumerate(seqs):
            engine.submit(DESRequest(request_id=idx, arrival_ms=0.0, isl=len(seq), osl=1))
        return engine.run()

    @staticmethod
    def _decode_start_ms(rows: list[dict[str, str]]) -> float:
        starts = [
            float(row["virtual_time_ms"])
            for row in rows
            if row["stage"].endswith("_start")
            and row.get("resource", "")
            and row["stage"] != "prefill_start"
        ]
        if not starts:
            starts = [
                float(row["virtual_time_ms"])
                for row in rows
                if row["stage"] == "decode_start"
            ]
        return min(starts) if starts else 0.0

    def _record_stage_events(self, rows: list[dict[str, str]], decode_start_ms: float):
        seen = set()
        for row in rows:
            stage = row["stage"]
            if not stage.endswith(("_start", "_end")) or stage.startswith("prefill"):
                continue
            dedupe_key = (
                row.get("virtual_time_ms", ""),
                stage,
                row.get("resource", ""),
                row.get("microbatch_id", ""),
                row.get("notes", ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            event_time = float(row["virtual_time_ms"]) - decode_start_ms
            notes = row.get("notes", "")
            resource = row.get("resource", "")
            if resource:
                notes = f"{notes};resource={resource}" if notes else f"resource={resource}"
            microbatch = row.get("microbatch_id", "")
            if microbatch and "microbatch=" not in notes:
                notes = f"{notes};microbatch={microbatch}" if notes else f"microbatch={microbatch}"
            self.last_stage_events.append(RunnerStageEvent(stage, event_time, event_time, notes))
