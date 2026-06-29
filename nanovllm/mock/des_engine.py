import csv
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanovllm.engine.pipeline_sim import PipelineStage, simulate_discrete_pipeline, split_into_microbatches
from nanovllm.mock.timing import build_timing_backend


TRACE_COLUMNS = [
    "virtual_time_ms",
    "request_id",
    "stage",
    "batch_size",
    "isl",
    "generated_tokens",
    "token_id",
    "kv_tokens_used",
    "kv_blocks_used",
    "kv_blocks_free",
    "kv_capacity_tokens",
    "resource",
    "queue_delay_ms",
    "session_id",
    "prefix_id",
    "notes",
]


@dataclass(frozen=True)
class DESRequest:
    request_id: int
    arrival_ms: float
    isl: int
    osl: int
    session_id: str = ""
    prefix_id: str = ""


@dataclass
class RequestState:
    spec: DESRequest
    generated_tokens: int = 0
    kv_tokens: int = 0
    finished: bool = False


@dataclass
class DESConfig:
    mode: str = "colocated"
    prefill_base_ms: float = 1.0
    prefill_ms_per_token: float = 0.01
    decode_base_ms: float = 0.5
    decode_ms_per_token: float = 0.02
    attention_ms_base: float = 0.4
    attention_ms_per_token: float = 0.02
    attention_ms_per_isl_token: float = 0.0001
    cs_rest_ms_base: float = 0.6
    cs_rest_ms_per_token: float = 0.03
    link_ms_one_way: float = 0.1
    attention_replicas: int = 1
    gpu_to_cs_link_resources: int = 1
    cs_rest_resources: int = 1
    cs_to_gpu_link_resources: int = 1
    mock_block_size: int = 256
    mock_kv_capacity_tokens: int = 1_000_000_000
    trace_output: str | None = None
    mock_token_base: int = 1000
    timing_backend: str = "parametric"
    roofline_gpu_arch: str = "helios"
    roofline_gpu_backend: str = "measured"
    roofline_tp_g: int = 1
    attention_groups: int = 1
    chunk_batch: int = 1
    gpu_cs_link_us: float = 12.0
    des_batch_decode: bool = False
    des_max_batch_size: int = 512

    def __post_init__(self):
        assert self.mode in ("colocated", "afd")
        assert self.attention_replicas > 0
        assert self.gpu_to_cs_link_resources > 0
        assert self.cs_rest_resources > 0
        assert self.cs_to_gpu_link_resources > 0
        assert self.mock_block_size > 0
        assert self.mock_kv_capacity_tokens > 0
        assert self.timing_backend in ("parametric", "gptoss_roofline")
        assert self.roofline_gpu_arch in ("helios", "rubin", "b200")
        assert self.roofline_gpu_backend in ("measured", "roofline")
        assert self.roofline_tp_g > 0
        assert self.gpu_cs_link_us >= 0
        assert self.des_max_batch_size > 0


@dataclass(order=True)
class Event:
    time_ms: float
    order: int
    kind: str = field(compare=False)
    request_id: int = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class ResourcePool:

    def __init__(self, name: str, count: int):
        self.name = name
        self.available_ms = [0.0] * count
        self.busy_ms = [0.0] * count

    def reserve(self, ready_ms: float, duration_ms: float, preferred: int | None = None):
        if preferred is not None:
            resource_id = preferred % len(self.available_ms)
        else:
            resource_id = min(range(len(self.available_ms)), key=lambda idx: (self.available_ms[idx], idx))
        start_ms = max(ready_ms, self.available_ms[resource_id])
        end_ms = start_ms + duration_ms
        self.available_ms[resource_id] = end_ms
        self.busy_ms[resource_id] += duration_ms
        return resource_id, start_ms, end_ms


class DESTraceWriter:

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.rows: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=TRACE_COLUMNS).writeheader()

    def emit(self, **row):
        full = {column: row.get(column, "") for column in TRACE_COLUMNS}
        if isinstance(full["virtual_time_ms"], float):
            full["virtual_time_ms"] = f"{full['virtual_time_ms']:.6f}"
        if isinstance(full["queue_delay_ms"], float):
            full["queue_delay_ms"] = f"{full['queue_delay_ms']:.6f}"
        self.rows.append(full)
        if self.path:
            with self.path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=TRACE_COLUMNS).writerow(full)


class DESEngine:

    def __init__(self, config: DESConfig):
        self.config = config
        self.clock_ms = 0.0
        self._counter = 0
        self.events: list[Event] = []
        self.requests: dict[int, RequestState] = {}
        self.trace = DESTraceWriter(config.trace_output)
        self.attention = ResourcePool("decode_attention", config.attention_replicas)
        self.gpu_to_cs = ResourcePool("gpu_to_cs_link", config.gpu_to_cs_link_resources)
        self.cs_rest = ResourcePool("cs_rest", config.cs_rest_resources)
        self.cs_to_gpu = ResourcePool("cs_to_gpu_link", config.cs_to_gpu_link_resources)
        self.colocated_decode = ResourcePool("decode", 1)
        self.prefill = ResourcePool("prefill", 1)
        self.used_kv_blocks = 0
        self.timing = build_timing_backend(config)
        self.decode_ready: list[tuple[int, float]] = []
        self.decode_batch_scheduled = False

    def submit(self, request: DESRequest):
        self.requests[request.request_id] = RequestState(request)
        self._push(request.arrival_ms, "arrival", request.request_id)

    def run(self):
        while self.events:
            event = heapq.heappop(self.events)
            self.clock_ms = event.time_ms
            getattr(self, f"_handle_{event.kind}")(event)
        return self.trace.rows

    def _push(self, time_ms: float, kind: str, request_id: int, **payload):
        self._counter += 1
        heapq.heappush(self.events, Event(time_ms, self._counter, kind, request_id, payload))

    def _state(self, request_id: int) -> RequestState:
        return self.requests[request_id]

    def _handle_arrival(self, event: Event):
        state = self._state(event.request_id)
        state.kv_tokens = state.spec.isl
        self._refresh_kv_blocks()
        self._emit(state, "request_arrival", event.time_ms, notes="des_arrival")
        duration = self.timing.prefill_ms(1, state.spec.isl)
        resource_id, start, end = self.prefill.reserve(event.time_ms, duration)
        self._emit_resource(state, "prefill", start, end, resource_id, event.time_ms)
        self._push(end, "prefill_done", event.request_id)

    def _handle_prefill_done(self, event: Event):
        state = self._state(event.request_id)
        self._emit(state, "prefill_end", event.time_ms)
        self._schedule_decode_token(state, event.time_ms)

    def _schedule_decode_token(self, state: RequestState, ready_ms: float):
        if self.config.des_batch_decode:
            self._mark_decode_ready(state, ready_ms)
            return

        if self.config.mode == "colocated":
            duration = self.timing.colocated_decode_ms(1, state.spec.isl + state.generated_tokens)
            resource_id, start, end = self.colocated_decode.reserve(ready_ms, duration)
            self._emit_resource(state, "decode", start, end, resource_id, ready_ms)
            self._push(end, "token_emit", state.spec.request_id)
            return

        context_len = state.spec.isl + state.generated_tokens
        timings = self.timing.afd_decode_stages_ms(1, context_len)
        attention_resource = state.spec.request_id % self.config.attention_replicas
        res, start, end = self.attention.reserve(ready_ms, timings.attention_ms, preferred=attention_resource)
        self._emit_resource(state, "decode_attention", start, end, res, ready_ms, timings.notes)
        self._push(end, "attention_done", state.spec.request_id, timings=timings)

    def _mark_decode_ready(self, state: RequestState, ready_ms: float):
        self.decode_ready.append((state.spec.request_id, ready_ms))
        if not self.decode_batch_scheduled:
            self.decode_batch_scheduled = True
            self._push(ready_ms, "decode_batch", -1)

    def _handle_decode_batch(self, event: Event):
        self.decode_batch_scheduled = False
        ready, future = [], []
        for request_id, ready_ms in self.decode_ready:
            if ready_ms <= event.time_ms and not self._state(request_id).finished:
                ready.append((request_id, ready_ms))
            else:
                future.append((request_id, ready_ms))
        if not ready:
            self.decode_ready = future
            if future:
                self.decode_batch_scheduled = True
                self._push(min(ready_ms for _, ready_ms in future), "decode_batch", -1)
            return

        batch = ready[:self.config.des_max_batch_size]
        self.decode_ready = ready[self.config.des_max_batch_size:] + future
        states = [self._state(request_id) for request_id, _ in batch]
        context_len = max(state.spec.isl + state.generated_tokens for state in states)
        if self.config.mode == "colocated":
            duration = self.timing.colocated_decode_ms(len(states), context_len)
            resource_id, start, end = self.colocated_decode.reserve(event.time_ms, duration)
            for state, (_, ready_ms) in zip(states, batch):
                self._emit_resource(
                    state,
                    "decode",
                    start,
                    end,
                    resource_id,
                    ready_ms,
                    notes="des_batch_decode",
                    batch_size=len(states),
                )
                self._push(end, "token_emit", state.spec.request_id)
        else:
            start, end = self._schedule_afd_decode_batch(states, batch, context_len, event.time_ms)
            for state in states:
                self._push(end, "token_emit", state.spec.request_id)
        if self.decode_ready:
            self.decode_batch_scheduled = True
            self._push(end, "decode_batch", -1)

    def _schedule_afd_decode_batch(
        self,
        states: list[RequestState],
        batch: list[tuple[int, float]],
        context_len: int,
        ready_ms: float,
    ) -> tuple[float, float]:
        microbatch_sizes = split_into_microbatches(len(states), self.config.chunk_batch)
        stages = [
            PipelineStage(
                "decode_attention",
                lambda mb: self.timing.afd_decode_stages_ms(mb, context_len).attention_ms,
                resources=self.config.attention_replicas,
                routing="round_robin",
            ),
            PipelineStage(
                "gpu_to_cs_link",
                lambda mb: self.timing.afd_decode_stages_ms(mb, context_len).gpu_to_cs_link_ms,
                resources=self.config.gpu_to_cs_link_resources,
            ),
            PipelineStage(
                "cs_rest",
                lambda mb: self.timing.afd_decode_stages_ms(mb, context_len).cs_rest_ms,
                resources=self.config.cs_rest_resources,
            ),
            PipelineStage(
                "cs_to_gpu_link",
                lambda mb: self.timing.afd_decode_stages_ms(mb, context_len).cs_to_gpu_link_ms,
                resources=self.config.cs_to_gpu_link_resources,
            ),
        ]
        result = simulate_discrete_pipeline(stages, microbatch_sizes)
        start = max(
            ready_ms,
            max(
                self.attention.available_ms
                + self.gpu_to_cs.available_ms
                + self.cs_rest.available_ms
                + self.cs_to_gpu.available_ms
            ),
        )
        batch_size = len(states)
        state = states[0]
        batch_ready_ms = min(ready for _, ready in batch)
        for pipeline_event in result.events:
            resource_pool = self._resource_pool_for_stage(pipeline_event.stage)
            resource_id = pipeline_event.resource_id % len(resource_pool.available_ms)
            event_start = start + pipeline_event.start_ms
            event_end = start + pipeline_event.end_ms
            resource_pool.available_ms[resource_id] = max(resource_pool.available_ms[resource_id], event_end)
            resource_pool.busy_ms[resource_id] += pipeline_event.end_ms - pipeline_event.start_ms
            self._emit_resource(
                state,
                pipeline_event.stage,
                event_start,
                event_end,
                resource_id,
                batch_ready_ms,
                notes=(
                    "des_batch_decode;"
                    f"microbatch={pipeline_event.microbatch_id};"
                    f"microbatch_size={pipeline_event.microbatch_size}"
                ),
                batch_size=batch_size,
            )
        return start, start + result.total_ms

    def _resource_pool_for_stage(self, stage: str) -> ResourcePool:
        if stage == "decode_attention":
            return self.attention
        if stage == "gpu_to_cs_link":
            return self.gpu_to_cs
        if stage == "cs_rest":
            return self.cs_rest
        if stage == "cs_to_gpu_link":
            return self.cs_to_gpu
        raise ValueError(f"unknown AFD stage {stage}")

    def _handle_attention_done(self, event: Event):
        state = self._state(event.request_id)
        timings = event.payload["timings"]
        res, start, end = self.gpu_to_cs.reserve(event.time_ms, timings.gpu_to_cs_link_ms)
        self._emit_resource(state, "gpu_to_cs_link", start, end, res, event.time_ms, timings.notes)
        self._push(end, "gpu_to_cs_done", state.spec.request_id, timings=timings)

    def _handle_gpu_to_cs_done(self, event: Event):
        state = self._state(event.request_id)
        timings = event.payload["timings"]
        res, start, end = self.cs_rest.reserve(event.time_ms, timings.cs_rest_ms)
        self._emit_resource(state, "cs_rest", start, end, res, event.time_ms, timings.notes)
        self._push(end, "cs_rest_done", state.spec.request_id, timings=timings)

    def _handle_cs_rest_done(self, event: Event):
        state = self._state(event.request_id)
        timings = event.payload["timings"]
        res, start, end = self.cs_to_gpu.reserve(event.time_ms, timings.cs_to_gpu_link_ms)
        self._emit_resource(state, "cs_to_gpu_link", start, end, res, event.time_ms, timings.notes)
        self._push(end, "token_emit", state.spec.request_id)

    def _handle_token_emit(self, event: Event):
        state = self._state(event.request_id)
        state.generated_tokens += 1
        state.kv_tokens = state.spec.isl + state.generated_tokens
        self._refresh_kv_blocks()
        token_id = self.config.mock_token_base + state.spec.request_id * 100000 + state.generated_tokens
        self._emit(state, "token_emit", event.time_ms, token_id=token_id, notes="des_token")
        if state.generated_tokens >= state.spec.osl:
            state.finished = True
            state.kv_tokens = 0
            self._refresh_kv_blocks()
            self._emit(state, "request_finish", event.time_ms, token_id=token_id, notes="des_finished")
        else:
            self._schedule_decode_token(state, event.time_ms)

    def _refresh_kv_blocks(self):
        total_tokens = sum(state.kv_tokens for state in self.requests.values() if not state.finished)
        self.used_kv_blocks = (total_tokens + self.config.mock_block_size - 1) // self.config.mock_block_size

    def _emit_resource(
        self,
        state: RequestState,
        stage: str,
        start_ms: float,
        end_ms: float,
        resource_id: int,
        ready_ms: float,
        notes: str = "",
        batch_size: int = 1,
    ):
        resource = f"{stage}_{resource_id}"
        self._emit(
            state,
            f"{stage}_start",
            start_ms,
            resource=resource,
            queue_delay_ms=start_ms - ready_ms,
            notes=notes,
            batch_size=batch_size,
        )
        self._emit(state, f"{stage}_end", end_ms, resource=resource, notes=notes, batch_size=batch_size)

    def _emit(
        self,
        state: RequestState,
        stage: str,
        time_ms: float,
        token_id: int | str = "",
        resource: str = "",
        queue_delay_ms: float = 0.0,
        notes: str = "",
        batch_size: int = 1,
    ):
        kv_capacity = self.config.mock_kv_capacity_tokens
        self.trace.emit(
            virtual_time_ms=time_ms,
            request_id=state.spec.request_id,
            stage=stage,
            batch_size=batch_size,
            isl=state.spec.isl,
            generated_tokens=state.generated_tokens,
            token_id=token_id,
            kv_tokens_used=state.kv_tokens,
            kv_blocks_used=self.used_kv_blocks,
            kv_blocks_free=max(0, (kv_capacity // self.config.mock_block_size) - self.used_kv_blocks),
            kv_capacity_tokens=kv_capacity,
            resource=resource,
            queue_delay_ms=queue_delay_ms,
            session_id=state.spec.session_id,
            prefix_id=state.spec.prefix_id,
            notes=notes,
        )
