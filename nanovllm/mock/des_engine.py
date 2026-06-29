import csv
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    def __post_init__(self):
        assert self.mode in ("colocated", "afd")
        assert self.attention_replicas > 0
        assert self.gpu_to_cs_link_resources > 0
        assert self.cs_rest_resources > 0
        assert self.cs_to_gpu_link_resources > 0
        assert self.mock_block_size > 0
        assert self.mock_kv_capacity_tokens > 0


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
        duration = self.config.prefill_base_ms + state.spec.isl * self.config.prefill_ms_per_token
        resource_id, start, end = self.prefill.reserve(event.time_ms, duration)
        self._emit_resource(state, "prefill", start, end, resource_id, event.time_ms)
        self._push(end, "prefill_done", event.request_id)

    def _handle_prefill_done(self, event: Event):
        state = self._state(event.request_id)
        self._emit(state, "prefill_end", event.time_ms)
        self._schedule_decode_token(state, event.time_ms)

    def _schedule_decode_token(self, state: RequestState, ready_ms: float):
        if self.config.mode == "colocated":
            duration = self.config.decode_base_ms + self.config.decode_ms_per_token
            resource_id, start, end = self.colocated_decode.reserve(ready_ms, duration)
            self._emit_resource(state, "decode", start, end, resource_id, ready_ms)
            self._push(end, "token_emit", state.spec.request_id)
            return

        context_len = state.spec.isl + state.generated_tokens
        attention_ms = (
            self.config.attention_ms_base
            + self.config.attention_ms_per_token
            + self.config.attention_ms_per_isl_token * context_len
        )
        attention_resource = state.spec.request_id % self.config.attention_replicas
        res, start, end = self.attention.reserve(ready_ms, attention_ms, preferred=attention_resource)
        self._emit_resource(state, "decode_attention", start, end, res, ready_ms)
        self._push(end, "attention_done", state.spec.request_id)

    def _handle_attention_done(self, event: Event):
        state = self._state(event.request_id)
        res, start, end = self.gpu_to_cs.reserve(event.time_ms, self.config.link_ms_one_way)
        self._emit_resource(state, "gpu_to_cs_link", start, end, res, event.time_ms)
        self._push(end, "gpu_to_cs_done", state.spec.request_id)

    def _handle_gpu_to_cs_done(self, event: Event):
        state = self._state(event.request_id)
        duration = self.config.cs_rest_ms_base + self.config.cs_rest_ms_per_token
        res, start, end = self.cs_rest.reserve(event.time_ms, duration)
        self._emit_resource(state, "cs_rest", start, end, res, event.time_ms)
        self._push(end, "cs_rest_done", state.spec.request_id)

    def _handle_cs_rest_done(self, event: Event):
        state = self._state(event.request_id)
        res, start, end = self.cs_to_gpu.reserve(event.time_ms, self.config.link_ms_one_way)
        self._emit_resource(state, "cs_to_gpu_link", start, end, res, event.time_ms)
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
    ):
        resource = f"{stage}_{resource_id}"
        self._emit(state, f"{stage}_start", start_ms, resource=resource, queue_delay_ms=start_ms - ready_ms)
        self._emit(state, f"{stage}_end", end_ms, resource=resource)

    def _emit(
        self,
        state: RequestState,
        stage: str,
        time_ms: float,
        token_id: int | str = "",
        resource: str = "",
        queue_delay_ms: float = 0.0,
        notes: str = "",
    ):
        kv_capacity = self.config.mock_kv_capacity_tokens
        self.trace.emit(
            virtual_time_ms=time_ms,
            request_id=state.spec.request_id,
            stage=stage,
            batch_size=1,
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
