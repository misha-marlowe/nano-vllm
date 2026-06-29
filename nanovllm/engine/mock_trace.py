import csv
from pathlib import Path
from typing import Any


TRACE_COLUMNS = [
    "virtual_time_ms",
    "request_id",
    "event_scope",
    "microbatch_id",
    "resource",
    "stage",
    "batch_size",
    "isl",
    "generated_tokens",
    "token_id",
    "kv_tokens_used",
    "kv_blocks_used",
    "kv_blocks_free",
    "kv_capacity_tokens",
    "notes",
]


class VirtualClock:

    def __init__(self):
        self.time_ms = 0.0

    def advance(self, delta_ms: float):
        self.time_ms += delta_ms


class MockTraceWriter:

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.rows: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
                writer.writeheader()

    def emit(
        self,
        virtual_time_ms: float,
        request_id: int | str,
        stage: str,
        event_scope: str = "request",
        microbatch_id: int | str = "",
        resource: str = "",
        batch_size: int = 0,
        isl: int = 0,
        generated_tokens: int = 0,
        token_id: int | str = "",
        kv_tokens_used: int = 0,
        kv_blocks_used: int = 0,
        kv_blocks_free: int = 0,
        kv_capacity_tokens: int = 0,
        notes: str = "",
    ):
        row = {
            "virtual_time_ms": f"{virtual_time_ms:.6f}",
            "request_id": request_id,
            "event_scope": event_scope,
            "microbatch_id": microbatch_id,
            "resource": resource,
            "stage": stage,
            "batch_size": batch_size,
            "isl": isl,
            "generated_tokens": generated_tokens,
            "token_id": token_id,
            "kv_tokens_used": kv_tokens_used,
            "kv_blocks_used": kv_blocks_used,
            "kv_blocks_free": kv_blocks_free,
            "kv_capacity_tokens": kv_capacity_tokens,
            "notes": notes,
        }
        self.rows.append(row)
        if self.path:
            with self.path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
                writer.writerow(row)
