import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_trace(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return int(value) if value not in ("", None) else default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[index]


def pair_stage_durations(rows: list[dict[str, str]], start_stage: str, end_stage: str) -> float:
    starts: list[float] = []
    total = 0.0
    for row in rows:
        stage = row["stage"]
        if stage == start_stage:
            starts.append(as_float(row, "virtual_time_ms"))
        elif stage == end_stage and starts:
            total += as_float(row, "virtual_time_ms") - starts.pop(0)
    return total


def compute_metrics(rows: list[dict[str, str]]) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    by_request: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        request_id = row.get("request_id", "")
        if request_id != "":
            by_request[request_id].append(row)

    request_metrics: list[dict[str, float | int | str]] = []
    all_tbt_ms: list[float] = []
    emitted_tokens = 0

    for request_id in sorted(by_request, key=lambda item: int(item) if item.isdigit() else item):
        req_rows = by_request[request_id]
        arrivals = [as_float(row, "virtual_time_ms") for row in req_rows if row["stage"] == "request_arrival"]
        prefill_starts = [as_float(row, "virtual_time_ms") for row in req_rows if row["stage"] == "prefill_start"]
        token_emit_times = [as_float(row, "virtual_time_ms") for row in req_rows if row["stage"] == "token_emit"]

        arrival = arrivals[0] if arrivals else 0.0
        first_prefill_start = prefill_starts[0] if prefill_starts else arrival
        ttft_ms = token_emit_times[0] - arrival if token_emit_times else 0.0
        tbts = [
            token_emit_times[i] - token_emit_times[i - 1]
            for i in range(1, len(token_emit_times))
        ]
        all_tbt_ms.extend(tbts)
        emitted_tokens += len(token_emit_times)

        request_metrics.append({
            "request_id": request_id,
            "ttft_ms": ttft_ms,
            "mean_tbt_ms": mean(tbts) if tbts else 0.0,
            "tpot_ms": (token_emit_times[-1] - token_emit_times[0]) / (len(token_emit_times) - 1)
            if len(token_emit_times) > 1 else 0.0,
            "queueing_delay_ms": first_prefill_start - arrival,
            "prefill_time_ms": pair_stage_durations(req_rows, "prefill_start", "prefill_end"),
            "decode_time_ms": pair_stage_durations(req_rows, "decode_start", "decode_end"),
            "output_tokens": len(token_emit_times),
        })

    times = [as_float(row, "virtual_time_ms") for row in rows]
    arrival_times = [
        as_float(row, "virtual_time_ms")
        for row in rows
        if row["stage"] == "request_arrival"
    ]
    batch_sizes = [
        as_int(row, "batch_size")
        for row in rows
        if row["stage"] in ("prefill_start", "decode_start") and as_int(row, "batch_size") > 0
    ]
    duration_ms = (max(times) - min(arrival_times)) if times and arrival_times else 0.0
    summary = {
        "num_requests": len(request_metrics),
        "output_tokens": emitted_tokens,
        "mean_tbt_ms": mean(all_tbt_ms) if all_tbt_ms else 0.0,
        "p50_tbt_ms": median(all_tbt_ms) if all_tbt_ms else 0.0,
        "p95_tbt_ms": percentile(all_tbt_ms, 95),
        "throughput_tokens_per_sec": emitted_tokens / (duration_ms / 1000.0) if duration_ms > 0 else 0.0,
        "avg_batch_size": mean(batch_sizes) if batch_sizes else 0.0,
        "max_kv_tokens_used": max((as_int(row, "kv_tokens_used") for row in rows), default=0),
    }
    return request_metrics, summary


def write_request_metrics(path: str | Path, request_metrics: list[dict[str, float | int | str]]):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "ttft_ms",
        "mean_tbt_ms",
        "tpot_ms",
        "queueing_delay_ms",
        "prefill_time_ms",
        "decode_time_ms",
        "output_tokens",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request_metrics)


def format_summary(request_metrics: list[dict[str, float | int | str]], summary: dict[str, float | int]) -> str:
    lines = ["Per-request metrics:"]
    lines.append("request_id,ttft_ms,mean_tbt_ms,tpot_ms,queueing_delay_ms,prefill_time_ms,decode_time_ms,output_tokens")
    for row in request_metrics:
        lines.append(
            f"{row['request_id']},{row['ttft_ms']:.6f},{row['mean_tbt_ms']:.6f},"
            f"{row['tpot_ms']:.6f},{row['queueing_delay_ms']:.6f},"
            f"{row['prefill_time_ms']:.6f},{row['decode_time_ms']:.6f},{row['output_tokens']}"
        )
    lines.append("")
    lines.append("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key},{value:.6f}")
        else:
            lines.append(f"{key},{value}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from a mock backend trace CSV.")
    parser.add_argument("trace_csv")
    parser.add_argument("--csv-output")
    args = parser.parse_args()

    rows = read_trace(args.trace_csv)
    request_metrics, summary = compute_metrics(rows)
    print(format_summary(request_metrics, summary))
    if args.csv_output:
        write_request_metrics(args.csv_output, request_metrics)


if __name__ == "__main__":
    main()
