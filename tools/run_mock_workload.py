import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import LLM, SamplingParams
from tools.mock_trace_metrics import compute_metrics, read_trace, write_request_metrics
from tools.mock_timing_cli import add_timing_backend_args, timing_backend_kwargs


@dataclass(frozen=True)
class WorkloadRequest:
    arrival_ms: float
    isl: int
    osl: int
    session_id: str = ""
    prefix_id: str = ""


def fixed_or_lognormal(rng: random.Random, dist: str, fixed: int, mean: float, sigma: float) -> int:
    if dist == "fixed":
        return fixed
    value = rng.lognormvariate(math.log(mean), sigma)
    return max(1, int(round(value)))


def generate_arrivals(rng: random.Random, args) -> list[float]:
    if args.arrival_process == "burst":
        return [0.0] * args.num_requests

    arrivals = []
    current = 0.0
    for _ in range(args.num_requests):
        current += rng.expovariate(args.arrival_rate_per_s) * 1000.0
        arrivals.append(current)
    return arrivals


def generate_workload(args) -> list[WorkloadRequest]:
    rng = random.Random(args.seed)
    arrivals = generate_arrivals(rng, args)
    requests = []
    for idx, arrival_ms in enumerate(arrivals):
        isl = fixed_or_lognormal(rng, args.isl_dist, args.fixed_isl, args.isl_lognormal_mean, args.isl_lognormal_sigma)
        osl = fixed_or_lognormal(rng, args.osl_dist, args.fixed_osl, args.osl_lognormal_mean, args.osl_lognormal_sigma)
        requests.append(WorkloadRequest(
            arrival_ms=arrival_ms,
            isl=isl,
            osl=osl,
            session_id=f"session-{idx}" if args.include_session_ids else "",
            prefix_id=f"prefix-{idx % args.num_prefixes}" if args.num_prefixes else "",
        ))
    return sorted(requests, key=lambda req: req.arrival_ms)


def write_workload_csv(path: Path, workload: list[WorkloadRequest]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_ms", "isl", "osl", "session_id", "prefix_id"])
        writer.writeheader()
        for req in workload:
            writer.writerow(req.__dict__)


def run_workload(args):
    output_dir = Path(args.output_dir)
    trace_path = Path(args.trace_output) if args.trace_output else output_dir / "mock_trace.csv"
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_dir / "mock_metrics.csv"
    workload_path = output_dir / "mock_workload.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    workload = generate_workload(args)
    write_workload_csv(workload_path, workload)

    max_model_len = max(args.max_model_len, max((req.isl + req.osl + 1 for req in workload), default=args.max_model_len))
    llm = LLM(
        "__mock__",
        mock_backend=True,
        mock_mode=args.mode,
        virtual_time=True,
        trace_output=str(trace_path),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        prefill_base_ms=args.prefill_base_ms,
        prefill_ms_per_token=args.prefill_ms_per_token,
        decode_base_ms=args.decode_base_ms,
        decode_ms_per_token=args.decode_ms_per_token,
        mock_kv_capacity_tokens=args.mock_kv_capacity_tokens,
        mock_block_size=args.mock_block_size,
        attention_ms_base=args.attention_ms_base,
        attention_ms_per_token=args.attention_ms_per_token,
        attention_ms_per_isl_token=args.attention_ms_per_isl_token,
        cs_rest_ms_base=args.cs_rest_ms_base,
        cs_rest_ms_per_token=args.cs_rest_ms_per_token,
        link_ms_one_way=args.link_ms_one_way,
        num_layers=args.num_layers,
        pipeline_mode=args.pipeline_mode,
        microbatch_size=args.microbatch_size,
        attention_replicas=args.attention_replicas,
        gpu_to_cs_link_resources=args.gpu_to_cs_link_resources,
        cs_rest_resources=args.cs_rest_resources,
        cs_to_gpu_link_resources=args.cs_to_gpu_link_resources,
        **timing_backend_kwargs(args),
    )

    pending = list(workload)
    outputs = {}
    next_prompt_token = 1
    while pending or not llm.is_finished():
        while pending and pending[0].arrival_ms <= llm.clock.time_ms:
            req = pending.pop(0)
            prompt = list(range(next_prompt_token, next_prompt_token + req.isl))
            next_prompt_token += req.isl
            llm.add_request(prompt, SamplingParams(max_tokens=req.osl, ignore_eos=True))
        if llm.is_finished():
            if pending:
                llm.clock.time_ms = pending[0].arrival_ms
            continue
        step_outputs, _ = llm.step()
        outputs.update(step_outputs)

    rows = read_trace(trace_path)
    request_metrics, summary = compute_metrics(rows)
    write_request_metrics(metrics_path, request_metrics)
    write_summary_csv(output_dir / "mock_summary.csv", summary)
    write_plots(output_dir, rows, request_metrics)

    print(f"wrote workload: {workload_path}")
    print(f"wrote trace: {trace_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote plots: {output_dir}")
    print(f"requests: {len(outputs)} tokens: {sum(len(tokens) for tokens in outputs.values())}")


def write_summary_csv(path: Path, summary: dict[str, float | int]):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def write_plots(output_dir: Path, rows: list[dict[str, str]], request_metrics: list[dict[str, float | int | str]]):
    ttft = [float(row["ttft_ms"]) for row in request_metrics]
    tbt = [float(row["mean_tbt_ms"]) for row in request_metrics if float(row["mean_tbt_ms"]) > 0]
    token_times = [float(row["virtual_time_ms"]) for row in rows if row["stage"] == "token_emit"]
    kv_points = [
        (float(row["virtual_time_ms"]), int(row["kv_tokens_used"]))
        for row in rows
        if row["kv_tokens_used"] not in ("", None)
    ]
    batch_points = [
        (float(row["virtual_time_ms"]), int(row["batch_size"]))
        for row in rows
        if row["stage"] in ("prefill_start", "decode_start") and int(row["batch_size"]) > 0
    ]

    write_hist_svg(output_dir / "ttft_distribution.svg", ttft, "TTFT distribution", "TTFT (ms)")
    write_hist_svg(output_dir / "tbt_distribution.svg", tbt, "TBT distribution", "TBT (ms)")
    write_line_svg(output_dir / "throughput_over_time.svg", throughput_points(token_times), "Throughput over virtual time", "tokens/s")
    write_line_svg(output_dir / "kv_usage_over_time.svg", kv_points, "KV usage over virtual time", "KV tokens")
    write_line_svg(output_dir / "batch_size_over_time.svg", batch_points, "Batch size over virtual time", "batch size")


def throughput_points(token_times: list[float], window_ms: float = 1000.0) -> list[tuple[float, float]]:
    points = []
    for time_ms in token_times:
        start = time_ms - window_ms
        count = sum(1 for item in token_times if start < item <= time_ms)
        points.append((time_ms, count * 1000.0 / window_ms))
    return points


def write_hist_svg(path: Path, values: list[float], title: str, x_label: str, bins: int = 20):
    if not values:
        values = [0.0]
    min_v, max_v = min(values), max(values)
    span = max(max_v - min_v, 1e-9)
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - min_v) / span * bins))
        counts[idx] += 1
    points = [(idx, count) for idx, count in enumerate(counts)]
    write_bar_svg(path, points, title, x_label)


def write_bar_svg(path: Path, bars: list[tuple[int, int]], title: str, x_label: str):
    width, height, pad = 720, 360, 48
    max_y = max((bar[1] for bar in bars), default=1) or 1
    bar_w = (width - 2 * pad) / max(len(bars), 1)
    rects = []
    for idx, count in bars:
        x = pad + idx * bar_w
        h = (height - 2 * pad) * count / max_y
        y = height - pad - h
        rects.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.85:.2f}" height="{h:.2f}" fill="#3b82f6"/>')
    write_svg(path, title, x_label, "count", "\n".join(rects), width, height)


def write_line_svg(path: Path, points: list[tuple[float, float]], title: str, y_label: str):
    width, height, pad = 720, 360, 48
    if not points:
        points = [(0.0, 0.0)]
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    coords = []
    for x, y in points:
        sx = pad + (x - min_x) / span_x * (width - 2 * pad)
        sy = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        coords.append(f"{sx:.2f},{sy:.2f}")
    body = f'<polyline points="{" ".join(coords)}" fill="none" stroke="#0f766e" stroke-width="2"/>'
    for coord in coords:
        x, y = coord.split(",")
        body += f'\n<circle cx="{x}" cy="{y}" r="2.2" fill="#0f766e"/>'
    write_svg(path, title, "virtual time (ms)", y_label, body, width, height)


def write_svg(path: Path, title: str, x_label: str, y_label: str, body: str, width: int, height: int):
    path.write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>
<line x1="48" y1="{height - 48}" x2="{width - 24}" y2="{height - 48}" stroke="#111827"/>
<line x1="48" y1="36" x2="48" y2="{height - 48}" stroke="#111827"/>
<text x="{width / 2:.0f}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>
<text x="14" y="{height / 2:.0f}" text-anchor="middle" transform="rotate(-90 14 {height / 2:.0f})" font-family="sans-serif" font-size="12">{y_label}</text>
{body}
</svg>
""")


def main():
    parser = argparse.ArgumentParser(description="Generate and run a synthetic mock serving workload.")
    parser.add_argument("--mode", choices=["colocated", "afd"], default="colocated")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--arrival-process", choices=["poisson", "burst"], default="burst")
    parser.add_argument("--arrival-rate-per-s", type=float, default=10.0)
    parser.add_argument("--isl-dist", choices=["fixed", "lognormal"], default="fixed")
    parser.add_argument("--osl-dist", choices=["fixed", "lognormal"], default="fixed")
    parser.add_argument("--fixed-isl", type=int, default=128)
    parser.add_argument("--fixed-osl", type=int, default=8)
    parser.add_argument("--isl-lognormal-mean", type=float, default=128.0)
    parser.add_argument("--isl-lognormal-sigma", type=float, default=0.5)
    parser.add_argument("--osl-lognormal-mean", type=float, default=8.0)
    parser.add_argument("--osl-lognormal-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-session-ids", action="store_true")
    parser.add_argument("--num-prefixes", type=int, default=0)
    parser.add_argument("--output-dir", default="results/mock_workload")
    parser.add_argument("--trace-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prefill-base-ms", type=float, default=1.0)
    parser.add_argument("--prefill-ms-per-token", type=float, default=0.01)
    parser.add_argument("--decode-base-ms", type=float, default=0.5)
    parser.add_argument("--decode-ms-per-token", type=float, default=0.02)
    parser.add_argument("--mock-kv-capacity-tokens", type=int)
    parser.add_argument("--mock-block-size", type=int)
    parser.add_argument("--attention-ms-base", type=float, default=0.4)
    parser.add_argument("--attention-ms-per-token", type=float, default=0.02)
    parser.add_argument("--attention-ms-per-isl-token", type=float, default=0.0001)
    parser.add_argument("--cs-rest-ms-base", type=float, default=0.6)
    parser.add_argument("--cs-rest-ms-per-token", type=float, default=0.03)
    parser.add_argument("--link-ms-one-way", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--pipeline-mode", choices=["sequential", "ideal_pipeline", "discrete_pipeline"], default="sequential")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--attention-replicas", type=int, default=1)
    parser.add_argument("--gpu-to-cs-link-resources", type=int, default=1)
    parser.add_argument("--cs-rest-resources", type=int, default=1)
    parser.add_argument("--cs-to-gpu-link-resources", type=int, default=1)
    add_timing_backend_args(parser)
    run_workload(parser.parse_args())


if __name__ == "__main__":
    main()
