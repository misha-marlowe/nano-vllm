import argparse
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.mock import DESConfig, DESEngine, DESRequest
from tools.mock_trace_metrics import compute_metrics, read_trace, write_request_metrics
from tools.run_mock_workload import write_plots, write_summary_csv


def sample_length(rng, dist: str, fixed: int, mean: float, sigma: float) -> int:
    if dist == "fixed":
        return fixed
    return max(1, int(round(rng.lognormvariate(math.log(mean), sigma))))


def generate_requests(args) -> list[DESRequest]:
    rng = random.Random(args.seed)
    current_ms = 0.0
    requests = []
    for idx in range(args.num_requests):
        if args.arrival_process == "poisson":
            current_ms += rng.expovariate(args.arrival_rate_per_s) * 1000.0
        arrival_ms = 0.0 if args.arrival_process == "burst" else current_ms
        isl = sample_length(rng, args.isl_dist, args.fixed_isl, args.isl_lognormal_mean, args.isl_lognormal_sigma)
        osl = sample_length(rng, args.osl_dist, args.fixed_osl, args.osl_lognormal_mean, args.osl_lognormal_sigma)
        requests.append(DESRequest(
            request_id=idx,
            arrival_ms=arrival_ms,
            isl=isl,
            osl=osl,
            session_id=f"session-{idx}" if args.include_session_ids else "",
            prefix_id=f"prefix-{idx % args.num_prefixes}" if args.num_prefixes else "",
        ))
    return requests


def write_requests(path: Path, requests: list[DESRequest]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "arrival_ms", "isl", "osl", "session_id", "prefix_id"])
        writer.writeheader()
        for req in requests:
            writer.writerow(req.__dict__)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_output) if args.trace_output else output_dir / "des_trace.csv"
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_dir / "des_metrics.csv"
    requests = generate_requests(args)
    write_requests(output_dir / "des_workload.csv", requests)

    config = DESConfig(
        mode=args.mode,
        trace_output=str(trace_path),
        prefill_base_ms=args.prefill_base_ms,
        prefill_ms_per_token=args.prefill_ms_per_token,
        decode_base_ms=args.decode_base_ms,
        decode_ms_per_token=args.decode_ms_per_token,
        attention_ms_base=args.attention_ms_base,
        attention_ms_per_token=args.attention_ms_per_token,
        attention_ms_per_isl_token=args.attention_ms_per_isl_token,
        cs_rest_ms_base=args.cs_rest_ms_base,
        cs_rest_ms_per_token=args.cs_rest_ms_per_token,
        link_ms_one_way=args.link_ms_one_way,
        attention_replicas=args.attention_replicas,
        gpu_to_cs_link_resources=args.gpu_to_cs_link_resources,
        cs_rest_resources=args.cs_rest_resources,
        cs_to_gpu_link_resources=args.cs_to_gpu_link_resources,
        mock_block_size=args.mock_block_size,
        mock_kv_capacity_tokens=args.mock_kv_capacity_tokens,
    )
    engine = DESEngine(config)
    for req in requests:
        engine.submit(req)
    engine.run()

    rows = read_trace(trace_path)
    request_metrics, summary = compute_metrics(rows)
    write_request_metrics(metrics_path, request_metrics)
    write_summary_csv(output_dir / "des_summary.csv", summary)
    write_plots(output_dir, rows, request_metrics)

    print(f"wrote workload: {output_dir / 'des_workload.csv'}")
    print(f"wrote trace: {trace_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"requests: {len(requests)}")


def main():
    parser = argparse.ArgumentParser(description="Run the standalone DES mock serving harness.")
    parser.add_argument("--mode", choices=["colocated", "afd"], default="afd")
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
    parser.add_argument("--output-dir", default="results/des_workload")
    parser.add_argument("--trace-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--prefill-base-ms", type=float, default=1.0)
    parser.add_argument("--prefill-ms-per-token", type=float, default=0.01)
    parser.add_argument("--decode-base-ms", type=float, default=0.5)
    parser.add_argument("--decode-ms-per-token", type=float, default=0.02)
    parser.add_argument("--attention-ms-base", type=float, default=0.4)
    parser.add_argument("--attention-ms-per-token", type=float, default=0.02)
    parser.add_argument("--attention-ms-per-isl-token", type=float, default=0.0001)
    parser.add_argument("--cs-rest-ms-base", type=float, default=0.6)
    parser.add_argument("--cs-rest-ms-per-token", type=float, default=0.03)
    parser.add_argument("--link-ms-one-way", type=float, default=0.1)
    parser.add_argument("--attention-replicas", type=int, default=1)
    parser.add_argument("--gpu-to-cs-link-resources", type=int, default=1)
    parser.add_argument("--cs-rest-resources", type=int, default=1)
    parser.add_argument("--cs-to-gpu-link-resources", type=int, default=1)
    parser.add_argument("--mock-block-size", type=int, default=256)
    parser.add_argument("--mock-kv-capacity-tokens", type=int, default=1_000_000_000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
