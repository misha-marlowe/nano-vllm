import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import LLM, SamplingParams
from tools.mock_timing_cli import add_timing_backend_args, timing_backend_kwargs


@dataclass
class SyntheticRequest:
    arrival_ms: float
    isl: int
    osl: int


def parse_csv_numbers(value: str, cast):
    return [cast(item) for item in value.split(",") if item]


def expand(values, count: int):
    if len(values) == 1:
        return values * count
    assert len(values) == count, f"expected 1 or {count} values, got {len(values)}"
    return values


def build_requests(args) -> list[SyntheticRequest]:
    arrivals = expand(parse_csv_numbers(args.arrivals_ms, float), args.num_requests)
    isls = expand(parse_csv_numbers(args.isl, int), args.num_requests)
    osls = expand(parse_csv_numbers(args.osl, int), args.num_requests)
    return [
        SyntheticRequest(arrival, isl, osl)
        for arrival, isl, osl in sorted(zip(arrivals, isls, osls))
    ]


def run(args):
    requests = build_requests(args)
    llm = LLM(
        "__mock__",
        mock_backend=args.mock_backend,
        mock_mode=args.mock_mode,
        virtual_time=args.virtual_time,
        trace_output=args.trace_output,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
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

    pending = list(requests)
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

    print(f"wrote trace: {args.trace_output}")
    for seq_id in sorted(outputs):
        print(f"request {seq_id}: {len(outputs[seq_id])} tokens")


def main():
    parser = argparse.ArgumentParser(description="Run a CPU-only nano-vLLM mock trace.")
    parser.add_argument("--mock-backend", action="store_true", default=True)
    parser.add_argument("--mock-mode", choices=["colocated", "afd"], default="colocated")
    parser.add_argument("--virtual-time", action="store_true", default=True)
    parser.add_argument("--trace-output", default="traces/mock_trace.csv")
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--arrivals-ms", default="0")
    parser.add_argument("--isl", default="128")
    parser.add_argument("--osl", default="8")
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
