<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

## CPU Mock Serving Simulator

This fork includes a CPU/Mac-runnable mock backend for learning and validating
serving flow without CUDA, model weights, NCCL, flash-attn, or Triton.

The mock path keeps nano-vLLM's request lifecycle, scheduler, batching,
sequence state, and KV/block manager active. It replaces model execution with
deterministic token generation and virtual latency.

### Mac Setup

Use Python 3.10 or newer. The default macOS `/usr/bin/python3` may be too old.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional analysis packages:

```bash
pip install pandas matplotlib rich
```

Run tests:

```bash
python -m pytest tests/mock_backend
```

### Single Synthetic Trace

Colocated decode:

```bash
python tools/run_mock_trace.py \
  --mock-backend \
  --mock-mode colocated \
  --virtual-time \
  --trace-output traces/mock_trace.csv \
  --num-requests 1 \
  --isl 128 \
  --osl 8
```

AFD sequential decode:

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --pipeline-mode sequential \
  --trace-output traces/mock_afd_trace.csv \
  --num-requests 1 \
  --isl 128 \
  --osl 8
```

AFD ideal pipeline:

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --pipeline-mode ideal_pipeline \
  --microbatch-size 4 \
  --trace-output traces/mock_afd_pipeline_trace.csv \
  --num-requests 16 \
  --arrivals-ms 0 \
  --isl 128 \
  --osl 8
```

Metrics:

```bash
python tools/mock_trace_metrics.py traces/mock_trace.csv \
  --csv-output traces/mock_metrics.csv
```

### Synthetic Workloads

Colocated burst:

```bash
python tools/run_mock_workload.py \
  --mode colocated \
  --num-requests 32 \
  --arrival-process burst \
  --fixed-isl 128 \
  --fixed-osl 8
```

AFD workload:

```bash
python tools/run_mock_workload.py \
  --mode afd \
  --pipeline-mode ideal_pipeline \
  --microbatch-size 4 \
  --num-requests 64 \
  --arrival-process poisson \
  --arrival-rate-per-s 20 \
  --isl-dist lognormal \
  --osl-dist lognormal
```

Workload outputs include trace CSV, metrics CSV, summary CSV, and SVG plots for
TTFT, TBT, throughput, KV usage, and batch size.

### GPT-OSS Timing Backend

The default timing backend is `parametric`, which preserves the original mock
latency formulas. For GPT-OSS-120B decode studies, use
`--timing-backend gptoss_roofline`. It maps the vendored AC_PerfModel decode
model onto the same mock stages: GPU-only decode for colocated mode, and GPU
attention / GPU↔CS link / CS rest for AFD mode. Prefill remains parametric.

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --timing-backend gptoss_roofline \
  --isl 8192 \
  --osl 8 \
  --trace-output traces/gptoss_afd_trace.csv

python tools/validate_roofline_backend.py \
  --output-dir results/roofline_validation
```

### What Is Real

- `LLMEngine.step()` and the request lifecycle.
- `Scheduler.schedule()` batching, prefill/decode selection, queueing, and
  preemption behavior.
- `Sequence` token state.
- `BlockManager` allocation, append, deallocation, and prefix-cache hashing.
- Trace and metrics generation from virtual-time events.

### What Is Mocked

- Model weights and real logits.
- CUDA tensors, CUDA graphs, NCCL, Triton, and flash-attn.
- Token generation, which is deterministic dummy token IDs.
- Latency, which is virtual time rather than `sleep()`.
- AFD CS execution, which contributes timing but holds no KV.

### What Is Not Validated

- Numerical model correctness.
- Real GPU/CS kernel behavior.
- Real network or PCIe/NIC behavior.
- Real tokenizer/model compatibility in mock mode.
- Production vLLM API serving or HTTP behavior.

### How To Interpret Results

Trace timestamps are virtual milliseconds. Wall-clock runtime of the simulator
is unrelated to modeled serving latency.

Colocated mode uses:

```text
prefill_ms = prefill_base_ms + isl * prefill_ms_per_token * batch_size
decode_ms = decode_base_ms + decode_ms_per_token * batch_size
```

AFD sequential mode uses:

```text
attention -> GPU-to-CS link -> CS rest -> CS-to-GPU link
```

AFD ideal pipeline mode is an optimistic steady-state model. For small
microbatch counts, use `pipeline_mode=discrete_pipeline` or the
`pipeline_sim.py` tests as the stricter event-level reference.

More detail lives in `docs/mock_backend.md`.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
