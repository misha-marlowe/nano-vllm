# Mock Backend

This document tracks the CPU/Mac-runnable mock backend work. Each phase gets a
section here so the design stays in one place as the latency model evolves.

## Phase 0: Code Map

### Request Entry Point

- `nanovllm/llm.py`
  - `LLM` is a thin subclass of `LLMEngine`.
- `nanovllm/engine/llm_engine.py`
  - `LLMEngine.generate(prompts, sampling_params, use_tqdm=True)` is the batch
    user-facing API.
  - `LLMEngine.add_request(prompt, sampling_params)` tokenizes string prompts,
    constructs a `Sequence`, and appends it to the scheduler waiting queue.

### Engine Loop

- `nanovllm/engine/llm_engine.py`
  - `LLMEngine.step()` is the core serving iteration:
    1. `Scheduler.schedule()` returns selected sequences and whether the batch
       is prefill or decode.
    2. `model_runner.call("run", seqs, is_prefill)` executes the model runner.
    3. `Scheduler.postprocess()` updates KV/block accounting and appends one
       sampled token per sequence when appropriate.
    4. Finished sequence outputs are returned to `generate()`.

### Scheduler Entry Point

- `nanovllm/engine/scheduler.py`
  - `Scheduler.add(seq)` appends new requests to `waiting`.
  - `Scheduler.schedule()` implements prefill and decode scheduling.
  - Prefill respects `max_num_seqs`, `max_num_batched_tokens`, and block
    availability, and it supports chunked prefill for the first sequence.
  - Decode schedules one token per running sequence and may preempt when append
    blocks are unavailable.
  - `Scheduler.postprocess(seqs, token_ids, is_prefill)` hashes completed
    blocks, advances cached-token state, appends generated tokens, detects
    finish, and deallocates blocks.

### Block / KV Manager

- `nanovllm/engine/block_manager.py`
  - `BlockManager.can_allocate(seq)` checks whether prompt KV blocks can be
    allocated and accounts for prefix-cache hits.
  - `BlockManager.allocate(seq, num_cached_blocks)` fills `seq.block_table` and
    updates `seq.num_cached_tokens`.
  - `BlockManager.can_append(seq)` and `may_append(seq)` reserve new blocks for
    decode growth.
  - `BlockManager.hash_blocks(seq)` records full-block hashes for prefix reuse.
  - `BlockManager.deallocate(seq)` releases blocks when a sequence is preempted
    or finished.

### Model Runner Boundary

- `nanovllm/engine/model_runner.py`
  - `ModelRunner.call(method_name, *args)` is the engine-facing dispatch
    boundary.
  - `ModelRunner.run(seqs, is_prefill)` prepares tensors, runs the model, samples
    token IDs, resets attention context, and returns one token ID per sequence.

The recommended CPU mock injection point is a config-gated runner selection in
`LLMEngine.__init__()`: normal mode keeps `ModelRunner`; mock mode constructs a
runner that implements `call("run", seqs, is_prefill) -> list[int]`.

### CUDA / NCCL Assumptions

- `nanovllm/config.py` normally requires a local model directory and Hugging Face
  config.
- `nanovllm/engine/llm_engine.py` normally constructs `ModelRunner`, spawns
  tensor-parallel workers, and loads `AutoTokenizer`.
- `nanovllm/engine/model_runner.py` initializes NCCL, sets CUDA devices, loads
  model weights, allocates CUDA KV cache, and optionally captures CUDA graphs.
- `nanovllm/layers/*` and `nanovllm/models/qwen3.py` assume real model tensors
  and attention kernels.

## Phase 1: CPU-Only Colocated Mock Backend

Phase 1 adds a Mac-runnable mock serving path that keeps nano-vLLM's engine,
scheduler, sequence lifecycle, and block manager active while replacing real
model execution with deterministic token generation and virtual latency.

### What Runs

- `LLMEngine.step()` remains the serving loop.
- `Scheduler.schedule()` still chooses prefill and decode batches.
- `BlockManager` still allocates, appends, hashes, and releases KV blocks.
- `Scheduler.postprocess()` still advances sequence state and finishes requests.
- `FakeColocatedRunner` replaces CUDA model execution when
  `mock_backend=True`.

### What Is Mocked

- No model weights are loaded.
- No CUDA tensors are created.
- No NCCL, Triton, flash-attn, CUDA graphs, or real attention kernels are used.
- Token IDs are deterministic:

```text
mock_token_base + seq_id * 100000 + current_completion_tokens + 1
```

For mock mode, completion tokens are emitted only from decode iterations, so an
output length of 8 means exactly 8 decode steps and 8 `token_emit` events. The
real CUDA path keeps upstream nano-vLLM's prefill-token behavior.

### Timing Model

The mock uses virtual time only. It never sleeps.

```text
prefill_ms = prefill_base_ms + isl * prefill_ms_per_token * batch_size
decode_ms = decode_base_ms + decode_ms_per_token * batch_size
```

Current defaults:

```text
prefill_base_ms = 1.0
prefill_ms_per_token = 0.01
decode_base_ms = 0.5
decode_ms_per_token = 0.02
```

### Trace Events

Phase 1 stages:

- `request_arrival`
- `prefill_start`
- `prefill_end`
- `decode_start`
- `decode_end`
- `token_emit`
- `request_finish`

### Run Command

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

## Phase 2: KV / Block Accounting Validation

Phase 2 makes KV/block state visible and validates that the mock backend follows
nano-vLLM's real block manager path.

### Config

- `mock_kv_capacity_tokens`
  - Converts token capacity to `num_kvcache_blocks` for mock runs.
  - The actual capacity is rounded up to a whole number of blocks.
- `mock_block_size`
  - Overrides `kvcache_block_size` for mock runs.
  - Small values are useful for deterministic block-pressure tests on CPU.

The CLI exposes these as:

```bash
--mock-kv-capacity-tokens
--mock-block-size
```

### Trace Columns

Trace rows include:

```text
virtual_time_ms, request_id, event_scope, microbatch_id, resource, stage,
batch_size, isl, generated_tokens, token_id, kv_tokens_used, kv_blocks_used,
kv_blocks_free, kv_capacity_tokens, notes
```

`kv_tokens_used` is per-request for normal request events. It is `0` on
`request_finish` after blocks have been released. Block counts are global
block-manager state at the event time.

### Additional Trace Events

- `admission_wait`
  - Emitted when the waiting request at the head of the queue cannot allocate
    KV blocks.
- `kv_preempt`
  - Emitted when decode preempts a running sequence because append-block
    capacity is unavailable.

### Expected KV Behavior

- After prefill, request KV tokens equal `ISL`.
- After decode step `k`, request KV tokens equal `ISL + k`.
- After request finish, KV blocks are released.
- With limited KV capacity, requests wait or are preempted according to the
  existing scheduler and block manager behavior.

### Prefix Cache

Prefix-cache behavior is preserved. The mock still calls
`BlockManager.hash_blocks()` and `BlockManager.can_allocate()`, so repeated
prompts can reuse cached full blocks and reduce scheduled prefill tokens.

### Validation

Run:

```bash
python -m pytest tests/mock_backend
```

Phase 2 tests validate KV growth, block release, admission waits under limited
capacity, and prefix-cache reuse.

On macOS, make sure the active Python is 3.10 or newer. The system
`/usr/bin/python3` may be Python 3.9, which is too old for this package. A local
setup that keeps CUDA-only dependencies out of the mock path is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/mock_backend
```

## Phase 3: Metrics From Traces

Phase 3 adds a small metrics script:

```bash
python tools/mock_trace_metrics.py traces/mock_trace.csv
```

Optional per-request CSV output:

```bash
python tools/mock_trace_metrics.py traces/mock_trace.csv \
  --csv-output traces/mock_metrics.csv
```

### Metrics

Per request:

- `ttft_ms`
  - First `token_emit` time minus `request_arrival` time.
- `mean_tbt_ms`
  - Mean interval between consecutive `token_emit` events for the request.
- `tpot_ms`
  - `(last_token_emit - first_token_emit) / (output_tokens - 1)` when there are
    at least two output tokens.
- `queueing_delay_ms`
  - First `prefill_start` time minus `request_arrival` time.
- `prefill_time_ms`
  - Sum of `prefill_end - prefill_start` durations for the request.
- `decode_time_ms`
  - Sum of `decode_end - decode_start` durations for the request.
- `output_tokens`
  - Number of `token_emit` events.

Summary:

- `num_requests`
- `output_tokens`
- `mean_tbt_ms`
- `p50_tbt_ms`
- `p95_tbt_ms`
- `throughput_tokens_per_sec`
- `avg_batch_size`
- `max_kv_tokens_used`

Throughput uses virtual time:

```text
output_tokens / ((max_trace_time - first_arrival_time) / 1000)
```

### Validation

Phase 3 tests generate a single-request colocated trace and compare metrics to
closed-form expectations:

```text
prefill = 1.0 + 128 * 0.01 = 2.28 ms
decode = 0.5 + 0.02 = 0.52 ms
total = 2.28 + 8 * 0.52 = 6.44 ms
```

## Pipeline Validation Utility

`nanovllm/engine/pipeline_sim.py` provides a small discrete pipeline simulator
for validating small microbatch counts before relying on the steady-state
formula used by the later AFD pipeline model.

The steady-state formula:

```text
t_pipe = M * s_max + sum_{i != argmax} s_i / L
```

is useful for checking the ideal formula implementation, but it is optimistic
for small `M` because fill and drain are not fully amortized. For `M=4` or
`M=8`, use `simulate_discrete_pipeline()` as the event-level reference.

### Discrete Model

Each pipeline stage has one resource. Microbatch `j` can enter stage `i` only
after:

- microbatch `j` has completed stage `i - 1`; and
- stage `i` has finished the previous microbatch.

The simulator returns:

- total elapsed time;
- one event per `(microbatch, stage)` with start and end timestamps;
- the actual microbatch size used for that event.

### Variable Microbatch Sizes

Stage cost is a function of microbatch size:

```python
PipelineStage("attention", lambda mb: attention_base + attention_per_token * mb)
PipelineStage("cs_rest", lambda mb: cs_base + cs_per_token * mb)
```

`split_into_microbatches(batch_size, microbatch_size)` handles tails:

```text
split_into_microbatches(18, 8) == [8, 8, 2]
```

For non-uniform microbatches, the discrete simulator is the correctness
reference. The uniform steady-state helper is intentionally limited to equal
microbatch sizes.

### Example

```python
from nanovllm.engine.pipeline_sim import PipelineStage, simulate_discrete_pipeline

stages = [
    PipelineStage("attention", lambda mb: 1.0),
    PipelineStage("gpu_to_cs_link", lambda mb: 0.5),
    PipelineStage("cs_rest", lambda mb: 2.0),
    PipelineStage("cs_to_gpu_link", lambda mb: 0.5),
]

result = simulate_discrete_pipeline(stages, [8, 8, 8, 8])
assert result.total_ms == 10.0
```

The matching steady-state formula with `L=32` gives `8.0625 ms`, which is lower
because it assumes ideal cross-layer overlap and amortized bubbles.

## Phase 4: AFD Fake Backend

Phase 4 adds `mock_mode="afd"` with decode split into explicit AFD stages:

```text
GPU attention -> GPU-to-CS link -> CS rest -> CS-to-GPU link -> token_emit
```

Prefill still uses the colocated Phase 1 timing model. KV remains on the fake
GPU side; the CS stage only contributes timing.

### Sequential Decode Timing

For `pipeline_mode="sequential"`:

```text
attention_ms = attention_ms_base
             + attention_ms_per_token * batch_size
             + attention_ms_per_isl_token * context_len * batch_size

cs_rest_ms = cs_rest_ms_base
           + cs_rest_ms_per_token * batch_size

decode_step_ms = attention_ms
               + link_ms_one_way
               + cs_rest_ms
               + link_ms_one_way
```

`context_len` is the current per-request decode context length before the new
token is appended, so attention cost can grow across decode steps.

### AFD Trace Events

Sequential AFD emits:

- `decode_attention_start`
- `decode_attention_end`
- `gpu_to_cs_link_start`
- `gpu_to_cs_link_end`
- `cs_rest_start`
- `cs_rest_end`
- `cs_to_gpu_link_start`
- `cs_to_gpu_link_end`
- `token_emit`

The regular `decode_start` and `decode_end` rows still bracket the whole decode
step for compatibility with Phase 3 metrics.

### CLI Example

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --pipeline-mode sequential \
  --trace-output traces/mock_afd_trace.csv \
  --num-requests 1 \
  --isl 128 \
  --osl 8 \
  --attention-ms-base 0.4 \
  --attention-ms-per-token 0.02 \
  --attention-ms-per-isl-token 0.0001 \
  --cs-rest-ms-base 0.6 \
  --cs-rest-ms-per-token 0.03 \
  --link-ms-one-way 0.1
```

### Validation

Phase 4 tests validate:

- exact stage ordering for a single request;
- two one-way link stages per decode step;
- token timestamps matching the closed-form sequential sum;
- colocated-equivalence when AFD stage costs are configured to sum to the
  colocated decode latency and link latency is zero;
- monotonic sensitivities for link latency, context length, and CS rest time;
- KV capacity is still enforced by the fake GPU-side block manager.

## Phase 5: Pipeline Mode

Phase 5 adds two pipeline timing modes for AFD decode:

- `pipeline_mode="ideal_pipeline"`
  - Uses the steady-state cross-layer formula for multi-microbatch batches.
  - Preserves full round-trip latency for a single microbatch.
- `pipeline_mode="discrete_pipeline"`
  - Uses the event-level simulator from `pipeline_sim.py`.
  - Intended as the correctness reference for small `M`, such as 4 or 8
    microbatches.

### Microbatching

The decode batch is split with:

```text
split_into_microbatches(batch_size, microbatch_size)
```

Examples:

```text
batch_size=4, microbatch_size=1 -> [1, 1, 1, 1]
batch_size=18, microbatch_size=8 -> [8, 8, 2]
```

Stage costs are evaluated against each actual microbatch size.

### Multi-Resource Stages

The discrete pipeline mode can model multiple resources per stage:

```bash
--attention-replicas 2
--gpu-to-cs-link-resources 1
--cs-rest-resources 1
--cs-to-gpu-link-resources 1
```

This supports the common AFD sketch:

```text
attention_replica_0 \
                     -> shared CS rest
attention_replica_1 /
```

Resource IDs are written into trace notes for discrete pipeline stage events:

```text
microbatch=2;microbatch_size=4;resource=decode_attention_0
```

Discrete pipeline stage rows are resource-scoped, not request-scoped. They are
emitted once per actual microbatch/resource event with `request_id=""` and
`event_scope="resource"`. Request lifecycle rows such as `token_emit` remain
request-scoped. This avoids multiplying large-batch traces by batch size.

The default is still one resource per stage, matching the original Phase 5
behavior.

### Ideal Pipeline Formula

For uniform or non-uniform microbatches, the implementation computes:

```text
steady_state_ms = sum_j max_i s_i[j]
bubble_ms = sum_{i != bottleneck} max_j s_i[j] / L
t_ideal = steady_state_ms + bubble_ms
```

For one microbatch, ideal mode intentionally falls back to the full sequential
round trip:

```text
t_one_microbatch = attention + link + cs_rest + link
```

This keeps interactivity honest while still allowing throughput improvement for
larger batches.

### Discrete Pipeline Reference

`pipeline_mode="discrete_pipeline"` schedules every `(microbatch, stage)` event
on single-resource stages. This is the stricter small-`M` model and is expected
to be greater than or equal to the ideal formula.

For stages:

```text
attention = 1.0 ms
gpu_to_cs_link = 0.5 ms
cs_rest = 2.0 ms
cs_to_gpu_link = 0.5 ms
M = 4
```

The discrete model gives:

```text
first microbatch round trip = 4.0 ms
remaining bottleneck CS work = 3 * 2.0 ms
total = 10.0 ms
```

The ideal formula with `L=32` gives:

```text
4 * 2.0 + (1.0 + 0.5 + 0.5) / 32 = 8.0625 ms
```

### Validation

Phase 5 tests validate:

- sequential timing equals the direct stage sum;
- ideal pipeline does not reduce single-microbatch round-trip latency;
- ideal pipeline improves multi-microbatch decode time;
- discrete pipeline matches the explicit small-`M` event schedule;
- increasing link latency affects single-token interactivity much more than
  steady-state throughput when CS rest remains the bottleneck.

## Phase 6: Synthetic Workload Runner

Phase 6 adds:

```bash
python tools/run_mock_workload.py
```

The runner generates synthetic arrivals, prompt lengths, output lengths, runs
the mock engine, computes metrics, and writes simple SVG plots.

### Workload Inputs

- `--mode colocated|afd`
- `--num-requests N`
- `--arrival-process burst|poisson`
- `--arrival-rate-per-s`
- `--isl-dist fixed|lognormal`
- `--osl-dist fixed|lognormal`
- `--fixed-isl`
- `--fixed-osl`
- `--seed`
- `--include-session-ids`
- `--num-prefixes`

AFD and KV knobs from earlier phases are also supported.

### Outputs

By default, outputs go under `results/mock_workload/`:

- `mock_workload.csv`
- `mock_trace.csv`
- `mock_metrics.csv`
- `mock_summary.csv`
- `ttft_distribution.svg`
- `tbt_distribution.svg`
- `throughput_over_time.svg`
- `kv_usage_over_time.svg`
- `batch_size_over_time.svg`

The SVG plots are generated with the Python standard library so the workload
runner does not require pandas or matplotlib.

### Examples

Colocated burst:

```bash
python tools/run_mock_workload.py \
  --mode colocated \
  --num-requests 32 \
  --arrival-process burst \
  --fixed-isl 128 \
  --fixed-osl 8
```

AFD ideal pipeline with Poisson arrivals:

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

## Timing Backends

All mock virtual durations now flow through a timing backend.

`timing_backend="parametric"` is the default and preserves the original closed
form mock formulas:

```text
prefill = prefill_base_ms + isl * prefill_ms_per_token * batch_size
decode  = decode_base_ms + decode_ms_per_token * batch_size
AFD     = attention + GPU->CS link + CS rest + CS->GPU link
```

`timing_backend="gptoss_roofline"` adapts the GPT-OSS-120B decode equations
from the [original analytical model](perf_model.pdf) into the same mock runner
contract. It is decode-focused: prefill still uses the parametric formula until
a prefill roofline model is supplied.
The backend models GPU-only decode for colocated mode and GPU FMHA attention,
GPU↔CS-4 link, and CS-4 non-attention stages for AFD mode.

Useful flags:

```bash
--timing-backend parametric|gptoss_roofline
--roofline-gpu-arch helios|rubin|b200
--roofline-gpu-backend measured|roofline
--tp-g 1
--gpu-cs-link-us 12
```

Example:

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --timing-backend gptoss_roofline \
  --roofline-gpu-backend measured \
  --tp-g 1 \
  --gpu-cs-link-us 12 \
  --num-requests 4 \
  --isl 8192 \
  --osl 4 \
  --trace-output traces/gptoss_afd_trace.csv
```

The standalone DES harness uses the same timing flags:

```bash
python tools/run_des_workload.py \
  --mode afd \
  --timing-backend gptoss_roofline \
  --fixed-isl 8192 \
  --fixed-osl 16 \
  --num-requests 16 \
  --output-dir results/des_gptoss
```

For GPU-only Pareto comparisons, add `--des-batch-decode` so DES groups ready
decode requests into one colocated timing call with batch size `B`. Without that
flag, DES intentionally models individual request-token jobs flowing through a
resource queue, which is useful for resource-overlap studies but not equivalent
to a batched GPU-only roofline point.

### Roofline Validation

`tools/validate_roofline_backend.py` validates the GPT-OSS timing adapter
against the vendored raw data from `perf_model_rawdata.txt`.

It first reproduces a few Section 5 raw points, then regenerates the ISL=8K
frontiers for MI455X-only and hybrid link-latency sweeps.

```bash
python tools/validate_roofline_backend.py \
  --output-dir results/roofline_validation
```

Outputs:

- `reproduced_points.csv`
- `section5_8k_frontier.csv`
- `section5_8k_reproduction.svg`

The point checks are intentionally small and fast. They catch unit conversion,
throughput-per-GPU, link-latency, and backend-selection mistakes before trying
to interpret a full Pareto plot.

`tools/validate_afd_pareto.py` takes the hybrid Section 5 analytical frontier
points and replays each point through both the in-engine nano-vLLM mock runner
and the standalone DES harness:

```bash
python tools/validate_afd_pareto.py \
  --output-dir results/roofline_validation/afd_pareto_sim
```

For large-context AFD plots, the validator keeps the in-engine nano-vLLM mock
replay enabled by using compact synthetic prompts and a coarse mock block size.
That preserves the engine/scheduler lifecycle without allocating giant
prompt-token arrays:

```bash
python tools/validate_afd_pareto.py \
  --isl 1000000 \
  --link-us 12 \
  --output-dir results/roofline_validation/afd_pareto_1m
```

Outputs:

- `afd_analytical_des_comparison.csv`
- `afd_analytical_vs_des_replay.svg`
- `afd_link12_analytical_vs_des_overlay.svg`
- `isl8192_link12_analytical_vs_des_overlay.svg`
- `isl1000000_link12_analytical_vs_des_overlay.svg`

The analytical columns are the closed-form Frontier-style Pareto model. The
nano-vLLM mock and DES columns use finite discrete microbatch schedules with the
same roofline stage costs, so they should match each other and can sit below the
analytical curve when fill/drain overhead is not fully amortized.

The SVG names the DES side as a replay because it only evaluates configurations
that were selected by the analytical Pareto model. A true DES Pareto can differ
and requires sweeping the full design space under DES timing.
