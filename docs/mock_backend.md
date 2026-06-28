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
virtual_time_ms, request_id, stage, batch_size, isl, generated_tokens,
token_id, kv_tokens_used, kv_blocks_used, kv_blocks_free,
kv_capacity_tokens, notes
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
pytest tests/mock_backend
```

Phase 2 tests validate KV growth, block release, admission waits under limited
capacity, and prefix-cache reuse.

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
