# Phase 1: CPU-Only Colocated Mock Backend

Phase 1 adds a Mac-runnable mock serving path that keeps nano-vLLM's engine,
scheduler, sequence lifecycle, and block manager active while replacing real
model execution with deterministic token generation and a virtual latency model.

## What Runs

- `LLMEngine.step()` remains the serving loop.
- `Scheduler.schedule()` still chooses prefill and decode batches.
- `BlockManager` still allocates, appends, hashes, and releases KV blocks.
- `Scheduler.postprocess()` still advances sequence state and finishes requests.
- `FakeColocatedRunner` replaces CUDA model execution when
  `mock_backend=True`.

## What Is Mocked

- No model weights are loaded.
- No CUDA tensors are created.
- No NCCL, Triton, flash-attn, CUDA graphs, or real attention kernels are used.
- Token IDs are deterministic:

```text
mock_token_base + seq_id * 100000 + current_completion_tokens + 1
```

For Phase 1 mock mode, completion tokens are emitted only from decode iterations
so an output length of 8 means exactly 8 decode steps and 8 `token_emit` events.
The real CUDA path keeps upstream nano-vLLM's prefill-token behavior.

## Timing Model

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

## Trace Events

The trace writer emits CSV rows with these columns:

```text
virtual_time_ms, request_id, stage, batch_size, isl, generated_tokens,
token_id, kv_tokens_used, notes
```

Phase 1 stages:

- `request_arrival`
- `prefill_start`
- `prefill_end`
- `decode_start`
- `decode_end`
- `token_emit`
- `request_finish`

## Run Command

From the repo root:

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

Example multi-request batch:

```bash
python tools/run_mock_trace.py \
  --trace-output traces/mock_batch_trace.csv \
  --num-requests 3 \
  --arrivals-ms 0 \
  --isl 64 \
  --osl 4
```

Example staggered arrivals:

```bash
python tools/run_mock_trace.py \
  --trace-output traces/mock_staggered_trace.csv \
  --num-requests 3 \
  --arrivals-ms 0,2.5,8.0 \
  --isl 128,32,32 \
  --osl 4,2,2
```

## Validation

Phase 1 tests live under `tests/mock_backend/`:

```bash
pytest tests/mock_backend
```

They validate:

- one ISL=128, OSL=8 request has one prefill, eight decode steps, eight emitted
  tokens, and closed-form virtual timing;
- same-time requests batch together when scheduler limits allow it;
- staggered arrivals appear in the trace and are scheduled after arrival.

## Notes

- `pyproject.toml` keeps CPU dependencies required and moves CUDA-only packages
  (`triton`, `flash-attn`) into the optional `cuda` extra.
- Prefix cache and deeper KV-capacity validation are left for Phase 2.
