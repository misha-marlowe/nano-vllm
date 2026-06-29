# DES Harness And nano-vLLM-DES

The repository has two DES paths:

- `tools/run_des_workload.py` runs the standalone DES harness directly.
- `--mock-runner des` runs nano-vLLM through `LLMEngine.step()` and delegates
  each scheduled decode batch to the same DES timing engine.
- `nanovllm/mock/global_pipeline.py` provides saturated global AFD replay for
  Pareto validation. It replays several ready batches through persistent
  attention, link, and CS resources to approximate cross-batch pipeline overlap.

The simple mock remains centered on:

```text
LLMEngine.step()
  Scheduler.schedule()
  FakeRunner.run()
  Scheduler.postprocess()
```

The DES harness is centered on:

```text
event_queue.pop()
  update request/resource state
  reserve downstream resource
  push completion event
```

The in-engine nano-vLLM-DES runner is centered on:

```text
LLMEngine.step()
  Scheduler.schedule()
  FakeDESRunner.run()
    DESEngine.run() for the scheduled decode batch
  Scheduler.postprocess()
```

Both DES paths are additive: they do not replace the simple fake runner.
The global replay helper is also additive and is currently used by
`tools/validate_afd_pareto.py`; it is not yet an `LLMEngine` control loop.

## Entry Point

In-engine nano-vLLM-DES:

```bash
python tools/run_mock_trace.py \
  --mock-mode afd \
  --mock-runner des \
  --timing-backend gptoss_roofline \
  --num-requests 16 \
  --isl 8192 \
  --osl 8 \
  --trace-output traces/mock_afd_des_trace.csv
```

Standalone DES:

```bash
python tools/run_des_workload.py \
  --mode afd \
  --num-requests 16 \
  --arrival-process burst \
  --attention-replicas 2 \
  --gpu-to-cs-link-resources 1 \
  --cs-rest-resources 1 \
  --cs-to-gpu-link-resources 1
```

## What It Models

- Global event queue ordered by virtual time.
- Per-request state, including `session_id` and `prefix_id` fields.
- Explicit resources:
  - `prefill`
  - `decode` for colocated mode
  - `decode_attention_N`
  - `gpu_to_cs_link_N`
  - `cs_rest_N`
  - `cs_to_gpu_link_N`
- Resource queue delay.
- Resource IDs in trace rows.
- AFD transfer events as queued resources rather than runner-internal markers.
- Optional colocated decode batching with `--des-batch-decode`, which groups
  ready decode requests into one `colocated_decode_ms(B, context)` timing call.
- The in-engine nano-vLLM-DES runner uses the same batch-decode model for the
  decode batch selected by `Scheduler.schedule()`.
- Global AFD replay keeps stage resources persistent across multiple ready
  batches, which amortizes fill/drain bubbles and models saturated cross-batch
  overlap more closely than the batch-scoped runner.

## Comparison Against Simple Mock

The tests intentionally compare both harnesses.

Expected matches:

- Single-request colocated timing matches exactly.
- Single-request sequential AFD timing matches exactly.

Expected gaps:

- Multi-request AFD can diverge.
- The simple mock batches a scheduled decode batch and computes one batch-level
  AFD latency.
- nano-vLLM-DES batches at the nano-vLLM scheduler boundary and then runs DES
  timing inside that boundary.
- Global AFD replay can improve throughput relative to nano-vLLM-DES when the
  workload has enough ready batches to keep attention/link/CS resources fed.
  It is optimistic because all replay batches are available at time zero.
- The DES harness models each request as work flowing through explicit resources,
  so shared CS/link resources can queue and increase tail latency.
- In colocated mode, DES uses one-token resource jobs by default. Enable
  `--des-batch-decode` for GPU-only Pareto reproduction.

This gap is useful, not a bug. It shows where the simple mock is optimistic and
where the DES harness exposes resource contention.

## Current Limits

- Standalone DES does not reuse nano-vLLM's `Scheduler` or `BlockManager`.
- nano-vLLM-DES reuses `Scheduler` and `BlockManager`, but DES timing is scoped
  to one scheduled decode batch at a time.
- Standalone DES KV accounting is token/block-level, not the exact nano-vLLM
  block table.
- Global AFD replay is timing-only. It does not run nano-vLLM request admission,
  token postprocess, or KV/block allocation; use nano-vLLM-DES tests for those.
- `--des-batch-decode` is a compact batching model for colocated decode, not a
  full copy of `Scheduler.schedule()`. Independent per-role batch schedulers can
  be added next.
- Runtime adapters such as speculative decoding are intentionally out of scope
  for this branch.

## Validation

Run:

```bash
python -m pytest tests/mock_backend/test_des_harness.py
python -m pytest tests/mock_backend
```

GPU-only Pareto-style colocated decode:

```bash
python tools/run_des_workload.py \
  --mode colocated \
  --des-batch-decode \
  --des-max-batch-size 256 \
  --timing-backend gptoss_roofline \
  --fixed-isl 8192 \
  --fixed-osl 8 \
  --num-requests 256 \
  --prefill-base-ms 0 \
  --prefill-ms-per-token 0
```

The comparison tests verify:

- DES equals simple mock for single-request colocated.
- DES equals simple mock for single-request AFD.
- DES explains multi-request AFD divergence through explicit CS queue delay.
- More attention replicas improve an attention-bottleneck DES workload.
- nano-vLLM-DES emits DES resource events while preserving nano-vLLM scheduler
  and token postprocess flow.
- Global AFD replay equals batched DES for one replay batch and improves
  throughput for CS-heavy multi-batch replay by amortizing pipeline fill/drain.
