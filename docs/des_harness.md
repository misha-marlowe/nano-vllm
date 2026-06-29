# DES Harness

The `des-harness` branch adds a standalone discrete-event simulator next to the
simple nano-vLLM mock backend.

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

It is additive: it does not replace `LLMEngine.step()` or the simple mock.

## Entry Point

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

## Comparison Against Simple Mock

The tests intentionally compare both harnesses.

Expected matches:

- Single-request colocated timing matches exactly.
- Single-request sequential AFD timing matches exactly.

Expected gaps:

- Multi-request AFD can diverge.
- The simple mock batches a scheduled decode batch and computes one batch-level
  AFD latency.
- The DES harness models each request as work flowing through explicit resources,
  so shared CS/link resources can queue and increase tail latency.

This gap is useful, not a bug. It shows where the simple mock is optimistic and
where the DES harness exposes resource contention.

## Current Limits

- DES does not yet reuse nano-vLLM's `Scheduler` or `BlockManager`.
- KV accounting is token/block-level, not the exact nano-vLLM block table.
- DES batches are per-request today; independent per-role batch schedulers can
  be added next.
- Runtime adapters such as speculative decoding are intentionally out of scope
  for this branch.

## Validation

Run:

```bash
python -m pytest tests/mock_backend/test_des_harness.py
python -m pytest tests/mock_backend
```

The comparison tests verify:

- DES equals simple mock for single-request colocated.
- DES equals simple mock for single-request AFD.
- DES explains multi-request AFD divergence through explicit CS queue delay.
- More attention replicas improve an attention-bottleneck DES workload.
