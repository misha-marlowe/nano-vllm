# Mock Backend Code Map

This map captures the request lifecycle in upstream nano-vLLM before adding the
CPU-only mock backend. The goal is to preserve the scheduler, batching, sequence
state, and block manager paths while replacing only the CUDA model execution
boundary.

## Request Entry Point

- `nanovllm/llm.py`
  - `LLM` is a thin subclass of `LLMEngine`.
- `nanovllm/engine/llm_engine.py`
  - `LLMEngine.generate(prompts, sampling_params, use_tqdm=True)` is the batch
    user-facing API.
  - `LLMEngine.add_request(prompt, sampling_params)` tokenizes string prompts,
    constructs a `Sequence`, and appends it to the scheduler waiting queue.

## Engine Loop

- `nanovllm/engine/llm_engine.py`
  - `LLMEngine.step()` is the core serving iteration:
    1. `Scheduler.schedule()` returns selected sequences and whether the batch
       is prefill or decode.
    2. `model_runner.call("run", seqs, is_prefill)` executes the model runner.
    3. `Scheduler.postprocess()` updates KV/block accounting and appends one
       sampled token per sequence when appropriate.
    4. Finished sequence outputs are returned to `generate()`.
  - `LLMEngine.is_finished()` delegates to `Scheduler.is_finished()`.

## Scheduler Entry Point

- `nanovllm/engine/scheduler.py`
  - `Scheduler.add(seq)` appends new requests to `waiting`.
  - `Scheduler.schedule()` implements both scheduling phases:
    - Prefill pulls from `waiting`, respects `max_num_seqs` and
      `max_num_batched_tokens`, allocates blocks, supports chunked prefill for
      the first sequence, and moves fully-prefilled sequences to `running`.
    - Decode pulls from `running`, ensures one append slot via the block
      manager, may preempt if no block is available, and schedules one token per
      running sequence.
  - `Scheduler.postprocess(seqs, token_ids, is_prefill)` hashes completed blocks,
    advances cached-token state, appends generated tokens, detects finish, and
    deallocates blocks.

## Block / KV Manager

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

## Sequence State

- `nanovllm/engine/sequence.py`
  - `Sequence` owns prompt and completion token IDs, `num_cached_tokens`,
    `num_scheduled_tokens`, `block_table`, and status.
  - `append_token()` is the only normal token-growth path used after runner
    output.
  - `num_blocks`, `last_block_num_tokens`, and `block(i)` are the block manager's
    view of KV storage requirements.

## Model Runner Boundary

- `nanovllm/engine/model_runner.py`
  - `ModelRunner.call(method_name, *args)` is the engine-facing dispatch
    boundary.
  - `ModelRunner.run(seqs, is_prefill)` prepares tensors, runs the model, samples
    token IDs, resets attention context, and returns one token ID per sequence.
  - This is the best injection point for a CPU-only fake runner because the
    scheduler only needs a list of generated token IDs and does not depend on
    model tensors directly.

## Token Postprocessing

- `nanovllm/engine/scheduler.py`
  - `Scheduler.postprocess()` consumes runner token IDs.
  - During full or final prefill, it appends the first generated token for each
    sequence.
  - During decode, it appends one generated token per scheduled sequence.
  - It marks requests finished when EOS is generated or
    `seq.num_completion_tokens == seq.max_tokens`.

## CUDA / NCCL Assumptions

- `nanovllm/config.py`
  - `Config.__post_init__()` requires `model` to be a local directory and loads
    Hugging Face config via `AutoConfig.from_pretrained()`.
- `nanovllm/engine/llm_engine.py`
  - Always constructs `ModelRunner`.
  - Spawns tensor-parallel workers for `tensor_parallel_size > 1`.
  - Always loads `AutoTokenizer.from_pretrained()`.
- `nanovllm/engine/model_runner.py`
  - Initializes `torch.distributed` with NCCL.
  - Calls `torch.cuda.set_device()`.
  - Sets default device to CUDA.
  - Allocates real model weights.
  - Uses CUDA tensors and pinned-memory transfers.
  - Uses CUDA memory queries to size KV cache.
  - Optionally captures CUDA graphs.
- `nanovllm/layers/*` and `nanovllm/models/qwen3.py`
  - Real model path assumes CUDA tensors and attention kernels.

## Recommended Injection Point

Add a config-gated runner selection in `LLMEngine.__init__()`:

- Normal mode keeps the existing `ModelRunner` path unchanged.
- `mock_backend=True` constructs a CPU-only `FakeColocatedRunner` instead.

The fake runner should implement the same minimal interface as `ModelRunner`:

```python
runner.call("run", seqs, is_prefill) -> list[int]
runner.call("exit")
```

This keeps `LLMEngine.step()`, `Scheduler.schedule()`,
`Scheduler.postprocess()`, `Sequence`, and `BlockManager` on the real serving
lifecycle. The mock path should also bypass tokenizer/model config loading when
prompts are already token IDs so tests and synthetic workloads can run on macOS
without CUDA, NCCL, Triton, flash-attn, or local model weights.
