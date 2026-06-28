import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.fake_runner import FakeColocatedRunner
from nanovllm.engine.mock_trace import MockTraceWriter, VirtualClock


class MockTokenizer:

    eos_token_id = -1

    def encode(self, prompt: str) -> list[int]:
        return [ord(ch) % 256 for ch in prompt] or [0]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.ps = []
        self.events = []
        self.clock = VirtualClock() if config.mock_backend else None
        self.trace = MockTraceWriter(config.trace_output) if config.mock_backend else None
        if config.mock_backend:
            self.model_runner = FakeColocatedRunner(config)
            self.tokenizer = MockTokenizer()
        else:
            from nanovllm.engine.model_runner import ModelRunner

            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        self._trace_event(seq, "request_arrival", isl=seq.num_prompt_tokens, notes="queued")

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        stage = "prefill" if is_prefill else "decode"
        self._trace_batch(seqs, f"{stage}_start")
        completion_counts = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self._advance_mock_time()
        self._trace_batch(seqs, f"{stage}_end")
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self._trace_token_emits(seqs, token_ids, completion_counts)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        for seq in seqs:
            if seq.is_finished:
                self._trace_event(seq, "request_finish", token_id=seq.last_token, notes="finished")
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def _advance_mock_time(self):
        if not self.config.mock_backend:
            return
        self.clock.advance(self.model_runner.last_latency_ms)

    def _trace_batch(self, seqs: list[Sequence], stage: str):
        if not self.trace:
            return
        batch_size = len(seqs)
        isl = max(seq.num_prompt_tokens for seq in seqs)
        kv_tokens = sum(len(seq) for seq in self.scheduler.running) + sum(len(seq) for seq in self.scheduler.waiting)
        for seq in seqs:
            self.trace.emit(
                self.clock.time_ms,
                seq.seq_id,
                stage,
                batch_size=batch_size,
                isl=isl,
                generated_tokens=seq.num_completion_tokens,
                kv_tokens_used=kv_tokens,
            )

    def _trace_token_emits(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        completion_counts: dict[int, int],
    ):
        if not self.trace:
            return
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_completion_tokens <= completion_counts[seq.seq_id]:
                continue
            self._trace_event(seq, "token_emit", token_id=token_id, notes="deterministic")

    def _trace_event(
        self,
        seq: Sequence,
        stage: str,
        batch_size: int = 0,
        isl: int | None = None,
        token_id: int | str = "",
        notes: str = "",
    ):
        if not self.trace:
            return
        self.trace.emit(
            self.clock.time_ms,
            seq.seq_id,
            stage,
            batch_size=batch_size,
            isl=seq.num_prompt_tokens if isl is None else isl,
            generated_tokens=seq.num_completion_tokens,
            token_id=token_id,
            kv_tokens_used=0 if seq.is_finished else len(seq),
            notes=notes,
        )
