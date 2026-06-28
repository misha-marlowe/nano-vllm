from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence


class FakeColocatedRunner:
    """CPU-only runner that preserves the engine/scheduler contract.

    The real runner returns one sampled token per scheduled sequence. This
    runner does the same without tensors, CUDA state, model weights, or sleeps.
    The engine consumes `last_latency_ms` to advance virtual time and write
    traces.
    """

    def __init__(self, config: Config):
        self.config = config
        self.last_latency_ms = 0.0

    def call(self, method_name, *args):
        method = getattr(self, method_name)
        return method(*args)

    def exit(self):
        return None

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        batch_size = len(seqs)
        if is_prefill:
            isl = max(seq.num_scheduled_tokens for seq in seqs)
            self.last_latency_ms = (
                self.config.prefill_base_ms
                + isl * self.config.prefill_ms_per_token * batch_size
            )
        else:
            self.last_latency_ms = (
                self.config.decode_base_ms
                + self.config.decode_ms_per_token * batch_size
            )
        return [self._next_token_id(seq) for seq in seqs]

    def _next_token_id(self, seq: Sequence) -> int:
        return self.config.mock_token_base + seq.seq_id * 100000 + seq.num_completion_tokens + 1
