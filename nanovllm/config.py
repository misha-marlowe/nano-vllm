import os
from math import ceil
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mock_backend: bool = False
    mock_mode: str = "colocated"
    virtual_time: bool = False
    trace_output: str = "traces/mock_trace.csv"
    prefill_base_ms: float = 1.0
    prefill_ms_per_token: float = 0.01
    decode_base_ms: float = 0.5
    decode_ms_per_token: float = 0.02
    mock_token_base: int = 1000

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.mock_backend:
            assert self.mock_mode == "colocated", "Phase 1 only supports colocated mock mode"
            if self.num_kvcache_blocks == -1:
                total_tokens = self.max_num_seqs * self.max_model_len
                self.num_kvcache_blocks = max(1, ceil(total_tokens / self.kvcache_block_size))
            return
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
