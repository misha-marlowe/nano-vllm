from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AFDStageDurations:
    attention_ms: float
    gpu_to_cs_link_ms: float
    cs_rest_ms: float
    cs_to_gpu_link_ms: float
    notes: str = ""


class TimingBackend(Protocol):
    name: str

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        ...

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        ...

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        ...


class ParametricTimingBackend:
    """Compatibility backend for the original mock timing formulas."""

    name = "parametric"

    def __init__(self, config):
        self.config = config

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        return self.config.decode_base_ms + self.config.decode_ms_per_token * batch_size

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        attention_ms = (
            self.config.attention_ms_base
            + self.config.attention_ms_per_token * microbatch_size
            + self.config.attention_ms_per_isl_token * context_len * microbatch_size
        )
        cs_rest_ms = self.config.cs_rest_ms_base + self.config.cs_rest_ms_per_token * microbatch_size
        return AFDStageDurations(
            attention_ms=attention_ms,
            gpu_to_cs_link_ms=self.config.link_ms_one_way,
            cs_rest_ms=cs_rest_ms,
            cs_to_gpu_link_ms=self.config.link_ms_one_way,
            notes="timing_backend=parametric",
        )


def build_timing_backend(config) -> TimingBackend:
    if config.timing_backend == "parametric":
        return ParametricTimingBackend(config)
    if config.timing_backend == "gptoss_roofline":
        from nanovllm.mock.timing.gptoss_roofline import GPTOSSRooflineTimingBackend

        return GPTOSSRooflineTimingBackend(config)
    raise ValueError(f"unknown timing backend: {config.timing_backend}")
