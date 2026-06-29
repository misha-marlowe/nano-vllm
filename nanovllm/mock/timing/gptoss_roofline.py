from __future__ import annotations

from dataclasses import dataclass

from nanovllm.mock.timing.backends import AFDStageDurations
from nanovllm.mock.timing.ac_model.hbm_efficiency_model import hbm_efficiency_pct
from nanovllm.mock.timing.ac_model import cs4_offload as CS4M


DTYPE_BYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}


@dataclass(frozen=True)
class Arch:
    name: str
    hbm_gbps: float
    hbm_capacity_gb: float
    peak_tflops: dict[str, float]
    gpus_per_rack: int
    link_gbps: float
    link_lat_us: float
    rack_power_kw: float = 0.0
    attn_hbm_eff: float = 0.866


HELIOS = Arch(
    "Helios MI455X",
    hbm_gbps=19600.0,
    hbm_capacity_gb=432.0,
    peak_tflops={"bf16": 4000.0, "fp8": 20000.0, "fp4": 40000.0},
    gpus_per_rack=72,
    link_gbps=3600.0,
    link_lat_us=0.7,
    rack_power_kw=246.0,
)
RUBIN = Arch(
    "Rubin NVL72",
    hbm_gbps=22000.0,
    hbm_capacity_gb=288.0,
    peak_tflops={"bf16": 4000.0, "fp8": 17500.0, "fp4": 50000.0},
    gpus_per_rack=72,
    link_gbps=3600.0,
    link_lat_us=0.7,
    rack_power_kw=250.0,
)
B200 = Arch(
    "B200 (DGX node)",
    hbm_gbps=8000.0,
    hbm_capacity_gb=192.0,
    peak_tflops={"bf16": 2250.0, "fp8": 4500.0, "fp4": 9000.0},
    gpus_per_rack=8,
    link_gbps=1800.0,
    link_lat_us=0.5,
)
ARCHES = {"helios": HELIOS, "rubin": RUBIN, "b200": B200}


@dataclass(frozen=True)
class Model:
    name: str = "gpt-oss-120B"
    L: int = 36
    d: int = 2880
    q_dim: int = 4096
    kv_dim: int = 512
    n_kv_heads: int = 8
    qkv_out: int = 5120
    n_experts: int = 128
    topk: int = 4
    fc1: int = 5760
    inter: int = 2880
    vocab: int = 201088
    sliding_window: int = 128
    sliding_every: int = 2
    w_proj: str = "bf16"
    w_moe: str = "fp4"
    kv: str = "fp8"
    act: str = "bf16"


GPTOSS = Model()
P_MI455X_KW = HELIOS.rack_power_kw / HELIOS.gpus_per_rack
_MGPU = None


def _mgpu():
    global _MGPU
    if _MGPU is None:
        from nanovllm.mock.timing.ac_model.measured_gpu import MeasuredGPU

        _MGPU = MeasuredGPU()
    return _MGPU


def layer_split(m: Model):
    n_sliding = m.L // m.sliding_every
    return m.L - n_sliding, n_sliding


def gemm_time(arch: Arch, M: float, N: float, B: int, w_dtype: str, act_dtype: str = "bf16") -> float:
    bytes_ = M * N * DTYPE_BYTES[w_dtype] + B * (M + N) * DTYPE_BYTES[act_dtype]
    eff = hbm_efficiency_pct(B, clamp=True) / 100.0
    return max(
        bytes_ / (arch.hbm_gbps * 1e9 * eff),
        (2.0 * M * N * B) / (arch.peak_tflops[w_dtype] * 1e12),
    )


def fmha_time(arch: Arch, m: Model, B: int, kv_len: int, kv_shard: int, head_par: int) -> float:
    kv_bytes = B * kv_len * (2 * m.kv_dim / kv_shard) * DTYPE_BYTES[m.kv]
    t_mem = kv_bytes / (arch.hbm_gbps * 1e9 * arch.attn_hbm_eff)
    t_comp = (4.0 * B * kv_len * m.q_dim / head_par) / (arch.peak_tflops["fp8"] * 1e12)
    return max(t_mem, t_comp)


def experts_active(B: int, E: int, k: int) -> float:
    return E * (1.0 - (1.0 - k / E) ** B)


def tp_allreduce(arch: Arch, S: float, P: int) -> float:
    if P <= 1 or arch.link_gbps <= 0:
        return 0.0
    return 2 * arch.link_lat_us * 1e-6 + (2 * (P - 1) / P) * S / (arch.link_gbps * 1e9)


def all_to_all(arch: Arch, S_egress: float, P: int) -> float:
    if P <= 1 or arch.link_gbps <= 0:
        return 0.0
    return arch.link_lat_us * 1e-6 + ((P - 1) / P) * S_egress / (arch.link_gbps * 1e9)


def all_gather(arch: Arch, S: float, P: int) -> float:
    if P <= 1 or arch.link_gbps <= 0:
        return 0.0
    return arch.link_lat_us * 1e-6 + ((P - 1) / P) * S / (arch.link_gbps * 1e9)


def attn_layer(arch: Arch, m: Model, B: int, kv_len: int, P: int) -> float:
    kvs = min(P, m.n_kv_heads)
    t_qkv = gemm_time(arch, m.d, m.qkv_out / P, B, m.w_proj)
    t_fmha = fmha_time(arch, m, B, kv_len, kvs, P)
    t_o = gemm_time(arch, m.q_dim / P, m.d, B, m.w_proj)
    comm = tp_allreduce(arch, B * m.d * DTYPE_BYTES[m.act], P)
    return t_qkv + t_fmha + t_o + comm


def moe_layer(arch: Arch, m: Model, B: int, P: int) -> tuple[float, str]:
    active = experts_active(B, m.n_experts, m.topk)
    pe_params = m.d * m.fc1 + m.inter * m.d
    routes = B * m.topk
    b_e = max(1.0, routes / active)
    act_b = B * (m.d + m.fc1) * DTYPE_BYTES[m.act]

    def compute(par: float) -> float:
        w = active * pe_params * DTYPE_BYTES[m.w_moe] / par
        flops = 2.0 * routes * pe_params / par
        return max(
            (w + act_b) / (arch.hbm_gbps * 1e9 * hbm_efficiency_pct(b_e, clamp=True) / 100.0),
            flops / (arch.peak_tflops[m.w_moe] * 1e12),
        )

    t_tp = compute(P) + tp_allreduce(arch, B * m.d * DTYPE_BYTES[m.act], P)
    egress = (B / P) * m.topk * m.d * DTYPE_BYTES[m.act]
    t_ep = compute(min(active, P)) + 2.0 * all_to_all(arch, egress, P)
    return (t_tp, "TP") if t_tp <= t_ep else (t_ep, "EP")


def tpot_seconds(
    arch: Arch,
    m: Model,
    B: int,
    isl: int,
    P: int,
    *,
    backend: str = "measured",
    want_strategy: bool = False,
    all_full: bool = False,
) -> float | tuple[float, str]:
    if backend == "measured":
        a, r = _mgpu().layer_times(arch, B, isl, P)
        lm = gemm_time(arch, m.d, m.vocab / P, B, m.w_proj) + all_gather(
            arch, B * m.vocab * DTYPE_BYTES[m.act], P
        )
        total = (a + r) * m.L + lm
        return (total, "meas") if want_strategy else total
    n_full, n_sld = (m.L, 0) if all_full else layer_split(m)
    attn_full = attn_layer(arch, m, B, isl, P)
    attn_sld = attn_layer(arch, m, B, min(isl, m.sliding_window), P)
    moe_t, strat = moe_layer(arch, m, B, P)
    lm = gemm_time(arch, m.d, m.vocab / P, B, m.w_proj) + all_gather(
        arch, B * m.vocab * DTYPE_BYTES[m.act], P
    )
    total = n_full * attn_full + n_sld * attn_sld + m.L * moe_t + lm
    return (total, strat) if want_strategy else total


def gpu_fmha_total(arch: Arch, m: Model, B: int, isl: int, P: int, *, backend: str = "measured") -> float:
    if backend == "measured":
        return _mgpu().layer_times(arch, B, isl, P)[0] * m.L
    n_full, n_sld = layer_split(m)
    kvs = min(P, m.n_kv_heads)
    return (
        n_full * fmha_time(arch, m, B, isl, kvs, P)
        + n_sld * fmha_time(arch, m, B, min(isl, m.sliding_window), kvs, P)
    )


def stage_times(arch: Arch, m: Model, ck: int, isl: int, tp_g: int, *, backend: str = "measured"):
    a = gpu_fmha_total(arch, m, ck, isl, tp_g, backend=backend) / m.L
    f = CS4M.cs4_nonattn_us(ck) * 1e-6
    c = CS4M.comm_us(ck) * 1e-6 / 2.0
    return a, f, c


def pipe_closed(a: float, f: float, c: float, s: int):
    latency = a + 2 * c + f
    bottleneck = max(a, f)
    block = max(s * bottleneck, latency)
    return {"L": latency, "b": bottleneck, "block": block, "exposed": max(0.0, latency - s * bottleneck)}


def hybrid_point(
    arch: Arch,
    m: Model,
    gb: int,
    isl: int,
    tp_g: int,
    a_g: int,
    ck: int,
    *,
    backend: str = "measured",
):
    s = gb // (a_g * ck)
    if s < 1 or a_g * ck * s != gb:
        return None
    a, f, c = stage_times(arch, m, ck, isl, tp_g, backend=backend)
    pc = pipe_closed(a, a_g * f, c, s)
    thr = gb / (m.L * pc["block"])
    intv = 1.0 / (m.L * pc["L"])
    pw = a_g * tp_g * P_MI455X_KW + CS4M.P_CS4_UNIT_KW
    return {
        "x": intv,
        "y": thr / (a_g * tp_g),
        "thr": thr,
        "tpg": thr / (a_g * tp_g),
        "eff": thr / pw,
        "gb": gb,
        "tp_g": tp_g,
        "a_g": a_g,
        "ck": ck,
        "s": s,
        "gpus": a_g * tp_g,
        "pw": pw,
        "exposed": pc["exposed"],
    }


def gpu_only_point(arch: Arch, m: Model, B: int, isl: int, tp_g: int, *, backend: str = "measured"):
    t = tpot_seconds(arch, m, B, isl, tp_g, backend=backend)
    return {
        "x": 1.0 / t,
        "y": (B / t) / tp_g,
        "thr": B / t,
        "tpg": (B / t) / tp_g,
        "B": B,
        "tp_g": tp_g,
        "gpus": tp_g,
        "pw": tp_g * P_MI455X_KW,
        "eff": (B / t) / (tp_g * P_MI455X_KW),
    }


def pareto_uplr(points: list[dict], ykey: str = "y") -> list[dict]:
    return sorted(
        [
            p
            for p in points
            if not any(
                q is not p
                and q["x"] >= p["x"]
                and q[ykey] >= p[ykey]
                and (q["x"] > p["x"] or q[ykey] > p[ykey])
                for q in points
            )
        ],
        key=lambda p: p["x"],
    )


class GPTOSSRooflineTimingBackend:
    """GPT-OSS-120B decode timing backend adapted from the original analytical model.

    The backend is decode-focused. Prefill intentionally remains on the
    parametric mock formula until a prefill roofline model is supplied.
    """

    name = "gptoss_roofline"

    def __init__(self, config):
        self.config = config
        self.model = GPTOSS
        self.arch = ARCHES[config.roofline_gpu_arch]

    @property
    def backend(self) -> str:
        return self.config.roofline_gpu_backend

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        return tpot_seconds(
            self.arch,
            self.model,
            batch_size,
            context_len,
            self.config.roofline_tp_g,
            backend=self.backend,
        ) * 1e3

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        old_link = CS4M.CLOS_LAT_US
        CS4M.CLOS_LAT_US = self.config.gpu_cs_link_us
        try:
            attention_s, cs_rest_s, link_s = stage_times(
                self.arch,
                self.model,
                microbatch_size,
                context_len,
                self.config.roofline_tp_g,
                backend=self.backend,
            )
        finally:
            CS4M.CLOS_LAT_US = old_link
        return AFDStageDurations(
            attention_ms=attention_s * 1e3,
            gpu_to_cs_link_ms=link_s * 1e3,
            cs_rest_ms=cs_rest_s * 1e3,
            cs_to_gpu_link_ms=link_s * 1e3,
            notes=(
                "timing_backend=gptoss_roofline;"
                f"arch={self.config.roofline_gpu_arch};"
                f"gpu_backend={self.backend};tp_g={self.config.roofline_tp_g};"
                f"link_us={self.config.gpu_cs_link_us}"
            ),
        )
