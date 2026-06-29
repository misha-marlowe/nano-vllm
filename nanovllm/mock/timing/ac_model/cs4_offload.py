#!/usr/bin/env python3
"""
CS-4 non-attention offload model for the GPT-OSS-120B decode hybrid (AMD MI455X attention + CS-4 "rest").
Ported from latency-study/hybrid_pareto_model.py (`cs3_nonattn_us`, on-wafer-movement version) and
latency_summary_report.py, switched from CS-3 -> CS-4 (meshgemv_arch_model.CS4) and given a 3-level-CLOS
GPU<->CS-4 link.

Offload split (per layer): GPU keeps FMHA; CS-4 does QKV-proj, O-proj, MoE router, MoE experts (fc1/fc2),
plus WaferLLM on-wafer data movement (MoE attn<->expert all-to-all + input replication + inter-wafer EP).
CS-4 is ONE 2-wafer unit, never replicated -> cs4_nonattn_us depends on B only (not on the GPU TP degree).
"""
from .meshgemv_arch_model import model_batched, CS4, moe_a2a_cycles, bcast_cycles

# ---- GPT-OSS-120B dims (same as latency-study) ----
D, Q_DIM, KV_DIM = 2880, 4096, 512
QKV_OUT, EXPERT_INTER, FC1 = 5120, 2880, 5760
EXPERTS, TOPK, N_LAYERS = 128, 4, 36

# ---- CS-4 on-wafer + link ----
FREQ        = CS4["freq"]              # 1.1 GHz (cycles -> seconds)
CS4_IO_BW   = 2.4e12 / 8               # off-wafer I/O: 2.4 Tbps -> 300 GB/s (also the inter-wafer link)
CS4_SRAM_BW = 41e15                    # ON-wafer aggregate SRAM BW = 41 PB/s/wafer (datasheet; CS-3 = 21 PB/s)
WAFERS_PER_UNIT = 2                    # MoE ~57 GB MXFP4 -> 2 wafers hold all experts (EP unit)
P_CS4_WAFER_KW = 47.0                  # max system power per wafer (datasheet; CS-3 = 27 kW)
P_CS4_UNIT_KW  = P_CS4_WAFER_KW * WAFERS_PER_UNIT   # 94 kW per 2-wafer EP unit
MODEL_ONWAFER_MOVE = True              # charge WaferLLM on-wafer movement (False -> compute-only)
IW_LINK_LAT_US = 3.0                   # ASSUMED inter-wafer (wafer<->wafer) one-way latency

# ---- GPU<->CS-4 link: CLOS of Broadcom switches + NICs (real latencies) ----
SWITCH_HOP_NS    = 800.0               # per switch hop: Broadcom TH6 = 800 ns (TH5 = 700)
CLOS_HOPS_ONEWAY = 5                   # 3-level (leaf-spine-ssp-spine-leaf) = 5 hops; 2-level (leaf-spine-leaf) = 3
NIC_US           = 1.0                 # per NIC; a sender->receiver flow crosses 2 (sender + receiver)
FLOWS_PER_LAYER  = 2                   # Q,K,V (CS-4->GPU) + attn-out (GPU->CS-4), both exposed per layer
# one-way flow = hops*hop_lat + 2 NICs (= 5*0.8 + 2 = 6.0 us TH6/3-level); x2 transfers -> 12.0 us/layer
CLOS_LAT_US      = FLOWS_PER_LAYER * (CLOS_HOPS_ONEWAY * SWITCH_HOP_NS / 1000.0 + 2 * NIC_US)

# ---- expert replication, capacity-bounded (never full); default OFF (sharded) ----
CS4_SRAM_GB    = 44.0 * WAFERS_PER_UNIT
_per_expert_gb = N_LAYERS * (D * FC1 + EXPERT_INTER * D) * 0.5 / 1e9
_moe_gb        = EXPERTS * _per_expert_gb
_dense_gb      = N_LAYERS * (D * QKV_OUT + Q_DIM * D + D * EXPERTS) * 2 / 1e9
_spare_gb      = CS4_SRAM_GB - _moe_gb - _dense_gb
MAX_REPL_FRAC  = max(0.0, min(0.95, (_spare_gb / _per_expert_gb) / EXPERTS))
EXPERT_REPL      = False
EXPERT_REPL_FRAC = MAX_REPL_FRAC if EXPERT_REPL else 0.0


def experts_active(B):
    return max(1, EXPERTS * (1.0 - (1.0 - TOPK / EXPERTS) ** B))


def gemv_us(B, M, N, w_bytes):
    """Best-P batched MeshGEMV time (us) on CS-4. FP4 weights (w_bytes=0.5) shrink the read term."""
    best, P = None, 8
    while P <= 950:
        mt, nt = M / P, N / P
        if mt >= 1 and nt >= 1 and mt * nt * w_bytes <= 48 * 1024:   # weight tile fits 48 KB/PE
            _, _, tot = model_batched(B, M, N, P, CS4, w_bytes=w_bytes)
            us = tot / (FREQ * 1e6) * 1e3
            best = us if best is None or us < best else best
        P += 2
    return best


def cs4_nonattn_parts(B):
    """Per-layer CS-4 non-attention split into (onwafer_us, interwafer_us):
      - onwafer  = GEMVs (QKV/O/router/MoE) + aux + on-wafer movement (MoE all-to-all + input replication)
      - interwafer = the wafer<->wafer EP crossing (~half the routed token-copies cross the 2-wafer link)."""
    qkv = gemv_us(B, D, QKV_OUT, 2.0)            # BF16 dense
    o   = gemv_us(B, Q_DIM, D, 2.0)              # BF16 O-proj
    rt  = gemv_us(B, D, EXPERTS, 2.0)            # BF16 router gate
    ea  = experts_active(B); b_e = max(1, round(B * TOPK / ea))
    moe = gemv_us(b_e, D, FC1, 0.5) + gemv_us(b_e, EXPERT_INTER, D, 0.5)   # FP4 weights, FP16 compute
    aux_bytes = (B * D * 2 + B * D * 0.5) + (B * TOPK * D * 2 + B * D * 2)  # quant + finalize, elementwise
    aux = aux_bytes / CS4_SRAM_BW * 1e6
    onwafer = qkv + o + rt + moe + aux
    interwafer = 0.0
    if MODEL_ONWAFER_MOVE:
        a2a_cy   = moe_a2a_cycles(B, D, TOPK, CS4, repl_frac=EXPERT_REPL_FRAC)   # scatter+gather, full wafer
        bcast_cy = bcast_cycles(B, D, CS4)                                       # one input replicate / layer
        onwafer += (a2a_cy + bcast_cy) / (FREQ * 1e6) * 1e3                      # on-wafer movement
        copies_cross = B * TOPK * (1.0 - EXPERT_REPL_FRAC) * 0.5                  # ~half cross the wafer link
        iw_bw  = (copies_cross * D * 2) / CS4_IO_BW * 1e6
        iw_lat = 2 * IW_LINK_LAT_US if copies_cross > 1e-9 else 0.0              # fixed inter-wafer round trip
        interwafer = iw_bw + iw_lat
    return onwafer, interwafer


def cs4_nonattn_us(B):
    """CS-4 time (us) for ALL non-attention work of ONE layer (on-wafer compute+movement + inter-wafer EP)."""
    ow, iw = cs4_nonattn_parts(B)
    return ow + iw


def comm_us(B):
    """GPU<->CS-4 transfer time (us) per layer over the 3-level CLOS: (Q,K,V CS-4->GPU) + (attn-out GPU->CS-4)
    bandwidth on the 300 GB/s link + the full round-trip CLOS latency (HOPS_ROUND_TRIP * T_HOP_US)."""
    bytes_q_k_v   = B * QKV_OUT * 2          # CS-4 -> GPU  (Q,K,V), bf16
    bytes_attn_out = B * Q_DIM * 2           # GPU -> CS-4  (O),     bf16
    bw_us = (bytes_q_k_v + bytes_attn_out) / CS4_IO_BW * 1e6
    return bw_us + CLOS_LAT_US
