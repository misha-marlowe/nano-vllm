#!/usr/bin/env python3
"""
Simple MeshGEMV performance model for Cerebras CS-2 / CS-3, built from the
published arch params plus the WaferLLM K-tree all-reduce cost formula.

MeshGEMV:  y[1,N] = x[1,M] @ W[M,N]  on a P x P PE grid (square M = N here).
  per-PE weight tile = (N/P) x (N/P) fp16 ;  reduction is 1-D over P cores.

------------------------------------------------------------------------------
MODEL = LOCAL GEMV (compute/memory roofline)  +  K-tree ALL-REDUCE (comm)
------------------------------------------------------------------------------

LOCAL GEMV (per PE) -- roofline of the three local rates:
    compute = 2*(N/P)^2 FLOPs          / Peak_FLOP_per_cy
    read    = 2*(N/P)^2 bytes (W tile) / SRAM_read_B_per_cy
    write   = 2*(N/P)   bytes (y tile) / SRAM_write_B_per_cy
    t_local = max(compute, read, write)

  NOTE on Peak_FLOP:  the table value is the *hardware ceiling*
  (CS-2 = 8 FLOP/cy = 4 fp16 MAC/cy ; CS-3 = 16 FLOP/cy, confirmed by
  SemiAnalysis: 15.625 PFLOP/s / 1.1 GHz / ~900k cores ~= 16 FLOP/cy/PE).
  GEMV does NOT reach this ceiling -- it is memory/latency-bound. The paper
  (Sec 7.5) reports 606x vs A100 against a 7,000x ideal (the 22 PB/s vs GPU
  bandwidth ratio), i.e. it realizes only ~1/12 of peak, due to (i) WSE-2
  cores not fully overlapping mem/compute, (ii) edge-core underutilization,
  (iii) residual NoC overhead. So expect the compute-bound corner (large N,
  small P) to read ~2x low at the literal peak; the *achieved* GEMV rate is
  ~half (set COMPUTE_EFFICIENCY < 1 to model that).

K-tree ALL-REDUCE of the length-(N/P) partial vector over the P cores
(WaferLLM paper Fig.8 formula):
    T_comm = alpha*N + (K * N^(1/K) / 2) * beta ,   with N = P, K = 2
    term 1 (alpha*P)               = critical-path LATENCY (data must cross P hops)
    term 2 ((K*P^(1/K)/2)*beta)    = the reduce-ADD/bandwidth work (K-tree shrinks
                                     it from O(P) stages down to O(sqrt(P)) for K=2)
    beta = 2*(N/P) bytes / PE2PE_bw  (1-hop transfer time of the partial vector)

  *** CUT-THROUGH ROUTING -- why alpha is ~1, not the 5-cy spec value ***
  The alpha*P term is the LATENCY for the reduce/broadcast to traverse P hops.
  Pipelining does NOT remove this (you can't pipeline away latency -- it only
  decouples *throughput* from distance, which is already in the beta term). BUT
  the per-hop coefficient alpha must be the CUT-THROUGH (wormhole) relay cost:
  the WSE NoC forwards one 32-bit wavelet/cycle and relays it through routers
  without buffering whole messages, so an intermediate hop is ~1 cycle.
  The spec-sheet "PE-to-PE latency" (CS-2 = 5 cy, CS-3 = 10 cy) is the
  END-TO-END single-message latency CE->router->link->router->CE -- it includes
  the inject/eject at the two compute engines, which in a multi-hop relay is
  paid ONCE at the endpoints, NOT at every hop. Multiplying 5*P charges that
  endpoint overhead at all P hops (store-and-forward) and over-predicts ~5x.
  Empirically (Fig.10) comm stays ~flat across P -> effective alpha ~= 1 cy,
  consistent with cut-through; 5 cy/hop would predict 5*P (up to 3000 cy at
  P=600) vs the ~900-1450 cy measured.
  (Only the K-tree *reduction nodes* are store-and-forward + add -- that is the
   beta term; the forwarding/broadcast hops are cut-through -- the alpha term.)

  p2p_lat below = the EFFECTIVE per-hop (cut-through) latency used for alpha.
  p2p_lat_spec  = the published end-to-end single-hop latency (documentation only).
------------------------------------------------------------------------------
"""
import math

KTREE_K = 2            # number of K-tree phases (paper's choice, K=2)
KTREE_BEST_K = False   # if True, pick K that minimizes the reduce-work term K·P^(1/K) (subject to KTREE_KMAX)
KTREE_KMAX = 16        # routing-resource cap on K (larger K needs O(K) routing paths per core)
def best_ktree_k(P):   # argmin_K K·P^(1/K) over 1..KTREE_KMAX  (≈ ln P, e.g. ~7 for P≈949)
    P = max(P, 2.0)
    return min(range(1, KTREE_KMAX + 1), key=lambda K: K * P ** (1.0 / K))

# Local-GEMV compute term: instead of a peak-FLOP roofline, use the MEASURED
# per-PE compute from the compute-only micro-benchmark (flops_interp.py):
#   compute_cycles(t) = fit_a*t^2 + fit_b*t + fit_c   (square t x t tile, t = N/P)
# These are simulator-measured (square-tile sweep t=4..128, P=4); see
# flops_interp_results.md.  CAVEAT: the quadratic fits LARGE t well (R^2~.99)
# but OVER-predicts at very small t (decode-size tiles, t<~17) -- it was fit on
# the whole range and its intercept (fit_c) is too high for tiny tiles.
USE_MEASURED_COMPUTE = True   # False -> fall back to the 2*t^2/peak roofline
COMPUTE_EFFICIENCY  = 1.0     # only used when USE_MEASURED_COMPUTE is False

#  fit_a/b/c   = measured square-tile compute fit (CS-2: ~4.13 FLOP/cy, CS-3: ~7.93);
#  p2p_lat     = per-hop (cut-through) latency for alpha (MEASURED = 1);
#  startup     = comm startup intercept c0 (MEASURED: CS-2=5, CS-3=7);
#  p2p_lat_spec= published end-to-end figure (NOT used in math);
#  io_bw_tbps  = host<->wafer system I/O bandwidth (Tb/s) -- stored for reference, NOT
#                used in the on-wafer model (kept here for completeness/future use).
#  PE_SIDE      = mesh side = sqrt(#cores) (WSE-2 ~850k -> 922; WSE-3 ~900k -> 949). Used only by the
#                 on-wafer movement model (moe_a2a_cycles / bcast_cycles), NOT by model_batched.
CS2 = dict(name="CS-2", freq=1.1, peak_flop=8,  sram_r=16, sram_w=8,
           p2p_bw=4, p2p_lat=1, startup=5.0, p2p_lat_spec=5, io_bw_tbps=1.2, PE_SIDE=922,
           fit_a=0.484279, fit_b=-3.237346, fit_c=164.0736)
CS3 = dict(name="CS-3", freq=1.1, peak_flop=16, sram_r=16, sram_w=8,
           p2p_bw=4, p2p_lat=1, startup=7.0, p2p_lat_spec=10, io_bw_tbps=1.2, PE_SIDE=949,
           fit_a=0.252110, fit_b=2.589384,  fit_c=61.5972)
# CS-4: PROJECTION (no sim). vs CS-3: peak FLOP doubled (16->32), memory BW (SRAM r/w)
# doubled, PE-to-PE BW doubled (4->8), comm startup doubled (7->14), system I/O doubled.
# Compute fit: the t^2 (MAC-work) coeff is HALVED (FLOPs doubled -> compute-work term
# halves); the linear/const terms (per-fmach issue + setup overhead) are FP-rate-
# independent so inherited from CS-3. Clock & per-hop latency unchanged.
CS4 = dict(name="CS-4", freq=1.1, peak_flop=32, sram_r=32, sram_w=16,
           p2p_bw=8, p2p_lat=1, startup=14.0, p2p_lat_spec=20, io_bw_tbps=2.4, PE_SIDE=949,
           fit_a=0.126055, fit_b=2.589384,  fit_c=61.5972)   # fit_a = CS-3/2 (FLOPs doubled)
# CS-3 effective alpha is ASSUMED ~1 (same cut-through router-forward mechanism);
# UNVALIDATED -- there is no CS-3/CS-4 figure to calibrate against.

SIZES = {"4K": 4096, "8K": 8192, "16K": 16384}
PS    = [120, 240, 360, 480, 600]

# MeshGEMV total cycles read from WaferLLM Fig.10 (CS-2 / WSE-2) -- the comparison target.
FIGURE = {
    "4K":  {120: 1050, 240: 720,  360: 650,  480: 770,  600: 920},
    "8K":  {120: 2780, 240: 1200, 360: 970,  480: 1020, 600: 1120},
    "16K": {120: 10000,240: 3100, 360: 1800, 480: 1500, 600: 1450},
}


def model_batched(B, M, N, P, hw, w_bytes=2.0):
    """Batched MeshGEMV cost (cycles) for  Y[B,N] = X[B,M] @ W[M,N]  on a P x P PE grid.

    Returns (t_local, t_comm, total) in cycles.  B = small batch (B << N).

    w_bytes = WEIGHT storage/read bytes/elem (default 2.0 = BF16; 0.5 = MXFP4).  FP4 weights only shrink
    the per-PE WEIGHT-READ term; COMPUTE stays FP16 (dequant->fp16 MAC) and the K-tree all-reduce stays
    FP16 activations -- both unaffected by weight precision.  At w_bytes=2.0 this is byte-identical to the
    fp16 model.  (Read-portion-of-the-fit decomposition is approximate -- see below.)

    MAPPING (unchanged from the 1-row case):
      * W[M,N] is tiled M/P x N/P, one tile resident per PE (read once, reused for all B rows).
      * X[B,M] is REPLICATED across the N-axis (same placement as the single x[1,M] vector).
      * Each PE computes a partial Y[B, N/P] = X_local[B, M/P] @ W_tile[M/P, N/P].
      * A K-tree ALL-REDUCE along the M-axis sums the partials (reduce) and broadcasts the
        result (the reduce+broadcast pipeline).  The reduced/broadcast message is now B*(N/P)
        elements instead of (N/P) -- so ONLY the comm bandwidth term scales with B; the per-hop
        LATENCY and startup are unchanged (same #hops, same endpoints).

    WHAT SCALES WITH B:
      * local compute: B MAC passes over the *same* resident weight tile (weight read amortized);
                       so only the MAC work (+ tiny per-row activation I/O) grows with B.
      * comm bandwidth: the partial/broadcast vector is B*Nt, so beta scales with B.

    B=1, M=N reduces EXACTLY to the original single-vector model() (see below).
    """
    mt = M / P                                         # per-PE rows of W (contraction slice)
    nt = N / P                                         # per-PE cols of W (= output tile len, "Nt")

    # --- LOCAL GEMV compute ---
    if USE_MEASURED_COMPUTE:
        # B=1 realized cost: measured square-tile fit, generalized to a rectangular mt x nt tile
        # (a*tile^2 -> a*mt*nt ; b*tile -> b*mt ; same as offload_model.gemv_cs3_ms).
        fit1 = hw["fit_a"]*mt*nt + hw["fit_b"]*mt + hw["fit_c"]
        # FP4 weights: shrink ONLY the weight-read portion baked into the fit (approx: the fit's read
        # term ~= 2*mt*nt/sram_r at fp16; replace 2 -> w_bytes). w_bytes=2 -> unchanged.
        fit1 -= (2.0 - w_bytes) * mt * nt / hw["sram_r"]
        # each EXTRA batch row reuses the resident weight tile -> just one more MAC pass (compute-
        # bound), or its input-read+output-write if those dominate.  (B-1) -> 0 at B=1.
        incr = max(2*mt*nt / hw["peak_flop"],          # MAC work for one extra row (FP16 compute)
                   2*mt / hw["sram_r"] + 2*nt / hw["sram_w"])   # X-row read + Y-row write (FP16 act)
        t_local = fit1 + (B - 1) * incr
    else:
        # peak-FLOP roofline: compute (x B) vs weight read (shared) vs output write (x B)
        compute = B * 2*mt*nt / (hw["peak_flop"] * COMPUTE_EFFICIENCY)   # FP16 MAC
        read    =     w_bytes*mt*nt / hw["sram_r"]      # weight tile (w_bytes/elem) read ONCE, reused
        write   = B * 2*nt    / hw["sram_w"]            # FP16 activations
        t_local = max(compute, read, write)

    # --- K-tree ALL-REDUCE (every coefficient MEASURED by comm_bench) ---
    #   T_comm = startup + per_hop*P + (K * P^(1/K) / 2) * (vector_bytes / link_bw)
    #     startup (c0)        : 5 cy (CS-2) / 7 cy (CS-3)   -- one-time inject/eject endpoint cost
    #     per_hop  (p2p_lat)  : 1 cy/hop                    -- cut-through router forward
    #     link_bw  (p2p_bw)   : 4 B/cy                      -- one 32-bit wavelet/cycle
    #   per_hop*P  = data traversal of the reduce+broadcast critical path (~P hops) -- B-INDEPENDENT.
    #   (K*P^(1/K)/2)*beta = the K-tree reduce-add stages, each moving the B*Nt fp16 vector.
    per_hop = hw["p2p_lat"]                            # = 1 cy  (MEASURED)
    beta    = (2 * B * nt) / hw["p2p_bw"]              # vector_bytes(2*B*Nt) / link_bw(4)  per stage
    K = best_ktree_k(P) if KTREE_BEST_K else KTREE_K
    t_comm = hw["startup"] + per_hop * P + (K * P ** (1.0 / K) / 2.0) * beta   # term1 = latency, term2 = reduce work

    return t_local, t_comm, t_local + t_comm


# ============ on-wafer DATA MOVEMENT (WaferLLM, OSDI'25 / arXiv 2502.04563) ============
# WaferLLM's DECODE path pre-optimizes the weight layout (BEyLx: embedding along Y, sequence
# replicated along X) so consecutive dense GEMVs (QKV->O-proj->router->fc1->fc2) chain with NO
# inter-op matrix transpose ("Pre-optimize the model weight layout for decode...to eliminate matrix
# transpose").  So the ONLY on-wafer movements we charge for decode are:
#   (a) the per-layer INPUT replication along X (the BEyLx replicate), and
#   (b) the MoE attention<->expert ALL-TO-ALL ("the main difference is the all-to-all communication
#       between attention and expert layers, which we implement using WSE NoC multi-cast operations").
# Both use the paper's PLMR alpha-beta cost  alpha*(Nw+Nh) + beta*r  + bandwidth, with the SAME fabric
# constants as the K-tree all-reduce: alpha = p2p_lat (cut-through, 1 cy/hop), startup = c0, link
# bandwidth = p2p_bw (B/cy).  Returns CYCLES (convert via /(freq*1e6)*1e3 for us).

def bcast_cycles(B, D, hw, act_bytes=2.0):
    """Per-layer BEyLx input replication: broadcast incoming hidden [B,D] along the X-axis (PE_SIDE
    cores).  D is sharded along Y, so each row broadcasts only its B*(D/PE_SIDE) slice along its row
    (cut-through, pipelined): startup + PE_SIDE hops + (B*(D/PE_SIDE)*act_bytes)/link_bw."""
    P = hw["PE_SIDE"]
    return hw["startup"] + P * hw["p2p_lat"] + (B * (D / P) * act_bytes) / hw["p2p_bw"]


def moe_a2a_cycles(B, D, topk, hw, repl_frac=0.0, act_bytes=2.0):
    """MoE attention<->expert ALL-TO-ALL (NoC multicast): scatter B tokens' hidden [.,D] to their
    top-k experts + gather/combine outputs back, across the full wafer mesh.  PLMR alpha-beta:
        one_way = startup + (Nw+Nh)*p2p_lat + (routed_copies*D*act_bytes)/bisection_bw
        total   = 2 * one_way                        (scatter + gather)
    with Nw+Nh = 2*PE_SIDE (max Manhattan hops -> the 'non-uniform latency' L of PLMR) and bisection
    bandwidth = PE_SIDE links * p2p_bw.  At decode the data is tiny, so this is LATENCY-dominated.
    repl_frac in [0,1]: hot-expert REPLICATION shortens hops and reduces the routed fraction (a copy
    is found nearer), but cannot reach 0 on a capacity-bound wafer (the cold-expert tail + the permute
    remain).  repl_frac=0 -> fully sharded (full a2a)."""
    P = hw["PE_SIDE"]
    hops      = 2 * P * (1.0 - repl_frac)            # Nw+Nh, reduced by replication locality
    bisect_bw = P * hw["p2p_bw"]                     # PE_SIDE links cross the mesh bisection
    copies    = B * topk * (1.0 - repl_frac)         # token-copies still needing a cross-fabric move
    one_way = hw["startup"] + hops * hw["p2p_lat"] + (copies * D * act_bytes) / bisect_bw
    return 2.0 * one_way                             # scatter + gather


def model(N, P, hw):
    """Single-vector MeshGEMV  y[1,N] = x[1,M] @ W[M,N]  (square M=N); cycles (t_local,t_comm,total).
    Thin wrapper over model_batched(B=1, M=N, N, P, hw) so the two paths can never drift."""
    return model_batched(1, N, N, P, hw)


def compare_configs(configs):
    """Print a side-by-side comparison of the arch configs."""
    print("=" * 92)
    print("  CEREBRAS CONFIG COMPARISON   (CS-2 & CS-3 measured; CS-4 projected: FLOP/memBW/P2P-BW/startup x2 vs CS-3)")
    print("=" * 92)
    hdr = (f"{'arch':>5} {'GHz':>4} {'peakFLOP/cy':>12} {'SRAM r/w B/cy':>14} {'P2P B/cy':>9} "
           f"{'hop_lat':>8} {'startup':>8} {'sysIO Tb/s':>11} {'GEMV FLOP/cy(2/a)':>18}")
    print(hdr); print("-" * len(hdr))
    for hw in configs:
        achieved = 2.0 / hw["fit_a"]                       # asymptotic GEMV rate = 2/a
        srw = f"{hw['sram_r']}/{hw['sram_w']}"
        print(f"{hw['name']:>5} {hw['freq']:>4} {hw['peak_flop']:>12} {srw:>14} {hw['p2p_bw']:>9} "
              f"{hw['p2p_lat']:>8} {hw['startup']:>8.0f} {hw['io_bw_tbps']:>11} {achieved:>18.2f}")
    print()


def report(hw, compare=True):
    f = hw["freq"]
    eff = "" if COMPUTE_EFFICIENCY == 1.0 else f", compute_eff {COMPUTE_EFFICIENCY}"
    print("=" * 88)
    print(f"  MeshGEMV model -- {hw['name']}  (compute: measured square-tile fit; "
          f"comm: startup {hw['startup']:.0f}cy + {hw['p2p_lat']:.0f}cy/hop*P + 2Nt/{hw['p2p_bw']:.0f}B/cy; "
          f"sysIO {hw['io_bw_tbps']} Tb/s [unused]; {f} GHz)")
    print("=" * 88)
    hdr = (f"{'size':>4} {'P':>4} {'tile':>6} | {'total_cy':>9} {'total_ns':>9} "
           f"{'comm_cy':>8} {'comm_ns':>8}")
    if compare:
        hdr += f" | {'fig_cy':>7} {'delta_cy':>9} {'delta_%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, N in SIZES.items():
        for P in PS:
            tl, tc, tot = model(N, P, hw)
            row = (f"{name:>4} {P:>4} {N/P:>6.1f} | {tot:>9.1f} {tot/f:>9.1f} "
                   f"{tc:>8.1f} {tc/f:>8.1f}")
            if compare:
                fig = FIGURE[name][P]
                d = tot - fig
                row += f" | {fig:>7} {d:>+9.1f} {100*d/fig:>+7.1f}%"
            print(row)
        print("-" * len(hdr))
    print()


def report_batched(hw, M, N, P, Bs=(1, 2, 4, 8, 16, 32)):
    """Batched MeshGEMV  Y[B,N]=X[B,M]@W[M,N]  vs batch B, on a fixed P x P grid."""
    f = hw["freq"]
    mt, nt = M / P, N / P
    print("=" * 78)
    print(f"  BATCHED MeshGEMV -- {hw['name']}  Y[B,N] = X[B,M] @ W[M,N]   "
          f"(M={M}, N={N}, P={P}, tile {mt:.0f}x{nt:.0f})")
    print("=" * 78)
    hdr = (f"{'B':>4} | {'local_cy':>9} {'comm_cy':>9} {'total_cy':>9} {'total_ns':>9} "
           f"| {'vs B=1':>7} {'cy/row':>8}")
    print(hdr); print("-" * len(hdr))
    base = None
    for B in Bs:
        tl, tc, tot = model_batched(B, M, N, P, hw)
        if base is None:
            base = tot
        print(f"{B:>4} | {tl:>9.1f} {tc:>9.1f} {tot:>9.1f} {tot/f:>9.1f} "
              f"| {tot/base:>6.2f}x {tot/B:>8.1f}")
    print()


if __name__ == "__main__":
    compare_configs([CS2, CS3, CS4])   # config comparison table up front
    report(CS2, compare=True)    # figure is CS-2 -> compare + delta
    report(CS3, compare=False)   # CS-3 projection (no figure to compare against)
    report(CS4, compare=False)   # CS-4 projection (memory BW, P2P BW, startup doubled vs CS-3)

    # --- B=1 equivalence: model_batched(1,N,N,P) must equal the original single-vector model ---
    print("=" * 78)
    print("  B=1 reduction check: model_batched(1,N,N,P,hw) == model(N,P,hw) ?")
    print("=" * 78)
    ok = True
    for hw in (CS2, CS3, CS4):
        for name, N in SIZES.items():
            for P in PS:
                a = model_batched(1, N, N, P, hw)
                b = model(N, P, hw)
                if max(abs(x - y) for x, y in zip(a, b)) > 1e-9:
                    ok = False
                    print(f"  MISMATCH {hw['name']} {name} P={P}: {a} vs {b}")
    print(f"  B=1 reduction: {'PASS' if ok else 'FAIL'}  "
          f"(all {len(SIZES)*len(PS)} sizes x 3 archs identical)\n")

    # --- batched scaling demo: CS-3 MoE fc1 GEMV [2880] -> [5760] ---
    report_batched(CS3, M=2880, N=5760, P=240)
