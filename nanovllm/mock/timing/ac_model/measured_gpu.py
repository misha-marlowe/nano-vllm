#!/usr/bin/env python3
"""
Measurement-grounded GPU performance model for GPT-OSS-120B decode.

Takes the MEASURED B200 per-kernel times in PROJECTIONS_AND_COMPARE.md ("Per-kernel cross-config comparison"
regimes), buckets them into 4 bottleneck classes, fits physically-structured surrogates (continuous in
TP/batch/ISL), and PROJECTS to any arch (e.g. MI455X) by bottleneck class. An alternative to the analytical
roofline in perf_model.py.

Bottleneck classes (per layer, µs):
  a_hbm  = "Attention" (FMHA) ............... HBM-bound, ∝ B·ISL / min(TP, n_kv=8)   -> projects by HBM ratio
  a_floor= "RoPE+KVUpdate" .................. launch-floor (kept on GPU with attention)
  r_hbm  = "GEMM" + "MoE-bmm" ............... HBM-bound, batch-scaling, sharded /TP^η -> projects by HBM ratio
  r_floor= "MoE-quant"+"MoE-router"+"GEMM-reduce" .. launch-floor (fixed)
  comm   = "Comm" (all-reduce) .............. NVLink; TP>8 -> analytical MultiShot

GPU attention(per layer) = a_hbm·hbm_ratio + a_floor ;  GPU rest = r_hbm·hbm_ratio + r_floor + comm.
With arch=B200 (ratio 1, NVLink comm) this reproduces the raw measured B200.
Caveat: measured attention is ALL-CAUSAL (as benchmarked); the real arch's sliding-window every-other-layer is
not in this data (the roofline backend models it).
"""
import os, re, math

# ---- model dims / refs ----
D, EXPERTS, TOPK, N_KV, L = 2880, 128, 4, 8, 36
SLIDING_EVERY, WINDOW = 2, 128                       # GPT-OSS: every other layer is sliding-window(128 tokens)
N_SLD = L // SLIDING_EVERY; N_FULL = L - N_SLD; FULL_FRAC = N_FULL / L   # 18 full-causal + 18 sliding -> 0.5 each
BLEND_SLIDING = True                                 # attention = 18 full-causal + 18 sliding (False = all-causal)
SPLIT_RHBM = True                                    # r_hbm: overhead fixed + bandwidth ×HBM-ratio (False = whole ×ratio, pre-fix)
OVERLAP = "allreduce"                                # {"none","allreduce","allreduce_bmm"}: hide all-reduce (+MoE-BMM) behind compute
B200_HBM_GBPS = 8000.0
B200_LINK_GBPS = 1800.0          # NVLink-5 BIDIRECTIONAL (= 900 GB/s/GPU unidirectional). link_gbps dicts are bidir.
DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROJECTIONS_AND_COMPARE_alllayers.gptoss120b.md")

def experts_active(B, E=EXPERTS, k=TOPK): return E * (1.0 - (1.0 - k / E) ** B)

# ----------------------------------------------------------------------------- parse measured data
def parse_measured(path=DOC, drop_conc=(256,)):
    """Per (tp,conc,isl) from the ALL-36-LAYER profile (18 causal + 18 sliding), in µs/LAYER:
      a_hbm     = causal attention/layer  (derived: causal-layers-total/18 − non-attention/layer)
      a_hbm_sld = sliding attention/layer (derived: sliding-layers-total/18 − non-attention/layer)
      a_floor   = RoPE+KV/layer ;  r_hbm = (GEMM+MoE+other)/layer ;  comm = all-reduce/layer (raw, pre-overlap)
      total     = all-layer kernel sum / 36 (avg blended layer)
      f_ar      = fraction of comm hidden behind compute (all-reduce overlap / comm)
      bmm_pl    = incremental MoE-BMM overlap/layer (ms-union, after all-reduce overlap)
    Non-attention is identical for causal & sliding layers, so it cancels out of the attention derivation."""
    out, hdr = [], None
    for ln in open(path).read().splitlines():
        s = ln.strip()
        if not s.startswith("|"): continue
        c = [x.strip() for x in s.strip("|").split("|")]
        if c and c[0] == "regime": hdr = c; continue                 # header row -> column names
        if hdr is None or not c[0].startswith(("12 kernels", "13 kernels")): continue
        d = dict(zip(hdr, c)); tp, conc, isl = int(d["tp"]), int(d["conc"]), int(d["isl"])
        if conc in drop_conc: continue
        g = lambda k: float(d[k]); U = 1e3                           # ms -> µs
        att, gemm, rope = g("attention total"), g("gemm total"), g("rope total")
        comm, moe, other = g("comm total"), g("moe total"), g("other total")
        allsum, ar, arbmm = g("all-layer kernel sum"), g("all-reduce overlap"), g("all-reduce + MoE BMM overlap")
        ctot, stot = g("causal layers total"), g("sliding-window layers total")
        rest_pl = (allsum - att) / L * U                             # non-attention per layer (same both variants)
        out.append(dict(tp=tp, conc=conc, isl=isl,                   # isl already in tokens
            a_hbm=ctot / N_FULL * U - rest_pl, a_hbm_sld=stot / N_SLD * U - rest_pl,
            a_floor=rope / L * U, r_hbm=(gemm + moe + other) / L * U, comm=comm / L * U,
            total=allsum / L * U, f_ar=(ar / comm if comm > 1e-9 else 0.0), bmm_pl=max(0.0, arbmm - ar) / L * U))
    return out

# ----------------------------------------------------------------------------- pure-python (weighted) least squares
def _linreg(xs, ys, w=None):                               # y = m*x + b ; w=None -> ordinary; else weighted (WLS)
    if w is None: w = [1.0] * len(xs)
    Sw = sum(w); Swx = sum(wi * x for wi, x in zip(w, xs)); Swy = sum(wi * y for wi, y in zip(w, ys))
    Swxx = sum(wi * x * x for wi, x in zip(w, xs)); Swxy = sum(wi * x * y for wi, x, y in zip(w, xs, ys))
    den = Sw * Swxx - Swx * Swx
    if abs(den) < 1e-30: return 0.0, Swy / Sw
    m = (Sw * Swxy - Swx * Swy) / den
    return m, (Swy - m * Swx) / Sw

def _relw(ys):                                             # relative-error weights ~ 1/y^2 (small configs count equally)
    return [1.0 / max(v, 1e-6) ** 2 for v in ys]

def _wls2(f1, f2, y, w):                                   # min Σ w (c1·f1 + c2·f2 − y)² (through origin) -> c1,c2
    S11 = sum(wi * a * a for wi, a in zip(w, f1)); S22 = sum(wi * b * b for wi, b in zip(w, f2))
    S12 = sum(wi * a * b for wi, a, b in zip(w, f1, f2))
    S1y = sum(wi * a * yy for wi, a, yy in zip(w, f1, y)); S2y = sum(wi * b * yy for wi, b, yy in zip(w, f2, y))
    det = S11 * S22 - S12 * S12
    if abs(det) < 1e-30: return 0.0, 0.0
    return (S22 * S1y - S12 * S2y) / det, (S11 * S2y - S12 * S1y) / det

def _wls3(F, y, w):                                        # min Σ w (Σ_k c_k F[k] − y)² -> [c1,c2,c3] (3 features)
    A = [[sum(w[i] * F[a][i] * F[b][i] for i in range(len(y))) for b in range(3)] for a in range(3)]
    z = [sum(w[i] * F[a][i] * y[i] for i in range(len(y))) for a in range(3)]
    # solve 3x3 A c = z (Gaussian elimination with partial pivoting)
    M = [A[r][:] + [z[r]] for r in range(3)]
    for col in range(3):
        p = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        if abs(M[col][col]) < 1e-30: continue
        for r in range(3):
            if r != col:
                f = M[r][col] / M[col][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(4)]
    return [M[r][3] / M[r][r] if abs(M[r][r]) > 1e-30 else 0.0 for r in range(3)]

# ----------------------------------------------------------------------------- fit the surrogates
class MeasuredGPU:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else parse_measured()
        R = self.rows
        # all fits minimize RELATIVE error (weight ~1/y^2) so small-batch configs aren't drowned by large ones
        # a_hbm = a0 + a1·(B·ISL)/min(TP,8) + c2·(ISL)/min(TP,8).  a0 = measured base at smallest B·ISL (FMHA
        # launch); a1 = per-token KV-read (large-batch, BW-saturated); c2 = batch-INDEPENDENT KV-scan that
        # gives small-batch its steeper ISL-slope (small-batch FMHA under-utilizes HBM). Relative-weighted.
        kv = [min(r["tp"], N_KV) for r in R]
        f1 = [r["conc"] * r["isl"] / k for r, k in zip(R, kv)]; f2 = [r["isl"] / k for r, k in zip(R, kv)]
        y = [r["a_hbm"] for r in R]; wy = _relw(y)
        small = sorted(range(len(R)), key=lambda i: f1[i])[:4]     # 4 smallest-x configs (b1, short ISL)
        self.a0 = max(0.0, sum(y[i] for i in small) / len(small))
        self.a1, self.c2 = _wls2(f1, f2, [yi - self.a0 for yi in y], wy)
        # a_floor (RoPE+KV launch): linear in B (mild). (No separate r_floor — the all-layer profile folds the small
        # launch kernels into gemm/moe, so the r_hbm overhead term OH absorbs them.)
        yaf = [r["a_floor"] for r in R]; self.af_m, self.af_b = _linreg([r["conc"] for r in R], yaf, _relw(yaf))
        # r_hbm = r_oh + (Gd + Gm*ea(B))/P^β  — OVERHEAD floor (launch-bound, TP-independent, fixed across arch) +
        # BANDWIDTH weight-read (×HBM-ratio, shards /P^β). Physical split: at small batch the rest is launch-bound
        # (barely shards, <50% HBM peak), at large batch the MoE-bmm is weight-bandwidth-bound (~99% HBM peak).
        # The projection scales ONLY the bandwidth part by the HBM ratio (overhead stays fixed, like the floors).
        # β = bandwidth sharding exponent: decode GEMMs shard SUB-linearly (~/P^0.6, tile/wave quantization on
        # small matrices), not perfectly /P. β is FITTED by a grid that minimizes the relative-weighted r_hbm SSE;
        # OH/Gd/Gm are the linear LSQ for that β. (r_hbm is ISL-independent — pure weight read.)
        ea_ = [experts_active(r["conc"]) for r in R]; yrh = [r["r_hbm"] for r in R]; wrh = _relw(yrh)
        best = None
        bgrid = [0.25 + 0.025 * i for i in range(31)]               # β ∈ [0.25, 1.00]
        for b in bgrid:
            invPb = [1.0 / r["tp"] ** b for r in R]
            F = [[1.0] * len(R), invPb, [e * ip for e, ip in zip(ea_, invPb)]]   # OH, Gd/P^β, Gm*ea/P^β
            oh, Gd, Gm = _wls3(F, yrh, wrh); oh = max(0.0, oh)
            sse = sum(wi * (oh + (Gd + Gm * e) / r["tp"] ** b - y) ** 2
                      for wi, e, r, y in zip(wrh, ea_, R, yrh))
            if best is None or sse < best[0]: best = (sse, b, oh, Gd, Gm)
        _, self.beta, self.r_oh, self.Gd, self.Gm = best
        # comm: measured average per TP (NVLink, single node); overlap (fraction of comm hidden, + MoE-BMM/layer) per TP
        self.comm_tp = {}; self.f_ar_tp = {}; self.bmm_tp = {}
        for tp in sorted({r["tp"] for r in R}):
            sub = [r for r in R if r["tp"] == tp]
            self.comm_tp[tp] = sum(r["comm"] for r in sub) / len(sub)
            self.f_ar_tp[tp] = sum(r["f_ar"] for r in sub) / len(sub)      # all-reduce hidden behind compute
            self.bmm_tp[tp]  = sum(r["bmm_pl"] for r in sub) / len(sub)    # incremental MoE-BMM overlap / layer
        # sliding-window attention: reads a FIXED 128-token window -> flat in ISL for all swept ISLs (>=8K).
        # Fit a_hbm_sld(B) = s0 + s1*B (relative-weighted), parallel to a_floor. (P-independent: §4c shows it
        # barely shards — it's launch/window-floor bound, not KV-bandwidth bound.)
        ysl = [r["a_hbm_sld"] for r in R]; self.s1, self.s0 = _linreg([r["conc"] for r in R], ysl, _relw(ysl))

    # ---- per-class predictions (µs/layer), measured-B200 frame ----
    def a_hbm(self, B, isl, P): return max(0.0, self.a0 + (self.a1 * B + self.c2) * isl / min(P, N_KV))
    def a_hbm_sld(self, B, P=None): return max(0.0, self.s0 + self.s1 * B)   # sliding: flat in ISL (window=128)
    def a_floor(self, B):        return max(0.0, self.af_b + self.af_m * B)
    def r_hbm_bw(self, B, P):    return max(0.0, (self.Gd + self.Gm * experts_active(B)) / P ** self.beta)  # bandwidth (shards /P^β, ×HBM-ratio)
    def r_hbm_oh(self):          return max(0.0, self.r_oh)                                     # overhead floor (fixed, no HBM speedup)
    def r_hbm(self, B, P):       return self.r_hbm_oh() + self.r_hbm_bw(B, P)                   # total (B200 frame)
    def _nearest_tp(self, d, P): return d.get(P) if P in d else d[min(d, key=lambda t: abs(t - P))]
    def f_ar(self, P):           return 0.0 if P <= 1 else self._nearest_tp(self.f_ar_tp, P)    # comm fraction hidden
    def bmm_pl(self, P):         return self._nearest_tp(self.bmm_tp, P)                         # MoE-BMM overlap µs/layer
    def comm(self, arch, B, P):
        if P <= 8:                                          # measured single-node NVLink (scale to arch link)
            c = self.comm_tp.get(P) or self.comm_tp[min(self.comm_tp, key=lambda t: abs(t - P))]
            if arch is None or arch.link_gbps <= 0: return c
            return c * (B200_LINK_GBPS / arch.link_gbps)    # scale by link ratio (both bidirectional -> uni /2 cancels)
        # TP>8: analytical MultiShot all-reduce, 2x/layer ([B,d] bf16), µs
        S = B * D * 2.0
        uni = arch.link_gbps / 2.0 * 1e9                 # all-reduce egress is one-directional -> unidirectional BW
        one = 2 * arch.link_lat_us + (2 * (P - 1) / P) * S / uni * 1e6
        return 2.0 * one

    # ---- projected per-layer GPU times (seconds) ----
    def layer_times(self, arch, B, isl, P, blend=None, overlap=None):
        """Per-layer (attn, rest) in seconds. blend -> 18 full-causal + 18 sliding attention mix (False = all-causal).
        overlap in {none, allreduce, allreduce_bmm}: hide all-reduce (and MoE-BMM) behind compute (default module OVERLAP).
        a_floor (RoPE+KV) is variant-independent; only the a_hbm term is blended. The 'rest' = r_hbm + comm."""
        if blend is None: blend = BLEND_SLIDING
        if overlap is None: overlap = OVERLAP
        ratio = B200_HBM_GBPS / arch.hbm_gbps               # HBM-bound work scales by HBM ratio
        a_full = self.a_hbm(B, isl, P)
        a_eff = (FULL_FRAC * a_full + (1 - FULL_FRAC) * self.a_hbm_sld(B, P)) if blend else a_full
        attn = (a_eff * ratio + self.a_floor(B)) * 1e-6
        # rest: bandwidth weight-read scales by HBM ratio (+ shards /P^β inside r_hbm_bw); overhead floor stays fixed.
        # SPLIT_RHBM=False (fix OFF) projects the WHOLE r_hbm by the HBM ratio — overhead included (pre-fix behaviour).
        r_hbm_proj = (self.r_hbm_bw(B, P) * ratio + self.r_hbm_oh()) if SPLIT_RHBM else (self.r_hbm(B, P) * ratio)
        comm = self.comm(arch, B, P)
        if overlap in ("allreduce", "allreduce_bmm"): comm *= (1.0 - self.f_ar(P))   # all-reduce hidden behind compute
        rest_us = r_hbm_proj + comm
        if overlap == "allreduce_bmm": rest_us = max(0.0, rest_us - self.bmm_pl(P))  # MoE-BMM also hidden
        rest = rest_us * 1e-6
        return attn, rest                                    # µs->s, per layer

# ----------------------------------------------------------------------------- validation
def validate():
    rows = parse_measured(); fit = MeasuredGPU(rows)
    # (1) leave-one-out total-time error (refit without each row, predict it; B200 frame)
    class _B200:  # duck arch
        hbm_gbps = B200_HBM_GBPS; link_gbps = B200_LINK_GBPS; link_lat_us = 0.5   # bidirectional (matches dicts)
    b200 = _B200()
    errs = []; byB = {}; byBa = {}                           # per-batch total + attention-bucket LOO error
    for i, r in enumerate(rows):
        f = MeasuredGPU(rows[:i] + rows[i + 1:])
        a, rest = f.layer_times(b200, r["conc"], r["isl"], r["tp"], blend=True, overlap="none")  # raw 36-layer kernel sum
        et = abs((a + rest) * 1e6 - r["total"]) / r["total"]
        errs.append(et); byB.setdefault(r["conc"], []).append(et)
        am = f.a_hbm(r["conc"], r["isl"], r["tp"]) + f.a_floor(r["conc"]); meas_a = r["a_hbm"] + r["a_floor"]
        byBa.setdefault(r["conc"], []).append(abs(am - meas_a) / meas_a)
    errs.sort(); n = len(errs)
    loo = dict(mean=sum(errs) / n, median=errs[n // 2], p90=errs[int(0.9 * n)], maxv=errs[-1], n=n)
    loo_by_batch = {B: dict(total=sum(byB[B]) / len(byB[B]), attn=sum(byBa[B]) / len(byBa[B]), n=len(byB[B]))
                    for B in sorted(byB)}
    # (2) class sums vs measured total (in-sample, raw 36-layer kernel sum)
    sum_err = []
    for r in rows:
        a, rest = fit.layer_times(b200, r["conc"], r["isl"], r["tp"], blend=True, overlap="none")
        sum_err.append(abs((a + rest) * 1e6 - r["total"]) / r["total"])
    # (3) sliding-window attention fit: predicted a_hbm_sld vs measured, + ISL-flatness (max/min across ISL per (tp,conc))
    sld_err = [abs(fit.a_hbm_sld(r["conc"], r["tp"]) - r["a_hbm_sld"]) / r["a_hbm_sld"] for r in rows]
    by_tc = {}
    for r in rows: by_tc.setdefault((r["tp"], r["conc"]), []).append(r["a_hbm_sld"])
    flat = [max(v) / min(v) for v in by_tc.values() if min(v) > 0]            # ~1.0 if truly flat in ISL
    sliding = dict(s0=fit.s0, s1=fit.s1, fit_mean=sum(sld_err) / len(sld_err),
                   fit_max=max(sld_err), isl_flat_max=max(flat))
    return dict(fit=fit, rows=rows, loo=loo, loo_by_batch=loo_by_batch, insample_mean=sum(sum_err) / len(sum_err),
                sliding=sliding,
                coeffs=dict(a0=fit.a0, a1=fit.a1, c2=fit.c2, r_oh=fit.r_oh, Gd=fit.Gd, Gm=fit.Gm, beta=fit.beta,
                            af_b=fit.af_b, af_m=fit.af_m, comm_tp=fit.comm_tp, f_ar_tp=fit.f_ar_tp,
                            bmm_tp=fit.bmm_tp, s0=fit.s0, s1=fit.s1))

if __name__ == "__main__":
    v = validate(); c = v["coeffs"]
    print(f"parsed {len(v['rows'])} measured configs")
    print("fitted coefficients:")
    print(f"  a_hbm = {c['a0']:.3f} + ({c['a1']:.3e}*B + {c['c2']:.3e})*ISL/min(TP,8)   us/layer")
    print(f"  r_hbm = {c['r_oh']:.2f} (overhead, fixed) + ({c['Gd']:.2f} + {c['Gm']:.3f}*ea(B))/TP^{c['beta']:.3f} (bandwidth, ×HBM-ratio)")
    print(f"  a_floor = {c['af_b']:.2f} + {c['af_m']:.4f}*B  (RoPE+KV; r_floor folded into r_hbm overhead)")
    print(f"  comm/TP (us, NVLink) = {dict((k, round(val,3)) for k,val in c['comm_tp'].items())}")
    print(f"  overlap: f_ar(comm hidden)/TP = {dict((k, round(val,2)) for k,val in c['f_ar_tp'].items())}  "
          f"bmm/layer/TP = {dict((k, round(val,3)) for k,val in c['bmm_tp'].items())} us  [mode default '{OVERLAP}']")
    lo = v["loo"]; print(f"leave-one-out total-time error: mean {lo['mean']*100:.1f}%  median {lo['median']*100:.1f}%  "
                          f"p90 {lo['p90']*100:.1f}%  max {lo['maxv']*100:.1f}%  (n={lo['n']})")
    print("per-batch LOO (total / attn-bucket):")
    for B, d in v["loo_by_batch"].items():
        print(f"  b{B:3d}: {d['total']*100:5.1f}% / {d['attn']*100:5.1f}%  (n={d['n']})")
    print(f"in-sample (4-class sum vs measured Sum) mean err: {v['insample_mean']*100:.1f}%")
