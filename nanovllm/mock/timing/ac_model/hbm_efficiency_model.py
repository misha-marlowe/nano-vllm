#!/usr/bin/env python3
"""Single-call GB200 FP16 GEMM HBM-efficiency model.

Use this module from a performance model instead of reading the benchmark or
analysis CSVs at runtime.

Modeled shape:

    A[N, N] @ X[N, B] -> Y[N, B]

The fitted efficiency is normalized to the measured B=1 logical HBM efficiency
from the GB200 benchmark run:

    B1_HBM_EFFICIENCY_PCT = 86.620417

The model returns expected smooth-trend HBM efficiency versus B. It intentionally
does not reproduce the slow cuBLAS skinny-GEMM outliers observed at:

    B = 2, 4, 20, 28, 36, 44, 52, 60

For those B values, the function returns the fitted trend value, not the slow
measured value.

Example:

    from hbm_efficiency_model import hbm_efficiency_pct, hbm_gbps

    eff = hbm_efficiency_pct(128)
    bw = hbm_gbps(128, peak_hbm_gbps=8000.0)
"""

from __future__ import annotations

import argparse
import bisect


B1_HBM_EFFICIENCY_PCT = 86.620417
DEFAULT_PEAK_HBM_GBPS = 8000.0
OUTLIER_B_VALUES = (2, 4, 20, 28, 36, 44, 52, 60)

# Sparse monotone isotonic fit points. The value is relative to B=1 efficiency.
# Linear interpolation is used between adjacent points.
_MODEL_POINTS = (
    (1.0, 1.00000000),
    (8.0, 0.93810998),
    (16.0, 0.93694273),
    (24.0, 0.92785492),
    (32.0, 0.92785492),
    (40.0, 0.87295934),
    (48.0, 0.87295934),
    (56.0, 0.87295934),
    (64.0, 0.87295934),
    (72.0, 0.87295934),
    (80.0, 0.87295934),
    (88.0, 0.87295934),
    (96.0, 0.87295934),
    (104.0, 0.86504940),
    (112.0, 0.86504940),
    (120.0, 0.86034762),
    (128.0, 0.86034762),
    (144.0, 0.75092214),
    (160.0, 0.74624560),
    (176.0, 0.69319936),
    (192.0, 0.67984292),
    (208.0, 0.64646941),
    (224.0, 0.63999389),
    (240.0, 0.63200437),
    (256.0, 0.63110171),
    (288.0, 0.53365425),
    (320.0, 0.48143880),
    (352.0, 0.46843495),
    (384.0, 0.46772386),
    (416.0, 0.38204651),
    (448.0, 0.38204651),
    (480.0, 0.37694541),
    (512.0, 0.37378088),
    (576.0, 0.30528760),
    (640.0, 0.30203459),
    (704.0, 0.25635544),
    (768.0, 0.25206845),
    (832.0, 0.21449175),
    (896.0, 0.21299781),
    (960.0, 0.21014789),
    (1024.0, 0.20937790),
)

_MODEL_B = tuple(point[0] for point in _MODEL_POINTS)
_MODEL_REL_EFF = tuple(point[1] for point in _MODEL_POINTS)


def relative_hbm_efficiency_vs_b1(b: float, *, clamp: bool = False) -> float:
    """Return fitted HBM efficiency normalized to B=1.

    Args:
        b: GEMM right-hand-side width B. Integer and fractional values are
            supported; fractional values are linearly interpolated.
        clamp: If False, B outside the fitted range [1, 1024] raises ValueError.
            If True, values outside the range use the nearest endpoint.
    """
    b = float(b)
    min_b = _MODEL_B[0]
    max_b = _MODEL_B[-1]

    if b < min_b:
        if not clamp:
            raise ValueError(f"B={b:g} is below fitted range [{min_b:g}, {max_b:g}]")
        return _MODEL_REL_EFF[0]
    if b > max_b:
        if not clamp:
            raise ValueError(f"B={b:g} is above fitted range [{min_b:g}, {max_b:g}]")
        return _MODEL_REL_EFF[-1]

    index = bisect.bisect_left(_MODEL_B, b)
    if index < len(_MODEL_B) and _MODEL_B[index] == b:
        return _MODEL_REL_EFF[index]

    left_b = _MODEL_B[index - 1]
    right_b = _MODEL_B[index]
    left_eff = _MODEL_REL_EFF[index - 1]
    right_eff = _MODEL_REL_EFF[index]
    alpha = (b - left_b) / (right_b - left_b)
    return left_eff + alpha * (right_eff - left_eff)


def hbm_efficiency_pct(b: float, *, clamp: bool = False) -> float:
    """Return fitted logical HBM efficiency percentage for GEMM width B."""
    return B1_HBM_EFFICIENCY_PCT * relative_hbm_efficiency_vs_b1(b, clamp=clamp)


def hbm_gbps(
    b: float,
    *,
    peak_hbm_gbps: float = DEFAULT_PEAK_HBM_GBPS,
    clamp: bool = False,
) -> float:
    """Return fitted logical HBM bandwidth in GB/s for GEMM width B."""
    return peak_hbm_gbps * hbm_efficiency_pct(b, clamp=clamp) / 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict fitted GB200 FP16 GEMM HBM efficiency for one B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("b", type=float, help="GEMM right-hand-side width B")
    parser.add_argument(
        "--peak-hbm-gbps",
        type=float,
        default=DEFAULT_PEAK_HBM_GBPS,
        help="peak HBM bandwidth for GB/s reconstruction",
    )
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="clamp B outside fitted range instead of raising an error",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    relative_efficiency = relative_hbm_efficiency_vs_b1(args.b, clamp=args.clamp)
    efficiency_pct = hbm_efficiency_pct(args.b, clamp=args.clamp)
    bandwidth_gbps = hbm_gbps(
        args.b,
        peak_hbm_gbps=args.peak_hbm_gbps,
        clamp=args.clamp,
    )

    print(f"B: {args.b:g}")
    print(f"relative_hbm_efficiency_vs_b1: {relative_efficiency:.8f}")
    print(f"hbm_efficiency_pct: {efficiency_pct:.6f}")
    print(f"hbm_gbps_at_peak_{args.peak_hbm_gbps:g}: {bandwidth_gbps:.3f}")


if __name__ == "__main__":
    main()
