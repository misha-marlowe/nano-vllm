import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.mock.timing.ac_model import cs4_offload as CS4M
from nanovllm.mock.timing.gptoss_roofline import (
    HELIOS,
    GPTOSS,
    gpu_only_point,
    hybrid_point,
    pareto_uplr,
)


DEFAULT_RAWDATA = ROOT / "tests/mock_backend/fixtures/perf_model_rawdata.txt"
P_GRID = [1, 2, 4, 8, 9, 12, 18, 24, 36, 72]
B_GRID = [1, 4, 8, 16, 64, 128, 256]
GB_GRID = [8, 16, 32, 64, 128, 256]
TP_G_HYB = [1, 2, 4, 8]
A_G_GRID = [1, 2, 4, 8, 16, 32]
CK_SET = [1, 2, 4, 8, 16, 32, 64, 128, 256]


RAW_POINT_CASES = [
    {
        "name": "gpu_only_tp1_b256",
        "kind": "gpu_only",
        "expected_x": 163.7,
        "expected_y": 41896.2,
        "kwargs": {"B": 256, "tp_g": 1},
    },
    {
        "name": "gpu_only_tp2_b256",
        "kind": "gpu_only",
        "expected_x": 223.8,
        "expected_y": 28650.4,
        "kwargs": {"B": 256, "tp_g": 2},
    },
    {
        "name": "hybrid_link12_tp1_ag1_ck128_gb256",
        "kind": "hybrid",
        "expected_x": 327.8,
        "expected_y": 83920.8,
        "link_us": 12.0,
        "kwargs": {"gb": 256, "tp_g": 1, "a_g": 1, "ck": 128},
    },
    {
        "name": "hybrid_link4_tp1_ag1_ck128_gb256",
        "kind": "hybrid",
        "expected_x": 362.0,
        "expected_y": 92669.8,
        "link_us": 4.0,
        "kwargs": {"gb": 256, "tp_g": 1, "a_g": 1, "ck": 128},
    },
]


def rel_err(got: float, expected: float) -> float:
    return abs(got - expected) / max(abs(expected), 1e-12)


def with_link_us(link_us: float, fn):
    old_link = CS4M.CLOS_LAT_US
    CS4M.CLOS_LAT_US = link_us
    try:
        return fn()
    finally:
        CS4M.CLOS_LAT_US = old_link


def compute_raw_point(case: dict, isl: int, backend: str):
    if case["kind"] == "gpu_only":
        point = gpu_only_point(HELIOS, GPTOSS, isl=isl, backend=backend, **case["kwargs"])
    else:
        point = with_link_us(
            case["link_us"],
            lambda: hybrid_point(HELIOS, GPTOSS, isl=isl, backend=backend, **case["kwargs"]),
        )
    return point["x"], point["y"]


def validate_points(args) -> list[dict]:
    rows = []
    for case in RAW_POINT_CASES:
        got_x, got_y = compute_raw_point(case, args.isl, args.gpu_backend)
        err_x = rel_err(got_x, case["expected_x"])
        err_y = rel_err(got_y, case["expected_y"])
        rows.append({
            "case": case["name"],
            "expected_interactivity": case["expected_x"],
            "got_interactivity": got_x,
            "interactivity_rel_error": err_x,
            "expected_toks_per_gpu": case["expected_y"],
            "got_toks_per_gpu": got_y,
            "toks_per_gpu_rel_error": err_y,
            "pass": err_x <= args.tolerance and err_y <= args.tolerance,
        })
    return rows


def sweep_gpu_only(isl: int, backend: str):
    return [
        gpu_only_point(HELIOS, GPTOSS, B=B, isl=isl, tp_g=tp_g, backend=backend)
        for tp_g in P_GRID
        for B in B_GRID
    ]


def sweep_hybrid(isl: int, link_us: float, backend: str):
    def run():
        points = []
        for tp_g in TP_G_HYB:
            for a_g in A_G_GRID:
                if a_g * tp_g > HELIOS.gpus_per_rack:
                    continue
                for gb in GB_GRID:
                    for ck in CK_SET:
                        point = hybrid_point(HELIOS, GPTOSS, gb, isl, tp_g, a_g, ck, backend=backend)
                        if point:
                            points.append(point)
        return points

    return with_link_us(link_us, run)


def frontier_rows(args):
    rows = []
    gpu_front = pareto_uplr(sweep_gpu_only(args.isl, args.gpu_backend), "y")
    for point in gpu_front:
        rows.append({
            "family": "gpu_only",
            "link_us": "",
            "interactivity": point["x"],
            "tok_s_per_gpu": point["y"],
            "tp_g": point["tp_g"],
            "a_g": "",
            "ck": "",
            "gb": point["B"],
            "s": "",
        })
    for link_us in args.link_us:
        hybrid_front = pareto_uplr(sweep_hybrid(args.isl, link_us, args.gpu_backend), "y")
        for point in hybrid_front:
            rows.append({
                "family": "hybrid",
                "link_us": link_us,
                "interactivity": point["x"],
                "tok_s_per_gpu": point["y"],
                "tp_g": point["tp_g"],
                "a_g": point["a_g"],
                "ck": point["ck"],
                "gb": point["gb"],
                "s": point["s"],
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 900, 560, 64
    xs = [float(row["interactivity"]) for row in rows]
    ys = [float(row["tok_s_per_gpu"]) for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def sx(x): return pad + (x - min_x) / span_x * (width - 2 * pad)
    def sy(y): return height - pad - (y - min_y) / span_y * (height - 2 * pad)

    colors = {"gpu_only": "#111827", "4.0": "#2563eb", "6.0": "#0f766e", "12.0": "#7c3aed", "24.0": "#d97706", "36.0": "#dc2626"}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = "gpu_only" if row["family"] == "gpu_only" else str(float(row["link_us"]))
        groups.setdefault(key, []).append(row)
    body = []
    for key, group in groups.items():
        group = sorted(group, key=lambda row: float(row["interactivity"]))
        color = colors.get(key, "#64748b")
        points = " ".join(f'{sx(float(row["interactivity"])):.2f},{sy(float(row["tok_s_per_gpu"])):.2f}' for row in group)
        label = "MI455X-only" if key == "gpu_only" else f"hybrid link={key}us"
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in group:
            body.append(f'<circle cx="{sx(float(row["interactivity"])):.2f}" cy="{sy(float(row["tok_s_per_gpu"])):.2f}" r="3" fill="{color}"/>')
        y = 36 + 18 * list(groups).index(key)
        body.append(f'<text x="{width - 220}" y="{y}" font-family="sans-serif" font-size="12" fill="{color}">{label}</text>')
    path.write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Section 5 ISL=8K Pareto reproduction</text>
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#111827"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#111827"/>
<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">interactivity (tok/s/user)</text>
<text x="18" y="{height / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.0f})" font-family="sans-serif" font-size="13">output tok/s/GPU</text>
{chr(10).join(body)}
</svg>
""")


def main():
    parser = argparse.ArgumentParser(description="Validate the GPT-OSS roofline timing backend against rawdata.")
    parser.add_argument("--isl", type=int, default=8192)
    parser.add_argument("--gpu-backend", choices=["measured", "roofline"], default="measured")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--rawdata", type=Path, default=DEFAULT_RAWDATA)
    parser.add_argument("--output-dir", type=Path, default=Path("results/roofline_validation"))
    parser.add_argument("--link-us", type=float, nargs="+", default=[4.0, 6.0, 12.0, 24.0, 36.0])
    args = parser.parse_args()

    if not args.rawdata.exists():
        raise FileNotFoundError(args.rawdata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_rows = validate_points(args)
    write_csv(args.output_dir / "reproduced_points.csv", point_rows)
    front_rows = frontier_rows(args)
    write_csv(args.output_dir / "section5_8k_frontier.csv", front_rows)
    write_svg(args.output_dir / "section5_8k_reproduction.svg", front_rows)

    print("Raw point checks")
    for row in point_rows:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{status} {row['case']}: "
            f"x={row['got_interactivity']:.1f} vs {row['expected_interactivity']:.1f} "
            f"({row['interactivity_rel_error'] * 100:.3f}%), "
            f"y={row['got_toks_per_gpu']:.1f} vs {row['expected_toks_per_gpu']:.1f} "
            f"({row['toks_per_gpu_rel_error'] * 100:.3f}%)"
        )
    print(f"wrote {args.output_dir / 'reproduced_points.csv'}")
    print(f"wrote {args.output_dir / 'section5_8k_frontier.csv'}")
    print(f"wrote {args.output_dir / 'section5_8k_reproduction.svg'}")
    if not all(row["pass"] for row in point_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
