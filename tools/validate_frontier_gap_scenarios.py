import argparse
import csv
import sys
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.mock.frontier_gap_scenarios import ParetoPoint, SCENARIOS, timed_scenario
from nanovllm.mock.timing.gptoss_roofline import pareto_uplr
from tools.validate_roofline_backend import frontier_rows


DEFAULT_OUTPUT_DIR = ROOT / "results/roofline_validation/frontier_gap_scenarios"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_points(args) -> list[dict]:
    frontier_args = SimpleNamespace(**vars(args))
    frontier_args.link_us = [args.link_us]
    rows = frontier_rows(frontier_args)
    return [
        row for row in rows
        if row["family"] == "hybrid" and float(row["link_us"]) == float(args.link_us)
    ]


def run_scenarios(args) -> tuple[list[dict], list[dict]]:
    rows = []
    timing = defaultdict(float)
    points = load_points(args)
    for idx, point in enumerate(points, start=1):
        print(
            f"[{idx}/{len(points)}] link={point['link_us']} tp={point['tp_g']} "
            f"a_g={point['a_g']} ck={point['ck']} gb={point['gb']}",
            flush=True,
        )
        pareto_point = ParetoPoint(
            link_us=float(point["link_us"]),
            tp_g=int(point["tp_g"]),
            a_g=int(point["a_g"]),
            ck=int(point["ck"]),
            gb=int(point["gb"]),
            isl=int(args.isl),
        )
        rows.append({
            "scenario": "analytical",
            "link_us": float(point["link_us"]),
            "tp_g": int(point["tp_g"]),
            "a_g": int(point["a_g"]),
            "ck": int(point["ck"]),
            "gb": int(point["gb"]),
            "interactivity": float(point["interactivity"]),
            "tok_s_per_gpu": float(point["tok_s_per_gpu"]),
            "wall_time_s": "",
            "notes": "closed_form",
        })
        for scenario in SCENARIOS:
            result = timed_scenario(
                scenario,
                pareto_point,
                batches=args.global_des_batches,
                gpu_backend=args.gpu_backend,
            )
            timing[result.name] += result.wall_time_s
            rows.append({
                "scenario": result.name,
                "link_us": float(point["link_us"]),
                "tp_g": int(point["tp_g"]),
                "a_g": int(point["a_g"]),
                "ck": int(point["ck"]),
                "gb": int(point["gb"]),
                "interactivity": result.interactivity,
                "tok_s_per_gpu": result.tok_s_per_gpu,
                "first_microbatch_ms": result.first_microbatch_ms,
                "effective_batch_ms": result.effective_batch_ms,
                "wall_time_s": result.wall_time_s,
                "rel_x_err_vs_analytical": (result.interactivity - float(point["interactivity"])) / float(point["interactivity"]),
                "rel_y_err_vs_analytical": (result.tok_s_per_gpu - float(point["tok_s_per_gpu"])) / float(point["tok_s_per_gpu"]),
                "notes": result.notes,
            })
    timing_rows = [
        {
            "scenario": name,
            "seconds": seconds,
            "points": len(points),
            "seconds_per_point": seconds / len(points) if points else "",
        }
        for name, seconds in sorted(timing.items())
    ]
    return rows, timing_rows


def write_svg(path: Path, rows: list[dict], *, isl: int, link_us: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 640
    pad_left, pad_right, pad_top, pad_bottom = 82, 250, 56, 72
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    points = [
        {"x": float(row["interactivity"]), "y": float(row["tok_s_per_gpu"])}
        for row in rows
    ]
    x_ticks = nice_ticks(min(point["x"] for point in points), max(point["x"] for point in points), 6)
    y_ticks = nice_ticks(0.0, max(point["y"] for point in points), 6)
    min_x, max_x = x_ticks[0], x_ticks[-1]
    min_y, max_y = y_ticks[0], y_ticks[-1]

    def sx(x):
        return pad_left + (x - min_x) / max(max_x - min_x, 1e-9) * plot_w

    def sy(y):
        return pad_top + plot_h - (y - min_y) / max(max_y - min_y, 1e-9) * plot_h

    colors = {
        "analytical": "#111827",
        "afd_global_des": "#16a34a",
        "context_growth": "#2563eb",
        "prefill_interference": "#dc2626",
        "sparse_arrivals": "#d97706",
        "operator_overheads": "#7c3aed",
        "roofline_backend": "#0891b2",
        "collective_contention": "#be123c",
        "replica_imbalance": "#4f46e5",
        "kv_transfer": "#9333ea",
        "runtime_optimizations": "#15803d",
    }
    dash = {
        "analytical": "",
        "afd_global_des": "10 2 2 2",
        "runtime_optimizations": "4 2",
    }
    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">ISL={isl:,}, link={link_us:g}us: AFD global DES baseline plus Frontier-gap scenarios</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" stroke="#111827"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#111827"/>',
    ]
    for tick in x_ticks:
        x = sx(tick)
        body.append(f'<line x1="{x:.2f}" y1="{pad_top + plot_h:.2f}" x2="{x:.2f}" y2="{pad_top + plot_h + 6:.2f}" stroke="#111827"/>')
        body.append(f'<line x1="{x:.2f}" y1="{pad_top:.2f}" x2="{x:.2f}" y2="{pad_top + plot_h:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.2f}" y="{pad_top + plot_h + 24:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick:g}</text>')
    for tick in y_ticks:
        y = sy(tick)
        body.append(f'<line x1="{pad_left - 6:.2f}" y1="{y:.2f}" x2="{pad_left:.2f}" y2="{y:.2f}" stroke="#111827"/>')
        body.append(f'<line x1="{pad_left:.2f}" y1="{y:.2f}" x2="{pad_left + plot_w:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{pad_left - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:g}</text>')

    for scenario in colors:
        group = [
            {"x": float(row["interactivity"]), "y": float(row["tok_s_per_gpu"])}
            for row in rows
            if row["scenario"] == scenario
        ]
        if not group:
            continue
        frontier = pareto_uplr(group)
        coords = " ".join(f'{sx(point["x"]):.2f},{sy(point["y"]):.2f}' for point in frontier)
        dash_attr = f' stroke-dasharray="{dash[scenario]}"' if scenario in dash and dash[scenario] else ""
        body.append(f'<polyline points="{coords}" fill="none" stroke="{colors[scenario]}" stroke-width="2.0"{dash_attr}/>')
        for point in frontier:
            body.append(f'<circle cx="{sx(point["x"]):.2f}" cy="{sy(point["y"]):.2f}" r="2.8" fill="white" stroke="{colors[scenario]}" stroke-width="1.4"/>')

    legend_x, legend_y = pad_left + plot_w + 28, pad_top + 8
    for idx, scenario in enumerate(colors):
        y = legend_y + idx * 22
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{colors[scenario]}" stroke-width="2.2"/>')
        body.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="sans-serif" font-size="11">{scenario}</text>')
    body.append(f'<text x="{pad_left + plot_w / 2:.0f}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="13">interactivity (tok/s/user)</text>')
    body.append(f'<text x="18" y="{pad_top + plot_h / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {pad_top + plot_h / 2:.0f})" font-family="sans-serif" font-size="13">output tok/s/GPU</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if max_value <= min_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(count - 1, 1)
    magnitude = 10 ** int(f"{raw_step:e}".split("e")[1])
    step = min([1, 2, 2.5, 5, 10], key=lambda c: abs(c * magnitude - raw_step)) * magnitude
    start = step * int(min_value // step)
    if start > min_value:
        start -= step
    ticks = []
    value = start
    while value < max_value + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def main():
    parser = argparse.ArgumentParser(description="Plot Frontier-gap scenario overlays against AFD global DES.")
    parser.add_argument("--isl", type=int, default=8192)
    parser.add_argument("--gpu-backend", choices=["measured", "roofline"], default="measured")
    parser.add_argument("--link-us", type=float, default=12.0)
    parser.add_argument("--global-des-batches", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, timing_rows = run_scenarios(args)
    stem = f"isl{args.isl}_link{int(args.link_us)}"
    csv_path = args.output_dir / f"{stem}_frontier_gap_scenarios.csv"
    timing_path = args.output_dir / f"{stem}_frontier_gap_timing.csv"
    svg_path = args.output_dir / f"{stem}_frontier_gap_scenarios.svg"
    write_csv(csv_path, rows)
    write_csv(timing_path, timing_rows)
    write_svg(svg_path, rows, isl=args.isl, link_us=args.link_us)
    print(f"WROTE {csv_path}")
    print(f"WROTE {timing_path}")
    print(f"WROTE {svg_path}")


if __name__ == "__main__":
    main()
