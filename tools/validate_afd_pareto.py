import argparse
import csv
import sys
from itertools import count
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.mock import DESConfig, DESEngine, DESRequest
from tools.validate_roofline_backend import frontier_rows, sweep_gpu_only
from nanovllm.mock.timing.gptoss_roofline import pareto_uplr


NUM_LAYERS = 36
DEFAULT_OUTPUT_DIR = ROOT / "results/roofline_validation/afd_pareto_sim"
DEFAULT_GPU_ONLY_CSV = ROOT / "results/roofline_validation/gpu_only_pareto_sim/gpu_only_pareto_direct_mock_des.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


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


def write_side_by_side_svg(path: Path, rows: list[dict]):
    """Plot analytical Pareto curves beside DES replay points/frontiers.

    The DES panel is intentionally named as a replay frontier. The rows only
    contain configurations selected by the analytical Pareto model, so
    the non-dominated DES line is not necessarily the true DES Pareto over the
    full design space.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    pad_left, pad_right, pad_top, pad_bottom = 72, 34, 60, 64
    panel_gap = 72
    panel_w = (width - pad_left - pad_right - panel_gap) / 2
    plot_h = height - pad_top - pad_bottom
    left_x0 = pad_left
    right_x0 = pad_left + panel_w + panel_gap
    y0 = pad_top

    points = []
    for row in rows:
        points.append((float(row["direct_interactivity"]), float(row["direct_tok_s_per_gpu"])))
        points.append((float(row["des_interactivity"]), float(row["des_tok_s_per_gpu"])))
        if row.get("mock_interactivity") not in ("", None) and row.get("mock_tok_s_per_gpu") not in ("", None):
            points.append((float(row["mock_interactivity"]), float(row["mock_tok_s_per_gpu"])))
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = 0.0
    max_y = max(y for _, y in points)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def sx(panel_x0: float, x: float) -> float:
        return panel_x0 + (x - min_x) / span_x * panel_w

    def sy(y: float) -> float:
        return y0 + plot_h - (y - min_y) / span_y * plot_h

    colors = {
        4.0: "#2563eb",
        6.0: "#0f766e",
        12.0: "#7c3aed",
        24.0: "#d97706",
        36.0: "#dc2626",
    }
    body = []
    body.append(f'<rect width="100%" height="100%" fill="white"/>')
    body.append(f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">AFD Section 5: analytical vs nano-vLLM mock vs DES replay</text>')
    body.append(panel_axes(left_x0, y0, panel_w, plot_h, "Analytical Pareto"))
    body.append(panel_axes(right_x0, y0, panel_w, plot_h, "DES replay of analytical configs"))

    for link_us in sorted({float(row["link_us"]) for row in rows}):
        group = [row for row in rows if float(row["link_us"]) == link_us]
        color = colors.get(link_us, "#64748b")
        direct_group = sorted(group, key=lambda row: float(row["direct_interactivity"]))
        direct_points = " ".join(
            f'{sx(left_x0, float(row["direct_interactivity"])):.2f},{sy(float(row["direct_tok_s_per_gpu"])):.2f}'
            for row in direct_group
        )
        body.append(f'<polyline points="{direct_points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for row in direct_group:
            body.append(circle(sx(left_x0, float(row["direct_interactivity"])), sy(float(row["direct_tok_s_per_gpu"])), color, 3.0, 0.95))

        des_group = sorted(group, key=lambda row: float(row["des_interactivity"]))
        for row in des_group:
            body.append(circle(sx(right_x0, float(row["des_interactivity"])), sy(float(row["des_tok_s_per_gpu"])), color, 2.5, 0.25))
        des_frontier = pareto_uplr([
            {
                "x": float(row["des_interactivity"]),
                "y": float(row["des_tok_s_per_gpu"]),
                "row": row,
            }
            for row in des_group
        ])
        des_points = " ".join(f'{sx(right_x0, p["x"]):.2f},{sy(p["y"]):.2f}' for p in des_frontier)
        if des_points:
            body.append(f'<polyline points="{des_points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            for p in des_frontier:
                body.append(circle(sx(right_x0, p["x"]), sy(p["y"]), color, 3.2, 0.95))
        mock_frontier = pareto_uplr([
            {
                "x": float(row["mock_interactivity"]),
                "y": float(row["mock_tok_s_per_gpu"]),
                "row": row,
            }
            for row in des_group
            if row.get("mock_interactivity") not in ("", None)
            and row.get("mock_tok_s_per_gpu") not in ("", None)
        ])
        mock_points = " ".join(f'{sx(right_x0, p["x"]):.2f},{sy(p["y"]):.2f}' for p in mock_frontier)
        if mock_points:
            body.append(f'<polyline points="{mock_points}" fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="8 4"/>')
            for p in mock_frontier:
                body.append(open_square(sx(right_x0, p["x"]), sy(p["y"]), color, 3.0))

    legend_x = width - 184
    for idx, link_us in enumerate(sorted(colors)):
        y = 76 + idx * 20
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{colors[link_us]}" stroke-width="2.4"/>')
        body.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#111827">link={link_us:.0f}us</text>')

    body.append(f'<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">interactivity (tok/s/user)</text>')
    body.append(f'<text x="18" y="{height / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.0f})" font-family="sans-serif" font-size="13">output tok/s/GPU</text>')
    body.append(f'<text x="{right_x0 + panel_w / 2:.0f}" y="{height - 38}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">right panel: solid circles=DES, dashed squares=nano-vLLM mock</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def write_link_overlay_svg(
    path: Path,
    rows: list[dict],
    link_us: float,
    isl: int,
    colocated_rows: list[dict] | None = None,
):
    """Overlay analytical and DES curves for one link latency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    group = [row for row in rows if float(row["link_us"]) == float(link_us)]
    if not group:
        return
    colocated_rows = colocated_rows or []

    afd_direct = sorted(
        [
            {
                "x": float(row["direct_interactivity"]),
                "y": float(row["direct_tok_s_per_gpu"]),
            }
            for row in group
        ],
        key=lambda point: point["x"],
    )
    afd_des_all = [
        {
            "x": float(row["des_interactivity"]),
            "y": float(row["des_tok_s_per_gpu"]),
        }
        for row in group
    ]
    afd_mock_all = [
        {
            "x": float(row["mock_interactivity"]),
            "y": float(row["mock_tok_s_per_gpu"]),
        }
        for row in group
        if row.get("mock_interactivity") not in ("", None)
        and row.get("mock_tok_s_per_gpu") not in ("", None)
    ]
    afd_des_frontier = pareto_uplr(afd_des_all)
    afd_mock_frontier = pareto_uplr(afd_mock_all) if afd_mock_all else []
    colocated_direct = colocated_curve(colocated_rows, "direct")
    colocated_des = colocated_curve(colocated_rows, "des")

    width, height = 880, 560
    pad_left, pad_right, pad_top, pad_bottom = 82, 34, 58, 72
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    all_points = afd_direct + afd_des_all + afd_mock_all + colocated_direct + colocated_des
    min_x = min(point["x"] for point in all_points)
    max_x = max(point["x"] for point in all_points)
    min_y = 0.0
    max_y = max(point["y"] for point in all_points)
    x_ticks = nice_ticks(min_x, max_x, 6)
    y_ticks = nice_ticks(min_y, max_y, 6)
    min_x, max_x = x_ticks[0], x_ticks[-1]
    min_y, max_y = y_ticks[0], y_ticks[-1]

    def sx(x: float) -> float:
        return pad_left + (x - min_x) / max(max_x - min_x, 1e-9) * plot_w

    def sy(y: float) -> float:
        return pad_top + plot_h - (y - min_y) / max(max_y - min_y, 1e-9) * plot_h

    def polyline(points: list[dict], color: str, dash: str = "") -> str:
        coords = " ".join(f'{sx(point["x"]):.2f},{sy(point["y"]):.2f}' for point in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        linecap = ' stroke-linecap="round"' if dash else ""
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6"{dash_attr}{linecap}/>'

    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">ISL={format_int(isl)}, link={link_us:.0f}us: colocated + AFD analytical/mock/DES</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" stroke="#111827"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#111827"/>',
    ]

    for tick in x_ticks:
        x = sx(tick)
        body.append(f'<line x1="{x:.2f}" y1="{pad_top + plot_h:.2f}" x2="{x:.2f}" y2="{pad_top + plot_h + 6:.2f}" stroke="#111827"/>')
        body.append(f'<line x1="{x:.2f}" y1="{pad_top:.2f}" x2="{x:.2f}" y2="{pad_top + plot_h:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.2f}" y="{pad_top + plot_h + 24:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#111827">{tick:g}</text>')
    for tick in y_ticks:
        y = sy(tick)
        body.append(f'<line x1="{pad_left - 6:.2f}" y1="{y:.2f}" x2="{pad_left:.2f}" y2="{y:.2f}" stroke="#111827"/>')
        body.append(f'<line x1="{pad_left:.2f}" y1="{y:.2f}" x2="{pad_left + plot_w:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{pad_left - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#111827">{tick:g}</text>')

    body.append(polyline(colocated_direct, "#111827"))
    body.append(polyline(colocated_des, "#111827", "2 5"))
    body.append(polyline(afd_direct, "#2563eb"))
    for point in afd_direct:
        body.append(circle(sx(point["x"]), sy(point["y"]), "#2563eb", 3.2, 0.9))
    if afd_mock_frontier:
        body.append(polyline(afd_mock_frontier, "#2563eb", "8 4"))
        for point in afd_mock_frontier:
            body.append(open_square(sx(point["x"]), sy(point["y"]), "#2563eb", 4.0))
    body.append(polyline(afd_des_frontier, "#2563eb", "2 5"))
    for point in afd_des_frontier:
        body.append(circle(sx(point["x"]), sy(point["y"]), "#2563eb", 3.4, 0.95))
    for point in colocated_direct:
        body.append(circle(sx(point["x"]), sy(point["y"]), "#111827", 3.0, 0.85))
    for point in colocated_des:
        body.append(open_circle(sx(point["x"]), sy(point["y"]), "#111827", 4.1))

    legend_x, legend_y = pad_left + plot_w - 254, pad_top + 24
    body.extend([
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 30}" y2="{legend_y}" stroke="#2563eb" stroke-width="2.6"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">AFD analytical</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 22}" x2="{legend_x + 30}" y2="{legend_y + 22}" stroke="#2563eb" stroke-width="2.6" stroke-dasharray="2 5"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 26}" font-family="sans-serif" font-size="12">AFD DES</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 44}" x2="{legend_x + 30}" y2="{legend_y + 44}" stroke="#2563eb" stroke-width="2.6" stroke-dasharray="8 4"/>',
        f'<rect x="{legend_x + 11}" y="{legend_y + 40}" width="8" height="8" fill="white" stroke="#2563eb" stroke-width="1.5"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 48}" font-family="sans-serif" font-size="12">AFD nano-vLLM mock</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 66}" x2="{legend_x + 30}" y2="{legend_y + 66}" stroke="#111827" stroke-width="2.6"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 70}" font-family="sans-serif" font-size="12">colocated analytical</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 88}" x2="{legend_x + 30}" y2="{legend_y + 88}" stroke="#111827" stroke-width="2.6" stroke-dasharray="2 5"/>',
        f'<circle cx="{legend_x + 15}" cy="{legend_y + 88}" r="4.1" fill="white" stroke="#111827" stroke-width="1.5"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 92}" font-family="sans-serif" font-size="12">colocated DES</text>',
        f'<text x="{width / 2:.0f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="13">interactivity (tok/s/user)</text>',
        f'<text x="18" y="{height / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.0f})" font-family="sans-serif" font-size="13">output tok/s/GPU</text>',
    ])
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def colocated_curve(rows: list[dict], case: str) -> list[dict]:
    aliases = {"des": {"des", "des_batched"}, "direct": {"direct"}}
    accepted = aliases.get(case, {case})
    return sorted(
        [
            {
                "x": float(row["interactivity"]),
                "y": float(row["tok_s_per_gpu"]),
            }
            for row in rows
            if row.get("case") in accepted
        ],
        key=lambda point: point["x"],
    )


def build_colocated_rows(args) -> list[dict]:
    direct_points = pareto_uplr(sweep_gpu_only(args.isl, args.gpu_backend), "y")
    rows = []
    for point in direct_points:
        direct_tbt_ms = 1000.0 / float(point["x"])
        rows.append({
            "case": "direct",
            "B": int(point["B"]),
            "tp_g": int(point["tp_g"]),
            "interactivity": float(point["x"]),
            "tok_s_per_gpu": float(point["y"]),
            "tpot_or_tbt_ms": direct_tbt_ms,
        })
        des_x, des_y, des_tbt_ms = run_des_colocated_point(point, args)
        rows.append({
            "case": "des",
            "B": int(point["B"]),
            "tp_g": int(point["tp_g"]),
            "interactivity": des_x,
            "tok_s_per_gpu": des_y,
            "tpot_or_tbt_ms": des_tbt_ms,
            "rel_x_err_vs_analytical": (des_x - float(point["x"])) / float(point["x"]),
            "rel_y_err_vs_analytical": (des_y - float(point["y"])) / float(point["y"]),
        })
    return rows


def run_des_colocated_point(point: dict, args) -> tuple[float, float, float]:
    batch_size = int(point["B"])
    tp_g = int(point["tp_g"])
    config = DESConfig(
        mode="colocated",
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        timing_backend="gptoss_roofline",
        roofline_gpu_backend=args.gpu_backend,
        roofline_tp_g=tp_g,
        des_batch_decode=True,
        des_max_batch_size=batch_size,
    )
    engine = DESEngine(config)
    for request_id in range(batch_size):
        engine.submit(DESRequest(request_id=request_id, arrival_ms=0.0, isl=args.isl, osl=1))
    rows = engine.run()
    token_times = [float(row["virtual_time_ms"]) for row in rows if row["stage"] == "token_emit"]
    start_times = [
        float(row["virtual_time_ms"])
        for row in rows
        if row["stage"] == "decode_start" and row.get("resource", "") == "decode_0"
    ]
    if not token_times or not start_times:
        raise ValueError("trace did not contain a complete colocated decode block")
    tbt_ms = max(token_times) - min(start_times)
    interactivity = 1000.0 / tbt_ms
    tok_s_per_gpu = batch_size * 1000.0 / (tbt_ms * tp_g)
    return interactivity, tok_s_per_gpu, tbt_ms


def nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if max_value <= min_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(count - 1, 1)
    magnitude = 10 ** int(f"{raw_step:e}".split("e")[1])
    candidates = [1, 2, 2.5, 5, 10]
    step = min(candidates, key=lambda candidate: abs(candidate * magnitude - raw_step)) * magnitude
    start = step * int(min_value // step)
    if start > min_value:
        start -= step
    ticks = []
    value = start
    while value < max_value + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def panel_axes(x0: float, y0: float, w: float, h: float, title: str) -> str:
    return "\n".join([
        f'<line x1="{x0:.2f}" y1="{y0 + h:.2f}" x2="{x0 + w:.2f}" y2="{y0 + h:.2f}" stroke="#111827"/>',
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + h:.2f}" stroke="#111827"/>',
        f'<text x="{x0 + w / 2:.2f}" y="{y0 - 16:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111827">{title}</text>',
    ])


def circle(x: float, y: float, color: str, r: float, opacity: float) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.1f}" fill="{color}" fill-opacity="{opacity:.2f}"/>'


def open_circle(x: float, y: float, color: str, r: float) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.1f}" fill="white" stroke="{color}" stroke-width="1.5"/>'


def open_square(x: float, y: float, color: str, r: float) -> str:
    side = 2 * r
    return f'<rect x="{x - r:.2f}" y="{y - r:.2f}" width="{side:.1f}" height="{side:.1f}" fill="white" stroke="{color}" stroke-width="1.5"/>'


def fmt(value, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def pct(value) -> str:
    return f"{float(value) * 100:.1f}%"


def format_int(value: int) -> str:
    return f"{int(value):,}"


def isl_tag(value: int) -> str:
    return f"isl{int(value)}"


def load_direct_hybrid_points(args) -> list[dict]:
    rows = frontier_rows(args)
    return [row for row in rows if row["family"] == "hybrid"]


def run_mock_point(point: dict, args) -> tuple[float, float, float]:
    trace_path = args.output_dir / "_mock_tmp.csv"
    if trace_path.exists():
        trace_path.unlink()
    Sequence.counter = count()
    gb = int(point["gb"])
    ck = int(point["ck"])
    isl = int(args.isl)
    llm = LLM(
        "__mock__",
        mock_backend=True,
        mock_mode="afd",
        virtual_time=True,
        trace_output=str(trace_path),
        max_num_seqs=gb,
        max_num_batched_tokens=max(gb * (isl + 1), 1),
        max_model_len=isl + 2,
        mock_kv_capacity_tokens=gb * (isl + 1 + 256),
        mock_block_size=isl + 2,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        pipeline_mode="discrete_pipeline",
        microbatch_size=ck,
        attention_replicas=int(point["a_g"]),
        gpu_to_cs_link_resources=1,
        cs_rest_resources=1,
        cs_to_gpu_link_resources=1,
        timing_backend="gptoss_roofline",
        roofline_gpu_backend=args.gpu_backend,
        roofline_tp_g=int(point["tp_g"]),
        gpu_cs_link_us=float(point["link_us"]),
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    prompt = range(isl)
    for _ in range(gb):
        llm.add_request(prompt, params)
    while not llm.is_finished():
        llm.step()
    rows = read_rows(trace_path)
    trace_path.unlink()
    first_ms, total_ms = trace_decode_times(rows)
    return normalize(first_ms, total_ms, gb, int(point["tp_g"]))


def run_des_point(point: dict, args) -> tuple[float, float, float]:
    gb = int(point["gb"])
    config = DESConfig(
        mode="afd",
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        attention_replicas=int(point["a_g"]),
        gpu_to_cs_link_resources=1,
        cs_rest_resources=1,
        cs_to_gpu_link_resources=1,
        timing_backend="gptoss_roofline",
        roofline_gpu_backend=args.gpu_backend,
        roofline_tp_g=int(point["tp_g"]),
        gpu_cs_link_us=float(point["link_us"]),
        chunk_batch=int(point["ck"]),
        des_batch_decode=True,
        des_max_batch_size=gb,
    )
    engine = DESEngine(config)
    for request_id in range(gb):
        engine.submit(DESRequest(request_id=request_id, arrival_ms=0.0, isl=args.isl, osl=1))
    rows = engine.run()
    first_ms, total_ms = trace_decode_times(rows)
    return normalize(first_ms, total_ms, gb, int(point["tp_g"]))


def trace_decode_times(rows: list[dict[str, str]]) -> tuple[float, float]:
    resource_rows = [row for row in rows if is_resource_row(row)]
    first_end = [
        float(row["virtual_time_ms"])
        for row in resource_rows
        if row["stage"] == "cs_to_gpu_link_end" and microbatch_id(row) == "0"
    ]
    token_times = [float(row["virtual_time_ms"]) for row in rows if row["stage"] == "token_emit"]
    start_times = [float(row["virtual_time_ms"]) for row in resource_rows if row["stage"].endswith("_start")]
    if not first_end or not token_times or not start_times:
        raise ValueError("trace did not contain a complete AFD decode block")
    decode_start = min(start_times)
    return min(first_end) - decode_start, max(token_times) - decode_start


def is_resource_row(row: dict[str, str]) -> bool:
    if row.get("event_scope") == "resource":
        return True
    return row.get("resource", "") != ""


def microbatch_id(row: dict[str, str]) -> str:
    if row.get("microbatch_id") not in ("", None):
        return str(row["microbatch_id"])
    for item in row.get("notes", "").split(";"):
        if item.startswith("microbatch="):
            return item.split("=", 1)[1]
    return ""


def normalize(first_layer_ms: float, block_layer_ms: float, gb: int, tp_g: int) -> tuple[float, float, float]:
    interactivity = 1000.0 / (first_layer_ms * NUM_LAYERS)
    tok_s_per_gpu = gb * 1000.0 / (block_layer_ms * NUM_LAYERS * tp_g)
    return interactivity, tok_s_per_gpu, block_layer_ms


def compare_rows(points: list[dict], args) -> list[dict]:
    rows = []
    for idx, point in enumerate(points, start=1):
        print(
            f"[{idx}/{len(points)}] "
            f"link={point['link_us']} tp={point['tp_g']} a_g={point['a_g']} "
            f"ck={point['ck']} gb={point['gb']} s={point['s']}",
            flush=True,
        )
        des_x, des_y, des_block_ms = run_des_point(point, args)
        if args.skip_mock:
            mock_x = mock_y = mock_block_ms = ""
            mock_x_err = mock_y_err = ""
        else:
            mock_x, mock_y, mock_block_ms = run_mock_point(point, args)
            mock_x_err = (mock_x - float(point["interactivity"])) / float(point["interactivity"])
            mock_y_err = (mock_y - float(point["tok_s_per_gpu"])) / float(point["tok_s_per_gpu"])
        direct_x = float(point["interactivity"])
        direct_y = float(point["tok_s_per_gpu"])
        rows.append({
            "link_us": float(point["link_us"]),
            "tp_g": int(point["tp_g"]),
            "a_g": int(point["a_g"]),
            "ck": int(point["ck"]),
            "gb": int(point["gb"]),
            "s": int(point["s"]),
            "direct_interactivity": direct_x,
            "direct_tok_s_per_gpu": direct_y,
            "mock_interactivity": mock_x,
            "mock_tok_s_per_gpu": mock_y,
            "mock_x_err": mock_x_err,
            "mock_y_err": mock_y_err,
            "des_interactivity": des_x,
            "des_tok_s_per_gpu": des_y,
            "des_x_err": (des_x - direct_x) / direct_x,
            "des_y_err": (des_y - direct_y) / direct_y,
            "mock_layer_block_ms": mock_block_ms,
            "des_layer_block_ms": des_block_ms,
        })
    return rows


def print_summary(rows: list[dict]):
    print("\nSummary by link")
    print("| link_us | points | max |mock y err| | max |DES y err| | mean mock y err | mean DES y err |")
    print("|---:|---:|---:|---:|---:|---:|")
    for link_us in sorted({row["link_us"] for row in rows}):
        group = [row for row in rows if row["link_us"] == link_us]
        mock_errs = [float(row["mock_y_err"]) for row in group if row["mock_y_err"] != ""]
        des_errs = [row["des_y_err"] for row in group]
        max_mock = f"{max(abs(err) for err in mock_errs) * 100:.2f}%" if mock_errs else "n/a"
        mean_mock = f"{sum(mock_errs) / len(mock_errs) * 100:.2f}%" if mock_errs else "n/a"
        print(
            f"| {link_us:.0f} | {len(group)} | "
            f"{max_mock} | "
            f"{max(abs(err) for err in des_errs) * 100:.2f}% | "
            f"{mean_mock} | "
            f"{sum(des_errs) / len(des_errs) * 100:.2f}% |"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate AFD Pareto points through analytical, nano-vLLM mock, and DES paths.")
    parser.add_argument("--isl", type=int, default=8192)
    parser.add_argument("--gpu-backend", choices=["measured", "roofline"], default="measured")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--link-us", type=float, nargs="+", default=[4.0, 6.0, 12.0, 24.0, 36.0])
    parser.add_argument("--from-csv", type=Path, help="Skip replay and plot/report an existing comparison CSV.")
    parser.add_argument("--overlay-link-us", type=float, default=12.0)
    parser.add_argument("--gpu-only-csv", type=Path, default=DEFAULT_GPU_ONLY_CSV)
    parser.add_argument("--skip-mock", action="store_true", help="Replay only DES against direct points; useful for very large ISL.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        rows = [
            {
                key: maybe_number(value)
                for key, value in row.items()
            }
            for row in read_rows(args.from_csv)
        ]
    else:
        points = load_direct_hybrid_points(args)
        rows = compare_rows(points, args)
    csv_path = args.output_dir / "afd_analytical_des_comparison.csv"
    svg_path = args.output_dir / "afd_analytical_vs_des_replay.svg"
    overlay_svg_path = args.output_dir / f"{isl_tag(args.isl)}_link{int(args.overlay_link_us)}_analytical_vs_des_overlay.svg"
    legacy_overlay_svg_path = args.output_dir / f"afd_link{int(args.overlay_link_us)}_analytical_vs_des_overlay.svg"
    if args.gpu_only_csv.exists() and args.isl == 8192:
        colocated_rows = read_rows(args.gpu_only_csv)
    else:
        colocated_rows = build_colocated_rows(args)
        write_csv(args.output_dir / f"{isl_tag(args.isl)}_colocated_analytical_des.csv", colocated_rows)
    write_csv(csv_path, rows)
    write_side_by_side_svg(svg_path, rows)
    write_link_overlay_svg(overlay_svg_path, rows, args.overlay_link_us, args.isl, colocated_rows)
    if args.isl == 8192:
        write_link_overlay_svg(legacy_overlay_svg_path, rows, args.overlay_link_us, args.isl, colocated_rows)
    print_summary(rows)
    print(f"\nWROTE {csv_path}")
    print(f"WROTE {svg_path}")
    print(f"WROTE {overlay_svg_path}")
    if args.isl == 8192:
        print(f"WROTE {legacy_overlay_svg_path}")


def maybe_number(value: str):
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


if __name__ == "__main__":
    main()
