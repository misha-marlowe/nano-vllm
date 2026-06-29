import csv
import subprocess
import sys


def test_run_mock_workload_writes_trace_metrics_and_plots(tmp_path):
    output_dir = tmp_path / "results"
    cmd = [
        sys.executable,
        "tools/run_mock_workload.py",
        "--mode",
        "afd",
        "--pipeline-mode",
        "ideal_pipeline",
        "--num-requests",
        "4",
        "--arrival-process",
        "burst",
        "--fixed-isl",
        "16",
        "--fixed-osl",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=".")

    expected_files = [
        "mock_workload.csv",
        "mock_trace.csv",
        "mock_metrics.csv",
        "mock_summary.csv",
        "ttft_distribution.svg",
        "tbt_distribution.svg",
        "throughput_over_time.svg",
        "kv_usage_over_time.svg",
        "batch_size_over_time.svg",
    ]
    for name in expected_files:
        assert (output_dir / name).exists()

    with (output_dir / "mock_metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == 4
    assert {row["output_tokens"] for row in metrics} == {"2"}

    trace_text = (output_dir / "mock_trace.csv").read_text()
    assert "pipeline_fill_start" in trace_text
