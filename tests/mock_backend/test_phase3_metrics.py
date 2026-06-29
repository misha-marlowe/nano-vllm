import csv
from itertools import count

import pytest

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from tools.mock_trace_metrics import compute_metrics, read_trace, write_request_metrics


def reset_sequence_ids():
    Sequence.counter = count()


def make_single_request_trace(path):
    reset_sequence_ids()
    llm = LLM(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        virtual_time=True,
        trace_output=str(path),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        decode_base_ms=0.5,
        decode_ms_per_token=0.02,
    )
    llm.add_request(list(range(128)), SamplingParams(max_tokens=8, ignore_eos=True))
    while not llm.is_finished():
        llm.step()


def test_single_request_metrics_match_closed_form(tmp_path):
    trace_path = tmp_path / "trace.csv"
    make_single_request_trace(trace_path)

    request_metrics, summary = compute_metrics(read_trace(trace_path))
    row = request_metrics[0]
    prefill_ms = 1.0 + 128 * 0.01
    decode_ms = 0.5 + 0.02
    total_ms = prefill_ms + 8 * decode_ms

    assert row["request_id"] == "0"
    assert row["ttft_ms"] == pytest.approx(prefill_ms + decode_ms)
    assert row["mean_tbt_ms"] == pytest.approx(decode_ms)
    assert row["tpot_ms"] == pytest.approx(decode_ms)
    assert row["queueing_delay_ms"] == pytest.approx(0.0)
    assert row["prefill_time_ms"] == pytest.approx(prefill_ms)
    assert row["decode_time_ms"] == pytest.approx(8 * decode_ms)
    assert row["output_tokens"] == 8

    assert summary["num_requests"] == 1
    assert summary["output_tokens"] == 8
    assert summary["mean_tbt_ms"] == pytest.approx(decode_ms)
    assert summary["p50_tbt_ms"] == pytest.approx(decode_ms)
    assert summary["p95_tbt_ms"] == pytest.approx(decode_ms)
    assert summary["throughput_tokens_per_sec"] == pytest.approx(8 / (total_ms / 1000.0))
    assert summary["avg_batch_size"] == pytest.approx(1.0)
    assert summary["max_kv_tokens_used"] == 136


def test_metrics_csv_output(tmp_path):
    trace_path = tmp_path / "trace.csv"
    metrics_path = tmp_path / "metrics.csv"
    make_single_request_trace(trace_path)

    request_metrics, _ = compute_metrics(read_trace(trace_path))
    write_request_metrics(metrics_path, request_metrics)

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["request_id"] == "0"
    assert float(rows[0]["ttft_ms"]) > 0
