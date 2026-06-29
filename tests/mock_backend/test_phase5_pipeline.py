import csv
from itertools import count

import pytest

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def reset_sequence_ids():
    Sequence.counter = count()


def read_trace(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def make_afd_pipeline(trace_path, *, pipeline_mode, num_requests, **kwargs):
    config = dict(
        mock_backend=True,
        mock_mode="afd",
        virtual_time=True,
        trace_output=str(trace_path),
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        attention_ms_base=0.0,
        attention_ms_per_token=1.0,
        attention_ms_per_isl_token=0.0,
        cs_rest_ms_base=0.0,
        cs_rest_ms_per_token=2.0,
        link_ms_one_way=0.5,
        pipeline_mode=pipeline_mode,
        microbatch_size=1,
        num_layers=32,
        max_num_seqs=num_requests,
        max_num_batched_tokens=4096,
    )
    config.update(kwargs)
    llm = LLM("__mock__", **config)
    for idx in range(num_requests):
        llm.add_request([idx], SamplingParams(max_tokens=1, ignore_eos=True))
    while not llm.is_finished():
        llm.step()
    return read_trace(trace_path)


def first_decode_duration(rows):
    start = next(row for row in rows if row["stage"] == "decode_start")
    end = next(row for row in rows if row["stage"] == "decode_end")
    return float(end["virtual_time_ms"]) - float(start["virtual_time_ms"])


def test_ideal_pipeline_preserves_single_microbatch_round_trip(tmp_path):
    reset_sequence_ids()
    sequential = make_afd_pipeline(tmp_path / "seq_one.csv", pipeline_mode="sequential", num_requests=1)
    reset_sequence_ids()
    ideal = make_afd_pipeline(tmp_path / "ideal_one.csv", pipeline_mode="ideal_pipeline", num_requests=1)

    assert first_decode_duration(sequential) == pytest.approx(4.0)
    assert first_decode_duration(ideal) == pytest.approx(4.0)


def test_ideal_pipeline_improves_multi_microbatch_decode_time(tmp_path):
    reset_sequence_ids()
    sequential = make_afd_pipeline(tmp_path / "seq_many.csv", pipeline_mode="sequential", num_requests=4)
    reset_sequence_ids()
    ideal = make_afd_pipeline(tmp_path / "ideal_many.csv", pipeline_mode="ideal_pipeline", num_requests=4)

    assert first_decode_duration(sequential) == pytest.approx(13.0)
    assert first_decode_duration(ideal) == pytest.approx(8.0625)
    assert first_decode_duration(ideal) < first_decode_duration(sequential)


def test_discrete_pipeline_matches_small_m_event_reference(tmp_path):
    reset_sequence_ids()
    rows = make_afd_pipeline(tmp_path / "discrete.csv", pipeline_mode="discrete_pipeline", num_requests=4)
    decode_duration = first_decode_duration(rows)
    cs_starts = [row for row in rows if row["stage"] == "cs_rest_start" and row["request_id"] == "0"]

    assert decode_duration == pytest.approx(10.0)
    assert any("microbatch=0" in row["notes"] for row in cs_starts)


def test_link_latency_affects_single_round_trip_more_than_steady_state(tmp_path):
    reset_sequence_ids()
    ideal_low_link = make_afd_pipeline(
        tmp_path / "ideal_low_link.csv",
        pipeline_mode="ideal_pipeline",
        num_requests=8,
        link_ms_one_way=0.5,
    )
    reset_sequence_ids()
    ideal_high_link = make_afd_pipeline(
        tmp_path / "ideal_high_link.csv",
        pipeline_mode="ideal_pipeline",
        num_requests=8,
        link_ms_one_way=1.0,
    )
    reset_sequence_ids()
    single_low_link = make_afd_pipeline(
        tmp_path / "single_low_link.csv",
        pipeline_mode="ideal_pipeline",
        num_requests=1,
        link_ms_one_way=0.5,
    )
    reset_sequence_ids()
    single_high_link = make_afd_pipeline(
        tmp_path / "single_high_link.csv",
        pipeline_mode="ideal_pipeline",
        num_requests=1,
        link_ms_one_way=1.0,
    )

    steady_state_delta = first_decode_duration(ideal_high_link) - first_decode_duration(ideal_low_link)
    single_delta = first_decode_duration(single_high_link) - first_decode_duration(single_low_link)

    assert steady_state_delta == pytest.approx(1.0 / 32.0)
    assert single_delta == pytest.approx(1.0)
