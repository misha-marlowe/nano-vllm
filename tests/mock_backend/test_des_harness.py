import csv
from itertools import count

import pytest

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.mock import DESConfig, DESEngine, DESRequest


def reset_sequence_ids():
    Sequence.counter = count()


def read_trace(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finish_time(rows, request_id=0):
    return float(next(
        row["virtual_time_ms"]
        for row in rows
        if row["request_id"] == str(request_id) and row["stage"] == "request_finish"
    ))


def run_simple_mock(trace_path, *, mode="colocated", num_requests=1, **kwargs):
    reset_sequence_ids()
    config = dict(
        mock_backend=True,
        mock_mode=mode,
        virtual_time=True,
        trace_output=str(trace_path),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        decode_base_ms=0.5,
        decode_ms_per_token=0.02,
        attention_ms_base=0.4,
        attention_ms_per_token=0.02,
        attention_ms_per_isl_token=0.0001,
        cs_rest_ms_base=0.6,
        cs_rest_ms_per_token=0.03,
        link_ms_one_way=0.1,
        pipeline_mode="sequential",
        max_num_seqs=num_requests,
        max_num_batched_tokens=4096,
    )
    config.update(kwargs)
    llm = LLM("__mock__", **config)
    for idx in range(num_requests):
        llm.add_request(list(range(idx * 100, idx * 100 + 16)), SamplingParams(max_tokens=2, ignore_eos=True))
    while not llm.is_finished():
        llm.step()
    return read_trace(trace_path)


def run_nanovllm_des_mock(trace_path, *, mode="colocated", num_requests=1, **kwargs):
    return run_simple_mock(
        trace_path,
        mode=mode,
        num_requests=num_requests,
        mock_runner="des",
        **kwargs,
    )


def run_des(trace_path, *, mode="colocated", num_requests=1, **kwargs):
    config_kwargs = dict(
        mode=mode,
        trace_output=str(trace_path),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        decode_base_ms=0.5,
        decode_ms_per_token=0.02,
        attention_ms_base=0.4,
        attention_ms_per_token=0.02,
        attention_ms_per_isl_token=0.0001,
        cs_rest_ms_base=0.6,
        cs_rest_ms_per_token=0.03,
        link_ms_one_way=0.1,
    )
    config_kwargs.update(kwargs)
    config = DESConfig(**config_kwargs)
    engine = DESEngine(config)
    for idx in range(num_requests):
        engine.submit(DESRequest(idx, 0.0, 16, 2))
    engine.run()
    return read_trace(trace_path)


def test_des_matches_simple_mock_for_single_colocated_request(tmp_path):
    simple = run_simple_mock(tmp_path / "simple.csv", mode="colocated")
    des = run_des(tmp_path / "des.csv", mode="colocated")

    assert finish_time(des) == pytest.approx(finish_time(simple))


def test_des_batched_decode_matches_simple_mock_for_colocated_decode_batch(tmp_path):
    simple = run_simple_mock(
        tmp_path / "simple_batch.csv",
        mode="colocated",
        num_requests=4,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
    )
    des = run_des(
        tmp_path / "des_batch.csv",
        mode="colocated",
        num_requests=4,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        des_batch_decode=True,
        des_max_batch_size=4,
    )

    simple_last = max(finish_time(simple, request_id) for request_id in range(4))
    des_last = max(finish_time(des, request_id) for request_id in range(4))

    assert des_last == pytest.approx(simple_last)
    assert any(row["stage"] == "decode_start" and row["batch_size"] == "4" for row in des)
    assert any(row["notes"] == "des_batch_decode" for row in des)


def test_des_matches_simple_mock_for_single_afd_request(tmp_path):
    simple = run_simple_mock(tmp_path / "simple_afd.csv", mode="afd")
    des = run_des(tmp_path / "des_afd.csv", mode="afd")

    assert finish_time(des) == pytest.approx(finish_time(simple))


def test_des_explains_gap_for_multi_request_afd(tmp_path):
    simple = run_simple_mock(tmp_path / "simple_many.csv", mode="afd", num_requests=4)
    des = run_des(tmp_path / "des_many.csv", mode="afd", num_requests=4)

    simple_last = max(finish_time(simple, request_id) for request_id in range(4))
    des_last = max(finish_time(des, request_id) for request_id in range(4))

    assert des_last > simple_last
    assert any(row["resource"] == "cs_rest_0" for row in des)
    assert any(float(row["queue_delay_ms"]) > 0 for row in des if row["stage"] == "cs_rest_start")


def test_des_attention_replicas_reduce_attention_bottleneck(tmp_path):
    one = run_des(
        tmp_path / "one.csv",
        mode="afd",
        num_requests=4,
        attention_ms_per_token=4.0,
        cs_rest_ms_per_token=1.0,
        link_ms_one_way=0.0,
    )
    two = run_des(
        tmp_path / "two.csv",
        mode="afd",
        num_requests=4,
        attention_ms_per_token=4.0,
        cs_rest_ms_per_token=1.0,
        link_ms_one_way=0.0,
        attention_replicas=2,
    )

    assert max(finish_time(two, request_id) for request_id in range(4)) < max(
        finish_time(one, request_id) for request_id in range(4)
    )
    assert any(row["resource"] == "decode_attention_1" for row in two)


def test_des_batched_afd_emits_microbatch_resource_events(tmp_path):
    rows = run_des(
        tmp_path / "afd_batch.csv",
        mode="afd",
        num_requests=4,
        des_batch_decode=True,
        des_max_batch_size=4,
        chunk_batch=2,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
    )

    assert any(row["stage"] == "decode_attention_start" and row["batch_size"] == "4" for row in rows)
    assert any("microbatch=0" in row["notes"] for row in rows)
    assert any("microbatch=1" in row["notes"] for row in rows)


def test_nanovllm_des_runner_uses_engine_scheduler_for_colocated_batch(tmp_path):
    rows = run_nanovllm_des_mock(
        tmp_path / "nanovllm_des_colocated.csv",
        mode="colocated",
        num_requests=4,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
    )

    resource_starts = [
        row for row in rows
        if row["event_scope"] == "resource" and row["stage"] == "decode_start"
    ]
    assert len(resource_starts) == 2
    assert all(row["batch_size"] == "4" for row in resource_starts)
    assert all("des_batch_decode" in row["notes"] for row in resource_starts)


def test_nanovllm_des_runner_emits_afd_resource_events(tmp_path):
    rows = run_nanovllm_des_mock(
        tmp_path / "nanovllm_des_afd.csv",
        mode="afd",
        num_requests=4,
        prefill_base_ms=0.0,
        prefill_ms_per_token=0.0,
        microbatch_size=2,
    )

    assert any(
        row["event_scope"] == "resource"
        and row["stage"] == "decode_attention_start"
        and row["batch_size"] == "4"
        for row in rows
    )
    assert any("microbatch=0" in row["notes"] for row in rows)
    assert any("microbatch=1" in row["notes"] for row in rows)
