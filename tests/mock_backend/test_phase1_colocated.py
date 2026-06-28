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


def make_llm(trace_path, **kwargs):
    return LLM(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        virtual_time=True,
        trace_output=str(trace_path),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        decode_base_ms=0.5,
        decode_ms_per_token=0.02,
        **kwargs,
    )


def drain(llm):
    outputs = {}
    while not llm.is_finished():
        step_outputs, _ = llm.step()
        outputs.update(step_outputs)
    return outputs


def test_single_request_closed_form_trace(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "single.csv"
    llm = make_llm(trace_path)

    llm.add_request(list(range(128)), SamplingParams(max_tokens=8, ignore_eos=True))
    outputs = drain(llm)
    rows = read_trace(trace_path)

    assert len(outputs[0]) == 8
    assert sum(row["stage"] == "prefill_start" for row in rows) == 1
    assert sum(row["stage"] == "prefill_end" for row in rows) == 1
    assert sum(row["stage"] == "decode_start" for row in rows) == 8
    assert sum(row["stage"] == "decode_end" for row in rows) == 8
    assert sum(row["stage"] == "token_emit" for row in rows) == 8

    prefill_end = next(row for row in rows if row["stage"] == "prefill_end")
    decode_ends = [row for row in rows if row["stage"] == "decode_end"]
    expected_prefill_ms = 1.0 + 128 * 0.01 * 1
    expected_decode_ms = 0.5 + 0.02 * 1
    assert float(prefill_end["virtual_time_ms"]) == pytest.approx(expected_prefill_ms)
    assert float(decode_ends[-1]["virtual_time_ms"]) == pytest.approx(
        expected_prefill_ms + 8 * expected_decode_ms
    )


def test_same_time_requests_batch_together(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "batch.csv"
    llm = make_llm(trace_path, max_num_seqs=8, max_num_batched_tokens=4096)

    for _ in range(3):
        llm.add_request(list(range(64)), SamplingParams(max_tokens=4, ignore_eos=True))
    outputs = drain(llm)
    rows = read_trace(trace_path)
    prefill_starts = [row for row in rows if row["stage"] == "prefill_start"]
    decode_starts = [row for row in rows if row["stage"] == "decode_start"]

    assert [len(outputs[seq_id]) for seq_id in sorted(outputs)] == [4, 4, 4]
    assert len(prefill_starts) == 3
    assert {row["batch_size"] for row in prefill_starts} == {"3"}
    assert any(row["batch_size"] == "3" for row in decode_starts)


def test_staggered_arrivals_are_visible_in_trace(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "staggered.csv"
    llm = make_llm(trace_path, max_num_seqs=4, max_num_batched_tokens=4096)
    pending = [(0.0, 128, 4), (2.5, 32, 2)]

    while pending or not llm.is_finished():
        while pending and pending[0][0] <= llm.clock.time_ms:
            _, isl, osl = pending.pop(0)
            llm.add_request(list(range(isl)), SamplingParams(max_tokens=osl, ignore_eos=True))
        if llm.is_finished():
            if pending:
                llm.clock.time_ms = pending[0][0]
            continue
        llm.step()

    rows = read_trace(trace_path)
    arrivals = [row for row in rows if row["stage"] == "request_arrival"]
    second_prefill = [
        row for row in rows
        if row["request_id"] == "1" and row["stage"] == "prefill_start"
    ][0]

    assert float(arrivals[0]["virtual_time_ms"]) == 0.0
    assert float(arrivals[1]["virtual_time_ms"]) > 0.0
    assert float(second_prefill["virtual_time_ms"]) >= float(arrivals[1]["virtual_time_ms"])
