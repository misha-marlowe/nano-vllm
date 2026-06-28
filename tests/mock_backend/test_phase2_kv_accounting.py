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


def rows_for(rows, request_id, stage):
    return [
        row for row in rows
        if row["request_id"] == str(request_id) and row["stage"] == stage
    ]


def test_kv_tokens_grow_after_prefill_and_each_decode(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "kv_growth.csv"
    llm = make_llm(trace_path, mock_block_size=4, mock_kv_capacity_tokens=64)

    llm.add_request(list(range(8)), SamplingParams(max_tokens=3, ignore_eos=True))
    drain(llm)
    rows = read_trace(trace_path)

    prefill_end = rows_for(rows, 0, "prefill_end")[0]
    emits = rows_for(rows, 0, "token_emit")

    assert int(prefill_end["kv_tokens_used"]) == 8
    assert [int(row["kv_tokens_used"]) for row in emits] == [9, 10, 11]


def test_finished_request_releases_kv_blocks(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "kv_release.csv"
    llm = make_llm(trace_path, mock_block_size=4, mock_kv_capacity_tokens=16)

    llm.add_request(list(range(4)), SamplingParams(max_tokens=1, ignore_eos=True))
    drain(llm)
    rows = read_trace(trace_path)
    finish = rows_for(rows, 0, "request_finish")[0]

    assert int(finish["kv_tokens_used"]) == 0
    assert int(finish["kv_blocks_used"]) == 0
    assert int(finish["kv_blocks_free"]) == 4


def test_limited_kv_capacity_causes_admission_wait(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "kv_wait.csv"
    llm = make_llm(
        trace_path,
        mock_block_size=4,
        mock_kv_capacity_tokens=12,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    )

    llm.add_request(list(range(8)), SamplingParams(max_tokens=1, ignore_eos=True))
    llm.add_request(list(range(100, 108)), SamplingParams(max_tokens=1, ignore_eos=True))
    outputs = drain(llm)
    rows = read_trace(trace_path)

    waits = rows_for(rows, 1, "admission_wait")
    second_prefill = rows_for(rows, 1, "prefill_start")[0]
    first_finish = rows_for(rows, 0, "request_finish")[0]

    assert [len(outputs[seq_id]) for seq_id in sorted(outputs)] == [1, 1]
    assert waits
    assert waits[0]["notes"] == "insufficient_kv_blocks"
    assert float(second_prefill["virtual_time_ms"]) >= float(first_finish["virtual_time_ms"])


def test_prefix_cache_is_preserved_for_mock_backend(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "prefix_cache.csv"
    llm = make_llm(
        trace_path,
        mock_block_size=4,
        mock_kv_capacity_tokens=32,
        max_num_seqs=1,
    )
    prompt = list(range(8))

    llm.add_request(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
    drain(llm)
    llm.add_request(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
    drain(llm)
    rows = read_trace(trace_path)

    prefill_starts = [row for row in rows if row["stage"] == "prefill_start"]
    prefill_ends = [row for row in rows if row["stage"] == "prefill_end"]
    first_latency = float(prefill_ends[0]["virtual_time_ms"]) - float(prefill_starts[0]["virtual_time_ms"])
    second_latency = float(prefill_ends[1]["virtual_time_ms"]) - float(prefill_starts[1]["virtual_time_ms"])

    assert first_latency == pytest.approx(1.0 + 8 * 0.01)
    assert second_latency == pytest.approx(1.0 + 4 * 0.01)
