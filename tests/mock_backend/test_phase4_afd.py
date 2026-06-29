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


def make_afd_llm(trace_path, **kwargs):
    config = dict(
        mock_backend=True,
        mock_mode="afd",
        virtual_time=True,
        trace_output=str(trace_path),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        attention_ms_base=0.4,
        attention_ms_per_token=0.02,
        attention_ms_per_isl_token=0.0001,
        cs_rest_ms_base=0.6,
        cs_rest_ms_per_token=0.03,
        link_ms_one_way=0.1,
        pipeline_mode="sequential",
    )
    config.update(kwargs)
    return LLM(
        "__mock__",
        **config,
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


def decode_step_ms(batch_size, context_len, *, link=0.1, cs_base=0.6, isl_factor=0.0001):
    attention = 0.4 + 0.02 * batch_size + isl_factor * context_len * batch_size
    cs_rest = cs_base + 0.03 * batch_size
    return attention + 2 * link + cs_rest


def test_afd_sequential_single_request_stage_order_and_timing(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "afd_seq.csv"
    llm = make_afd_llm(trace_path)

    llm.add_request(list(range(16)), SamplingParams(max_tokens=2, ignore_eos=True))
    outputs = drain(llm)
    rows = read_trace(trace_path)

    assert len(outputs[0]) == 2
    expected_stage_order = [
        "decode_attention_start",
        "decode_attention_end",
        "gpu_to_cs_link_start",
        "gpu_to_cs_link_end",
        "cs_rest_start",
        "cs_rest_end",
        "cs_to_gpu_link_start",
        "cs_to_gpu_link_end",
    ]
    first_decode_rows = [
        row for row in rows
        if row["stage"] in expected_stage_order
    ][:8]
    assert [row["stage"] for row in first_decode_rows] == expected_stage_order
    assert sum(row["stage"].endswith("link_start") for row in first_decode_rows) == 2

    prefill_ms = 1.0 + 16 * 0.01
    first_step = decode_step_ms(1, 16)
    second_step = decode_step_ms(1, 17)
    token_emits = rows_for(rows, 0, "token_emit")
    assert float(token_emits[0]["virtual_time_ms"]) == pytest.approx(prefill_ms + first_step)
    assert float(token_emits[1]["virtual_time_ms"]) == pytest.approx(prefill_ms + first_step + second_step)


def test_afd_disabled_equivalence_to_colocated_decode(tmp_path):
    reset_sequence_ids()
    colocated_trace = tmp_path / "colocated.csv"
    afd_trace = tmp_path / "afd_equiv.csv"

    colocated = LLM(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        virtual_time=True,
        trace_output=str(colocated_trace),
        prefill_base_ms=1.0,
        prefill_ms_per_token=0.01,
        decode_base_ms=0.5,
        decode_ms_per_token=0.02,
    )
    colocated.add_request(list(range(8)), SamplingParams(max_tokens=3, ignore_eos=True))
    drain(colocated)

    reset_sequence_ids()
    afd = make_afd_llm(
        afd_trace,
        attention_ms_base=0.2,
        attention_ms_per_token=0.0,
        attention_ms_per_isl_token=0.0,
        cs_rest_ms_base=0.32,
        cs_rest_ms_per_token=0.0,
        link_ms_one_way=0.0,
    )
    afd.add_request(list(range(8)), SamplingParams(max_tokens=3, ignore_eos=True))
    drain(afd)

    colocated_finish = rows_for(read_trace(colocated_trace), 0, "request_finish")[0]
    afd_finish = rows_for(read_trace(afd_trace), 0, "request_finish")[0]
    assert float(afd_finish["virtual_time_ms"]) == pytest.approx(float(colocated_finish["virtual_time_ms"]))


def test_afd_monotonic_sensitivities(tmp_path):
    def finish_time(name, **kwargs):
        reset_sequence_ids()
        trace_path = tmp_path / f"{name}.csv"
        isl = kwargs.pop("isl", 16)
        llm = make_afd_llm(trace_path, **kwargs)
        llm.add_request(list(range(isl)), SamplingParams(max_tokens=1, ignore_eos=True))
        drain(llm)
        return float(rows_for(read_trace(trace_path), 0, "request_finish")[0]["virtual_time_ms"])

    base = finish_time("base")
    higher_link = finish_time("higher_link", link_ms_one_way=0.5)
    longer_isl = finish_time("longer_isl", isl=64)
    faster_cs = finish_time("faster_cs", cs_rest_ms_base=0.1)

    assert higher_link > base
    assert longer_isl > base
    assert faster_cs < base


def test_afd_kv_capacity_still_uses_gpu_side_blocks(tmp_path):
    reset_sequence_ids()
    trace_path = tmp_path / "afd_kv.csv"
    llm = make_afd_llm(
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

    assert [len(outputs[seq_id]) for seq_id in sorted(outputs)] == [1, 1]
    assert rows_for(rows, 1, "admission_wait")
