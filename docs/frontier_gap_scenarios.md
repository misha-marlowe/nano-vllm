# Frontier Gap Scenarios

This document tracks the branch that adds Frontier-inspired scenario overlays
on top of the AFD global DES replay baseline.

The baseline is `afd_global_des`: saturated repeated AFD decode batches flowing
through persistent attention, GPU-to-CS link, CS rest, and CS-to-GPU link
resources. Each scenario changes one modeling assumption and is plotted as an
additional curve for ISL=8K and ISL=1M.

These scenarios are sensitivity models. They do not mean nano-vLLM now has the
full Frontier runtime. They make the gaps explicit and measurable.

## Step 0: Baseline Scenario Harness

- Added `nanovllm/mock/frontier_gap_scenarios.py`.
- Added `ParetoPoint`, `ScenarioResult`, and `baseline_global_des`.
- Added tests that baseline output is positive and more replay batches amortize
  fill/drain while preserving first-microbatch interactivity.

Validation:

```text
pytest tests/mock_backend/test_frontier_gap_scenarios.py -q
2 passed
```

Commit:

```text
c90d986 Add Frontier gap scenario baseline
```

## Step 0.1: Scenario Timing

- Added `timed_scenario` so each scenario reports wall-clock runtime.
- Scenario CSVs will include `wall_time_s`.

Validation:

```text
pytest tests/mock_backend/test_frontier_gap_scenarios.py -q
3 passed
```

Commit:

```text
c35ec06 Track timing for Frontier gap scenarios
```

## Step 1: Context Growth

Gap represented:

- Frontier-style decode simulations usually advance request state over many
  generated tokens. The original saturated replay used a constant context.

Model:

- Repeated decode batches use `context_len = ISL + batch_id`.

Expected effect:

- Throughput should not improve versus constant context.
- First microbatch interactivity is unchanged because the first context is the
  same.
- Effect is small for ISL=8K and tiny for ISL=1M over short replay depth.

Validation:

```text
pytest tests/mock_backend/test_frontier_gap_scenarios.py -q
4 passed
```

Commit:

```text
46db50e Add context-growth global DES scenario
```

## Step 2: Prefill Interference

Gap represented:

- Frontier models prefill and decode sharing runtime resources. AFD global DES
  baseline assumes decode has the pipeline to itself.

Model:

- Reserve the attention-side GPU resource for 10% of the baseline effective
  batch time before decode replay starts. This approximates prefill work
  delaying decode attention.

Expected effect:

- Throughput drops versus baseline.
- Interactivity drops because the first decode microbatch waits behind prefill.

Validation:

```text
pytest tests/mock_backend/test_frontier_gap_scenarios.py -q
5 passed
```
