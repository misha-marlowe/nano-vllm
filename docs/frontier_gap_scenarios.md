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

Commit:

```text
ee422aa Add prefill-interference global DES scenario
```

## Step 3: Sparse Arrivals

Gap represented:

- Frontier models workload arrivals and queue starvation. The saturated replay
  assumes every next decode batch is ready immediately.

Model:

- Add an inter-batch ready gap equal to 25% of baseline effective batch time.

Expected effect:

- Throughput cannot improve versus the saturated baseline.
- This curve shows how much steady-state assumptions depend on a full queue.

## Step 4: Operator Overheads

Gap represented:

- Frontier can model per-operator details instead of only coarse AFD stages.

Model:

- Add small fixed overheads to attention and CS-rest stages.

Expected effect:

- Throughput should not improve.
- Effect is larger when microbatches are small because fixed overheads are less
  amortized.

## Step 5: Roofline Backend Variant

Gap represented:

- Frontier can swap measured/kernel-backed predictors. Our baseline uses the
  measured GPT-OSS GPU fit.

Model:

- Re-run the global replay with `roofline_gpu_backend="roofline"`.

Expected effect:

- Direction is backend-dependent, so the test only validates that the scenario
  runs and is labeled. The plot shows whether analytical roofline is optimistic
  or pessimistic for each point.

## Step 6: Collective / Link Contention

Gap represented:

- Frontier models explicit collective and communication contention. The
  baseline uses simple AFD link stages.

Model:

- Scale both GPU-to-CS and CS-to-GPU link stages by 1.35.

Expected effect:

- Throughput should not improve.
- Link-heavy configurations should move more than attention-heavy ones.

## Step 7: Parallelism / Replica Imbalance

Gap represented:

- Frontier tracks richer parallelism semantics and replica scheduling. The
  baseline assumes clean attention-replica routing.

Model:

- Scale attention stage by 1.10 to approximate load imbalance or imperfect
  routing.

Expected effect:

- Throughput should not improve.
- Attention-bound configurations move the most.

## Step 8: KV / Cache Transfer

Gap represented:

- Frontier has richer KV/cache state and transfer modeling. The baseline keeps
  KV effects inside the attention timing and does not add cache movement.

Model:

- Add a small context-proportional overhead to both AFD link directions.

Expected effect:

- Throughput should not improve.
- 1M context should show a larger penalty than 8K context.

## Step 9: Runtime Optimizations

Gap represented:

- Frontier models runtime optimizations such as CUDA graphs and fusion. The
  baseline does not distinguish these runtime modes.

Model:

- Scale attention by 0.97 and CS rest by 0.95.

Expected effect:

- Throughput and interactivity should improve or stay equal.

Validation for Steps 3-9:

```text
pytest tests/mock_backend/test_frontier_gap_scenarios.py -q
12 passed
```

## Scenario Plots And Timing

Generated by:

```bash
python tools/validate_frontier_gap_scenarios.py \
  --isl 8192 \
  --link-us 12 \
  --output-dir results/roofline_validation/frontier_gap_scenarios

python tools/validate_frontier_gap_scenarios.py \
  --isl 1000000 \
  --link-us 12 \
  --output-dir results/roofline_validation/frontier_gap_scenarios
```

Artifacts:

- `results/roofline_validation/frontier_gap_scenarios/isl8192_link12_frontier_gap_scenarios.svg`
- `results/roofline_validation/frontier_gap_scenarios/isl8192_link12_frontier_gap_scenarios.csv`
- `results/roofline_validation/frontier_gap_scenarios/isl8192_link12_frontier_gap_timing.csv`
- `results/roofline_validation/frontier_gap_scenarios/isl1000000_link12_frontier_gap_scenarios.svg`
- `results/roofline_validation/frontier_gap_scenarios/isl1000000_link12_frontier_gap_scenarios.csv`
- `results/roofline_validation/frontier_gap_scenarios/isl1000000_link12_frontier_gap_timing.csv`

Mean throughput error versus analytical, link=12us:

| scenario | 8K mean y err | 8K max abs y err | 1M mean y err | 1M max abs y err |
|---|---:|---:|---:|---:|
| afd_global_des | -0.55% | 9.56% | -0.47% | 2.27% |
| context_growth | -0.56% | 9.48% | -0.47% | 2.27% |
| prefill_interference | -1.17% | 8.88% | -1.09% | 2.87% |
| sparse_arrivals | -19.94% | 22.64% | -19.65% | 22.16% |
| operator_overheads | -23.56% | 25.99% | -6.06% | 19.84% |
| roofline_backend | 0.24% | 26.06% | 10.25% | 54.14% |
| collective_contention | -0.71% | 8.94% | -0.55% | 2.65% |
| replica_imbalance | -0.98% | 11.15% | -9.48% | 10.97% |
| kv_transfer | -0.61% | 9.41% | -40.01% | 87.75% |
| runtime_optimizations | 4.56% | 12.94% | 2.61% | 3.05% |

Generation timing:

| scenario | 8K seconds | 8K sec/point | 1M seconds | 1M sec/point |
|---|---:|---:|---:|---:|
| afd_global_des | 0.072 | 0.00157 | 0.054 | 0.00110 |
| context_growth | 0.603 | 0.01310 | 0.617 | 0.01259 |
| prefill_interference | 0.145 | 0.00315 | 0.106 | 0.00217 |
| sparse_arrivals | 0.144 | 0.00314 | 0.106 | 0.00217 |
| operator_overheads | 0.072 | 0.00157 | 0.053 | 0.00109 |
| roofline_backend | 0.072 | 0.00156 | 0.053 | 0.00109 |
| collective_contention | 0.072 | 0.00157 | 0.054 | 0.00109 |
| replica_imbalance | 0.072 | 0.00157 | 0.053 | 0.00109 |
| kv_transfer | 0.072 | 0.00157 | 0.054 | 0.00109 |
| runtime_optimizations | 0.072 | 0.00156 | 0.054 | 0.00109 |

Interpretation:

- `sparse_arrivals` intentionally breaks the saturated queue assumption, so it
  drops throughput by about 20%.
- `kv_transfer` is mild at 8K but large at 1M because the added transfer term is
  context-proportional.
- `operator_overheads` hurts small-microbatch 8K points more strongly because
  fixed per-stage overheads are less amortized.
- `runtime_optimizations` is the only intentionally optimistic curve.
- `roofline_backend` can move either direction because it swaps the measured GPU
  fit for the analytical roofline backend.
