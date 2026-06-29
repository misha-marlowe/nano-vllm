# Roofline Validation Results

This directory contains validation artifacts for the GPT-OSS roofline timing
backend and the AFD serving simulators.

## Names

- **Analytical**: closed-form roofline / Frontier-style Pareto model.
- **nano-vLLM mock**: the in-engine fake backend that keeps nano-vLLM request,
  scheduler, and block-manager flow where practical.
- **nano-vLLM-DES**: the in-engine DES runner that keeps nano-vLLM scheduling
  and block/KV paths, then times each scheduled decode batch with DES.
- **DES**: the standalone discrete-event simulator with scalar request/KV state
  and explicit resources.
- **global DES replay**: saturated AFD replay over multiple ready batches with
  persistent attention/link/CS resources. It is plotted as **AFD global DES**.

For large-context AFD plots, nano-vLLM mock and nano-vLLM-DES use compact
synthetic prompts and coarse mock blocks so the engine/scheduler path can be
replayed without materializing giant prompt-token arrays.

## Key Plots

8K ISL, link=12us overlay:

- `afd_pareto_sim/isl8192_link12_analytical_vs_des_overlay.svg`

1M ISL, link=12us overlay:

- `afd_pareto_1m/isl1000000_link12_analytical_vs_des_overlay.svg`

All-link AFD replay comparison at 8K ISL:

- `afd_pareto_sim/afd_analytical_vs_des_replay.svg`

## Data Files

- `afd_pareto_sim/afd_analytical_des_comparison.csv`
  - 8K ISL AFD analytical, nano-vLLM mock, nano-vLLM-DES, global DES, and DES replay rows.
- `afd_pareto_sim/pareto_generation_timing.csv`
  - Wall-clock timing for generating each 8K Pareto path.
- `afd_pareto_sim/isl8192_colocated_analytical_des.csv`
  - 8K ISL colocated analytical, nano-vLLM-DES, and DES rows.
- `afd_pareto_1m/afd_analytical_des_comparison.csv`
  - 1M ISL AFD analytical, nano-vLLM mock, nano-vLLM-DES, global DES, and DES replay rows.
- `afd_pareto_1m/pareto_generation_timing.csv`
  - Wall-clock timing for generating each 1M Pareto path.
- `afd_pareto_1m/isl1000000_colocated_analytical_des.csv`
  - 1M ISL colocated analytical, nano-vLLM-DES, and DES rows.

Markdown tables were intentionally removed; CSVs are the canonical tabular
artifacts.

## Summary

AFD replay at 8K ISL:

| link_us | points | max nano-vLLM-DES err | max global DES err | max DES err | mean nano-vLLM-DES err | mean global DES err | mean DES err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 47 | 32.98% | 2.98% | 32.98% | -8.00% | -0.54% | -8.00% |
| 6 | 46 | 32.08% | 2.29% | 32.08% | -8.28% | -0.51% | -8.28% |
| 12 | 46 | 30.50% | 9.56% | 30.50% | -10.59% | -0.55% | -10.59% |
| 24 | 45 | 39.25% | 3.88% | 39.25% | -14.33% | -1.19% | -14.33% |
| 36 | 40 | 46.60% | 33.70% | 46.60% | -37.29% | -30.39% | -37.29% |

AFD replay at 1M ISL, link=12us:

| link_us | points | max nano-vLLM-DES err | max global DES err | max DES err | mean nano-vLLM-DES err | mean global DES err | mean DES err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 49 | 27.05% | 2.27% | 27.05% | -6.55% | -0.47% | -6.55% |

The batch-scoped DES and nano-vLLM-DES replays can sit below the analytical
curve because they pay finite microbatch/resource fill/drain per scheduled
batch. The global DES replay is closer to analytical because it amortizes those
bubbles across 16 ready batches. It can be slightly optimistic because every
replayed batch is ready at time zero.

Generation timing on this run:

| ISL | points | analytical | nano-vLLM mock | nano-vLLM-DES | standalone DES | global DES |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 224 | 2.114s | 7.959s | 8.453s | 0.830s | 0.322s |
| 1M | 49 | 0.423s | 1.565s | 1.780s | 0.198s | 0.054s |

## Reproduce

8K ISL:

```bash
python tools/validate_afd_pareto.py \
  --output-dir results/roofline_validation/afd_pareto_sim
```

1M ISL:

```bash
python tools/validate_afd_pareto.py \
  --isl 1000000 \
  --link-us 12 \
  --output-dir results/roofline_validation/afd_pareto_1m
```

Use `--global-des-batches N` to change the saturation depth for the global DES
line. The checked-in plots use the default `N=16`.
