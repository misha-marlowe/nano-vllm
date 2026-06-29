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
  - 8K ISL AFD analytical, nano-vLLM mock, nano-vLLM-DES, and DES replay rows.
- `afd_pareto_sim/isl8192_colocated_analytical_des.csv`
  - 8K ISL colocated analytical, nano-vLLM-DES, and DES rows.
- `afd_pareto_1m/afd_analytical_des_comparison.csv`
  - 1M ISL AFD analytical, nano-vLLM mock, nano-vLLM-DES, and DES replay rows.
- `afd_pareto_1m/isl1000000_colocated_analytical_des.csv`
  - 1M ISL colocated analytical, nano-vLLM-DES, and DES rows.

Markdown tables were intentionally removed; CSVs are the canonical tabular
artifacts.

## Summary

AFD replay at 8K ISL:

| link_us | points | max nano-vLLM-DES throughput error | max DES throughput error | mean nano-vLLM-DES throughput error | mean DES throughput error |
|---:|---:|---:|---:|---:|---:|
| 4 | 47 | 32.98% | 32.98% | -8.00% | -8.00% |
| 6 | 46 | 32.08% | 32.08% | -8.28% | -8.28% |
| 12 | 46 | 30.50% | 30.50% | -10.59% | -10.59% |
| 24 | 45 | 39.25% | 39.25% | -14.33% | -14.33% |
| 36 | 40 | 46.60% | 46.60% | -37.29% | -37.29% |

AFD replay at 1M ISL, link=12us:

| link_us | points | max nano-vLLM-DES throughput error | max DES throughput error | mean nano-vLLM-DES throughput error | mean DES throughput error |
|---:|---:|---:|---:|---:|---:|
| 12 | 49 | 27.05% | 27.05% | -6.55% | -6.55% |

The DES and nano-vLLM-DES replays can sit below the analytical curve because
they use finite microbatch/resource scheduling, while the analytical Pareto
model amortizes pipeline fill/drain more aggressively.

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
