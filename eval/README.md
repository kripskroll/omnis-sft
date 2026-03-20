# Omnis SFT Evaluation Framework

Automated evaluation of fine-tuned models against a canonical benchmark of 50 network analytics questions.

## What It Tests

The benchmark covers 10 workflow types that a network analytics assistant must handle:

| Workflow | Questions | Description |
|----------|-----------|-------------|
| health_triage | 5 | General network health assessment |
| dns_investigation | 6 | DNS failure root cause analysis |
| site_performance | 5 | Per-site performance comparison |
| app_performance | 5 | Application latency and TCP health |
| host_investigation | 5 | Per-host activity and connectivity |
| cross_site_path | 4 | WAN path tracing and hop analysis |
| bandwidth_analysis | 5 | Traffic patterns and top talkers |
| mysql_troubleshooting | 5 | Slow query detection and session tracing |
| security_scan | 5 | Anomaly detection and security audit |
| wrong_path_recovery | 5 | Correcting user assumptions with data |

Questions use real environment data (IPs, sites, applications, known issues) from the live Omnis deployment.

## Scoring

Four components, weighted per Section 6.4 of the project spec:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Tool selection accuracy | 40% | Did the model call the correct first tool? |
| Argument accuracy | 25% | Did it pass the right parameters? |
| Chain continuation rate | 20% | Did it continue investigating after receiving tool results? |
| Reasoning relevance | 15% | Does the `<think>` block reference data (numbers, IPs, percentages)? |

The composite score is the weighted sum, ranging from 0.0 (complete failure) to 1.0 (perfect).

## Running the Evaluation

```bash
# Default: uses training/output model and eval/benchmark.json
uv run python eval/eval_runner.py

# Custom model and benchmark paths
uv run python eval/eval_runner.py --model path/to/adapter --benchmark eval/benchmark.json

# All options
uv run python eval/eval_runner.py \
    --model training/output \
    --benchmark eval/benchmark.json \
    --results-dir eval/results \
    --max-seq-length 4096
```

Requires a GPU with 24GB+ VRAM (same as training).

## Output

The runner prints a summary report to stdout and saves detailed per-question results to `eval/results/eval_YYYYMMDD_HHMMSS.json`.

### Interpreting Scores

| Composite Score | Interpretation |
|-----------------|----------------|
| 0.90+ | Production-ready: strong tool routing and multi-step reasoning |
| 0.75-0.89 | Good: correct tool selection, minor argument or chaining gaps |
| 0.60-0.74 | Acceptable: needs more training data for weak workflows |
| Below 0.60 | Needs work: review training data coverage and quality |

Check the **by_workflow** breakdown to identify which workflows need more training examples. Check **by_difficulty** to see if the model handles complex multi-step investigations.

## Files

- `benchmark.json` — 50 canonical test questions with expected tools, args, and metadata
- `eval_runner.py` — Automated evaluation script (loads model, runs benchmark, scores)
- `results/` — Timestamped JSON result files for tracking progress across training runs
