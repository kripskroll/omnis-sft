# Omnis SFT — Claude Code Instructions

## Project overview

Fine-tuning Qwen 2.5 7B for autonomous network diagnostics using 31 Omnis MCP tools.
The model learns chained tool-calling with explicit `<think>` reasoning between steps.

## Python environment

Always use `uv` for dependency management. Never use `pip install` directly.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv run python <script>
```

## Key files

- `data/tool_catalog.json` — All 31 MCP tool definitions (source of truth for tool names/params)
- `data/environment_context.json` — Live environment snapshot (grounds synthetic data in reality)
- `data/seed_chains.json` — Hand-crafted gold-standard examples (quality bar)
- `data/generate_chains.py` — Synthetic data generator (uses Anthropic API)
- `data/validate_data.py` — Data quality checks

## Training data format

Training examples use ChatML format with `<think>` tags for reasoning:

```
<think>
DNS shows 20% error rate. I should investigate DNS anomalies.
</think>
<tool_call>
{"name": "get_anomalies", "arguments": {"application": "DNS"}}
</tool_call>
```

## Reusability constraint

All pipeline code must be dataset-agnostic. Only these three files change between domains:
- `data/tool_catalog.json`
- `data/environment_context.json`
- `data/seed_chains.json`

## Conventions

- Commit messages: `<type>: <description>` (e.g., `data: add tool catalog`)
- Every directory has a README explaining what it does and how to reproduce
- No secrets in repo — API keys via environment variables
