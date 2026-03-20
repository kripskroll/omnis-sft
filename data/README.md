# Data — Training Data Generation

## What's here

| File | Purpose |
|------|---------|
| `tool_catalog.json` | All 31 Omnis MCP tool definitions with parameters and response schemas |
| `environment_context.json` | Live environment snapshot — topology, apps, known issues, key IPs |
| `seed_chains.json` | 8 hand-crafted gold-standard chain examples (quality bar) |
| `generate_chains.py` | Synthetic data generator using Claude Sonnet API |
| `validate_data.py` | Quality checks on generated data |
| `generated/` | Output directory for generated training data |

## How to reproduce

### Prerequisites
- Python 3.11+ with uv
- Anthropic API key (`export ANTHROPIC_API_KEY=your_key`)

### Steps

```bash
# From repo root
uv venv && source .venv/bin/activate
uv pip install -e .

# Generate ~350 training examples (~$10-15)
uv run python data/generate_chains.py

# Validate quality
uv run python data/validate_data.py
```

### What the generator does

1. Loads tool catalog, environment context, and seed chains
2. Generates examples across 5 personas and 10 workflow types
3. Distribution: 40% single-turn, 40% 2-3 step chains, 15% deep investigations, 5% wrong path + recovery
4. Validates each example against tool catalog (correct names, valid params)
5. Outputs `generated/training_data.jsonl` in ChatML format

### Updating environment context

To refresh with current live data, pull from the Omnis MCP tools:
- `get_health_overview` — application metrics
- `get_topology_summary` — hosts, connections, sites
- `get_anomalies` — current issues
- `get_top_talkers` — bandwidth patterns
- `list_sites`, `list_communities`, `list_sensors` — infrastructure
