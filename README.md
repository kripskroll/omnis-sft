# Omnis SFT — Fine-tuned Network Diagnostics Model

Fine-tune Qwen 2.5 7B to autonomously diagnose network issues using NETSCOUT Omnis MCP tools. Enables on-premises AI-driven network diagnostics without cloud LLM dependencies.

## What this does

The model learns **chained reasoning**: call tool A, reason about results in `<think>` tags, extract values, call tool B, and continue until reaching a diagnosis. This is the capability gap between frontier and small models.

- **31 Omnis MCP tools** across 10 categories (health, performance, troubleshooting, topology, etc.)
- **350 synthetic training examples** generated via Claude Sonnet, grounded in real environment data
- **QLoRA fine-tuning** with Unsloth on AWS g5.2xlarge (~$30 total cost)
- **Autoresearch loop** for overnight hyperparameter optimization (60-80 experiments)

## Quickstart

```bash
# Clone and setup
git clone https://github.com/kripskroll/omnis-sft.git
cd omnis-sft
uv venv && source .venv/bin/activate
uv pip install -e .

# Generate training data (~$10-15 via Claude Sonnet API)
export ANTHROPIC_API_KEY=your_key_here
uv run python data/generate_chains.py

# Validate generated data
uv run python data/validate_data.py

# Fine-tune (requires GPU — see training/README.md)
uv run python training/finetune.py

# Evaluate
uv run python eval/eval_runner.py
```

## Project structure

```
omnis-sft/
├── data/                    # Training data generation
│   ├── tool_catalog.json    # 31 MCP tool definitions + signatures
│   ├── environment_context.json  # Live environment snapshot
│   ├── seed_chains.json     # Hand-crafted gold-standard examples
│   ├── generate_chains.py   # Synthetic data generator (Anthropic API)
│   ├── validate_data.py     # Data quality checks
│   └── generated/           # Output directory for training data
├── training/                # QLoRA fine-tuning with Unsloth
├── eval/                    # Benchmark and evaluation runner
├── autoresearch/            # Autonomous optimization loop
├── deploy/                  # GGUF export + Ollama deployment
└── infra/                   # AWS EC2 bootstrap scripts
```

## Reusability

This pipeline is dataset-agnostic. To adapt for a different domain (e.g., Mobile SP):
1. Replace `data/tool_catalog.json` with new tool definitions
2. Replace `data/environment_context.json` with new environment data
3. Replace `data/seed_chains.json` with domain-specific examples
4. Everything else (generator, training, eval, autoresearch) works unchanged

## Documentation

- [Implementation Plan](docs/implementation_plan.md) — Full reproduction guide
- [Project Specification](docs/omnis_spec.md) — Original project spec
- [Data Guide](data/README.md) — How to reproduce training data

## Cost summary

| Phase | Cost |
|-------|------|
| Training data generation (Claude Sonnet) | $10-15 |
| AWS fine-tuning + testing | $2-3 |
| Overnight autoresearch | $3-5 |
| **Total** | **~$30** |
