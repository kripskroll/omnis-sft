# Implementation Plan: Omnis AI Network Assistant (omnis-sft)

## Context

NETSCOUT enterprise customers (finance, defense, government) cannot send network telemetry to cloud LLMs. This project fine-tunes Qwen 2.5 7B to autonomously diagnose network issues using 31 Omnis MCP tools — enabling on-premises AI-driven network diagnostics at ~$30 total development cost.

The model must learn **chained reasoning**: call tool A, reason about results in `<think>` tags, extract values, call tool B, and continue until reaching a diagnosis. This is the capability gap between frontier and small models.

**Live environment validated (2026-03-20)**: 171 hosts, 640 connections, 8 sites, 11 apps — all spec data confirmed against live Omnis MCP.

---

## Reusability

This plan is designed as a **reusable template**. The same approach will be applied to a Mobile Service Provider (SP) dataset using Omnis SP MCP tools. The repo structure, generation pipeline, training scripts, and evaluation framework are all dataset-agnostic — only `tool_catalog.json`, `environment_context.json`, and `seed_chains.json` change between domains.

---

## Phase 1: Project Scaffold & Data Generation

### 1.1 — Repository initialization

```
omnis-sft/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── docs/
│   ├── implementation_plan.md
│   └── omnis_spec.md
├── data/
│   ├── README.md
│   ├── tool_catalog.json
│   ├── environment_context.json
│   ├── seed_chains.json
│   ├── generate_chains.py
│   ├── validate_data.py
│   └── generated/
│       └── .gitkeep
├── training/
├── eval/
├── autoresearch/
├── deploy/
└── infra/
```

### 1.2 — Tool catalog (`data/tool_catalog.json`)

31 Omnis MCP tools with names, descriptions, parameters (required/optional with types), example invocations, and response key fields. Source: spec + live MCP schemas.

### 1.3 — Environment context (`data/environment_context.json`)

Live environment snapshot: topology, applications, known issues, key IPs, traffic pairs, anomalies. This grounds synthetic data in reality.

### 1.4 — Seed chains (`data/seed_chains.json`)

8 hand-crafted gold-standard examples:

| # | Workflow | Steps | Teaching point |
|---|---------|-------|----------------|
| 1 | Health triage | 3 | Start broad → identify worst app → drill down |
| 2 | DNS investigation | 4 | Anomaly detection → query-level analysis → host profiling |
| 3 | Site performance | 3 | Site summary → app breakdown → connection health |
| 4 | MySQL troubleshooting | 3 | Healthy average hiding slow tail latency |
| 5 | Host investigation | 3 | Activity profile → flow details → bandwidth analysis |
| 6 | Cross-site path | 3 | Topology → path tracing → hub-and-spoke awareness |
| 7 | Security scan | 3 | Anomalies → DNS analysis → distinguish infra vs security |
| 8 | Wrong path + recovery | 3 | Try errors → none found → pivot to performance |

### 1.5 — Data generator (`data/generate_chains.py`)

~350 examples via Claude Sonnet API:
- 40% single-turn tool selection (140)
- 40% 2-3 step diagnostic chains (140)
- 15% 3-4 step deep investigations (52)
- 5% wrong path + recovery (18)

5 personas × 10 workflow types. Estimated cost: $10-15.

### 1.6 — Data validation (`data/validate_data.py`)

Checks: tool names, required params, think tags, IP validation, distribution, duplicates, message structure.

### 1.7 — Generate the dataset

Run generator, validate, commit output.

---

## Phase 2: Training Pipeline

### 2.1 — Training config (`training/config.json`)
LoRA rank 16, alpha 32, lr 2e-4, 3 epochs, batch size 4.

### 2.2 — Fine-tuning script (`training/finetune.py`)
Unsloth QLoRA, ChatML format, Qwen 2.5 7B.

### 2.3 — AWS infrastructure (`infra/`)
EC2 g5.2xlarge bootstrap, S3 helpers.

### 2.4 — Documentation
Step-by-step reproduction guide.

---

## Phase 3: Evaluation

### 3.1 — Benchmark (`eval/benchmark.json`)
50 canonical questions across all 10 workflows.

### 3.2 — Eval runner (`eval/eval_runner.py`)
Automated scoring: tool selection, argument accuracy, chain continuation, reasoning relevance.

---

## Phase 4: Autoresearch Loop

### 4.1 — Agent instructions (`autoresearch/program.md`)
### 4.2 — Runner (`autoresearch/runner.py`)
### 4.3 — Constants (`autoresearch/constants.py`)

---

## Phase 5: Deployment

### 5.1 — GGUF export (`deploy/export_gguf.py`)
### 5.2 — Ollama config (`deploy/Modelfile`)
### 5.3 — Orchestration loop
### 5.4 — Documentation

---

## Verification Plan

### Phase 1
- `uv run python data/validate_data.py` passes all checks
- Generated data has correct distribution (40/40/15/5 split)
- All tool names and parameters match catalog

### Phase 2
- Training completes without errors on g5.2xlarge
- Loss curve shows convergence

### Phase 3
- Eval runner produces score for all 50 benchmark questions
- Baseline score recorded

### Phase 4
- 60+ experiments overnight
- Best score improves over baseline

### Phase 5
- GGUF loads in Ollama
- Model responds to "How is the network doing?" with correct tool call
- End-to-end orchestration loop completes 3-step investigation
