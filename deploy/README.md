# Phase 5: Deployment

This directory contains everything needed to deploy the fine-tuned Omnis diagnostics
model via Ollama and connect it to live MCP tools.

## Overview

```
training/output/          # LoRA adapter (from Phase 3)
        |
  export_gguf.py          # Merge + quantize
        |
  omnis-sft.gguf          # Portable model file (~4.5 GB)
        |
  Modelfile               # Ollama configuration
        |
  ollama create            # Register in Ollama
        |
  orchestrator.py          # Agent loop: model <-> MCP tools
```

## Prerequisites

- **Ollama** installed: https://ollama.com/download
- **GPU** (for export step only -- inference runs on CPU too)
- **Omnis MCP server** running (for live tool calls)
- Python dependencies: `uv pip install requests`

## Step 1: Export Model to GGUF

Merge the LoRA adapters into the base model and quantize to GGUF format:

```bash
uv run python deploy/export_gguf.py \
    --adapter training/output \
    --output deploy/omnis-sft.gguf \
    --quant q4_k_m
```

**Quantization options:**

| Method    | Size (7B) | Quality | Use case                    |
|-----------|-----------|---------|------------------------------|
| `q3_k_m`  | ~3.5 GB  | Good    | Tight memory constraints     |
| `q4_k_m`  | ~4.5 GB  | Great   | Recommended default          |
| `q5_k_m`  | ~5.3 GB  | Better  | When you have extra RAM      |
| `q8_0`    | ~7.5 GB  | Best    | Near-lossless, more memory   |
| `f16`     | ~14 GB   | Perfect | Full precision, GPU only     |

The export requires a GPU (for loading the 4-bit base model and merging). The
resulting GGUF file runs on CPU or GPU.

## Step 2: Create Ollama Model

```bash
cd deploy
ollama create omnis-sft -f Modelfile
```

Verify it loaded:

```bash
ollama list              # Should show omnis-sft
ollama run omnis-sft     # Quick interactive test
```

## Step 3: Run the Orchestrator

The orchestrator connects the model to live Omnis MCP tools in an agent loop.

### Interactive mode (REPL)

```bash
uv run python deploy/orchestrator.py --model omnis-sft
```

This opens a prompt where you can type questions. The orchestrator:
1. Sends your question to the model
2. Parses any `<tool_call>` tags in the response
3. Calls the corresponding MCP tool
4. Feeds the result back to the model
5. Repeats until the model gives a final answer

### Single-question mode

```bash
uv run python deploy/orchestrator.py \
    --model omnis-sft \
    --question "Why is latency high for the DNS service in the last hour?"
```

### Configuration

| Environment variable | Default                  | Description              |
|---------------------|--------------------------|--------------------------|
| `OLLAMA_URL`        | `http://localhost:11434` | Ollama API endpoint      |
| `OMNIS_MCP_URL`     | `http://localhost:8080`  | Omnis MCP server URL     |

## Deployment Options

### Option A: Local laptop

Best for development and testing.

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run the orchestrator
export OMNIS_MCP_URL=http://10.149.9.4:8080   # Point to lab MCP
uv run python deploy/orchestrator.py
```

Requires: 8+ GB RAM for Q4_K_M quantization.

### Option B: EC2 alongside the lab

Deploy on the same subnet as the Omnis infrastructure for lowest latency.

```bash
# On the EC2 instance
ollama serve &
export OMNIS_MCP_URL=http://10.149.9.4:8080
uv run python deploy/orchestrator.py
```

### Option C: Docker

```bash
# Pull Ollama Docker image
docker run -d --name ollama -p 11434:11434 ollama/ollama

# Copy GGUF and Modelfile into container
docker cp deploy/omnis-sft.gguf ollama:/tmp/
docker cp deploy/Modelfile ollama:/tmp/
docker exec ollama ollama create omnis-sft -f /tmp/Modelfile

# Run orchestrator on the host
export OLLAMA_URL=http://localhost:11434
export OMNIS_MCP_URL=http://10.149.9.4:8080
uv run python deploy/orchestrator.py
```

## Troubleshooting

**"Cannot connect to Ollama"** -- Make sure `ollama serve` is running and the
`OLLAMA_URL` environment variable points to the right address.

**"Cannot connect to MCP server"** -- Check that the Omnis MCP server is running
and reachable from where you are running the orchestrator. Verify with:
```bash
curl -s $OMNIS_MCP_URL/health
```

**Model outputs garbled text** -- The GGUF may be corrupted. Re-export with
`export_gguf.py`. If using a very aggressive quantization (q3_k_m or lower),
try q4_k_m instead.

**Slow inference** -- On CPU, expect 5-15 tokens/second for a 7B Q4_K_M model.
Use a GPU for faster inference, or consider a smaller quantization.
