# Training — QLoRA Fine-tuning with Unsloth

## What this does

Fine-tunes Qwen 2.5 7B Instruct with QLoRA (4-bit quantized LoRA) to learn chained tool-calling with explicit `<think>` reasoning for network diagnostics.

## Prerequisites

- **GPU**: NVIDIA A10G (24GB VRAM) — AWS g5.2xlarge recommended
- **OS**: Ubuntu 22.04 with CUDA 12.x
- **Python**: 3.11+
- **Training data**: `data/generated/training_data.jsonl` (generated in Phase 1)

## Quick start (on GPU instance)

```bash
# 1. Bootstrap the instance (if not already set up)
bash infra/setup_instance.sh

# 2. Activate environment
cd omnis-sft
source .venv/bin/activate

# 3. Run fine-tuning (~6-8 minutes per epoch, ~20 min total)
uv run python training/finetune.py

# 4. Check output
ls -la training/output/
```

## Configuration

All hyperparameters are in `training/config.json`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_model` | `unsloth/Qwen2.5-7B-Instruct` | 4-bit quantized by Unsloth |
| `max_seq_length` | 4096 | Covers longest chain examples |
| `lora_rank` | 16 | Starting point — autoresearch will optimize |
| `lora_alpha` | 32 | 2x rank is standard |
| `learning_rate` | 2e-4 | QLoRA standard |
| `num_epochs` | 3 | Usually converges by epoch 2-3 |
| `per_device_train_batch_size` | 4 | Fits in 24GB with gradient accumulation |
| `gradient_accumulation_steps` | 4 | Effective batch size = 16 |
| `lr_scheduler_type` | cosine | Standard for SFT |

## Training data format

The script converts JSONL examples to Qwen 2.5's native ChatML format:

```
<|im_start|>system
You are a network diagnostics assistant...<|im_end|>
<|im_start|>user
anything wrong on the network?<|im_end|>
<|im_start|>assistant
<think>Let me check overall health first.</think>

<tool_call>{"name": "get_health_overview", "arguments": {}}</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"overall": {"total_transactions": 1214467, ...}}
</tool_response><|im_end|>
<|im_start|>assistant
DNS has a critical 20% error rate...<|im_end|>
```

Tool results are wrapped in `<tool_response>` tags and sent as user messages (Qwen 2.5 doesn't have a native tool role).

## Output

After training, `training/output/` contains:
- LoRA adapter weights (`adapter_model.safetensors`)
- Adapter config (`adapter_config.json`)
- Tokenizer files

These can be:
1. Merged with the base model for GGUF export (Phase 5)
2. Loaded directly for evaluation (Phase 3)
3. Uploaded to S3: `bash infra/upload_to_s3.sh upload-model`

## Verification

After training completes, the script runs a quick inference test:
- Sends "How is the network doing?" to the fine-tuned model
- Checks that the response contains `<think>` and `<tool_call>` tags
- Prints the generated response for manual review

## Cost

- ~20 minutes training on g5.2xlarge spot (~$0.15)
- ~$2-3 total including instance startup, data transfer, and testing
