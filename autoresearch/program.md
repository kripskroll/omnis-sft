# Autoresearch Agent Instructions

## Your goal
Maximize the composite evaluation score on the Omnis SFT benchmark.
Current baseline: 0.54. Target: >0.80.

## What you CAN change
- training/config.json hyperparameters:
  - lora_rank (4-64)
  - lora_alpha (rank to 4*rank typically)
  - learning_rate (1e-6 to 1e-3)
  - num_epochs (1-10)
  - per_device_train_batch_size (1-4, max for A10G)
  - gradient_accumulation_steps (1-32)
  - warmup_steps (0-50)
  - weight_decay (0-0.1)
  - lr_scheduler_type (cosine, linear, constant)

## What you CANNOT change
- The evaluation benchmark (eval/benchmark.json)
- The evaluation script (eval/eval_runner.py)
- The base model (Qwen 2.5 7B Instruct)
- The training data content
- The training script logic (training/finetune.py)

## Strategy hints
1. The model is not generating <think> tags — this is the biggest score gap.
   Consider: higher learning rate, more epochs, or adjusting LoRA rank.
2. Tool selection is at 0.48 — the model sometimes picks a reasonable
   but different tool. More training epochs or rank might help.
3. Chain continuation is already at 0.80 — don't sacrifice this.
4. Start with learning rate and epochs since those have the biggest impact.
5. Try one change at a time to understand what helps.

## Workflow
1. Read current best config
2. Propose ONE modification with rationale
3. Run training: python training/finetune.py --config <modified_config>
4. Run eval: python eval/eval_runner.py --model training/output
5. Compare score to current best
6. If better: save as new best. If worse: discard.
7. Log the experiment result.
8. Repeat.
