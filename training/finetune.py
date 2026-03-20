#!/usr/bin/env python3
"""Fine-tune Qwen 2.5 7B Instruct with QLoRA using Unsloth.

Loads training data in ChatML format and fine-tunes with LoRA adapters.
Hyperparameters are loaded from training/config.json.

Usage:
    uv run python training/finetune.py [--config training/config.json] [--data data/generated/training_data.jsonl]

Requires GPU with 24GB+ VRAM (e.g., NVIDIA A10G on AWS g5.2xlarge).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration — overridden by config.json
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 4096,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "output_dir": "training/output",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load training configuration from a JSON file.

    Falls back to DEFAULT_CONFIG for any keys not present in the file.
    """
    config = dict(DEFAULT_CONFIG)
    path = Path(config_path)
    if path.exists():
        logger.info("Loading config from %s", path)
        with open(path) as f:
            overrides = json.load(f)
        config.update(overrides)
    else:
        logger.warning(
            "Config file %s not found — using default configuration", path
        )
    return config


# ---------------------------------------------------------------------------
# Data loading and ChatML formatting
# ---------------------------------------------------------------------------


def load_training_data(data_path: str | Path) -> list[dict]:
    """Load JSONL training data.

    Each line is a JSON object with a ``messages`` key containing a list of
    chat messages, each with ``role`` and ``content`` fields.
    """
    path = Path(data_path)
    if not path.exists():
        logger.error("Training data file not found: %s", path)
        sys.exit(1)

    examples: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                continue
            if "messages" not in example:
                logger.warning("Skipping line %d — no 'messages' key", line_num)
                continue
            examples.append(example)

    logger.info("Loaded %d training examples from %s", len(examples), path)
    return examples


def format_to_chatml(example: dict) -> str:
    """Convert a training example to Qwen 2.5 ChatML format.

    Qwen 2.5 uses the ChatML template natively::

        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>

    Special handling:
    - ``<think>`` tags in assistant messages are preserved exactly.
    - ``<tool_call>`` tags in assistant messages are preserved exactly.
    - ``tool`` role messages (tool results) are mapped to a user message
      wrapped in ``<tool_response>...</tool_response>`` tags, since Qwen 2.5
      ChatML does not define a native ``tool`` role.
    """
    parts: list[str] = []

    for msg in example["messages"]:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "tool":
            # Map tool results into a user turn with explicit wrapper tags.
            # If the content is already a string, use it directly; otherwise
            # serialise it so the model sees valid JSON inside the wrapper.
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            )
        elif role in ("system", "user", "assistant"):
            # For assistant messages the content may contain <think> and
            # <tool_call> blocks — they are kept verbatim.
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        else:
            logger.warning("Unknown role '%s' — treating as user", role)
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")

    return "".join(parts)


def prepare_dataset(examples: list[dict]) -> Dataset:
    """Format all examples to ChatML and return a HuggingFace Dataset."""
    formatted = [{"text": format_to_chatml(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)
    logger.info("Dataset prepared — %d examples", len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Model loading and LoRA setup
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(config: dict[str, Any]):
    """Load the base model with 4-bit quantization via Unsloth."""
    logger.info("Loading model: %s", config["base_model"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
    )
    return model, tokenizer


def apply_lora(model, config: dict[str, Any]):
    """Apply QLoRA adapters to the model."""
    logger.info(
        "Applying LoRA — rank=%d, alpha=%d, dropout=%s, targets=%s",
        config["lora_rank"],
        config["lora_alpha"],
        config["lora_dropout"],
        config["target_modules"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    config: dict[str, Any],
) -> SFTTrainer:
    """Construct the SFTTrainer with TrainingArguments from config."""
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=config["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )
    return trainer


def train(trainer: SFTTrainer) -> None:
    """Run training and log summary statistics."""
    logger.info("Starting training …")
    result = trainer.train()

    metrics = result.metrics
    logger.info("Training complete.")
    logger.info("  Total steps:  %s", metrics.get("total_flos", "N/A"))
    logger.info("  Train loss:   %s", metrics.get("train_loss", "N/A"))
    logger.info("  Train runtime: %.1fs", metrics.get("train_runtime", 0))
    logger.info(
        "  Samples/sec:  %.2f", metrics.get("train_samples_per_second", 0)
    )


# ---------------------------------------------------------------------------
# Post-training verification
# ---------------------------------------------------------------------------


def report_adapter_files(output_dir: str) -> None:
    """Print the saved adapter file names and sizes."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist", output_dir)
        return

    logger.info("Saved adapter files in %s:", output_dir)
    total_size = 0
    for p in sorted(output_path.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total_size += size
            logger.info("  %-40s %s", p.relative_to(output_path), _fmt_size(size))
    logger.info("  Total adapter size: %s", _fmt_size(total_size))


def _fmt_size(size: int) -> str:
    """Return a human-readable file size."""
    n = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def run_test_inference(model, tokenizer) -> None:
    """Run a quick inference to verify the adapter produces expected tags.

    Sends a short prompt and checks that the model can generate ``<think>``
    and ``<tool_call>`` tags, confirming the fine-tuned behaviour is intact.
    """
    logger.info("Running test inference …")

    # Switch model to inference mode
    FastLanguageModel.for_inference(model)

    test_prompt = (
        "<|im_start|>system\n"
        "You are a network analytics assistant. Think step by step inside "
        "<think> tags, then respond or call tools using <tool_call> tags.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the top talker on site Prague in the last hour?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    logger.info("Test generation:\n%s", generated)

    has_think = "<think>" in generated
    has_tool_call = "<tool_call>" in generated
    logger.info(
        "Verification — <think>: %s, <tool_call>: %s",
        "FOUND" if has_think else "MISSING",
        "FOUND" if has_tool_call else "MISSING",
    )
    if not has_think and not has_tool_call:
        logger.warning(
            "Neither <think> nor <tool_call> appeared in test output. "
            "The model may need more training data or epochs."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen 2.5 7B Instruct with QLoRA (Unsloth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.json",
        help="Path to the JSON configuration file (default: training/config.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/generated/training_data.jsonl",
        help="Path to the JSONL training data (default: data/generated/training_data.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # 1. Load configuration
    config = load_config(args.config)
    logger.info("Configuration: %s", json.dumps(config, indent=2))

    # 2. Load and prepare data
    examples = load_training_data(args.data)
    dataset = prepare_dataset(examples)

    # 3. Load model + tokenizer with 4-bit quantization
    model, tokenizer = load_model_and_tokenizer(config)

    # 4. Apply LoRA adapters
    model = apply_lora(model, config)

    # 5. Build trainer
    trainer = build_trainer(model, tokenizer, dataset, config)

    # 6. Train
    train(trainer)

    # 7. Save adapter and tokenizer
    logger.info("Saving LoRA adapter to %s", config["output_dir"])
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # 8. Report saved files
    report_adapter_files(config["output_dir"])

    # 9. Quick verification inference
    run_test_inference(model, tokenizer)

    logger.info("Done.")


if __name__ == "__main__":
    main()
