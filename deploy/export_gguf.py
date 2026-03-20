#!/usr/bin/env python3
"""Export fine-tuned model to GGUF format for Ollama deployment.

Merges LoRA adapters with the base model and exports a quantized GGUF file.

Usage:
    uv run python deploy/export_gguf.py [--adapter training/output] [--output deploy/omnis-sft.gguf] [--quant Q4_K_M]

Requires GPU for merging. Output GGUF runs on CPU or GPU via Ollama.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_ADAPTER = "training/output"
DEFAULT_OUTPUT = "deploy/omnis-sft.gguf"
DEFAULT_QUANT = "q4_k_m"

# Supported quantization methods (llama.cpp / Unsloth naming)
SUPPORTED_QUANTS = [
    "q4_k_m",   # Good balance of quality and size (~4.5 GB for 7B)
    "q5_k_m",   # Higher quality, slightly larger (~5.3 GB)
    "q8_0",     # Near-lossless, larger (~7.5 GB)
    "f16",      # Full half-precision (largest, ~14 GB)
    "q4_k_s",   # Smaller variant of Q4
    "q3_k_m",   # More aggressive quantization
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and export to GGUF for Ollama.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model identifier for Unsloth (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the GGUF file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=DEFAULT_QUANT,
        choices=SUPPORTED_QUANTS,
        help=f"Quantization method (default: {DEFAULT_QUANT})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length used during training (default: 2048)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    adapter_path = Path(args.adapter)
    output_path = Path(args.output)

    # -- Validate adapter directory exists -----------------------------------
    if not adapter_path.exists():
        print(f"ERROR: Adapter directory not found: {adapter_path}", file=sys.stderr)
        print(
            "  Train a model first with: uv run python training/train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Ensure output directory exists --------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Import Unsloth (requires GPU) ---------------------------------------
    print(f"Loading base model: {args.base_model}")
    print(f"Loading LoRA adapter: {adapter_path}")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "ERROR: Unsloth is not installed. Install with:\n"
            "  uv pip install 'omnis-sft[training]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Step 1: Load base model + LoRA adapter ------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    print("Base model and adapter loaded successfully.")

    # -- Step 2: Merge adapters into base model ------------------------------
    print("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()
    print("Merge complete.")

    # -- Step 3: Export to GGUF ----------------------------------------------
    #
    # Unsloth's save_pretrained_gguf handles:
    #   - Converting merged weights to GGUF format
    #   - Applying the requested quantization
    #   - Writing the output file
    #
    # Under the hood it uses llama.cpp's convert scripts.
    print(f"Exporting to GGUF with quantization: {args.quant}")
    print(f"Output: {output_path}")

    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=args.quant,
    )

    # -- Verify output -------------------------------------------------------
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nExport successful: {output_path} ({size_mb:.1f} MB)")
        print(
            "\nNext steps:\n"
            "  1. Create the Ollama model:\n"
            f"     cd deploy && ollama create omnis-sft -f Modelfile\n"
            "  2. Test the model:\n"
            "     ollama run omnis-sft\n"
            "  3. Run with the orchestrator:\n"
            "     uv run python deploy/orchestrator.py --model omnis-sft\n"
        )
    else:
        print(
            f"WARNING: Expected output file not found at {output_path}.",
            file=sys.stderr,
        )
        print(
            "  Unsloth may have written it under a different name. "
            "Check the deploy/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
