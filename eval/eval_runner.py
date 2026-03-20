#!/usr/bin/env python3
"""Evaluate a fine-tuned Omnis SFT model against the benchmark.

Loads the model, runs each benchmark question, and computes a composite
accuracy score based on tool selection, argument accuracy, chain continuation,
and reasoning relevance.

Usage:
    uv run python eval/eval_runner.py [--model training/output] [--benchmark eval/benchmark.json]

Requires GPU. Outputs a single composite score (0.0-1.0).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a network analytics assistant with access to Omnis MCP tools. "
    "Think step by step inside <think> tags, then call tools using <tool_call> "
    "tags or respond directly. Always investigate thoroughly before concluding."
)

# Scoring weights (Section 6.4)
WEIGHT_TOOL_SELECTION = 0.40
WEIGHT_ARGUMENT_ACCURACY = 0.25
WEIGHT_CHAIN_CONTINUATION = 0.20
WEIGHT_REASONING_RELEVANCE = 0.15

# Synthetic tool result injected for chain continuation testing
SYNTHETIC_TOOL_RESULT = json.dumps(
    {
        "status": "success",
        "overall": {
            "total_transactions": 1214467,
            "total_errors": 97199,
            "error_rate_pct": 8.0,
            "avg_response_time_ms": 33.6,
        },
        "by_application": [
            {
                "application_name": "DNS",
                "transactions": 479492,
                "errors": 97191,
                "error_rate_pct": 20.27,
                "avg_response_time_ms": 2.8,
            },
            {
                "application_name": "MYSQL",
                "transactions": 446206,
                "errors": 0,
                "error_rate_pct": 0.0,
                "avg_response_time_ms": 8.1,
            },
            {
                "application_name": "SSH",
                "transactions": 152642,
                "errors": 0,
                "error_rate_pct": 0.0,
                "avg_response_time_ms": 92.8,
            },
            {
                "application_name": "Web Store",
                "transactions": 86198,
                "errors": 8,
                "error_rate_pct": 0.009,
                "avg_response_time_ms": 106.8,
            },
        ],
    },
    indent=2,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: str, max_seq_length: int = 4096):
    """Load the fine-tuned model (LoRA adapter + base) with Unsloth."""
    logger.info("Loading model from %s", model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    logger.info("Model loaded and set to inference mode.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(question: str) -> str:
    """Format a benchmark question as a ChatML prompt."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_chain_prompt(question: str, first_response: str) -> str:
    """Format a continuation prompt with the first response and a synthetic tool result."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{first_response}<|im_end|>\n"
        f"<|im_start|>user\n<tool_response>\n{SYNTHETIC_TOOL_RESULT}\n</tool_response><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model given a prompt string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )
    return generated


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """Extract the first <tool_call> from generated text.

    Returns (tool_name, arguments) or (None, None) if not found.
    """
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None, None
    try:
        call = json.loads(match.group(1))
        return call.get("name"), call.get("arguments", {})
    except json.JSONDecodeError:
        return None, None


def has_think_with_data(text: str) -> bool:
    """Check if <think> block contains data-like content (numbers, IPs, percentages)."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not think_match:
        return False
    think_content = think_match.group(1)
    # Check for numbers, IPs, or percentages
    has_numbers = bool(re.search(r"\d+", think_content))
    has_ips = bool(re.search(r"\d+\.\d+\.\d+\.\d+", think_content))
    has_percentages = bool(re.search(r"\d+%", think_content))
    return has_numbers or has_ips or has_percentages


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_tool_selection(predicted_tool: str | None, expected_tool: str) -> float:
    """Score 1.0 if the predicted first tool matches the expected tool, else 0.0."""
    if predicted_tool is None:
        return 0.0
    return 1.0 if predicted_tool == expected_tool else 0.0


def score_argument_accuracy(
    predicted_args: dict | None, expected_args: dict
) -> float:
    """Score argument match. Partial match: check that expected keys are present
    with correct values. Empty expected args scores 1.0 if any args were produced
    (or if expected is empty, always 1.0)."""
    if not expected_args:
        # No specific args required — full marks
        return 1.0
    if predicted_args is None:
        return 0.0

    matches = 0
    total = len(expected_args)
    for key, expected_val in expected_args.items():
        pred_val = predicted_args.get(key)
        if pred_val is not None and str(pred_val) == str(expected_val):
            matches += 1
    return matches / total if total > 0 else 1.0


def score_chain_continuation(
    model,
    tokenizer,
    question: str,
    first_response: str,
    min_chain_length: int,
    _expected_second_tool: str | None,
) -> float:
    """Score chain continuation: does the model generate another tool call
    after receiving a synthetic tool result?

    For single-turn questions (min_chain_length == 1), score 1.0 always.
    For chain questions, score 1.0 if the model calls another tool.
    """
    if min_chain_length <= 1:
        return 1.0

    continuation_prompt = format_chain_prompt(question, first_response)
    continuation = generate(model, tokenizer, continuation_prompt)
    second_tool, _ = parse_tool_call(continuation)

    if second_tool is not None:
        # Bonus: if it matches the expected second tool, still 1.0
        # (we just care that it continues)
        return 1.0
    return 0.0


def score_reasoning_relevance(text: str) -> float:
    """Score reasoning relevance: does the <think> block contain data-like content?"""
    return 1.0 if has_think_with_data(text) else 0.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_question(
    model,
    tokenizer,
    q: dict,
) -> dict[str, Any]:
    """Evaluate a single benchmark question. Returns a detailed result dict."""
    question_text = q["question"]
    expected_tool = q["expected_first_tool"]
    expected_args = q.get("expected_first_args", {})
    expected_second = q.get("expected_second_tool")
    min_chain = q.get("min_chain_length", 1)

    # Generate first response
    prompt = format_prompt(question_text)
    response = generate(model, tokenizer, prompt)

    # Parse first tool call
    pred_tool, pred_args = parse_tool_call(response)

    # Score components
    tool_score = score_tool_selection(pred_tool, expected_tool)
    arg_score = score_argument_accuracy(pred_args, expected_args)
    reasoning_score = score_reasoning_relevance(response)

    # Chain continuation — only test if min_chain > 1
    chain_score = score_chain_continuation(
        model, tokenizer, question_text, response, min_chain, expected_second
    )

    # Composite score for this question
    composite = (
        WEIGHT_TOOL_SELECTION * tool_score
        + WEIGHT_ARGUMENT_ACCURACY * arg_score
        + WEIGHT_CHAIN_CONTINUATION * chain_score
        + WEIGHT_REASONING_RELEVANCE * reasoning_score
    )

    return {
        "id": q["id"],
        "question": question_text,
        "workflow": q["workflow"],
        "difficulty": q["difficulty"],
        "expected_tool": expected_tool,
        "expected_args": expected_args,
        "predicted_tool": pred_tool,
        "predicted_args": pred_args,
        "expected_second_tool": expected_second,
        "min_chain_length": min_chain,
        "scores": {
            "tool_selection": tool_score,
            "argument_accuracy": arg_score,
            "chain_continuation": chain_score,
            "reasoning_relevance": reasoning_score,
            "composite": composite,
        },
        "raw_response_preview": response[:500],
    }


def run_evaluation(
    model,
    tokenizer,
    benchmark: dict,
) -> dict[str, Any]:
    """Run the full benchmark evaluation and return aggregated results."""
    questions = benchmark["questions"]
    results: list[dict] = []

    for i, q in enumerate(questions):
        logger.info(
            "Evaluating %d/%d: [%s] %s",
            i + 1,
            len(questions),
            q["id"],
            q["question"][:60],
        )
        result = evaluate_question(model, tokenizer, q)
        results.append(result)

        # Log per-question score
        scores = result["scores"]
        logger.info(
            "  -> tool=%s arg=%.2f chain=%.2f reason=%.2f composite=%.3f",
            "PASS" if scores["tool_selection"] == 1.0 else "FAIL",
            scores["argument_accuracy"],
            scores["chain_continuation"],
            scores["reasoning_relevance"],
            scores["composite"],
        )

    return aggregate_results(results)


def aggregate_results(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate scores from individual question results."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}

    # Component averages
    tool_scores = [r["scores"]["tool_selection"] for r in results]
    arg_scores = [r["scores"]["argument_accuracy"] for r in results]
    chain_scores = [r["scores"]["chain_continuation"] for r in results]
    reasoning_scores = [r["scores"]["reasoning_relevance"] for r in results]

    avg_tool = sum(tool_scores) / n
    avg_arg = sum(arg_scores) / n
    avg_chain = sum(chain_scores) / n
    avg_reasoning = sum(reasoning_scores) / n

    composite = (
        WEIGHT_TOOL_SELECTION * avg_tool
        + WEIGHT_ARGUMENT_ACCURACY * avg_arg
        + WEIGHT_CHAIN_CONTINUATION * avg_chain
        + WEIGHT_REASONING_RELEVANCE * avg_reasoning
    )

    # By difficulty
    by_difficulty: dict[str, list[float]] = {}
    for r in results:
        d = r["difficulty"]
        by_difficulty.setdefault(d, []).append(r["scores"]["composite"])
    difficulty_scores = {
        d: sum(s) / len(s) for d, s in sorted(by_difficulty.items())
    }
    difficulty_counts = {d: len(s) for d, s in sorted(by_difficulty.items())}

    # By workflow
    by_workflow: dict[str, list[float]] = {}
    for r in results:
        w = r["workflow"]
        by_workflow.setdefault(w, []).append(r["scores"]["composite"])
    workflow_scores = {
        w: sum(s) / len(s) for w, s in sorted(by_workflow.items())
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": n,
        "composite_score": round(composite, 4),
        "component_scores": {
            "tool_selection": round(avg_tool, 4),
            "argument_accuracy": round(avg_arg, 4),
            "chain_continuation": round(avg_chain, 4),
            "reasoning_relevance": round(avg_reasoning, 4),
        },
        "weights": {
            "tool_selection": WEIGHT_TOOL_SELECTION,
            "argument_accuracy": WEIGHT_ARGUMENT_ACCURACY,
            "chain_continuation": WEIGHT_CHAIN_CONTINUATION,
            "reasoning_relevance": WEIGHT_REASONING_RELEVANCE,
        },
        "by_difficulty": {
            d: {"score": round(s, 4), "count": difficulty_counts[d]}
            for d, s in difficulty_scores.items()
        },
        "by_workflow": {
            w: round(s, 4) for w, s in workflow_scores.items()
        },
        "individual_results": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(agg: dict[str, Any]) -> None:
    """Print a human-readable evaluation report to stdout."""
    print("\n=== Omnis SFT Evaluation Report ===\n")
    print(f"Overall composite score: {agg['composite_score']:.2f}\n")

    cs = agg["component_scores"]
    print("Component scores:")
    print(f"  Tool selection accuracy:  {cs['tool_selection']:.2f} (40% weight)")
    print(f"  Argument accuracy:        {cs['argument_accuracy']:.2f} (25% weight)")
    print(f"  Chain continuation rate:  {cs['chain_continuation']:.2f} (20% weight)")
    print(f"  Reasoning relevance:      {cs['reasoning_relevance']:.2f} (15% weight)")

    print("\nBy difficulty:")
    for d, info in agg["by_difficulty"].items():
        print(f"  {d.capitalize():<8} {info['score']:.2f} ({info['count']} questions)")

    print("\nBy workflow:")
    for w, s in agg["by_workflow"].items():
        print(f"  {w:<24} {s:.2f}")

    print()


def save_results(agg: dict[str, Any], output_dir: str) -> str:
    """Save detailed results to a timestamped JSON file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    return filepath


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Omnis SFT model against the benchmark.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="training/output",
        help="Path to the fine-tuned model (LoRA adapter directory). Default: training/output",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="eval/benchmark.json",
        help="Path to the benchmark JSON file. Default: eval/benchmark.json",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory to save detailed result JSON. Default: eval/results",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length for the model. Default: 4096",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        logger.error("Benchmark file not found: %s", benchmark_path)
        sys.exit(1)

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    logger.info(
        "Loaded benchmark v%s: %d questions",
        benchmark.get("version", "?"),
        len(benchmark["questions"]),
    )

    # Load model
    model, tokenizer = load_model(args.model, args.max_seq_length)

    # Run evaluation
    agg = run_evaluation(model, tokenizer, benchmark)

    # Print report
    print_report(agg)

    # Save detailed results
    results_path = save_results(agg, args.results_dir)
    print(f"Individual results saved to: {results_path}")


if __name__ == "__main__":
    main()
