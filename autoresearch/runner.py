#!/usr/bin/env python3
"""Autoresearch runner — autonomous hyperparameter optimization.

Runs a loop: modify config -> train -> evaluate -> keep/discard.
Implements a ratcheting mechanism: only keeps configs that improve the score.

Usage:
    uv run python autoresearch/runner.py [--max-experiments 60] [--hours 8]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constants import (
    BASE_CONFIG,
    BEST_CONFIG,
    EVAL_SCRIPT,
    EXPERIMENT_LOG,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    TRAINING_SCRIPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_sigint(_signum, _frame):
    global _shutdown_requested
    logger.warning("SIGINT received — finishing current experiment, then stopping.")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

# Each mutation is a callable: (config) -> (config, description)
# They modify the config in place and return a human-readable description.


def mutate_learning_rate(config: dict) -> str:
    old = config["learning_rate"]
    factor = random.choice([0.5, 0.75, 1.5, 2.0])
    new = old * factor
    new = max(MIN_LR, min(MAX_LR, new))
    config["learning_rate"] = new
    return f"learning_rate: {old:.1e} -> {new:.1e} (x{factor})"


def mutate_lora_rank(config: dict) -> str:
    old = config["lora_rank"]
    candidates = [8, 16, 32, 64]
    candidates = [r for r in candidates if r != old]
    new = random.choice(candidates)
    config["lora_rank"] = new
    # Also adjust alpha to maintain 2:1 ratio
    config["lora_alpha"] = new * 2
    return f"lora_rank: {old} -> {new} (alpha: {new * 2})"


def mutate_num_epochs(config: dict) -> str:
    old = config["num_epochs"]
    candidates = [2, 3, 4, 5, 6]
    candidates = [e for e in candidates if e != old and e <= MAX_EPOCHS]
    new = random.choice(candidates)
    config["num_epochs"] = new
    return f"num_epochs: {old} -> {new}"


def mutate_warmup_steps(config: dict) -> str:
    old = config["warmup_steps"]
    candidates = [0, 5, 10, 20, 50]
    candidates = [w for w in candidates if w != old]
    new = random.choice(candidates)
    config["warmup_steps"] = new
    return f"warmup_steps: {old} -> {new}"


def mutate_weight_decay(config: dict) -> str:
    old = config["weight_decay"]
    candidates = [0, 0.001, 0.01, 0.05, 0.1]
    candidates = [w for w in candidates if not math.isclose(w, old, abs_tol=1e-6)]
    new = random.choice(candidates)
    config["weight_decay"] = new
    return f"weight_decay: {old} -> {new}"


def mutate_lr_scheduler(config: dict) -> str:
    old = config["lr_scheduler_type"]
    candidates = ["cosine", "linear", "constant"]
    candidates = [s for s in candidates if s != old]
    new = random.choice(candidates)
    config["lr_scheduler_type"] = new
    return f"lr_scheduler_type: {old} -> {new}"


def mutate_batch_size(config: dict) -> str:
    """Mutate batch size + gradient accumulation to keep effective batch = 16."""
    old_bs = config["per_device_train_batch_size"]
    old_ga = config["gradient_accumulation_steps"]
    combos = [(1, 16), (2, 8), (4, 4)]
    combos = [(b, g) for b, g in combos if b != old_bs and b <= MAX_BATCH_SIZE]
    if not combos:
        combos = [(2, 8)]
    new_bs, new_ga = random.choice(combos)
    config["per_device_train_batch_size"] = new_bs
    config["gradient_accumulation_steps"] = new_ga
    return f"batch_size: {old_bs}x{old_ga} -> {new_bs}x{new_ga} (effective=16)"


MUTATIONS = [
    mutate_learning_rate,
    mutate_lora_rank,
    mutate_num_epochs,
    mutate_warmup_steps,
    mutate_weight_decay,
    mutate_lr_scheduler,
    mutate_batch_size,
]


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------


def load_best_config() -> tuple[dict, float]:
    """Load the best config and its score. Falls back to base config with score 0.0."""
    if Path(BEST_CONFIG).exists():
        with open(BEST_CONFIG) as f:
            data = json.load(f)
        return data["config"], data["score"]

    # Bootstrap from base config
    if Path(BASE_CONFIG).exists():
        with open(BASE_CONFIG) as f:
            config = json.load(f)
    else:
        logger.error("No base config found at %s", BASE_CONFIG)
        sys.exit(1)

    return config, 0.0


def save_best_config(config: dict, score: float) -> None:
    """Persist the best config and its score."""
    os.makedirs(os.path.dirname(BEST_CONFIG), exist_ok=True)
    with open(BEST_CONFIG, "w") as f:
        json.dump({"config": config, "score": score}, f, indent=2)
    logger.info("Saved best config (score=%.4f) to %s", score, BEST_CONFIG)


def write_temp_config(config: dict, experiment_id: int) -> str:
    """Write a temporary config file for this experiment. Returns the path."""
    os.makedirs("autoresearch/tmp", exist_ok=True)
    path = f"autoresearch/tmp/config_exp{experiment_id:04d}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def run_training(config_path: str) -> tuple[bool, float]:
    """Run finetune.py and return (success, duration_sec).

    The finetune script may fail during test inference even though
    training + adapter saving succeeded. We check for the adapter file
    to determine success rather than relying on the exit code.
    """
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, TRAINING_SCRIPT, "--config", config_path],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
        )
        duration = time.time() - start

        # Log output for debugging
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                logger.info("[train stdout] %s", line)
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.warning("[train stderr] %s", line)

        # Check if adapter was actually saved (training may succeed but
        # test inference at the end may crash — that is OK)
        adapter_path = Path("training/output/adapter_model.safetensors")
        if adapter_path.exists():
            logger.info("Adapter saved successfully (%.0fs)", duration)
            return True, duration

        logger.error("Adapter file not found after training")
        return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        logger.error("Training timed out after %.0fs", duration)
        return False, duration
    except Exception as exc:
        duration = time.time() - start
        logger.error("Training failed: %s", exc)
        return False, duration


def run_eval() -> tuple[float | None, str | None]:
    """Run eval_runner.py and return (composite_score, results_path).

    Returns (None, None) on failure.
    """
    try:
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT, "--model", "training/output"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            logger.error("Eval failed (exit %d)", result.returncode)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-5:]:
                    logger.error("[eval stderr] %s", line)
            return None, None

        # Parse composite score from the latest results file
        results_dir = Path("eval/results")
        if not results_dir.exists():
            logger.error("eval/results directory not found")
            return None, None

        # Find the most recent result file
        result_files = sorted(results_dir.glob("eval_*.json"), reverse=True)
        if not result_files:
            logger.error("No result files found in eval/results/")
            return None, None

        latest = result_files[0]
        with open(latest) as f:
            eval_data = json.load(f)

        score = eval_data.get("composite_score")
        if score is None:
            logger.error("No composite_score in %s", latest)
            return None, str(latest)

        logger.info("Eval composite score: %.4f (from %s)", score, latest.name)
        return score, str(latest)

    except subprocess.TimeoutExpired:
        logger.error("Eval timed out")
        return None, None
    except Exception as exc:
        logger.error("Eval failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def log_experiment(entry: dict[str, Any]) -> None:
    """Append an experiment entry to the JSONL log."""
    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_experiment_loop(max_experiments: int, max_hours: float) -> None:
    """Run the autonomous optimization loop."""
    deadline = time.time() + max_hours * 3600
    best_config, best_score = load_best_config()

    # If no prior score, use the known baseline
    if best_score == 0.0:
        best_score = 0.54
        logger.info("Using baseline score: %.2f", best_score)
    else:
        logger.info("Resuming from best score: %.4f", best_score)

    logger.info("Starting autoresearch loop: max %d experiments, %.1f hours",
                max_experiments, max_hours)

    for exp_id in range(1, max_experiments + 1):
        if _shutdown_requested:
            logger.info("Shutdown requested — stopping after %d experiments.", exp_id - 1)
            break

        if time.time() > deadline:
            logger.info("Time limit reached — stopping after %d experiments.", exp_id - 1)
            break

        logger.info("=" * 60)
        logger.info("EXPERIMENT %d / %d  (best so far: %.4f)", exp_id, max_experiments, best_score)
        logger.info("=" * 60)

        # 1. Copy best config and apply one mutation
        candidate = copy.deepcopy(best_config)
        mutation_fn = random.choice(MUTATIONS)
        mutation_desc = mutation_fn(candidate)
        logger.info("Mutation: %s", mutation_desc)

        # 2. Write temp config
        config_path = write_temp_config(candidate, exp_id)

        # 3. Train
        exp_start = time.time()
        train_ok, _ = run_training(config_path)

        if not train_ok:
            logger.warning("Training failed — skipping evaluation.")
            log_experiment({
                "experiment_id": exp_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mutation": mutation_desc,
                "config": candidate,
                "score": None,
                "best_score": best_score,
                "improved": False,
                "status": "train_failed",
                "duration_sec": round(time.time() - exp_start, 1),
            })
            continue

        # 4. Evaluate
        score, results_path = run_eval()

        if score is None:
            logger.warning("Evaluation failed — skipping this experiment.")
            log_experiment({
                "experiment_id": exp_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mutation": mutation_desc,
                "config": candidate,
                "score": None,
                "best_score": best_score,
                "improved": False,
                "status": "eval_failed",
                "duration_sec": round(time.time() - exp_start, 1),
            })
            continue

        # 5. Compare
        improved = score > best_score
        if improved:
            logger.info("IMPROVEMENT: %.4f -> %.4f (+%.4f)", best_score, score, score - best_score)
            best_score = score
            best_config = candidate
            save_best_config(best_config, best_score)
        else:
            logger.info("No improvement: %.4f <= %.4f (best)", score, best_score)

        # 6. Log
        duration = round(time.time() - exp_start, 1)
        log_experiment({
            "experiment_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mutation": mutation_desc,
            "config": candidate,
            "score": score,
            "best_score": best_score,
            "improved": improved,
            "status": "complete",
            "results_path": results_path,
            "duration_sec": duration,
        })

        logger.info("Experiment %d done in %.0fs. Best score: %.4f", exp_id, duration, best_score)

    # Final summary
    logger.info("=" * 60)
    logger.info("AUTORESEARCH COMPLETE")
    logger.info("  Best score: %.4f", best_score)
    logger.info("  Best config saved to: %s", BEST_CONFIG)
    logger.info("  Experiment log: %s", EXPERIMENT_LOG)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous hyperparameter optimization for Omnis SFT."
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=60,
        help="Maximum number of experiments to run (default: 60)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=8.0,
        help="Maximum wall-clock hours to run (default: 8.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = random)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Random seed: %d", args.seed)

    run_experiment_loop(args.max_experiments, args.hours)


if __name__ == "__main__":
    main()
