"""Immutable constants for autoresearch — never modified by the optimization agent."""

# Paths
BENCHMARK_PATH = "eval/benchmark.json"
EVAL_SCRIPT = "eval/eval_runner.py"
TRAINING_SCRIPT = "training/finetune.py"
TRAINING_DATA = "data/generated/training_data.jsonl"
BASE_CONFIG = "training/config.json"

# Model
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"

# Constraints
MAX_SEQ_LENGTH = 2048  # A10G 24GB limit
MAX_BATCH_SIZE = 4     # OOM above this on A10G
MIN_EXAMPLES = 100     # Don't train on fewer than this
MAX_EPOCHS = 10        # Prevent overfitting
MAX_LR = 1e-3          # Prevent divergence
MIN_LR = 1e-6          # Too low = no learning

# Experiment tracking
EXPERIMENT_LOG = "autoresearch/experiments.jsonl"
BEST_CONFIG = "autoresearch/best_config.json"
