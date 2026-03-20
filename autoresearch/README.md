# Autoresearch — Autonomous Hyperparameter Optimization

Inspired by Karpathy's autoresearch concept: an autonomous loop that modifies
training hyperparameters, retrains, evaluates, and keeps only improvements.

## Quick start

Launch an overnight run on the GPU instance:

```bash
# From the project root on the g5.2xlarge instance
uv run python autoresearch/runner.py --max-experiments 60 --hours 8
```

For a shorter test run:

```bash
uv run python autoresearch/runner.py --max-experiments 5 --hours 1 --seed 42
```

Use `nohup` or `tmux` for unattended runs so the process survives SSH disconnects:

```bash
nohup uv run python autoresearch/runner.py --hours 8 > autoresearch/run.log 2>&1 &
```

## What to expect

- Each experiment takes roughly 8-15 minutes (training ~5-10 min, eval ~3-5 min).
- An 8-hour run fits approximately 30-60 experiments.
- The runner applies one random hyperparameter mutation per experiment and only
  keeps changes that improve the composite eval score (ratcheting).
- Press Ctrl-C to stop gracefully after the current experiment finishes.

## Reading the experiment log

Each experiment is appended as a JSON line to `autoresearch/experiments.jsonl`:

```json
{
  "experiment_id": 1,
  "timestamp": "2026-03-20T02:15:00+00:00",
  "mutation": "learning_rate: 2.0e-04 -> 4.0e-04 (x2.0)",
  "config": {"lora_rank": 16, "learning_rate": 0.0004, "...": "..."},
  "score": 0.58,
  "best_score": 0.58,
  "improved": true,
  "status": "complete",
  "duration_sec": 720
}
```

Filter for improvements only:

```bash
grep '"improved": true' autoresearch/experiments.jsonl | python -m json.tool
```

## Finding the best config

The current best configuration and its score are saved to
`autoresearch/best_config.json` and updated automatically whenever
an experiment improves on the previous best:

```bash
cat autoresearch/best_config.json | python -m json.tool
```

To use the best config for a manual training run:

```bash
cp autoresearch/best_config.json training/config.json  # extract the "config" key first
uv run python training/finetune.py
```

## Files

| File | Purpose |
|------|---------|
| `constants.py` | Immutable constraints (paths, limits) — not modified by the agent |
| `program.md` | Strategy instructions for the optimization agent |
| `runner.py` | Main experiment loop script |
| `experiments.jsonl` | Append-only log of every experiment (created at runtime) |
| `best_config.json` | Current best config + score (created at runtime) |
| `tmp/` | Temporary per-experiment config files (created at runtime) |
