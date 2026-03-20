#!/usr/bin/env python3
"""Merge partitioned generation outputs into a single training_data.jsonl.

Handles:
- Re-numbering IDs sequentially (gen_001, gen_002, ...)
- Cross-partition deduplication (by first tool call)
- Final distribution report

Usage:
    uv run python data/merge_partitions.py
"""

import json
import hashlib
import argparse
from pathlib import Path


def compute_dedup_key(example: dict) -> str | None:
    """Compute a deduplication key based on first tool call + user question.

    Includes user question so different questions starting with the same
    no-arg tool (e.g., get_health_overview) are not flagged as duplicates.
    """
    user_q = ""
    first_tool_call = None

    for msg in example.get("messages", []):
        if msg.get("role") == "user" and not user_q:
            user_q = msg.get("content", "").strip().lower()
        if msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "") and first_tool_call is None:
            content = msg["content"]
            start = content.find("<tool_call>") + len("<tool_call>")
            end = content.find("</tool_call>")
            if end == -1:
                return None
            try:
                first_tool_call = json.loads(content[start:end].strip())
            except (json.JSONDecodeError, TypeError):
                return None

    if first_tool_call is None:
        return None

    key_str = json.dumps({"q": user_q, "call": first_tool_call}, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Merge partitioned generation outputs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/generated",
        help="Directory containing part_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/training_data.jsonl",
        help="Output merged file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # Find all partition files
    part_files = sorted(input_dir.glob("part_*.jsonl"))
    if not part_files:
        print(f"No part_*.jsonl files found in {input_dir}")
        return

    print(f"Found {len(part_files)} partition files:")
    for pf in part_files:
        print(f"  {pf.name}")

    # Load all examples
    all_examples = []
    for pf in part_files:
        count = 0
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_examples.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError:
                    pass
        print(f"  {pf.name}: {count} examples")

    print(f"\nTotal loaded: {len(all_examples)}")

    # Deduplicate across partitions
    dedup_keys = set()
    unique_examples = []
    dup_count = 0

    for ex in all_examples:
        dk = compute_dedup_key(ex)
        if dk and dk in dedup_keys:
            dup_count += 1
            continue
        if dk:
            dedup_keys.add(dk)
        unique_examples.append(ex)

    print(f"Duplicates removed: {dup_count}")
    print(f"Unique examples: {len(unique_examples)}")

    # Re-number IDs
    for i, ex in enumerate(unique_examples, start=1):
        ex["id"] = f"gen_{i:03d}"

    # Write merged output
    with open(output_path, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {output_path}")

    # Distribution report
    type_counts = {}
    persona_counts = {}
    for ex in unique_examples:
        t = ex.get("type", "unknown")
        p = ex.get("persona", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        persona_counts[p] = persona_counts.get(p, 0) + 1

    total = len(unique_examples)
    print(f"\nDistribution ({total} total):")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} ({c/total*100:.1f}%)")
    print(f"\nPersonas:")
    for p, c in sorted(persona_counts.items()):
        print(f"  {p}: {c} ({c/total*100:.1f}%)")


if __name__ == "__main__":
    main()
