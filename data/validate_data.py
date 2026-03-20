#!/usr/bin/env python3
"""Validate generated training data quality.

Checks tool names, parameters, think tags, IP addresses,
distribution, duplicates, and message structure.

Usage:
    uv run python data/validate_data.py [--data data/generated/training_data.jsonl]

Exit codes:
    0: All checks passed
    1: Validation errors found
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_json(path: Path) -> dict:
    """Load a JSON file, exit with a clear message on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path}: {exc}")
        sys.exit(1)


def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Load a JSONL file, returning (examples, errors)."""
    examples: list[dict] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"  Line {line_num}: malformed JSON — {exc}")
    return examples, errors


def extract_tool_calls(content: str) -> list[dict]:
    """Extract all parsed tool-call dicts from an assistant message."""
    calls = []
    for match in TOOL_CALL_RE.finditer(content):
        raw = match.group(1).strip()
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            calls.append(None)  # record that a call existed but was invalid
    return calls


def extract_ips_from_args(arguments: dict) -> set[str]:
    """Recursively extract IP addresses from tool call arguments."""
    ips: set[str] = set()
    for value in arguments.values():
        if isinstance(value, str):
            ips.update(IP_RE.findall(value))
        elif isinstance(value, dict):
            ips.update(extract_ips_from_args(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    ips.update(IP_RE.findall(item))
                elif isinstance(item, dict):
                    ips.update(extract_ips_from_args(item))
    return ips


# ---------------------------------------------------------------------------
# Build reference sets from catalog & environment
# ---------------------------------------------------------------------------

def build_tool_index(catalog: dict) -> dict[str, dict]:
    """Map tool_name -> {"required": set[str], "optional": set[str]}."""
    index: dict[str, dict] = {}
    for category in catalog.get("categories", []):
        for tool in category.get("tools", []):
            name = tool["name"]
            required = set(tool.get("parameters", {}).get("required", {}).keys())
            optional = set(tool.get("parameters", {}).get("optional", {}).keys())
            index[name] = {"required": required, "optional": optional}
    return index


def build_valid_ips(env: dict) -> set[str]:
    """Collect every IP mentioned anywhere in environment_context.json."""
    ips: set[str] = set()

    # key_hosts keys are IPs
    for ip in env.get("key_hosts", {}):
        ips.add(ip)

    # known_issues → affected_hosts, target
    for issue in env.get("known_issues", []):
        for host in issue.get("affected_hosts", []):
            ips.add(host)
        target = issue.get("target")
        if target:
            ips.add(target)

    # top_traffic_pairs
    for pair in env.get("top_traffic_pairs", []):
        if pair.get("client"):
            ips.add(pair["client"])
        if pair.get("server"):
            ips.add(pair["server"])

    # anomalies → high_error_rate, latency_spikes
    anomalies = env.get("anomalies", {})
    for entry in anomalies.get("high_error_rate", []):
        if entry.get("client"):
            ips.add(entry["client"])
        if entry.get("server"):
            ips.add(entry["server"])
    for entry in anomalies.get("latency_spikes", []):
        if entry.get("client"):
            ips.add(entry["client"])
        if entry.get("server"):
            ips.add(entry["server"])

    # topology → sensors
    for sensor in env.get("topology", {}).get("sensors", []):
        if sensor.get("ip"):
            ips.add(sensor["ip"])

    return ips


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_tool_names(examples: list[dict], tool_index: dict) -> tuple[int, int, list[str]]:
    """Return (valid, total, error_messages)."""
    valid = 0
    total = len(examples)
    errors: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        ex_ok = True
        for msg in ex.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for call in extract_tool_calls(msg.get("content", "")):
                if call is None:
                    errors.append(f"  Example {idx}: unparseable tool_call JSON")
                    ex_ok = False
                    continue
                name = call.get("name", "")
                if name not in tool_index:
                    errors.append(f"  Example {idx}: unknown tool '{name}'")
                    ex_ok = False
        if ex_ok:
            valid += 1

    return valid, total, errors


def check_required_params(examples: list[dict], tool_index: dict) -> tuple[int, int, list[str]]:
    """Return (valid, total, error_messages)."""
    valid = 0
    total = len(examples)
    errors: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        ex_ok = True
        for msg in ex.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for call in extract_tool_calls(msg.get("content", "")):
                if call is None:
                    continue
                name = call.get("name", "")
                if name not in tool_index:
                    continue  # already caught by tool-name check
                required = tool_index[name]["required"]
                provided = set(call.get("arguments", {}).keys())
                missing = required - provided
                if missing:
                    errors.append(
                        f"  Example {idx}: tool '{name}' missing required params: {sorted(missing)}"
                    )
                    ex_ok = False
        if ex_ok:
            valid += 1

    return valid, total, errors


def check_think_tags(examples: list[dict]) -> tuple[int, int, list[str]]:
    """Chain examples (chain_length > 1) must have <think> tags between tool calls.

    Returns (valid_chain_count, total_chain_count, errors).
    """
    valid = 0
    total = 0
    errors: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        chain_length = ex.get("chain_length", 1)
        if chain_length <= 1:
            continue
        total += 1

        # Collect assistant messages that contain a tool call (not the final answer)
        assistant_tool_msgs = []
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant" and TOOL_CALL_RE.search(msg.get("content", "")):
                assistant_tool_msgs.append(msg)

        # All assistant messages with a tool_call except the first should have <think>
        if len(assistant_tool_msgs) <= 1:
            # Only one tool-call message — check that it has think if chain > 1
            # (even the first can have think, but at minimum the 2nd+ must)
            valid += 1
            continue

        ex_ok = True
        for i, msg in enumerate(assistant_tool_msgs):
            if i == 0:
                continue  # first tool call may or may not have think
            if not THINK_RE.search(msg.get("content", "")):
                errors.append(
                    f"  Example {idx}: assistant tool-call #{i + 1} missing <think> tag"
                )
                ex_ok = False
        if ex_ok:
            valid += 1

    return valid, total, errors


def check_ip_validation(
    examples: list[dict], valid_ips: set[str]
) -> tuple[int, int, list[str], set[str]]:
    """Return (valid, total, warnings, unknown_ips)."""
    valid = 0
    total = len(examples)
    warnings: list[str] = []
    all_unknown: set[str] = set()

    for idx, ex in enumerate(examples, start=1):
        ex_ok = True
        for msg in ex.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for call in extract_tool_calls(msg.get("content", "")):
                if call is None:
                    continue
                found_ips = extract_ips_from_args(call.get("arguments", {}))
                unknown = found_ips - valid_ips
                if unknown:
                    warnings.append(
                        f"  Example {idx}: unknown IPs in tool args: {sorted(unknown)}"
                    )
                    all_unknown.update(unknown)
                    ex_ok = False
        if ex_ok:
            valid += 1

    return valid, total, warnings, all_unknown


def check_distribution(examples: list[dict]) -> tuple[bool, list[str]]:
    """Verify category distribution matches targets.

    Expects each example to have a 'category' field with one of:
        single_turn, chain_2_3, chain_3_4, wrong_path

    Falls back to inferring from chain_length if category is absent.
    """
    targets = {
        "single_turn": (40, 5),
        "chain_2_3": (40, 5),
        "chain_3_4": (15, 5),
        "wrong_path": (5, 3),
    }

    counts: Counter = Counter()
    total = len(examples)

    for ex in examples:
        # Check explicit type/category fields first
        cat = ex.get("type") or ex.get("category")
        if cat is None:
            # Infer from chain_length
            cl = ex.get("chain_length", 1)
            is_wrong = ex.get("wrong_path", False) or ex.get("recovery", False)
            if is_wrong:
                cat = "wrong_path"
            elif cl <= 1:
                cat = "single_turn"
            elif cl <= 3:
                cat = "chain_2_3"
            else:
                cat = "chain_3_4"
        counts[cat] += 1

    lines: list[str] = []
    all_pass = True
    for cat, (target_pct, tolerance) in targets.items():
        actual = counts.get(cat, 0)
        actual_pct = (actual / total * 100) if total else 0
        lo = target_pct - tolerance
        hi = target_pct + tolerance
        ok = lo <= actual_pct <= hi
        status = "ok" if ok else "OUT OF RANGE"
        if not ok:
            all_pass = False
        lines.append(
            f"    - {cat}: {actual} ({actual_pct:.1f}%) — target {target_pct}% ± {tolerance}% [{status}]"
        )

    # Report any unknown categories
    known = set(targets.keys())
    unknown_cats = set(counts.keys()) - known
    if unknown_cats:
        for uc in sorted(unknown_cats):
            lines.append(f"    - {uc}: {counts[uc]} (unknown category)")
        all_pass = False

    return all_pass, lines


def check_duplicates(examples: list[dict]) -> tuple[int, list[str]]:
    """Detect duplicate examples (same user question + first tool call)."""
    seen: dict[str, int] = {}  # canonical_key -> first example index
    duplicates: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        # Extract user question
        user_q = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                user_q = msg.get("content", "").strip().lower()
                break

        # Find the first assistant tool call
        first_call = None
        for msg in ex.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            calls = extract_tool_calls(msg.get("content", ""))
            if calls:
                first_call = calls[0]
                break

        if first_call is None or not isinstance(first_call, dict):
            continue

        # Canonical key: user question + tool call
        key = json.dumps(
            {"q": user_q, "name": first_call.get("name", ""), "arguments": first_call.get("arguments", {})},
            sort_keys=True,
        )

        if key in seen:
            duplicates.append(
                f"  Example {idx}: duplicate (same question + tool call as example {seen[key]})"
            )
        else:
            seen[key] = idx

    return len(duplicates), duplicates


def check_message_structure(examples: list[dict]) -> tuple[int, int, list[str]]:
    """Each example must have messages with system, user, assistant in correct order.

    Expected pattern:
        system, user, (assistant, tool)*, assistant
    where the final assistant has no tool_call.
    """
    valid = 0
    total = len(examples)
    errors: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        msgs = ex.get("messages", [])
        if not msgs:
            errors.append(f"  Example {idx}: no messages")
            continue

        problems: list[str] = []

        # First message must be system
        if msgs[0].get("role") != "system":
            problems.append("first message is not 'system'")

        # Must have at least one user message
        roles = [m.get("role") for m in msgs]
        if "user" not in roles:
            problems.append("no 'user' message found")

        # Must have at least one assistant message
        if "assistant" not in roles:
            problems.append("no 'assistant' message found")

        # The user message should come after system and before any assistant
        if "system" in roles and "user" in roles:
            sys_idx = roles.index("system")
            usr_idx = roles.index("user")
            if usr_idx <= sys_idx:
                problems.append("'user' message appears before 'system'")

        # Last message should be assistant
        if msgs[-1].get("role") != "assistant":
            problems.append(f"last message role is '{msgs[-1].get('role')}', expected 'assistant'")

        if problems:
            errors.append(f"  Example {idx}: {'; '.join(problems)}")
        else:
            valid += 1

    return valid, total, errors


def check_completeness(examples: list[dict]) -> tuple[int, int, list[str]]:
    """Final assistant message should not contain a <tool_call> — it should be a concluding answer."""
    valid = 0
    total = len(examples)
    errors: list[str] = []

    for idx, ex in enumerate(examples, start=1):
        msgs = ex.get("messages", [])
        if not msgs:
            continue

        # Find the last assistant message
        last_assistant = None
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                last_assistant = msg
                break

        if last_assistant is None:
            continue  # caught by structure check

        if TOOL_CALL_RE.search(last_assistant.get("content", "")):
            errors.append(f"  Example {idx}: final assistant message contains a <tool_call>")
        else:
            valid += 1

    return valid, total, errors


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_validation(data_path: Path, catalog_path: Path, env_path: Path) -> bool:
    """Run all checks and print a report. Returns True if all critical checks pass."""

    # Load references
    catalog = load_json(catalog_path)
    env = load_json(env_path)
    tool_index = build_tool_index(catalog)
    valid_ips = build_valid_ips(env)

    # Load data
    examples, parse_errors = load_jsonl(data_path)

    if parse_errors:
        print(f"ERROR: {len(parse_errors)} lines failed to parse in {data_path}:")
        for e in parse_errors[:20]:
            print(e)
        if len(parse_errors) > 20:
            print(f"  ... and {len(parse_errors) - 20} more")

    if not examples:
        print("ERROR: No valid examples found.")
        return False

    print("=" * 40)
    print("  Data Validation Report")
    print("=" * 40)
    print()
    print(f"Total examples: {len(examples)}")
    if parse_errors:
        print(f"Parse errors:   {len(parse_errors)}")
    print()

    failures = 0
    warnings = 0

    # 1. Tool names
    tn_valid, tn_total, tn_errors = check_tool_names(examples, tool_index)
    if tn_valid == tn_total:
        print(f"[PASS] Tool names: {tn_valid}/{tn_total} valid")
    else:
        print(f"[FAIL] Tool names: {tn_valid}/{tn_total} valid")
        for e in tn_errors:
            print(e)
        failures += 1

    # 2. Required params
    rp_valid, rp_total, rp_errors = check_required_params(examples, tool_index)
    if rp_valid == rp_total:
        print(f"[PASS] Required params: {rp_valid}/{rp_total} valid")
    else:
        print(f"[FAIL] Required params: {rp_valid}/{rp_total} valid")
        for e in rp_errors:
            print(e)
        failures += 1

    # 3. Think tags
    tt_valid, tt_total, tt_errors = check_think_tags(examples)
    if tt_total == 0:
        print("[PASS] Think tags: no chain examples found (nothing to check)")
    elif tt_valid == tt_total:
        print(f"[PASS] Think tags: {tt_valid}/{tt_total} chain examples have think tags")
    else:
        print(f"[FAIL] Think tags: {tt_valid}/{tt_total} chain examples have think tags")
        for e in tt_errors:
            print(e)
        failures += 1

    # 4. IP validation (WARNING, not FAIL)
    ip_valid, ip_total, ip_warnings, unknown_ips = check_ip_validation(examples, valid_ips)
    if ip_valid == ip_total:
        print(f"[PASS] IP validation: {ip_valid}/{ip_total} valid")
    else:
        print(
            f"[WARN] IP validation: {ip_valid}/{ip_total} valid "
            f"({len(unknown_ips)} unknown IP{'s' if len(unknown_ips) != 1 else ''})"
        )
        for w in ip_warnings[:20]:
            print(w)
        if len(ip_warnings) > 20:
            print(f"  ... and {len(ip_warnings) - 20} more")
        warnings += 1

    # 5. Distribution
    dist_ok, dist_lines = check_distribution(examples)
    if dist_ok:
        print("[PASS] Distribution:")
    else:
        print("[FAIL] Distribution:")
        failures += 1
    for line in dist_lines:
        print(line)

    # 6. Duplicates
    dup_count, dup_errors = check_duplicates(examples)
    if dup_count == 0:
        print(f"[PASS] Duplicates: 0 found")
    else:
        print(f"[FAIL] Duplicates: {dup_count} found")
        for e in dup_errors[:20]:
            print(e)
        if len(dup_errors) > 20:
            print(f"  ... and {len(dup_errors) - 20} more")
        failures += 1

    # 7. Message structure
    ms_valid, ms_total, ms_errors = check_message_structure(examples)
    if ms_valid == ms_total:
        print(f"[PASS] Message structure: {ms_valid}/{ms_total} valid")
    else:
        print(f"[FAIL] Message structure: {ms_valid}/{ms_total} valid")
        for e in ms_errors:
            print(e)
        failures += 1

    # 8. Completeness
    cp_valid, cp_total, cp_errors = check_completeness(examples)
    if cp_valid == cp_total:
        print(f"[PASS] Completeness: {cp_valid}/{cp_total} valid")
    else:
        print(f"[FAIL] Completeness: {cp_valid}/{cp_total} valid")
        for e in cp_errors:
            print(e)
        failures += 1

    # Summary
    print()
    if failures == 0 and warnings == 0:
        print("Result: PASSED")
    elif failures == 0:
        print(f"Result: PASSED ({warnings} warning{'s' if warnings != 1 else ''})")
    else:
        parts = []
        parts.append(f"{failures} failure{'s' if failures != 1 else ''}")
        if warnings:
            parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        print(f"Result: FAILED ({', '.join(parts)})")

    return failures == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate generated training data quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes:\n  0  All critical checks passed\n  1  Validation errors found",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated/training_data.jsonl"),
        help="Path to the JSONL training data file (default: data/generated/training_data.jsonl)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/tool_catalog.json"),
        help="Path to tool_catalog.json (default: data/tool_catalog.json)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path("data/environment_context.json"),
        help="Path to environment_context.json (default: data/environment_context.json)",
    )
    args = parser.parse_args()

    passed = run_validation(args.data, args.catalog, args.env)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
