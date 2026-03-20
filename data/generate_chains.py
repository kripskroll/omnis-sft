#!/usr/bin/env python3
"""Generate synthetic training data for Omnis SFT.

Uses Claude Code CLI (claude -p) to generate multi-turn tool-calling conversations
grounded in real environment data from a live Omnis deployment. This uses your
Claude Code subscription — no API key needed.

Usage:
    uv run python data/generate_chains.py [--count 350] [--output data/generated/training_data.jsonl]
"""

import json
import random
import subprocess
import tempfile
import time
import argparse
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FOR_GENERATED_EXAMPLES = (
    "You are a network diagnostics assistant with access to NETSCOUT Omnis "
    "monitoring tools. When investigating network issues:\n"
    "1. Start with broad health checks, then drill into specific problems\n"
    "2. Reason step by step in <think> tags before each tool call\n"
    "3. Extract specific values (IPs, error rates, latencies) from tool "
    "results for follow-up queries\n"
    "4. Always provide actionable recommendations in your final answer"
)

# Distribution targets (fractions of total count)
DISTRIBUTION = {
    "single_turn": 0.40,   # single-turn tool selection
    "chain_2_3": 0.40,     # 2-3 step diagnostic chains
    "chain_3_4": 0.15,     # 3-4 step deep investigations
    "wrong_path": 0.05,    # wrong path + recovery
}

# Personas with language style descriptions and example phrasings
PERSONAS = {
    "noc_operator": {
        "style": "Short, informal, casual language",
        "examples": [
            "any issues?",
            "DNS ok?",
            "what's up with the network?",
            "anything weird going on?",
            "how's everything looking?",
            "see any problems?",
        ],
    },
    "network_engineer": {
        "style": "Technical, references IPs and protocols directly",
        "examples": [
            "investigate DNS failures from 10.0.3.x subnet",
            "show me TCP retransmissions for MYSQL connections to 10.0.5.20",
            "what's the latency distribution for Web Store over the last 4 hours?",
            "trace the path from 10.0.4.12 to 151.101.2.132",
        ],
    },
    "it_manager": {
        "style": "Business-oriented, wants summaries and high-level status",
        "examples": [
            "give me a summary of network health",
            "how are our sites performing?",
            "any critical issues I should know about?",
            "what's the overall error rate trend?",
        ],
    },
    "security_analyst": {
        "style": "Focused on anomalies, suspicious activity, unusual patterns",
        "examples": [
            "check for suspicious activity",
            "any DGA domains?",
            "are there anomalous connection patterns?",
            "which hosts have unusual outbound traffic?",
        ],
    },
    "app_owner": {
        "style": "Service-focused, cares about specific application performance",
        "examples": [
            "how's MySQL doing?",
            "Web Store seems slow, can you check?",
            "are there errors on the Shop Backend API?",
            "is the database responding to queries normally?",
        ],
    },
}

# Workflow types
WORKFLOWS = [
    "health_triage",
    "dns_investigation",
    "site_performance",
    "app_performance",
    "host_investigation",
    "cross_site_path",
    "bandwidth_analysis",
    "mysql_troubleshooting",
    "security_scan",
    "wrong_path_recovery",
]

# Map example types to allowed workflows
TYPE_WORKFLOWS = {
    "single_turn": [
        "health_triage", "dns_investigation", "site_performance",
        "app_performance", "host_investigation", "bandwidth_analysis",
        "mysql_troubleshooting", "security_scan",
    ],
    "chain_2_3": [
        "health_triage", "dns_investigation", "site_performance",
        "app_performance", "host_investigation", "bandwidth_analysis",
        "mysql_troubleshooting", "security_scan",
    ],
    "chain_3_4": [
        "dns_investigation", "app_performance", "host_investigation",
        "cross_site_path", "mysql_troubleshooting", "security_scan",
    ],
    "wrong_path": [
        "wrong_path_recovery",
    ],
}

# Type-specific generation requirements
TYPE_REQUIREMENTS = {
    "single_turn": (
        "Generate a SINGLE-TURN example: the user asks a question and the "
        "assistant picks ONE correct tool, calls it, and provides a final "
        "answer. The conversation has exactly: system, user, assistant "
        "(with <think> + <tool_call>), tool result, assistant (final answer). "
        "The <think> block should explain WHY this tool is the right choice."
    ),
    "chain_2_3": (
        "Generate a 2-3 STEP diagnostic chain: the user asks a question, and "
        "the assistant uses 2 or 3 tool calls in sequence, reasoning after each "
        "result before deciding the next tool. Each assistant turn (except the "
        "final answer) must have <think> tags explaining what was learned and "
        "what to investigate next, followed by a <tool_call>. The final turn "
        "must synthesize all findings into an actionable diagnosis."
    ),
    "chain_3_4": (
        "Generate a 3-4 STEP deep investigation: a complex diagnostic chain "
        "where the assistant progressively narrows down a problem through 3-4 "
        "tool calls. Each step must extract SPECIFIC values (IPs, error rates, "
        "latency numbers) from the previous result and use them as parameters "
        "in the next call. The <think> blocks must reference concrete data "
        "points. The final answer must include a root cause analysis and "
        "specific remediation steps."
    ),
    "wrong_path": (
        "Generate a WRONG PATH + RECOVERY example: the assistant initially "
        "picks the wrong tool or wrong parameters (e.g., checking the wrong "
        "application, wrong IP, or using an overly broad tool when a specific "
        "one is needed). After seeing the unhelpful result, the <think> block "
        "must explicitly acknowledge the mistake ('This didn't give me what I "
        "need because...') and pivot to a better tool. Total: 2-3 tool calls. "
        "The example teaches the model to recover gracefully from wrong choices."
    ),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_context_files(data_dir: Path) -> dict:
    """Load tool_catalog.json, environment_context.json, and seed_chains.json."""
    tool_catalog_path = data_dir / "tool_catalog.json"
    env_context_path = data_dir / "environment_context.json"
    seed_chains_path = data_dir / "seed_chains.json"

    for p in [tool_catalog_path, env_context_path, seed_chains_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required context file not found: {p}")

    with open(tool_catalog_path) as f:
        tool_catalog = json.load(f)
    with open(env_context_path) as f:
        env_context = json.load(f)
    with open(seed_chains_path) as f:
        seed_chains = json.load(f)

    return {
        "tool_catalog": tool_catalog,
        "environment_context": env_context,
        "seed_chains": seed_chains,
    }


def get_all_tool_names(tool_catalog: dict) -> set:
    """Extract all valid tool names from the catalog."""
    names = set()
    for category in tool_catalog.get("categories", []):
        for tool in category.get("tools", []):
            names.add(tool["name"])
    return names


def get_tool_required_params(tool_catalog: dict) -> dict:
    """Build a mapping of tool_name -> set of required parameter names."""
    mapping = {}
    for category in tool_catalog.get("categories", []):
        for tool in category.get("tools", []):
            required = set(tool.get("parameters", {}).get("required", {}).keys())
            mapping[tool["name"]] = required
    return mapping


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_generation_prompt(
    example_type: str,
    persona: str,
    workflow: str,
    tool_catalog: dict,
    env_context: dict,
    seed_examples: list[dict],
) -> str:
    """Build the prompt sent to Claude CLI to generate one training example."""

    seed_text = json.dumps(seed_examples, indent=2)
    catalog_text = json.dumps(tool_catalog, indent=2)
    env_text = json.dumps(env_context, indent=2)

    persona_info = PERSONAS[persona]
    type_reqs = TYPE_REQUIREMENTS[example_type]

    prompt = f"""You are generating training data for a network diagnostics model.

<tool_catalog>
{catalog_text}
</tool_catalog>

<environment>
{env_text}
</environment>

<seed_examples>
{seed_text}
</seed_examples>

Generate a {example_type} example for a {persona} (style: {persona_info['style']}) asking about {workflow}.
Example phrasings for this persona: {json.dumps(persona_info['examples'])}

The conversation must use the ChatML format with system/user/assistant/tool roles.
The assistant must use <think> tags for reasoning and <tool_call> tags for tool calls.
Tool results must be realistic JSON matching the environment data.

The system message for the generated conversation MUST be exactly:
"{SYSTEM_PROMPT_FOR_GENERATED_EXAMPLES}"

Requirements:
- {type_reqs}
- Use IPs and values from the environment context (real IPs like 10.0.3.20, 10.0.5.20, 10.0.2.100, 8.8.8.8, etc.)
- Reference specific data values in <think> blocks (exact error rates, latency numbers, IP addresses)
- End with an actionable diagnosis or answer
- The user's question should feel natural for a {persona} — use their language style
- Do NOT copy seed examples verbatim — create a NEW scenario that covers {workflow}
- Tool result JSON should be realistic but NOT identical to seed examples

Return ONLY a valid JSON object with this structure (no markdown fencing, no extra text):
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "<think>...</think>\\n\\n<tool_call>{{...}}</tool_call>"}},
    {{"role": "tool", "name": "tool_name", "content": "{{...}}"}},
    ...
    {{"role": "assistant", "content": "Final answer with diagnosis..."}}
  ]
}}"""
    return prompt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_example(
    example: dict,
    valid_tool_names: set,
    tool_required_params: dict,
    example_type: str,
) -> tuple[bool, str]:
    """Validate a generated example. Returns (is_valid, reason)."""

    messages = example.get("messages", [])
    if not messages:
        return False, "No messages in example"

    if messages[0].get("role") != "system":
        return False, "First message must be system role"

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False, "No user message found"

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False, "No assistant messages found"

    # Check tool messages reference valid tools
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    for tm in tool_msgs:
        tool_name = tm.get("name", "")
        if tool_name not in valid_tool_names:
            return False, f"Unknown tool name: {tool_name}"

    # Check <think> tags
    if example_type != "single_turn":
        think_found = False
        for am in assistant_msgs:
            if "<think>" in am.get("content", ""):
                think_found = True
                break
        if not think_found:
            return False, "No <think> tags found in chain example"
    else:
        tool_calling_turns = [
            m for m in assistant_msgs if "<tool_call>" in m.get("content", "")
        ]
        for tc in tool_calling_turns:
            if "<think>" not in tc.get("content", ""):
                return False, "Tool-calling assistant turn missing <think> tags"

    # Check <tool_call> tags reference valid tools with required params
    for am in assistant_msgs:
        content = am.get("content", "")
        if "<tool_call>" in content:
            start = content.find("<tool_call>") + len("<tool_call>")
            end = content.find("</tool_call>")
            if end == -1:
                return False, "Unclosed <tool_call> tag"
            try:
                call_json = json.loads(content[start:end].strip())
            except json.JSONDecodeError:
                return False, "Invalid JSON inside <tool_call> tags"
            call_name = call_json.get("name", "")
            if call_name not in valid_tool_names:
                return False, f"Tool call references unknown tool: {call_name}"
            required = tool_required_params.get(call_name, set())
            call_args = call_json.get("arguments", {})
            for rp in required:
                if rp not in call_args:
                    return False, f"Tool {call_name} missing required param: {rp}"

    if messages[-1].get("role") != "assistant":
        return False, "Last message must be assistant (final answer)"

    if "<tool_call>" in messages[-1].get("content", ""):
        return False, "Final assistant message should not contain a tool call"

    return True, "ok"


def compute_dedup_key(example: dict) -> str | None:
    """Compute a deduplication key based on first tool call + user question.

    We include the user question in the key so that different questions
    starting with the same no-arg tool (e.g., get_health_overview) are
    not flagged as duplicates.
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


# ---------------------------------------------------------------------------
# Claude CLI generation
# ---------------------------------------------------------------------------


def call_claude_cli(prompt: str, model: str = "sonnet") -> str | None:
    """Call claude CLI in print mode and return the response text.

    Uses your Claude Code subscription — no API key needed.
    Disables all tools (--tools "") so Claude only generates text.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                "--model", model,
                "--output-format", "text",
                "--tools", "",
                "--no-session-persistence",
                "--mcp-config", '{"mcpServers":{}}',
                "--strict-mcp-config",
            ],
            stdin=open(prompt_file),
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout per generation
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f"  [CLI error] {stderr[:200]}")
            return None

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        print("  [timeout] Claude CLI timed out after 120s")
        return None
    except FileNotFoundError:
        print("  [error] 'claude' CLI not found. Make sure Claude Code is installed.")
        return None
    finally:
        Path(prompt_file).unlink(missing_ok=True)


def generate_one_example(
    example_type: str,
    persona: str,
    workflow: str,
    context: dict,
    seed_count: int,
    model: str,
) -> dict | None:
    """Generate a single training example via Claude CLI.

    Returns the parsed example dict, or None on failure.
    """
    # Pick random seed examples for reference
    all_seeds = context["seed_chains"].get("seed_chains", [])
    seeds_sample = random.sample(all_seeds, min(seed_count, len(all_seeds)))

    prompt = build_generation_prompt(
        example_type=example_type,
        persona=persona,
        workflow=workflow,
        tool_catalog=context["tool_catalog"],
        env_context=context["environment_context"],
        seed_examples=seeds_sample,
    )

    text = call_claude_cli(prompt, model=model)
    if text is None:
        return None

    # Strip markdown fencing if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        example = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                example = json.loads(text[start:end])
            except json.JSONDecodeError:
                print(f"  [parse error] Could not parse JSON ({len(text)} chars)")
                return None
        else:
            print(f"  [parse error] No JSON object found ({len(text)} chars)")
            return None

    return example


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------


def build_generation_plan(total_count: int) -> list[dict]:
    """Build a shuffled list of (type, persona, workflow) assignments."""
    plan = []

    for ex_type, fraction in DISTRIBUTION.items():
        type_count = round(total_count * fraction)
        allowed_workflows = TYPE_WORKFLOWS[ex_type]
        persona_names = list(PERSONAS.keys())

        for i in range(type_count):
            persona = persona_names[i % len(persona_names)]
            workflow = allowed_workflows[i % len(allowed_workflows)]
            plan.append({
                "type": ex_type,
                "persona": persona,
                "workflow": workflow,
            })

    # Trim or pad to exact count
    if len(plan) > total_count:
        plan = plan[:total_count]
    while len(plan) < total_count:
        plan.append({
            "type": "chain_2_3",
            "persona": random.choice(list(PERSONAS.keys())),
            "workflow": random.choice(TYPE_WORKFLOWS["chain_2_3"]),
        })

    random.shuffle(plan)
    return plan


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def run_generation(
    total_count: int,
    output_path: Path,
    batch_size: int,
    seed_count: int,
    model: str,
) -> None:
    """Main generation loop with batching, validation, dedup, and progress."""

    # Load context
    data_dir = Path(__file__).resolve().parent
    context = load_context_files(data_dir)

    valid_tool_names = get_all_tool_names(context["tool_catalog"])
    tool_required_params = get_tool_required_params(context["tool_catalog"])

    print(f"Loaded {len(valid_tool_names)} tools from catalog")
    print(f"Loaded {len(context['environment_context'].get('applications', []))} applications from environment")
    print(f"Loaded {len(context['seed_chains'].get('seed_chains', []))} seed chains")
    print(f"Target: {total_count} examples -> {output_path}")
    print()

    # Build generation plan
    plan = build_generation_plan(total_count)

    # Tracking
    results: list[dict] = []
    dedup_keys: set[str] = set()
    counts_by_type: dict[str, int] = {t: 0 for t in DISTRIBUTION}
    counts_by_persona: dict[str, int] = {p: 0 for p in PERSONAS}
    failed_count = 0
    skipped_dedup = 0
    skipped_invalid = 0

    # Load existing progress if output file exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    results.append(existing)
                    dk = compute_dedup_key(existing)
                    if dk:
                        dedup_keys.add(dk)
                    ex_type = existing.get("type", "chain_2_3")
                    ex_persona = existing.get("persona", "noc_operator")
                    counts_by_type[ex_type] = counts_by_type.get(ex_type, 0) + 1
                    counts_by_persona[ex_persona] = counts_by_persona.get(ex_persona, 0) + 1
                except json.JSONDecodeError:
                    pass
        print(f"Resumed from {len(results)} existing examples in {output_path}")
        plan = plan[len(results):]

    if not plan:
        print("All examples already generated. Nothing to do.")
        return

    print(f"Generating {len(plan)} remaining examples in batches of {batch_size}...")
    print()

    start_time = time.time()
    plan_idx = 0
    batch_num = 0

    while plan_idx < len(plan):
        batch_num += 1
        batch_end = min(plan_idx + batch_size, len(plan))
        batch_items = plan[plan_idx:batch_end]

        print(f"--- Batch {batch_num} ({len(batch_items)} examples) ---")

        for item in batch_items:
            ex_type = item["type"]
            persona = item["persona"]
            workflow = item["workflow"]
            example_id = f"gen_{len(results) + 1:03d}"

            # Retry loop (up to 3 attempts per example)
            max_retries = 3
            for attempt in range(max_retries):
                example = generate_one_example(
                    example_type=ex_type,
                    persona=persona,
                    workflow=workflow,
                    context=context,
                    seed_count=seed_count,
                    model=model,
                )

                if example is None:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  [{example_id}] Attempt {attempt + 1} failed, retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  [{example_id}] FAILED after {max_retries} attempts")
                        failed_count += 1
                        break

                # Validate
                is_valid, reason = validate_example(
                    example, valid_tool_names, tool_required_params, ex_type
                )
                if not is_valid:
                    if attempt < max_retries - 1:
                        print(f"  [{example_id}] Invalid ({reason}), regenerating...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"  [{example_id}] SKIPPED (invalid after {max_retries} attempts: {reason})")
                        skipped_invalid += 1
                        break

                # Deduplication
                dk = compute_dedup_key(example)
                if dk and dk in dedup_keys:
                    if attempt < max_retries - 1:
                        print(f"  [{example_id}] Duplicate first tool call, regenerating...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"  [{example_id}] SKIPPED (duplicate after {max_retries} attempts)")
                        skipped_dedup += 1
                        break

                # Success — enrich and store
                if dk:
                    dedup_keys.add(dk)

                tool_msg_count = sum(
                    1 for m in example.get("messages", []) if m.get("role") == "tool"
                )

                record = {
                    "id": example_id,
                    "workflow": workflow,
                    "persona": persona,
                    "type": ex_type,
                    "chain_length": tool_msg_count,
                    "messages": example["messages"],
                }
                results.append(record)
                counts_by_type[ex_type] = counts_by_type.get(ex_type, 0) + 1
                counts_by_persona[persona] = counts_by_persona.get(persona, 0) + 1

                print(f"  [{example_id}] {ex_type} | {persona} | {workflow} | {tool_msg_count} tools")
                break

        # Save progress after each batch
        with open(output_path, "w") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Progress report
        elapsed = time.time() - start_time
        rate = len(results) / elapsed * 60 if elapsed > 0 else 0
        print()
        print(f"Progress: {len(results)}/{total_count} generated ({elapsed:.0f}s elapsed, ~{rate:.1f}/min)")
        print(f"  By type:    {json.dumps(counts_by_type)}")
        print(f"  By persona: {json.dumps(counts_by_persona)}")
        print(f"  Failed: {failed_count} | Invalid: {skipped_invalid} | Dedup: {skipped_dedup}")
        print()

        plan_idx = batch_end

    # Final summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(results)}")
    print(f"Output file:    {output_path}")
    print(f"Time elapsed:   {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"By type:")
    for t, c in sorted(counts_by_type.items()):
        print(f"  {t}: {c}")
    print(f"By persona:")
    for p, c in sorted(counts_by_persona.items()):
        print(f"  {p}: {c}")
    print(f"Failed:          {failed_count}")
    print(f"Skipped invalid: {skipped_invalid}")
    print(f"Skipped dedup:   {skipped_dedup}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Omnis SFT using Claude Code CLI."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=350,
        help="Total number of examples to generate (default: 350)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/training_data.jsonl",
        help="Output JSONL file path (default: data/generated/training_data.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of examples per batch (default: 10)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=2,
        help="Number of seed examples to include in each generation prompt (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        help="Claude model to use (default: sonnet). Options: sonnet, opus, haiku",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print("Omnis SFT Training Data Generator")
    print("Engine: Claude Code CLI (subscription-based, no API key needed)")
    print(f"Model: {args.model}")
    print(f"Target count: {args.count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed examples per prompt: {args.seed_count}")
    print(f"Output: {output_path}")
    print()

    run_generation(
        total_count=args.count,
        output_path=output_path,
        batch_size=args.batch_size,
        seed_count=args.seed_count,
        model=args.model,
    )


if __name__ == "__main__":
    main()
