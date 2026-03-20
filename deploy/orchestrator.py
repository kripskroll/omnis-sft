#!/usr/bin/env python3
"""Orchestration loop: connects the fine-tuned model to live Omnis MCP tools.

The model generates <tool_call> tags -> this script parses them, calls the real
Omnis MCP server, and feeds the result back until the model produces a final answer.

Usage:
    uv run python deploy/orchestrator.py [--model omnis-sft] [--max-turns 10]
    uv run python deploy/orchestrator.py --question "Why is latency high on 10.0.1.5?"

Prerequisites:
    - Ollama running with the omnis-sft model loaded
    - Omnis MCP server accessible (configured via OMNIS_MCP_URL env var)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OMNIS_MCP_URL = os.environ.get("OMNIS_MCP_URL", "http://localhost:8080")
DEFAULT_MODEL = "omnis-sft"
DEFAULT_MAX_TURNS = 10


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------
def chat(model: str, messages: list[dict[str, str]]) -> str:
    """Send a chat request to the Ollama API and return the assistant message.

    Args:
        model: Name of the Ollama model to use.
        messages: Conversation history in OpenAI-compatible format.

    Returns:
        The assistant's response text.

    Raises:
        requests.HTTPError: If the Ollama API returns an error.
        ConnectionError: If Ollama is not reachable.
    """
    url = f"{OLLAMA_URL}/api/chat"
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
            timeout=300,  # Large models may take a while
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        print(
            f"ERROR: Cannot connect to Ollama at {OLLAMA_URL}.\n"
            "  Make sure Ollama is running: ollama serve",
            file=sys.stderr,
        )
        sys.exit(1)

    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------
# Matches <tool_call> JSON </tool_call> blocks in model output.
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from model output.

    Expected format inside <tool_call> tags:
        {"name": "tool_name", "arguments": {"arg1": "val1", ...}}

    Returns:
        List of parsed tool-call dicts, each with 'name' and 'arguments' keys.
        Returns an empty list if no valid tool calls are found.
    """
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if "name" in payload:
                calls.append(payload)
        except json.JSONDecodeError:
            # Model produced malformed JSON -- skip this call
            print(
                f"WARNING: Skipping malformed tool call: {match.group(1)[:120]}...",
                file=sys.stderr,
            )
    return calls


# ---------------------------------------------------------------------------
# MCP tool execution
# ---------------------------------------------------------------------------
def call_mcp_tool(name: str, arguments: dict[str, Any]) -> str:
    """Call a tool on the Omnis MCP server.

    Sends a JSON-RPC style request to the MCP endpoint and returns the
    result as a string suitable for feeding back to the model.

    Args:
        name: The MCP tool name (e.g., "get_health_overview").
        arguments: Tool arguments as a dict.

    Returns:
        JSON string of the tool result, or an error message on failure.
    """
    url = f"{OMNIS_MCP_URL}/tools/{name}"
    try:
        resp = requests.post(
            url,
            json=arguments,
            timeout=60,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except requests.ConnectionError:
        return json.dumps({
            "error": f"Cannot connect to MCP server at {OMNIS_MCP_URL}",
        })
    except requests.HTTPError as exc:
        return json.dumps({
            "error": f"MCP returned HTTP {exc.response.status_code}",
            "detail": exc.response.text[:500],
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------
def run_turn(
    model: str,
    messages: list[dict[str, str]],
    max_turns: int,
) -> str:
    """Run the agent loop until the model produces a final answer or hits the
    turn limit.

    Args:
        model: Ollama model name.
        messages: Initial conversation (system + user messages).
        max_turns: Maximum number of model-call rounds.

    Returns:
        The model's final answer text.
    """
    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn}/{max_turns} ---")

        # Get model response
        assistant_text = chat(model, messages)
        messages.append({"role": "assistant", "content": assistant_text})

        # Check for tool calls
        tool_calls = parse_tool_calls(assistant_text)

        if not tool_calls:
            # No tool calls -> this is the final answer
            return assistant_text

        # Execute each tool call and collect results
        results: list[str] = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("arguments", {})
            print(f"  Calling tool: {tool_name}({json.dumps(tool_args, indent=2)})")

            result = call_mcp_tool(tool_name, tool_args)
            results.append(
                f"<tool_response>\n"
                f'{{"name": "{tool_name}", "result": {result}}}\n'
                f"</tool_response>"
            )
            print(f"  -> Got result ({len(result)} chars)")

        # Feed results back as a user message
        tool_response_msg = "\n\n".join(results)
        messages.append({"role": "user", "content": tool_response_msg})

    # Hit the turn limit
    print(f"\nWARNING: Reached max turns ({max_turns}). Forcing final answer.")
    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of tool-call turns. "
            "Please provide your best answer now based on the information gathered so far."
        ),
    })
    return chat(model, messages)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def repl(model: str, max_turns: int) -> None:
    """Interactive REPL for asking questions to the agent.

    Maintains conversation history across questions within a session.
    Type 'quit', 'exit', or Ctrl-D to exit. Type 'clear' to reset history.
    """
    print(f"Omnis Diagnostics Agent (model: {model})")
    print(f"MCP endpoint: {OMNIS_MCP_URL}")
    print("Type your question, 'clear' to reset, or 'quit' to exit.\n")

    system_msg = {
        "role": "system",
        "content": (
            "You are a network diagnostics assistant with access to NETSCOUT Omnis "
            "monitoring tools. When investigating network issues:\n"
            "1. Start with broad health checks, then drill into specific problems\n"
            "2. Reason step by step in <think> tags before each tool call\n"
            "3. Extract specific values (IPs, error rates, latencies) from tool results "
            "for follow-up queries\n"
            "4. Always provide actionable recommendations in your final answer"
        ),
    }

    # Persistent conversation history for the session
    messages: list[dict[str, str]] = [system_msg]

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if question.lower() == "clear":
            messages = [system_msg]
            print("Conversation history cleared.\n")
            continue

        messages.append({"role": "user", "content": question})

        answer = run_turn(model, messages, max_turns)
        print(f"\nAssistant> {answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrator: connect the fine-tuned model to live Omnis MCP tools.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum tool-call turns per question (default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Single question to ask (non-interactive mode)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.question:
        # Single-shot mode: answer one question and exit
        system_msg = {
            "role": "system",
            "content": (
                "You are a network diagnostics assistant with access to NETSCOUT Omnis "
                "monitoring tools. When investigating network issues:\n"
                "1. Start with broad health checks, then drill into specific problems\n"
                "2. Reason step by step in <think> tags before each tool call\n"
                "3. Extract specific values (IPs, error rates, latencies) from tool results "
                "for follow-up queries\n"
                "4. Always provide actionable recommendations in your final answer"
            ),
        }
        messages = [
            system_msg,
            {"role": "user", "content": args.question},
        ]
        answer = run_turn(args.model, messages, args.max_turns)
        print(f"\n{answer}")
    else:
        # Interactive REPL mode
        repl(args.model, args.max_turns)


if __name__ == "__main__":
    main()
