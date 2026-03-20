#!/usr/bin/env python3
"""Quick test script to run sample questions through the fine-tuned model."""

from unsloth import FastLanguageModel
import torch

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="training/output",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

SYSTEM = (
    "You are a network diagnostics assistant with access to NETSCOUT Omnis "
    "monitoring tools. When investigating network issues:\n"
    "1. Start with broad health checks, then drill into specific problems\n"
    "2. Reason step by step in <think> tags before each tool call\n"
    "3. Extract specific values (IPs, error rates, latencies) from tool "
    "results for follow-up queries\n"
    "4. Always provide actionable recommendations in your final answer"
)

questions = [
    "How is the network doing?",
    "DNS seems broken, investigate",
    "What is host 10.0.4.12 doing?",
    "Our database seems fine but users complain about slow queries",
    "Check for any suspicious DNS activity",
]

SEP = "=" * 60

for q in questions:
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=300, do_sample=False, use_cache=False
        )
    out = tokenizer.decode(
        ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
    )
    if "<|im_end|>" in out:
        out = out[:out.index("<|im_end|>")]

    print(f"\n{SEP}")
    print(f"USER: {q}")
    print(SEP)
    print(out.strip())
    print()
