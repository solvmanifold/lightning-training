#!/usr/bin/env python3
"""
Generate DPO preference data using the BASE model (pre-SFT).

For each training prompt:
  - chosen   = ground-truth assistant response (from Super-generated dataset)
  - rejected = response sampled from the base model at temperature 0.9

This teaches DPO to prefer Super-quality answers over raw base-model answers.

Output format (one JSON object per line):
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage (inside training container):
  python3 data/generate_preferences.py \\
    --train data/train.jsonl \\
    --out   data/preferences.jsonl \\
    --n     200 \\
    --max-new-tokens 256 \\
    --temperature 0.9
"""
import argparse
import json
import random
import time
from pathlib import Path

# Must run before transformers imports.
import gb10_patches  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_hybrid_cache_class():
    import sys
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "HybridMambaAttentionDynamicCache"):
            return mod.HybridMambaAttentionDynamicCache
    raise RuntimeError("HybridMambaAttentionDynamicCache not found")


@torch.no_grad()
def manual_generate(model, tokenizer, prompt_text, max_new_tokens, temperature,
                    max_prompt_tokens, device):
    """Same KV-cached driver used by generate_responses.py."""
    Cache = _get_hybrid_cache_class()
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_id = tokenizer.eos_token_id

    base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") \
           else (model.base_model if hasattr(model, "base_model") else model)
    cache = Cache(base.config, batch_size=input_ids.shape[0], dtype=base.dtype, device=device)
    cache.conv_kernel_size = base.config.conv_kernel

    class _DeviceAwareList(list):
        @property
        def device(self):
            for t in self:
                if hasattr(t, "device"):
                    return t.device
            return torch.device("cuda")

    cache.conv_states = _DeviceAwareList(cache.conv_states)
    cache.ssm_states  = _DeviceAwareList(cache.ssm_states)

    seq_len = input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        cache_params=cache, cache_position=cache_position, use_cache=True,
    )
    logits = out.logits[:, -1, :]
    output_ids = input_ids

    for step in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=1
            )
        if next_token.item() == eos_id:
            break
        cache_position = torch.tensor([seq_len + step], device=device)
        out = model(
            input_ids=next_token, attention_mask=attention_mask,
            cache_params=cache, cache_position=cache_position, use_cache=True,
        )
        logits = out.logits[:, -1, :]

    return output_ids


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load training data and sample N prompts
    print(f"Loading {args.train}...")
    records = [json.loads(l) for l in open(args.train) if l.strip()]
    print(f"  Loaded {len(records)} records")

    if args.n < len(records):
        records = random.sample(records, args.n)
    print(f"  Using {len(records)} records for preference generation")

    print(f"Loading BASE model {args.model} in BF16 (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map='cpu' avoids a transformers 4.46.3 sharded-safetensors bug.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
        device_map="cpu",
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    completed = 0
    partial_records = []
    if partial_path.exists():
        partial_records = [
            json.loads(line) for line in open(partial_path) if line.strip()
        ]
        completed = len(partial_records)
        print(f"Resuming from {partial_path} after {completed} records")
    if completed > len(records):
        raise SystemExit(
            f"Partial output has {completed} rows but this run selected only {len(records)} records"
        )
    for i, prior in enumerate(partial_records):
        messages = records[i]["messages"]
        expected_prompt = [m for m in messages if m["role"] != "assistant"]
        expected_answer = next(
            (m["content"].strip() for m in messages if m["role"] == "assistant"),
            "",
        )
        prior_answer = prior.get("chosen", [{}])[0].get("content", "")
        if prior.get("prompt") != expected_prompt or prior_answer != expected_answer:
            raise SystemExit(
                f"Partial output no longer matches this run at row {i + 1}; "
                f"remove {partial_path} to restart"
            )

    n_written = completed
    with open(partial_path, "a" if completed else "w") as f:
        for i, rec in enumerate(records):
            if i < completed:
                continue
            msgs = rec["messages"]
            # Separate the user turn from the ground-truth assistant response
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            gt_response = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if not gt_response:
                continue

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_len = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                   max_length=512)["input_ids"].shape[1]
            # Seed each record independently so an interrupted/resumed run is
            # byte-for-byte equivalent to an uninterrupted run.
            torch.manual_seed(args.seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + i)
            t0 = time.perf_counter()
            out_ids = manual_generate(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_prompt_tokens=512,
                device=device,
            )
            elapsed = time.perf_counter() - t0

            # Slice off the prompt tokens to get just the new generation
            gen_ids = out_ids[0, prompt_len:]
            rejected = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Conversational DPO schema. TRL will apply the same tokenizer chat
            # template used for SFT and generation, including the system turn.
            record = {
                "prompt":   prompt_msgs,
                "chosen":   [{"role": "assistant", "content": gt_response.strip()}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_written += 1
            print(f"[{i+1}/{len(records)}] {elapsed:.1f}s  chosen={len(gt_response)}  rejected={len(rejected)}", flush=True)

    partial_path.replace(out_path)
    print(f"\nDone — wrote {n_written} preference records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          default="data/train.jsonl")
    parser.add_argument("--out",            default="data/preferences.jsonl")
    parser.add_argument("--model",          default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--n",              type=int,   default=200, help="Number of preference pairs to generate")
    parser.add_argument("--max-new-tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    main(parser.parse_args())
