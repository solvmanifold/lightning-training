#!/usr/bin/env python3
"""
Generate assistant responses from a fine-tuned (or base) model on a list of
test prompts. Used by the weekend eval pipeline — each method produces one
responses file which is later scored by the judge.

Output format (one JSON object per line):
  {"prompt": "...", "response": "...", "ground_truth": "...", "model": "..."}

Usage (inside training container):
  # Fine-tuned adapter:
  python3 scripts/generate_responses.py \\
    --adapter outputs/sft/checkpoints/lora_adapter_final \\
    --test    data/val.jsonl \\
    --out     outputs/comparison/responses/sft.jsonl \\
    --name    sft

  # Base model (no adapter):
  python3 scripts/generate_responses.py \\
    --adapter "" \\
    --test    data/val.jsonl \\
    --out     outputs/comparison/responses/base.jsonl \\
    --name    base
"""
import argparse
import json
import time
from pathlib import Path

# Must run before transformers/PEFT imports.
import gb10_patches  # noqa: F401

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_hybrid_cache_class():
    """Locate HybridMambaAttentionDynamicCache from the dynamically-loaded
    modeling_nemotron_h module."""
    import sys
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "HybridMambaAttentionDynamicCache"):
            return mod.HybridMambaAttentionDynamicCache
    raise RuntimeError("HybridMambaAttentionDynamicCache not found — model must be loaded first")


@torch.no_grad()
def manual_generate(model, tokenizer, prompt_text, max_new_tokens, temperature,
                    max_prompt_tokens, device):
    """
    Custom KV-cached generation for Nemotron-H.

    The bundled modeling_nemotron_h.py is inconsistent between forward (uses
    `cache_params`) and prepare_inputs_for_generation (returns `past_key_values`),
    so model.generate() can't actually use the cache. We drive forward directly,
    constructing a HybridMambaAttentionDynamicCache and threading it through.
    """
    Cache = _get_hybrid_cache_class()
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_id = tokenizer.eos_token_id

    # The PEFT wrapper proxies attribute access; reach the underlying base model
    # for config and dtype.
    base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") \
           else (model.base_model if hasattr(model, "base_model") else model)
    cache = Cache(base.config, batch_size=input_ids.shape[0], dtype=base.dtype, device=device)
    # Patch cache attributes that modeling_nemotron_h.py references but the
    # cache class itself doesn't define correctly:
    #   - cache.conv_kernel_size is read but never set
    #   - cache.conv_states.device / cache.ssm_states.device are read on a
    #     list (which has no .device attribute)
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
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache_params=cache,
        cache_position=cache_position,
        use_cache=True,
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
            input_ids=next_token,
            attention_mask=attention_mask,
            cache_params=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]

    return output_ids


def main(args):
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.base_model} in BF16...")
    # device_map='cpu' avoids a transformers 4.46.3 sharded-safetensors bug;
    # we move the model to GPU after PEFT adapter is applied.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
        device_map="cpu",
    )

    if args.adapter:
        print(f"Loading adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("No adapter — running base model only")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Loading test prompts from {args.test}...")
    records = [json.loads(l) for l in open(args.test) if l.strip()]
    if args.n > 0:
        records = records[: args.n]
    print(f"  Processing {len(records)} prompts")

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
        print(f"Resuming from {partial_path} after {completed} prompts")
    if completed > len(records):
        raise SystemExit(
            f"Partial output has {completed} rows but this run has only {len(records)} prompts"
        )
    for i, prior in enumerate(partial_records):
        messages = records[i]["messages"]
        expected_prompt = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        expected_gt = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )
        if (
            prior.get("model") != args.name
            or prior.get("prompt") != expected_prompt
            or prior.get("ground_truth") != expected_gt
        ):
            raise SystemExit(
                f"Partial output no longer matches this run at row {i + 1}; "
                f"remove {partial_path} to restart"
            )

    with open(partial_path, "a" if completed else "w") as f:
        for i, rec in enumerate(records):
            if i < completed:
                continue
            msgs = rec["messages"]
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            gt = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            user_text = next((m["content"] for m in msgs if m["role"] == "user"), "")

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_len = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                   max_length=args.max_prompt_tokens)["input_ids"].shape[1]

            torch.manual_seed(args.seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + i)
            t0 = time.perf_counter()
            out_ids = manual_generate(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_prompt_tokens=args.max_prompt_tokens,
                device=device,
            )
            elapsed = time.perf_counter() - t0

            gen_ids = out_ids[0, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            record = {
                "prompt":       user_text,
                "response":     response,
                "ground_truth": gt,
                "model":        args.name,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"[{i+1}/{len(records)}] {elapsed:.1f}s  response_len={len(response)}", flush=True)

    partial_path.replace(out_path)
    print(f"\nDone — wrote {len(records)} responses to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",           default="", help="PEFT adapter path (empty = base model)")
    parser.add_argument("--base-model",        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--test",              default="data/val.jsonl")
    parser.add_argument("--out",               required=True)
    parser.add_argument("--name",              required=True, help="Model name tag in output records")
    parser.add_argument("--n",                 type=int,   default=0, help="Cap on number of prompts (0 = all)")
    parser.add_argument("--max-new-tokens",    type=int,   default=512)
    parser.add_argument("--max-prompt-tokens", type=int,   default=512)
    parser.add_argument("--temperature",       type=float, default=0.3)
    parser.add_argument("--seed",              type=int,   default=42)
    main(parser.parse_args())
