#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for Nemotron-3.5-Lightning,
starting from a LoRA adapter produced by SFT.

Uses trl.GRPOTrainer with a ROUGE-L reward function: completions that score
higher ROUGE-L against the reference answer get higher reward. No reward
model needed — rule-based reward keeps this toy demo fast and deterministic.

Data format (one JSON object per line):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
The assistant turn is the reference answer used for reward computation.

Run via docker-compose.training.yml.
"""
import argparse
import inspect
import json

# Must run before transformers/PEFT/TRL imports.
import gb10_patches  # noqa: F401

from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# Module-level scorer so the reward fn doesn't re-create it every batch
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_reward(completions, references=None, **kwargs):
    """
    GRPO reward: ROUGE-L F1 between the completion and the reference answer.
    Completions come as a list of strings (already decoded); references is
    passed through GRPOTrainer from the dataset column.
    """
    if references is None:
        return [0.0] * len(completions)
    rewards = []
    for c, r in zip(completions, references):
        if isinstance(c, list):
            c = "\n".join(
                str(message.get("content", ""))
                for message in c if isinstance(message, dict)
            )
        if not c or not r:
            rewards.append(0.0)
            continue
        try:
            score = _ROUGE.score(r, c)["rougeL"].fmeasure
        except Exception:
            score = 0.0
        rewards.append(float(score))
    return rewards


def load_prompts(path: str) -> Dataset:
    records = [json.loads(l) for l in open(path) if l.strip()]
    rows = []
    for rec in records:
        msgs = rec["messages"]
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
        if not user_msg or not asst_msg:
            continue
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        rows.append({
            "prompt": prompt_msgs,
            "reference": asst_msg["content"],
        })
    return Dataset.from_list(rows)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    lc = cfg["lora"]
    tc = cfg["training"]
    dc = cfg["data"]
    oc = cfg["output"]

    tokenizer = AutoTokenizer.from_pretrained(mc["name"], trust_remote_code=mc["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model {mc['name']} in BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        mc["name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=mc["trust_remote_code"],
        use_cache=False,
        device_map="cpu",
    )
    try:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")
    except Exception as e:
        print(f"Gradient checkpointing unavailable ({e})")

    init_adapter = mc.get("init_adapter")
    if init_adapter:
        print(f"Initializing from SFT adapter at {init_adapter}...")
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        print("No init adapter — fresh LoRA")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lc["r"], lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
        )
        model = get_peft_model(base_model, peft_cfg)

    model.print_trainable_parameters()

    # ── Patch prepare_inputs_for_generation ──
    # GRPOTrainer does rollouts via model.generate(), and the bundled
    # modeling_nemotron_h.prepare_inputs_for_generation chokes on
    # cache_position=None which newer transformers' _prefill passes during
    # prefill. Wrap it to compute cache_position from past length when missing.
    import functools
    _orig_prep = model.prepare_inputs_for_generation
    @functools.wraps(_orig_prep)
    def _safe_prep(*args, **kwargs):
        if kwargs.get("cache_position") is None:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is not None:
                past_kv = kwargs.get("past_key_values")
                past_len = 0
                if past_kv is not None:
                    if hasattr(past_kv, "get_seq_length"):
                        past_len = past_kv.get_seq_length()
                    elif len(past_kv) > 0 and isinstance(past_kv[0], tuple) and len(past_kv[0]) > 0:
                        past_len = past_kv[0][0].shape[2]
                kwargs["cache_position"] = torch.arange(
                    past_len, past_len + input_ids.shape[1], device=input_ids.device
                )
        return _orig_prep(*args, **kwargs)
    model.prepare_inputs_for_generation = _safe_prep
    print("Patched prepare_inputs_for_generation to handle cache_position=None")

    print(f"Loading GRPO prompts from {dc['train_path']}...")
    train_ds = load_prompts(dc["train_path"])
    print(f"  {len(train_ds)} prompts")
    if tc.get("max_prompts", 0) and tc["max_prompts"] < len(train_ds):
        train_ds = train_ds.select(range(tc["max_prompts"]))
        print(f"  Truncated to {len(train_ds)} prompts (max_prompts)")

    grpo_config = GRPOConfig(
        output_dir=oc["checkpoint_dir"],
        num_train_epochs=tc.get("num_train_epochs", 1),
        max_steps=tc.get("max_steps", -1),
        per_device_train_batch_size=tc["micro_batch_size"],
        gradient_accumulation_steps=tc["grad_accumulation_steps"],
        learning_rate=tc["lr"],
        weight_decay=tc.get("weight_decay", 0.0),
        warmup_steps=tc.get("warmup_steps", 0),
        lr_scheduler_type="cosine",
        max_grad_norm=tc.get("grad_clip", 1.0),
        bf16=True,
        logging_steps=tc.get("logging_steps", 5),
        save_strategy="steps",
        save_steps=tc.get("save_every_n_steps", 50),
        save_total_limit=oc.get("save_top_k", 2),
        report_to="tensorboard",
        logging_dir=f"{oc['log_dir']}/{oc['run_name']}",
        # GRPO-specific
        num_generations=tc.get("num_generations", 4),    # K samples per prompt
        max_prompt_length=tc.get("max_prompt_len", 256),
        max_completion_length=tc.get("max_completion_len", 256),
        temperature=tc.get("temperature", 0.9),
        beta=tc.get("beta", 0.04),                       # KL coefficient
        gradient_checkpointing=False,                    # already enabled above
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model,
        reward_funcs=rouge_l_reward,
        args=grpo_config,
        train_dataset=train_ds,
    )
    processing_parameter = (
        "processing_class"
        if "processing_class" in inspect.signature(GRPOTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[processing_parameter] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)

    print("Starting GRPO training...")
    trainer.train()

    adapter_path = Path(oc["checkpoint_dir"]) / "lora_adapter_final"
    print(f"Saving final adapter to {adapter_path}...")
    try:
        model = model.cpu()
        torch.cuda.empty_cache()
        model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        print(f"Saved to {adapter_path}")
    except Exception as e:
        import sys, traceback
        print(f"FATAL: final adapter save failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    main(parser.parse_args())