#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training for Nemotron-3.5-Lightning, starting
from a LoRA adapter produced by SFT.

Uses trl.DPOTrainer. No Lightning — DPOTrainer is built on HF Trainer.

Data format (one JSON object per line):
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Run via docker-compose.training.yml — don't execute on the host directly.
"""
import argparse
import json
from pathlib import Path

# Must run before transformers/PEFT/TRL imports.
import gb10_patches  # noqa: F401

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Answer clearly and concisely."
)


def load_preferences(path: str) -> Dataset:
    records = [json.loads(l) for l in open(path) if l.strip()]
    # Migrate legacy flat records in memory. The rejected response was sampled
    # under this chat template, so training must score it under the same prompt
    # representation instead of concatenating it to a raw user string.
    for rec in records:
        if isinstance(rec.get("prompt"), str):
            rec["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rec["prompt"]},
            ]
        if isinstance(rec.get("chosen"), str):
            rec["chosen"] = [{"role": "assistant", "content": rec["chosen"]}]
        if isinstance(rec.get("rejected"), str):
            rec["rejected"] = [{"role": "assistant", "content": rec["rejected"]}]
    return Dataset.from_list(records)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    lc = cfg["lora"]
    tc = cfg["training"]
    dc = cfg["data"]
    oc = cfg["output"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], trust_remote_code=mc["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model in BF16
    print(f"Loading base model {mc['name']} in BF16...")
    # Loading via device_map='cpu' avoids a metadata=None bug in transformers
    # 4.46.3's sharded safetensors loader on this model. Unified memory on
    # GB10 makes the eventual move-to-GPU trivial.
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

    # Start from SFT adapter if provided, else fresh LoRA
    init_adapter = mc.get("init_adapter")
    if init_adapter:
        print(f"Initializing from SFT adapter at {init_adapter}...")
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        print("No init adapter — starting from fresh LoRA")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
        )
        model = get_peft_model(base_model, peft_cfg)

    model.print_trainable_parameters()

    # Dataset
    print(f"Loading preference data from {dc['pref_path']}...")
    train_ds = load_preferences(dc["pref_path"])
    print(f"  {len(train_ds)} preference pairs")

    # DPO config
    dpo_config = DPOConfig(
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
        beta=tc.get("beta", 0.1),
        max_length=tc["max_seq_len"],
        max_prompt_length=tc.get("max_prompt_len", 256),
        bf16=True,
        logging_steps=tc.get("logging_steps", 5),
        save_strategy="steps",
        save_steps=tc.get("save_every_n_steps", 50),
        save_total_limit=oc.get("save_top_k", 2),
        report_to="tensorboard",
        logging_dir=f"{oc['log_dir']}/{oc['run_name']}",
        remove_unused_columns=False,
        gradient_checkpointing=False,  # already enabled above
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,     # Will use the PEFT model with adapter disabled as reference
        args=dpo_config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final adapter
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
    parser.add_argument("--config", default="configs/dpo.yaml")
    main(parser.parse_args())