#!/usr/bin/env python3
"""
Convert a Lightning .ckpt file (containing only LoRA adapter weights) into a
PEFT adapter directory that can be loaded with PeftModel.from_pretrained.

train.py's on_save_checkpoint filter strips base-model weights from the state
dict, so Lightning checkpoints contain only LoRA params. This script reloads
the base model, applies a fresh LoRA, injects the saved weights, and calls
save_pretrained() to produce a standard PEFT adapter format.

Usage (inside the training container):
  python3 scripts/ckpt_to_adapter.py \\
    --ckpt   outputs/sft/checkpoints/step=212-val_loss=1.2030.ckpt \\
    --out    outputs/sft/checkpoints/lora_adapter_final \\
    --config configs/sft.yaml
"""
import argparse
from pathlib import Path

# Must run before transformers/PEFT imports.
import gb10_patches  # noqa: F401

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    lc = cfg["lora"]

    print(f"Loading base model {mc['name']} in BF16 on CPU...")
    # Load straight to CPU — we don't need GPU for this conversion, and it
    # avoids competing with any training that might restart soon.
    base_model = AutoModelForCausalLM.from_pretrained(
        mc["name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=mc["trust_remote_code"],
        use_cache=False,
        device_map="cpu",
    )

    print("Wrapping in LoRA...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_cfg)

    print(f"Loading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_state = ckpt["state_dict"]
    print(f"  Checkpoint state_dict has {len(ckpt_state)} entries")

    # Lightning wraps the PEFT model in LoRASFTModule.model, so keys are like:
    #   "model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    # PEFT model expects:
    #   "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    stripped = {k.removeprefix("model."): v for k, v in ckpt_state.items()}

    print("Loading LoRA weights into PEFT model (strict=False)...")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys. First 3: {unexpected[:3]}")

    # 'missing' is huge because base-model weights aren't in the ckpt — that's fine.
    # But any LoRA keys in 'missing' are a problem.
    lora_missing = [k for k in missing if "lora_" in k or "modules_to_save" in k]
    if lora_missing:
        print(f"  ERROR: {len(lora_missing)} LoRA keys missing. First 3: {lora_missing[:3]}")
        raise SystemExit(1)

    lora_loaded = sum(1 for k in stripped if "lora_" in k or "modules_to_save" in k)
    print(f"  Loaded {lora_loaded} LoRA params successfully")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving adapter to {out}...")
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(mc["name"], trust_remote_code=mc["trust_remote_code"])
    tokenizer.save_pretrained(out)

    print(f"Done. Load with: PeftModel.from_pretrained(base_model, '{out}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Lightning .ckpt file")
    parser.add_argument("--out",    required=True, help="Output PEFT adapter directory")
    parser.add_argument("--config", default="configs/sft.yaml", help="Training config (for LoRA settings)")
    main(parser.parse_args())
