#!/usr/bin/env python3
"""
LoRA SFT for Nemotron-3.5-Lightning using NeMo 2.0 + HuggingFace PEFT.

Uses NeMo's Lightning Trainer for checkpointing/logging and HF PEFT for LoRA,
since Nemotron 3.5 Lightning is a dense model not yet in NeMo's Megatron model registry.
Run via docker-compose.training.yml — don't execute on the host directly.
"""
import argparse
import json
from pathlib import Path

# Must run before transformers/PEFT imports.
import gb10_patches  # noqa: F401

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from benchmarking import BenchmarkCallback


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_dataset(records: list[dict], tokenizer, max_len: int, use_chat_template: bool):
    """
    Each record must have:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    Or the simpler flat form:
      {"prompt": "...", "response": "..."}
    prepare.py normalises raw data into the messages format.
    """
    input_ids_list, label_ids_list = [], []

    for rec in records:
        if "messages" in rec and use_chat_template:
            # Use apply_chat_template with tokenize=True to avoid double-BOS from
            # tokenizing the string representation of special tokens a second time.
            ids = tokenizer.apply_chat_template(
                rec["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]
            # Mask the prompt; supervise only the final assistant turn
            prompt_ids = tokenizer.apply_chat_template(
                rec["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
            truncated_from_left = max(ids.size(0) - max_len, 0)
            if truncated_from_left:
                # Preserve the assistant answer at the end of long examples;
                # right truncation can otherwise produce all-masked labels.
                ids = ids[-max_len:]
            labels = ids.clone()
            prompt_tokens = max(len(prompt_ids) - truncated_from_left, 0)
            labels[: min(prompt_tokens, len(labels))] = -100

        else:
            text = (
                "".join(m["content"] for m in rec["messages"])
                if "messages" in rec
                else rec["prompt"] + rec["response"]
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"][0]
            labels = ids.clone()

        input_ids_list.append(ids)
        label_ids_list.append(labels)

    return input_ids_list, label_ids_list


def collate(batch, pad_id: int):
    input_ids = [b[0] for b in batch]
    labels    = [b[1] for b in batch]
    max_len = max(x.size(0) for x in input_ids)

    padded_ids, padded_lbls, masks = [], [], []
    for ids, lbls in zip(input_ids, labels):
        pad = max_len - ids.size(0)
        padded_ids.append(torch.nn.functional.pad(ids,  (0, pad), value=pad_id))
        padded_lbls.append(torch.nn.functional.pad(lbls, (0, pad), value=-100))
        masks.append(torch.nn.functional.pad(torch.ones_like(ids), (0, pad), value=0))

    return {
        "input_ids":      torch.stack(padded_ids),
        "labels":         torch.stack(padded_lbls),
        # pad_token is often eos_token for decoder-only models, so comparing
        # token IDs would incorrectly mask real EOS tokens.
        "attention_mask": torch.stack(masks).long(),
    }


# ── Lightning module ───────────────────────────────────────────────────────────

class LoRASFTModule(L.LightningModule):
    def __init__(self, model, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, _):
        out = self.model(**batch)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=False)
        return out.loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            out = self.model(**batch)
        self.log("val_loss", out.loss, prog_bar=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        # Drop the frozen base-model weights — keep only LoRA adapter params.
        # The full base model is ~64 GB; serialising it would OOM and is useless
        # for resuming since we can always reload the original base weights.
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if "lora_" in k or "modules_to_save" in k
        }

    def configure_optimizers(self):
        t = self.cfg["training"]
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=t["lr"],
            weight_decay=t["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=t["warmup_steps"],
            num_training_steps=t.get("scheduler_steps", t["max_steps"]),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    lc = cfg["lora"]
    tc = cfg["training"]
    dc = cfg["data"]
    oc = cfg["output"]
    L.seed_everything(tc.get("seed", 42), workers=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], trust_remote_code=mc["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model — BF16 or 4-bit NF4 (QLoRA), controlled by model.use_qlora
    use_qlora = mc.get("use_qlora", False)
    if use_qlora:
        print(f"Loading base model {mc['name']} in 4-bit NF4 (QLoRA)...")
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            mc["name"],
            quantization_config=bnb_config,
            trust_remote_code=mc["trust_remote_code"],
            use_cache=False,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        print(f"Loading base model {mc['name']} in BF16 ...")
        # device_map='cpu' avoids a transformers 4.46.3 sharded-safetensors bug;
        # Lightning moves the model to GPU during fit().
        base_model = AutoModelForCausalLM.from_pretrained(
            mc["name"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=mc["trust_remote_code"],
            use_cache=False,
            device_map="cpu",
        )

    # Nemotron 3.5 Lightning is a dense transformer; gradient checkpointing should work
    try:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")
    except Exception as e:
        print(f"Gradient checkpointing unavailable ({e}), continuing without it")

    # LoRA via HF PEFT
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

    # Datasets
    use_chat = dc.get("use_chat_template", True)
    train_records = load_jsonl(dc["train_path"])
    val_records   = load_jsonl(dc["val_path"])

    train_ids, train_lbls = make_dataset(train_records, tokenizer, tc["max_seq_len"], use_chat)
    val_ids,   val_lbls   = make_dataset(val_records,   tokenizer, tc["max_seq_len"], use_chat)

    pad_id = tokenizer.pad_token_id
    train_dl = DataLoader(
        list(zip(train_ids, train_lbls)),
        batch_size=tc["micro_batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        list(zip(val_ids, val_lbls)),
        batch_size=tc["micro_batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
    )

    # Lightning module
    lit_model = LoRASFTModule(model, cfg)

    # Callbacks + logger
    ckpt_cb = ModelCheckpoint(
        dirpath=oc["checkpoint_dir"],
        filename="{step}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=oc["save_top_k"],
        save_last=oc["save_last"],
        mode="min",
    )
    logger = TensorBoardLogger(save_dir=oc["log_dir"], name=oc["run_name"])

    # Trainer
    overfit_batches = tc.get("overfit_batches", 0)
    benchmark = BenchmarkCallback(
        args.metrics, token_budget=tc.get("token_budget", 0)
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=tc["precision"],
        max_epochs=-1,       # unlimited epochs; max_steps is the sole stop criterion
        max_steps=tc["max_steps"],
        accumulate_grad_batches=tc["grad_accumulation_steps"],
        gradient_clip_val=tc["grad_clip"],
        check_val_every_n_epoch=None,   # count batches across epochs; val_check_interval controls frequency
        val_check_interval=tc["val_every_n_steps"],
        limit_val_batches=tc["val_batches"],
        overfit_batches=overfit_batches,
        log_every_n_steps=1 if overfit_batches else 10,
        callbacks=[ckpt_cb, benchmark],
        logger=logger,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(lit_model, train_dl, val_dl)
    except Exception as exc:
        benchmark.write(trainer, lit_model, status="failed", error=str(exc))
        raise
    benchmark.write(trainer, lit_model)

    if args.skip_final_save:
        print("Skipping final adapter save (benchmark mode)")
        return

    # Save final LoRA adapter weights separately for easy merging.
    # Move to CPU first to free the ~64 GB of GPU memory before serialising.
    adapter_path = Path(oc["checkpoint_dir"]) / "lora_adapter_final"
    print("Moving model to CPU for adapter save...")
    try:
        model = model.cpu()
        torch.cuda.empty_cache()
        model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        print(f"LoRA adapter saved to {adapter_path}")
    except Exception as e:
        import sys, traceback
        print(f"FATAL: final adapter save failed: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"Lightning checkpoints are still available at {oc['checkpoint_dir']}/.", file=sys.stderr)
        print(f"Recover with: python3 scripts/ckpt_to_adapter.py --ckpt <best.ckpt> --out {adapter_path}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--metrics", default="", help="Write benchmark metrics JSON")
    parser.add_argument("--skip-final-save", action="store_true")
    main(parser.parse_args())