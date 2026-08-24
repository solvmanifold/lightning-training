# Nemotron 3.5 Lightning fine-tuning on DGX Spark

This repository starts from NVIDIA's single-GPU BF16 LoRA recipe for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

The first milestone is deliberately narrow: reproduce NVIDIA's 20-step recipe
on one DGX Spark GB10 before changing the dataset, optimizer, LoRA settings, or
training framework.

## Provenance

The baseline configuration is adapted from
[NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)'s
[`nemotron_nano_v3_5_lightning_singlegpu_lora.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/060cc495ac23350d4882f67ddf96ba663dd3696c/examples/llm_finetune/nemotron/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml)
at upstream commit `060cc495ac23350d4882f67ddf96ba663dd3696c`.

It retains the important architecture-specific choices from that recipe:

- NeMo AutoModel's native Nemotron-H implementation.
- Single-GPU FSDP2 on GB10.
- BF16 base weights with memory-efficient LoRA.
- Activation checkpointing.
- Transformer Engine attention and `torch_mm` experts.
- Mamba `out_proj` exclusion.
- Repeated MTP training with a scaled auxiliary loss.

The container follows NVIDIA's
[`nemo-automodel` DGX Spark guidance](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nemo-fine-tune).

## Prerequisites

- DGX Spark with current drivers and Docker GPU access.
- Enough free unified memory for the run; stop local inference services first.
- Hugging Face access to the checkpoint and `HF_TOKEN` exported if required.

## Validate the wrapper

```bash
docker compose config --quiet
```

## Run the NVIDIA baseline

```bash
export HF_TOKEN=your_token_if_required
docker compose run --rm trainer
```

Checkpoints are written to `outputs/checkpoints/`. Hugging Face downloads are
cached in the host's standard cache directory.

## Development order

Do not change the baseline until the 20-step run completes and produces a
consolidated adapter checkpoint.

After that milestone:

1. Add a tiny local dataset adapter and one-batch overfit test.
2. Add held-out evaluation and reproducible seeds.
3. Measure throughput and peak unified-memory use.
4. Add controlled LoRA experiments.
5. Consider QLoRA only after BF16 LoRA is established.

DPO and GRPO are intentionally out of scope. NVIDIA's published Lightning RL
recipes target multi-node GPU clusters, not a single DGX Spark.
