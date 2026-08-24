# Evaluate and adapt Nemotron 3.5 Lightning on DGX Spark

This repository determines whether Nemotron 3.5 Lightning needs fine-tuning for
our workloads. The default adaptation order is prompting, constrained output,
tooling, and RAG. Fine-tuning is considered only after evaluation shows a
stable residual gap that those cheaper interventions do not solve.

Local inference already lives in `~/Dev/spark-inference`. Its verified NVIDIA
NIM serves the deployable NVFP4 checkpoint at
`http://localhost:8011/v1/chat/completions` as
`nvidia/nemotron-3.5-lightning`. See the [evaluation plan](docs/EVALUATION_PLAN.md).
The first experiment designed to give LoRA a fair opportunity is the
[behavioral-compression tool-routing study](docs/BEHAVIORAL_COMPRESSION_EXPERIMENT.md).

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

## Evaluation first

Start the existing inference service:

```bash
source ~/Dev/spark-inference/env.sh
model start nemotron-3.5-lightning
curl -s http://localhost:8011/v1/health/ready
```

Evaluation starts with a reproducible HellaSwag chat-MC harness smoke, then
moves to the decision-relevant workload suite: a versioned prompt ladder,
deterministic structured outputs, mock tool calls, and grounded QA. The
HellaSwag chat score is not presented as the standard likelihood-ranked
benchmark or used alone to justify fine-tuning. The locked workload test set
must never be used to create prompts, retrieval content, or training examples.

Prepare the pinned HellaSwag data and deterministic, LLM-free Atlas synthetic
smoke set without using the GPU:

```bash
python3 scripts/prepare_hellaswag.py
python3 scripts/generate_atlas_smoke.py
python3 scripts/verify_data.py
```

See the [data artifact notes](data/README.md) for provenance and layout.

## Fine-tuning baseline

The checked-in training configuration is retained as a conditional path, not
the next task. If evaluation justifies fine-tuning, it starts from NVIDIA's
single-GPU BF16 LoRA recipe for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

Validate its container wrapper with:

```bash
docker compose config --quiet
```

Do not run training while the NIM inference service is active; both compete for
the Spark's unified memory. Once the decision gate is met, stop inference and
run the unmodified NVIDIA baseline:

```bash
source ~/Dev/spark-inference/env.sh
model stop nemotron-3.5-lightning
export HF_TOKEN=your_token_if_required
docker compose run --rm trainer
```

Checkpoints are written to `outputs/checkpoints/`. Hugging Face downloads are
cached in the host's standard cache directory.

Before any LoRA run, we must also prove that its adapter can be served through
a practical Spark deployment path and evaluated under conditions comparable to
the current NVFP4 baseline. DPO, GRPO, and QLoRA remain out of scope.
