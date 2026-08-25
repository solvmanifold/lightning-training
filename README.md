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
The [evaluation runbook](docs/EVALUATION_RUNBOOK.md) documents checkpointed
execution, resumption, and the one-command GPU release procedure.
The first completed result is the
[2026-08-25 HellaSwag chat-MC report](reports/HELLASWAG_CHAT_MC_2026-08-25.md).
The first decision-relevant workload result is the
[2026-08-25 Atlas tool-routing smoke report](reports/ATLAS_SMOKE_2026-08-25.md).
The deterministic canonical-JSON experiment uses the generated Beacon dataset
under `data/synthetic/beacon_json/` and the checkpointed
`scripts/evaluate_beacon_json.py` runner. Its frozen base-model result and LoRA
decision gate are in the
[2026-08-25 Beacon canonical JSON report](reports/BEACON_JSON_2026-08-25.md).

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

Run the trainer as a named container so it can be paused safely:

```bash
docker compose run -d --name lightning-training-baseline trainer
./scripts/pause_training.sh lightning-training-baseline
```

The pause command sends SIGTERM and lets NeMo AutoModel finish the current
optimizer step and checkpoint; it never escalates to SIGKILL. AutoModel
restores the latest compatible checkpoint when the trainer is started again.

Before any LoRA run, we must also prove that its adapter can be served through
a practical Spark deployment path and evaluated under conditions comparable to
the current NVFP4 baseline. DPO, GRPO, and QLoRA remain out of scope.

That gate now passes: the 16-case Beacon adapter trained, saved, reloaded
through NVIDIA NIM, and scored 16/16 exact. NIM 2.0.9-variant requires
`--enforce-eager` for reliable LoRA restarts on this GB10; see the
[Beacon report](reports/BEACON_JSON_2026-08-25.md) for the result and serving
limitation.

The two precommitted full candidates differ only by seed. Start the first as a
named, pausable container:

```bash
docker compose run -d --name lightning-training-beacon-seed1111 trainer \
  automodel /workspace/configs/beacon_lora_seed1111.yaml --nproc-per-node 1
./scripts/pause_training.sh lightning-training-beacon-seed1111
```

AutoModel checkpoints every 64 optimizer steps and restores the latest
compatible checkpoint when the same command is launched again. Train and
checkpoint selection use only the training and development splits; the locked
test is opened once after the candidate and configuration are frozen.

Serve a completed checkpoint with the verified eager-mode LoRA profile:

```bash
./scripts/serve_lora_nim.sh \
  outputs/checkpoints/beacon-lora-seed1111/epoch_0_step_255/model \
  beacon-seed1111-step255
```

The launcher uses the shared offline NIM cache, exposes port 8012 by default,
and refuses to replace an existing container. Stop its exact container with
`docker rm -f lightning-lora-eval` after evaluation.
