# Lightning-training configuration lab

This repository exercises `NVIDIA-Nemotron-3.5-Lightning` fine-tuning paths on a DGX Spark GB10 using NVIDIA's official NeMo 2.0 + NeMo-Run recipes. The primary outputs are engineering measurements—what runs, how quickly, with how much memory, and how different adapter configurations optimize—not claims about downstream model quality.

The synthetic Q&A corpus is intentionally just a stable workload. Its subject matter and objective are not the experiment.

## Configuration sweeps

The curated matrix in `experiments/sweep.yaml` covers:

- LoRA ranks 4, 8, and 16.
- Attention-only, attention+MoE, attention+Mamba, and all projection targets.
- A learning-rate sensitivity control.
- QLoRA NF4, guarded by a real GPU kernel preflight.

Inspect a tier without running anything:

```bash
python3 scripts/run_sweep.py plan --tier smoke
```

Run one experiment, then produce/update the report:

```bash
python3 scripts/run_sweep.py run --tier smoke --experiment r8-attn-moe
python3 scripts/run_sweep.py report --tier smoke
```

Run a group with shell-style ID filters:

```bash
python3 scripts/run_sweep.py run --tier overfit --experiment 'r8-*'
python3 scripts/run_sweep.py run --tier compare --experiment r4-attn --experiment r16-attn-moe
```

Completed runs are skipped unless `--force` is passed. Add `--build` to rebuild the standard training image before a sweep. On DGX Spark, the runner also refuses to start when available unified memory is below 70 GiB for BF16 LoRA or 40 GiB for QLoRA. This prevents a sweep from displacing an active inference workload; `--ignore-memory-preflight` is available for an intentional override.

## Three experiment tiers

| Tier | Purpose | Workload |
|------|---------|----------|
| `smoke` | Imports, model load, forward/backward, optimizer | 3 steps on 4 examples |
| `overfit` | Confirm gradient flow and convergence | 50 steps on one repeated batch |
| `compare` | Compare performance and learning dynamics | Fixed 250k input-token budget |

The compare tier uses a token budget because optimizer steps are not equivalent when sequence length, microbatch size, or accumulation changes.

Each run writes a generated config, log, status, and `metrics.json` under `outputs/sweeps/lightning-config-lab/{tier}/{experiment}/`. The tier report contains `results.csv` and `results.md`.

Collected measurements include:

- Input tokens/second and batch time.
- Optimizer steps and tokens processed.
- Train/validation loss and gradient norms.
- Trainable and total parameter counts.
- Peak CUDA allocated/reserved memory.
- Peak process RSS, which is important on unified-memory GB10 systems.
- Structured preflight or runtime failure reasons.

Benchmark runs skip final adapter serialization and disable checkpoints so save spikes do not dominate the measurement.

## QLoRA

QLoRA is no longer disabled based on the GPU name. Before every QLoRA sweep run, the runner performs a real bitsandbytes NF4 forward/backward pass:

```bash
docker compose -f docker-compose.training.yml run --rm smoketest \
  python3 scripts/preflight_qlora.py
```

This currently passes on the local GB10 with bitsandbytes 0.49.2. The shared `gb10_patches.py` also fixes the safetensors loader path that caused the original QLoRA attempt to fail before reaching a quantization kernel.

## GRPO isolation

GRPO has its own opt-in image because NeMo 25.02 requires an older TRL release that does not contain `GRPOConfig`.

```bash
docker compose -f docker-compose.training.yml --profile grpo build grpo-trainer
docker compose -f docker-compose.training.yml --profile grpo run --rm grpo-trainer \
  python3 scripts/preflight_grpo.py
docker compose -f docker-compose.training.yml --profile grpo run --rm grpo-trainer
```

The image uses NVIDIA's ARM64-capable PyTorch container and TRL 0.15.1. Nemotron 3.5 Lightning is a dense model; the unsupported Mamba CUDA fast path is not relevant. The image is not built during normal SFT/DPO workflows. See the [DGX Spark NGC guide](https://docs.nvidia.com/dgx/dgx-spark/ngc.html) and [TRL 0.15.1 GRPO documentation](https://huggingface.co/docs/trl/v0.15.1/en/grpo_trainer).

## Data and optional quality evaluation

For configuration experiments, keep `data/train.jsonl` and `data/val.jsonl` fixed. The four-record `data/smoke.jsonl` is deliberately trivial.

The repository also retains an optional source-grouped test/judge pipeline for any future quality experiment. It is separate from sweeps and does not block training benchmarks.

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q train.py train_dpo.py train_grpo.py \
  benchmarking.py gb10_patches.py data scripts tests
bash -n scripts/run_weekend.sh scripts/run_experiments.sh
docker compose -f docker-compose.training.yml config --quiet
docker compose -f docker-compose.training.yml --profile grpo config --quiet
```

## Quick start

```bash
# Build the training image
docker compose -f docker-compose.training.yml build training-base

# Plan a smoke-tier sweep
python3 scripts/run_sweep.py plan --tier smoke

# Run one smoke experiment
python3 scripts/run_sweep.py run --tier smoke --experiment r8-attn-moe

# View report
python3 scripts/run_sweep.py report --tier smoke
```