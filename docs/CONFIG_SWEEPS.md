# Designing Nano3 configuration sweeps

## Experimental contract

These runs answer systems and optimization questions, not whether one adapter
produces a generally better assistant. Keep the data, seed, tier, and token
budget fixed while changing one configuration dimension whenever possible.

The most useful outcomes are:

1. Does the configuration complete forward, backward, optimizer, validation,
   and cleanup without unsupported kernels, NaNs, OOMs, or save failures?
2. How many adapter parameters does it introduce?
3. What throughput and memory cost does it impose?
4. Does loss move on the smoke/overfit workload?
5. At an equal token budget, how does its loss trajectory compare?

## Tiers

Start every new configuration at `smoke`. Promote successful runs to `overfit`,
then select only informative configurations for `compare`. This keeps failures
cheap on a 30B model.

The runner checks Linux `MemAvailable` before starting a container. Its default
floors are 70 GiB for BF16 and 40 GiB for QLoRA, chosen to avoid competing with
other unified-memory workloads on the Spark. A preflight refusal is recorded in
the report and is retried normally the next time the run command is issued.

The compare tier's 250k-token default is intentionally modest. Increase
`tiers.compare.training.token_budget` in the manifest when longer curves are
worth the wall time. `max_steps` is only a safety ceiling; `scheduler_steps`
controls the cosine schedule independently.

## Changing the matrix

Each entry in `experiments/sweep.yaml` is a recursive override of
`configs/sft.yaml` plus the selected tier. For example:

```yaml
- id: r8-mamba-lr1e4
  description: Mamba-only learning-rate probe
  lora:
    r: 8
    alpha: 16
    target_modules: [in_proj, out_proj]
  training:
    lr: 1.0e-4
```

Good single-variable comparisons include:

- Rank at a fixed target-module set.
- Target modules at fixed rank/alpha.
- Learning rate at fixed adapter shape.
- Sequence length at a fixed token budget.
- Accumulation at a fixed effective batch size.
- BF16 LoRA versus preflight-approved QLoRA.

Avoid interpreting SFT versus DPO/GRPO loss values as directly comparable: the
objectives have different scales. Those runs are most useful for compatibility,
throughput, memory, and stability comparisons.

## Reading reports

`tokens_per_second` measures non-padding input tokens across complete training
batches. `mean_batch_seconds` includes forward, backward, accumulation, and any
optimizer work for that microbatch. CUDA memory comes from PyTorch's allocator;
RSS captures the process-level memory footprint on the unified-memory system.

For failed runs, inspect `run_result.json` first and then `train.log`. Known
failure classes are normalized in the CSV so unsupported kernels and OOMs can
be compared across the matrix.
