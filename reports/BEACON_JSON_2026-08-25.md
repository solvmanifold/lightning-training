# Beacon canonical JSON evaluation — 2026-08-25

## Result

The frozen full-prompt condition scored **486/512 (94.92%) semantic exact**
on the locked test set. All 512 responses parsed as JSON and passed the schema;
top-level field precision and recall were both 99.49%. The Wilson 95%
confidence interval for exact success is 92.66–96.51%.

This result justifies a controlled LoRA experiment, not deployment of a LoRA.
The base model already solves most cases with prompting, while the selected
condition consumes about 1,215 input tokens per request. A LoRA now has a fair,
precommitted target: preserve that behavior under the compact prompt, or close
the remaining stable semantic gap.

## Protocol

- Model: `nvidia/nemotron-3.5-lightning`, served by the local NVIDIA NIM.
- Dataset: deterministic `beacon-job-v1`; no LLM generated requests or labels.
- Splits: 2,048 train, 256 development, and 512 locked test cases.
- Split construction: disjoint surface-language families with mechanically
  derived canonical JSON oracles.
- Decoding: thinking off, temperature 0, top-p 1, seed 35003501.
- Prompt examples: four fixed cases from the training split only.
- Scoring: exact canonical semantics after JSON parsing plus independent schema
  and field-level checks.
- Test SHA-256:
  `3b7f8129eb9f4b9c6779f6f0ea3e2c200ee97688d956677dd9e7e379c9f2ee9a`.

The final prompt was frozen and pushed at commit `bba175a` before the test was
opened. The locked test was then evaluated once as `final-test-01`; no prompt
was edited after observing it. Raw requests, responses, timings, and metadata
remain ignored under `outputs/evaluations/beacon-canonical-json/`.

## Prompt ladder

The prompt ladder used the same first 64 development cases. These scores were
used to select the final condition and are not independent test estimates.

| Condition | Semantic exact | Schema valid | Field precision | Mean input tokens |
| --- | ---: | ---: | ---: | ---: |
| Compact | 19/64 (29.69%) | 79.69% | 91.41% | 253 |
| Manual | 48/64 (75.00%) | 100% | 97.50% | 515 |
| Manual + fixed examples | 53/64 (82.81%) | 100% | 98.28% | 789 |
| Schema-constrained examples | 51/64 (79.69%) | 100% | 97.97% | 789 |
| Optimized | 57/64 (89.06%) | 100% | 98.91% | 1,025 |
| Schema-constrained optimized | 58/64 (90.63%) | 100% | 99.06% | 1,025 |
| Final | 62/64 (96.88%) | 100% | 99.69% | 1,216 |
| Schema-constrained final | 62/64 (96.88%) | 98.44% | 99.69% | 1,216 |

The unconstrained final condition was selected because it tied the constrained
condition on exactness and had better schema validity. NIM's JSON grammar did
not accept the schema's `uniqueItems` keyword, so constrained serving omitted
only that keyword while the independent validator retained it.

The frozen condition then scored 252/256 (98.44%) on full development and
486/512 (94.92%) on locked test:

| Split | Exact | JSON parse | Schema valid | Field precision/recall | Mean input tokens | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 252/256 (98.44%) | 100% | 100% | 99.84% | 1,216 | 1.128 s |
| Locked test | 486/512 (94.92%) | 100% | 100% | 99.49% | 1,215 | 1.137 s |

Test median latency was 1.136 seconds and p95 was 1.253 seconds. These are
single-request observations, not a cold-cache/warm-cache throughput study.

## Failure taxonomy

The 26 test failures were narrow and deterministic rather than structural:

| Error | Cases |
| --- | ---: |
| Snapshot `compression=false` rendered as `true` | 20 |
| Restore `compression=true` rendered as `false` | 3 |
| Default `urgency=normal` rendered as `low` | 2 |
| Notification addresses emitted in non-canonical order | 1 |

Snapshot was the weak action at 108/128 exact (84.38%). Replicate scored
127/128, restore 125/128, and retire 126/128. No failure involved malformed
JSON, an extra field, or an endpoint error.

The test errors are now observed and must not influence training examples,
prompt changes, hyperparameters, or checkpoint selection. Any LoRA is trained
only on the already-frozen 2,048-case training split and selected only on
development. Its configuration must be committed before its single test run.

## LoRA infrastructure and overfit gates

The pinned NeMo AutoModel 26.06 container completed NVIDIA's unmodified
20-step single-GPU recipe on HellaSwag. It loaded 61.31 GB of BF16 weights,
trained 7,072,256 LoRA parameters (0.02% of 32,920,335,424 total parameters),
and saved both adapter and resumable optimizer state. Steady-state training was
about nine seconds per optimizer step for the Beacon sequence lengths and used
about 62.5 GiB of unified GPU memory.

A separate 16-example Beacon overfit run used only the first 16 training cases.
Loss fell from 1.2832 to 0.1807 over 20 optimizer steps; validation on those
same examples fell to 0.1459. Served through NVIDIA NIM's NVFP4 LoRA profile,
the compact-prompt base condition scored 4/16 exact while the adapter scored
16/16 exact with 100% JSON parse, schema validity, and field accuracy. A fresh
NIM process reloaded the saved adapter and repeated 16/16 exactly.

The restart test also exposed a GB10 serving limitation: NIM 2.0.9-variant's
compiled LoRA path failed once during startup with `cudaErrorIllegalInstruction`
inside `lora_shrink`. Adding vLLM's `--enforce-eager` flag made clean adapter
reload reliable, at the cost of slower inference. This workaround and its
performance cost remain part of the operational deployment gate.

## Decision gate

Proceed with the NVIDIA-recipe infrastructure smoke and then a small LoRA
experiment. Do not call fine-tuning successful unless the adapter satisfies
the precommitted behavioral-compression rule:

- improve exact success over 94.92% by at least 5 percentage points with a
  paired 95% confidence interval excluding zero; **or**
- finish within 1 percentage point of 94.92% while reducing total input tokens
  by at least 60% relative to the 1,215-token final prompt.

The compact development prompt averaged 253 input tokens, about 79% fewer than
the final prompt, so the compression target is technically plausible. Prefix
KV caching may reduce prefill cost; therefore the operational gate still
requires measured cold-prefix, warm-prefix, and representative mixed-load
latency/throughput. Adapter serving, rollback, invalid-output rate below 1%,
and regression limits remain mandatory.

Aggregate metrics are in
[`beacon-json-2026-08-25.json`](beacon-json-2026-08-25.json).
