# Behavioral-compression LoRA experiment

## Question

Can a LoRA teach Nemotron 3.5 Lightning a stable, specialized tool-routing
behavior that otherwise requires a large instruction-and-example prefix?

This experiment is deliberately constructed so that fine-tuning could be the
right intervention. It is not constructed to guarantee that result. LoRA must
beat the strongest practical prompt, retrieval, constraint, and validation
baselines before we call it justified.

## Surrogate workload: Atlas workflow compiler

Create an invented service-operations domain named Atlas. A request contains a
natural-language instruction and deterministic account or resource state. The
model must produce one of:

- no action;
- a clarification request;
- one tool call; or
- an ordered plan of two or three tool calls.

The catalog contains 18-24 compact, executable mock tools with deliberately
overlapping purposes, such as suspending versus deleting an account, rotating
versus revoking a credential, changing a project role versus transferring
ownership, and previewing versus applying a configuration change. Arguments
include invented identifiers, enums, quantities, timestamps, and explicit
reason fields.

The executor, not the model, remains responsible for authorization, changing
policy, validation, idempotency, and side effects. The behavior being learned
is the stable mapping from language and state to the canonical action grammar.
That distinction matters: dynamic facts belong in context or tools, not model
weights.

## Why this could justify fine-tuning

The workload has all of the properties that favor behavioral fine-tuning:

- It is frequent and repeated.
- The action vocabulary and semantics are stable.
- Correct outputs have deterministic labels.
- The tools cannot decide when they should be called.
- Retrieval can supply examples but does not itself perform the mapping.
- A long prompt can plausibly solve the task, giving us a strong teacher and a
  measurable token-compression target.

The experiment does not use obscure facts or hide necessary information from
the prompt. That would manufacture a misleading advantage for fine-tuning.

## Dataset construction

Generate cases from a versioned latent specification rather than asking a
language model to invent labels. Each latent case determines:

- available state and permissions;
- requested outcome;
- canonical tool sequence and arguments, or the reason no action is valid;
- ambiguity requiring clarification;
- surface-language family, noise features, and difficulty tags.

Render the latent cases through controlled language templates. Include
paraphrases, shorthand, corrections, irrelevant details, conflicting requests,
missing arguments, explicit no-action cases, and near-neighbor tool choices.
Use invented names and identifiers throughout.

Proposed split:

| Split | Cases | Purpose |
| --- | ---: | --- |
| Train | 4,000 | LoRA training only |
| Development | 500 | Prompt, retrieval, and training selection |
| Locked test | 1,000 | Final decision only |
| Regression | 300 | General structured output and unrelated tool use |

Split on template families and scenario compositions, not individual rendered
examples. Hold out at least 25% of paraphrase families and multi-tool
compositions from training. Freeze the locked test artifact before prompt or
adapter optimization. Record generator revision, seed, manifest, and hashes.

Before generating all cases, manually audit 100 latent/rendered pairs and run
the generator's oracle through the same scorer used for model responses. The
oracle must score 100%.

## Compared conditions

Use identical tool schemas, model parameters, and test cases in every
applicable condition:

| ID | Model condition | Runtime context |
| --- | --- | --- |
| A | Base | Compact system prompt |
| B | Base | Full routing manual |
| C | Base | Full manual plus fixed few-shot examples |
| D | Base | Compact prompt plus retrieved training examples |
| E | Base | Best prompt with schema constraints, validation, and one retry |
| F | LoRA | Compact prompt, identical tools and validation policy |

Condition D is essential: it tests whether example retrieval provides the
benefit without changing weights. Condition E is the best non-training system,
not merely the best raw model response. Report first-attempt and post-retry
results separately.

Run the prompt baselines first and freeze them before LoRA training. Do not use
locked-test failures to create new prompt rules or training cases.

## Metrics

Primary metric: exact workflow success. A response succeeds only when the
decision to act, ordered tool names, normalized arguments, and stopping
behavior all match the oracle.

Also report:

- tool-selection accuracy;
- exact argument accuracy;
- clarification and no-action precision/recall;
- unsafe extra-action rate;
- invalid-output and retry rate;
- first-attempt versus post-validation success;
- total input tokens, including system prompt, examples, and tool schemas;
- output tokens;
- cold-cache and warm-cache latency and throughput;
- regression-suite deltas;
- bootstrap confidence intervals and paired per-case differences.

Measure caching rather than assuming its impact. Run cold-prefix, warm-prefix,
and mixed-concurrency trials. A shared prefix can reduce repeated prefill work,
but it does not make cache misses, context occupancy, or long-context decoding
free.

## Training protocol

1. Prove that the LoRA checkpoint can be served through a practical Spark path
   before committing to a full run.
2. Stop the NVFP4 NIM service and reproduce NVIDIA's unmodified 20-step
   single-GB10 LoRA recipe as the infrastructure smoke.
3. Convert only the training split to the required supervised chat/tool format.
4. Run a 16-case overfit test and verify save, reload, and identical scoring.
5. Train at least two LoRA seeds. Choose steps and checkpoints using the
   development set only.
6. Compare base and adapter in the same evaluation stack and precision where
   possible. Separately report any effect introduced by deployment or
   quantization differences.
7. Run the locked test once for the final decision.

HellaSwag remains an evaluation-harness smoke and regression signal. It is not
training data for this experiment.

## Precommitted decision rule

Recommend deploying LoRA only if it satisfies both the quality and operational
gates.

Quality gate:

- It improves exact workflow success over the best non-training condition by
  at least 5 percentage points, with a paired 95% confidence interval excluding
  zero; **or**
- it is within 1 percentage point of that condition while cutting total input
  tokens by at least 60%.
- Unsafe extra actions may not increase, invalid outputs must remain below 1%,
  and no regression category may fall by more than 2 percentage points.

Operational gate:

- The adapter has a reproducible serving path on Spark.
- Measured latency or throughput supports the claimed benefit under the actual
  cache-hit distribution.
- Adapter storage, loading, versioning, and rollback are acceptable.

If retrieved examples or the validated prompt meets the target without LoRA,
the experiment has still succeeded: it has shown that fine-tuning is not
operationally justified for this workload.

## Interpretation

- **F beats E on quality:** fine-tuning improves a stable behavior that runtime
  orchestration did not solve.
- **F matches E with materially fewer tokens:** fine-tuning acts as useful
  behavioral compression; deployment economics decide whether to use it.
- **D or E matches F:** prefer retrieval or prompting because behavior stays
  inspectable and easier to update.
- **All conditions fail the same cases:** revisit the interface, tool design, or
  available state instead of adding training examples blindly.
