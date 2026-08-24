# Evaluation-first adaptation plan

## Objective

Decide whether prompting, constrained generation, tools, or RAG can meet the
target workload before spending time or introducing risk through fine-tuning.
The evaluation target is the already-operational local NVIDIA NIM in
`~/Dev/spark-inference`, not a second inference stack in this repository.

The local service is:

- Endpoint: `http://localhost:8011/v1/chat/completions`
- Served model: `nvidia/nemotron-3.5-lightning`
- Checkpoint class: Nemotron 3.5 Lightning 30B-A3B NVFP4
- Default context profile: 262,144 tokens
- Reasoning control: `chat_template_kwargs.enable_thinking`
- Tool mode: OpenAI-compatible tools with automatic tool choice

The existing `spark-inference` benchmark establishes transport and performance
behavior. This project adds task-quality evaluation and decision criteria.

## Principles

1. Evaluate the deployable model configuration first.
2. Prefer the least stateful intervention that meets the requirement.
3. Keep prompts, retrieval indexes, tools, datasets, and scorers versioned.
4. Use deterministic scoring wherever possible; report manual judgment
   separately.
5. Lock the test set before optimizing prompts or building training data.
6. Preserve individual failures and categories, not only aggregate scores.
7. Never use the locked evaluation set as fine-tuning data.

## Evaluation suites

### HellaSwag harness smoke

Run HellaSwag first to validate the evaluation path against a familiar,
automatically scored task. This is a harness and reproducibility check, not the
evidence used to decide whether the model needs fine-tuning.

Protocol:

1. Pin the HellaSwag dataset revision, validation split, prompt text, and prompt
   hash.
2. Run the first 100 validation examples as a fixed smoke subset.
3. Use the deployed NVFP4 chat endpoint with thinking off, temperature zero,
   and a short output limit. Ask for exactly one label: `A`, `B`, `C`, or `D`.
4. Preserve every raw response and score a label only after strict, documented
   normalization. Count invalid or ambiguous responses separately.
5. Repeat the smoke subset and require identical normalized predictions before
   expanding to all 10,042 validation examples.

Report this result as `hellaswag-chat-mc`. It is not directly comparable to the
standard HellaSwag benchmark, which ranks candidate continuations by model
likelihood. Probe the serving endpoint for the log-probability or forced-
completion capabilities needed by a standard-compatible scorer; if supported,
implement and report that protocol separately.

HellaSwag may also have been present in, or exposed by material related to, the
model's broad pre-training corpus. Its score can expose harness bugs and provide
a repeatable model fingerprint, but cannot establish clean generalization or by
itself justify fine-tuning.

### Real-workload decision set

The primary suite will contain representative tasks from the intended product
or workflow. Each case must specify the user goal, permitted context, expected
properties, disallowed behavior, and acceptance threshold. Until this suite is
defined, benchmark results cannot justify fine-tuning.

Target size: 100-200 carefully reviewed cases, organized by capability and
difficulty rather than sampled indiscriminately.

### Deterministic structured output

Generate novel natural-language requests from latent specifications and score
the response against canonical JSON. Use invented entity names and values to
reduce memorization effects.

Metrics:

- JSON parse rate
- Schema-valid rate
- Canonical semantic exact match
- Field precision and recall
- Constraint satisfaction
- Unrequested-field rate

The initial locked test set should contain 512 examples. Prompt templates and
latent specifications must be disjoint from any future training set.

### Tool use

Expose deterministic mock tools with overlapping descriptions and typed
arguments. Cases cover selecting no tool, one tool, and a short sequence of
tools. Tool outputs contain generated facts unavailable in the prompt.

Metrics:

- Correct tool selection
- Exact normalized arguments
- Call ordering and stopping behavior
- Correct final answer from tool observations
- Fabricated call or argument rate

Target size: 150-250 cases.

### Grounded QA and RAG

Build a small versioned corpus containing synthetic policies, product facts,
and conflicting document revisions. Generated facts should not be recoverable
from model pre-training. Separate retrieval quality from answer quality.

Conditions:

- Closed-book baseline
- Oracle context supplied directly
- Retrieved context from the actual retrieval pipeline
- Retrieved context containing distractors or stale documents

Metrics:

- Retrieval recall at k
- Answer correctness
- Citation/evidence correctness
- Unsupported-claim rate
- Correct abstention when evidence is absent

Target size: 150-250 questions plus explicit unanswerable cases.

### Regression set

Maintain a small set of ordinary reasoning, coding, summarization, and safety
tasks. Its purpose is to detect damage from aggressive prompts or a future
adapter, not to claim broad benchmark leadership.

## Prompt ladder

Run the same locked cases through these conditions in order:

1. Minimal user prompt, thinking off.
2. Versioned system prompt, thinking off.
3. System prompt plus a small fixed set of examples.
4. Structured-output enforcement, if the local NIM capability probe confirms
   the required schema mode.
5. Tool-enabled prompt for action or calculation tasks.
6. Oracle context, then actual RAG, for knowledge-dependent tasks.
7. Repeat the best relevant conditions with thinking on.

Use temperature zero for deterministic task comparisons unless the endpoint or
task requires sampling. Sampling experiments must use fixed parameters and
multiple recorded trials.

## Artifact format

Each JSONL case should include:

```json
{
  "id": "structured-0001",
  "suite": "structured-output",
  "messages": [{"role": "user", "content": "..."}],
  "expected": {"...": "..."},
  "tools": [],
  "documents": [],
  "scorers": ["json_schema", "semantic_exact"],
  "tags": ["difficulty:1", "template:held-out-a"]
}
```

Every run records:

- Dataset and prompt-profile hashes
- `spark-inference` and evaluation-repository Git commits
- Served model ID and `/v1/models` response
- NIM image/profile when available
- Thinking, temperature, top-p, token limit, and tool settings
- Raw content, reasoning, tool calls, token usage, latency, and errors
- Per-case scores and aggregate confidence intervals

## Decision gate

Choose the simplest condition that meets the workload-specific acceptance
threshold. The exact thresholds belong to the product requirements, but they
must be written before examining final test results.

Do not fine-tune when failures are primarily:

- Missing or changing knowledge: use RAG.
- Deterministic computation or external actions: use tools.
- Output syntax: use constrained generation and validation.
- Ambiguous instructions: improve prompts or the interface.

Fine-tuning becomes a candidate only when all of the following hold:

1. A material, repeated behavioral gap remains under the best prompt/tool/RAG
   condition.
2. The gap is represented by enough licensed, high-quality examples.
3. A locked evaluation suite measures the gap directly.
4. The expected benefit outweighs regression and maintenance risk.
5. A practical adapter serving, merge, and quantization path exists on Spark.

The decision artifact is a short report recommending one of: no adaptation,
prompt-only, constrained output, tools, RAG, or LoRA. It must include failure
examples and operational cost, not just a headline score.

## Execution phases

### Phase 1: capability and reproducibility probe

- Start the existing Lightning service through `spark-inference`.
- Capture health, model metadata, NIM profile, and Git revisions.
- Confirm reasoning on/off behavior.
- Run the fixed 100-example HellaSwag chat-MC smoke twice and verify identical
  normalized predictions.
- If stable, run the full HellaSwag validation split and publish the result with
  its explicitly non-standard protocol label.
- Probe whether the endpoint exposes enough continuation likelihood information
  for a separate standard-compatible HellaSwag scorer.
- Probe schema-constrained output and tool-call request compatibility.
- Reuse the existing streaming benchmark for latency and throughput.

### Phase 2: deterministic evaluation harness

- Implement the JSONL case loader and OpenAI-compatible client.
- Preserve content, reasoning, and tool calls as separate fields.
- Implement exact structured-output and tool-call scorers.
- Generate and lock the initial structured-output and tool suites.

### Phase 3: prompt, tool, and RAG optimization

- Version prompt profiles rather than editing prompts in place.
- Evaluate the full prompt ladder on development cases.
- Freeze the selected conditions before running the locked test set.
- Add oracle-context and actual-retrieval comparisons.

### Phase 4: decision

- Publish the base-model evaluation report and failure taxonomy.
- Decide whether the residual failures justify fine-tuning.
- If not, retain the best prompt/tool/RAG configuration as the result.

### Phase 5: conditional LoRA experiment

Only after the gate passes:

- Verify adapter deployment compatibility with the Spark serving path.
- Stop the NIM service to free unified memory.
- Reproduce NVIDIA's unmodified 20-step BF16 LoRA recipe.
- Run a tiny overfit/save/reload test on separate training examples.
- Train against the identified failure class.
- Evaluate the adapter on the untouched suite and regression set under a
  serving configuration comparable to the NVFP4 baseline.

The first controlled candidate is the
[behavioral-compression experiment](BEHAVIORAL_COMPRESSION_EXPERIMENT.md), which
tests whether LoRA can replace a large tool-routing prompt while beating strong
prompting, retrieved examples, constrained output, and validation baselines.
