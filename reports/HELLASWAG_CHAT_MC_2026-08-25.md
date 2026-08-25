# HellaSwag chat-MC evaluation — 2026-08-25

## Result

Nemotron 3.5 Lightning answered 8,494 of 10,042 validation cases correctly:

- Accuracy: **84.58%**
- Wilson 95% confidence interval: **83.87%-85.28%**
- Valid-label rate: **100%**
- Request errors: **0**
- In-domain accuracy: **83.96%** (4,199/5,001)
- Zero-shot split accuracy: **85.20%** (4,295/5,041)

This result is named `hellaswag-chat-mc`. It uses a chat prompt that asks the
model to emit `A`, `B`, `C`, or `D`; it is not the standard continuation-
likelihood HellaSwag protocol and must not be compared directly with published
standard HellaSwag accuracy.

## Reproducibility gate

Two initial temperature-zero 100-case passes disagreed on 2 predictions. The
protocol was tightened with a fixed seed and captured top-label log
probabilities. Two clean passes from committed evaluator code then produced:

- 69/100 correct in both passes;
- 100/100 identical predictions;
- no errors or invalid labels.

The first 100 predictions of the subsequent full run also agreed exactly with
the seeded smoke pass. The full run used evaluator commit `e77533c` with a clean
worktree.

## Fixed protocol

- Model: `nvidia/nemotron-3.5-lightning`
- Endpoint: local NVIDIA NIM OpenAI-compatible chat completions
- Deployed checkpoint class: NVFP4
- Thinking: off
- Temperature: 0
- Top-p: 1
- Seed: 35,003,500
- Maximum output: 4 tokens
- Top log probabilities captured: 20
- Dataset SHA-256:
  `961fe851e978e4fcde703bf1fa8c44a770a8a6b7b4297a641a4dc197c23205f0`
- Run signature:
  `967ccd5a5b996f2e67eb9834b18c21c4d6dd45c556807d5361a5d57003292c78`

Strict scoring accepts only a response that normalizes to one capital label.
Raw responses, usage, timing, and top-token log probabilities were preserved
locally for every case.

## Runtime

Client-observed request latency was 113 ms mean, 116 ms median, and 124 ms p95.
The sum of per-request latency was 1,137.9 seconds. These numbers describe this
sequential quality run and are not a throughput benchmark.

## Interpretation

The evaluation transport, checkpoint/resume behavior, parsing, and strict
scoring are now validated at full-split scale. The score is a useful deployed-
model fingerprint, but it does not establish clean generalization, diagnose
our intended workload, or justify fine-tuning. The next decision-relevant step
is the deterministic Atlas tool-routing baseline.

Aggregate machine-readable results are in
[`hellaswag-chat-mc-2026-08-25.json`](hellaswag-chat-mc-2026-08-25.json). Raw
results remain ignored under
`outputs/evaluations/hellaswag-chat-mc/validation-full-01/` and can be resumed
or regenerated from the pinned dataset and evaluator.
