# Atlas tool-routing smoke — 2026-08-25

## Result

Prompting and runtime validation solved the six-tool smoke workload well enough
that this dataset does not currently justify fine-tuning.

| Condition | Development exact | Repeat agreement | Mean prompt tokens/case |
| --- | ---: | ---: | ---: |
| Compact prompt, run 1 | 27/40 (67.5%) |  | 1,210 |
| Compact prompt, run 2 | 25/40 (62.5%) | 90.0% | 1,210 |
| Routing manual, run 1 | 36/40 (90.0%) |  | 1,548 |
| Routing manual, run 2 | 36/40 (90.0%) | 97.5% | 1,548 |
| Manual plus two fixed training examples, run 1 | 40/40 (100%) |  | 1,906 |
| Manual plus two fixed training examples, run 2 | 40/40 (100%) | 100% | 1,906 |

The examples came from the training split and demonstrated only clarification
and no-action formatting. Development and test cases were not used as examples.

## Frozen smoke test

After the few-shot condition was committed, it scored 37/40 (92.5%) on the
untouched 40-case test split, with zero unsafe extra actions. The three failures
contained the correct function and arguments as plain text rather than NIM
structured tool calls:

- two `transfer_project_ownership` responses;
- one `delete_account` response.

This is an observable interface/format failure. A generic validator and one
retry reached 40/40 on a subsequent diagnostic run, using two retries. Because
that run reused an already-observed test split, it is engineering evidence, not
a new locked-test result.

## Regression interpretation

All 24 informational regression cases produced zero tool calls, so unintended
action rate was 0%. Only 6/24 emitted the exact artificial `NO_ACTION` sentinel;
the other 18 answered the informational request directly, for example by
summarizing the supplied state.

Treating those helpful no-tool answers as routing failures would manufacture a
fine-tuning advantage through the scorer. A generic format retry was also the
wrong intervention: it often converted a useful answer into `CLARIFY` while
still making no tool call. The next protocol should either route informational
requests outside the action router or accept free-form content as a valid
no-tool outcome.

## What the smoke established

- The NIM's structured tool-call path works for single and parallel calls.
- Prompt rules fixed stable delete and role-routing failures.
- Two fixed examples fixed ambiguity handling and made development predictions
  repeatable.
- A validator can detect textual function emission without oracle knowledge.
- Checkpoint, resume, and one-command GPU release are operational.

## Decision

Do not start LoRA from this smoke result. The best prompt is only about 696
input tokens longer than the compact condition, and it already closes the
measured behavioral gap. Prefix caching further weakens a token-compression
argument at this scale.

The next honest opportunity to justify fine-tuning is a larger Atlas v2 suite
with more tools, less templated language, held-out scenario compositions, and a
new locked test that is generated before further optimization. It should score
semantic no-tool behavior separately from exact response formatting. LoRA is a
candidate only if a stable residual gap remains after the few-shot, tool, and
validator conditions on that stronger suite.

Aggregate results are in
[`atlas-smoke-2026-08-25.json`](atlas-smoke-2026-08-25.json). Raw responses and
timings remain ignored under `outputs/evaluations/atlas-tool-routing/`.
