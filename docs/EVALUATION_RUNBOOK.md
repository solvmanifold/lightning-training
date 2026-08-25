# Evaluation runbook

## Start or resume the HellaSwag smoke

Start the existing inference service, wait for readiness, and run the first
checkpointed pass:

```bash
source ~/Dev/spark-inference/env.sh
model start nemotron-3.5-lightning
until curl -fsS http://localhost:8011/v1/health/ready >/dev/null; do sleep 5; done
python3 scripts/evaluate_hellaswag_chat.py --run-id smoke-01
```

Each completed case is appended and fsynced to
`outputs/evaluations/hellaswag-chat-mc/smoke-01/results.jsonl`. Running the same
command resumes at the first unfinished case. A second independent pass uses
`--run-id smoke-02`.

The fixed protocol uses temperature zero, seed `35003500`, thinking off, and
captures top-label log probabilities. Exact output labels remain the chat-MC
score; the log probabilities help diagnose near-boundary disagreements.

After two smoke passes have identical predictions, start or resume the full
validation run:

```bash
python3 scripts/evaluate_hellaswag_chat.py \
  --dataset data/derived/hellaswag/chat_mc_validation_10042.jsonl \
  --run-id validation-full-01
```

## Pause and release the GPU

From another terminal:

```bash
./scripts/release_evaluation_gpu.sh
```

The script asks the evaluator to stop after its current request, waits briefly
for the checkpoint, and stops the Lightning NIM. If the service must be stopped
immediately, `model stop nemotron-3.5-lightning` is still safe: at worst the
in-flight, uncheckpointed case is retried on resume.

Evaluation output is ignored by Git until a completed result is deliberately
promoted into a versioned report.
