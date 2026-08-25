#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for pid_file in "$repo_root"/outputs/evaluations/hellaswag-chat-mc/*/active.pid; do
    [[ -f "$pid_file" ]] || continue
    eval_pid=$(<"$pid_file")
    if [[ -r "/proc/$eval_pid/cmdline" ]] \
        && tr '\0' ' ' < "/proc/$eval_pid/cmdline" | grep -q 'evaluate_hellaswag_chat.py'; then
        kill -INT "$eval_pid"
        echo "Asked evaluation process $eval_pid to checkpoint and pause"
    fi
done

# Give the in-flight request a short opportunity to finish and fsync its result.
for _ in {1..15}; do
    active=false
    for pid_file in "$repo_root"/outputs/evaluations/hellaswag-chat-mc/*/active.pid; do
        [[ -f "$pid_file" ]] || continue
        eval_pid=$(<"$pid_file")
        if kill -0 "$eval_pid" 2>/dev/null; then
            active=true
        fi
    done
    [[ "$active" == false ]] && break
    sleep 1
done

source /home/areite/Dev/spark-inference/env.sh
model stop nemotron-3.5-lightning
echo "Nemotron 3.5 Lightning stopped; rerun the evaluator with the same run ID to resume"
