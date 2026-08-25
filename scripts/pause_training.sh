#!/usr/bin/env bash
set -euo pipefail

container_name="${1:-lightning-training-baseline}"

if ! docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "Training container does not exist: $container_name" >&2
    exit 1
fi

running="$(docker container inspect --format '{{.State.Running}}' "$container_name")"
if [[ "$running" != "true" ]]; then
    echo "Training container is already stopped: $container_name"
    exit 0
fi

echo "Sending SIGTERM to $container_name; AutoModel will finish the current step and checkpoint"
docker kill --signal=TERM "$container_name" >/dev/null

# Do not use `docker stop`: its timeout ends in SIGKILL and can interrupt a
# large checkpoint. Poll for up to 30 minutes without escalating the signal.
for _ in $(seq 1 360); do
    running="$(docker container inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)"
    if [[ "$running" != "true" ]]; then
        exit_code="$(docker container inspect --format '{{.State.ExitCode}}' "$container_name")"
        echo "Training stopped with exit code $exit_code"
        exit 0
    fi
    sleep 5
done

echo "Training is still checkpointing after 30 minutes; it was not force-killed" >&2
exit 1
