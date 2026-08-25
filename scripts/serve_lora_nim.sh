#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 ADAPTER_DIR ADAPTER_ID [PORT] [CONTAINER_NAME]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage

adapter_dir=$(readlink -f "$1")
adapter_id=$2
port=${3:-8012}
container_name=${4:-lightning-lora-eval}

[[ -d "$adapter_dir" ]] || { echo "adapter directory not found: $adapter_dir" >&2; exit 1; }
[[ -f "$adapter_dir/adapter_config.json" ]] || { echo "missing adapter_config.json" >&2; exit 1; }
[[ -f "$adapter_dir/adapter_model.safetensors" ]] || { echo "missing adapter_model.safetensors" >&2; exit 1; }
[[ "$adapter_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid adapter ID: $adapter_id" >&2; exit 1; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 1; }
[[ "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || {
  echo "invalid container name: $container_name" >&2
  exit 1
}

image=nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant
profile=15766f03e594ee013d9328b909c1b4bbd432f7f7fcdb5503372b354e34d14662

if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi

# AutoModel writes adapter weights mode 0600 as root. NIM runs unprivileged and
# needs read access; use the already-pinned image to make only these files
# readable without changing ownership.
docker run --rm --user root --entrypoint bash \
  -v "$adapter_dir:/adapter" \
  "$image" -lc 'chmod a+r /adapter/adapter_config.json /adapter/adapter_model.safetensors'

docker run -d \
  --name "$container_name" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "$port:8000" \
  -v nim-spark_nim-cache:/opt/nim/.cache \
  -v "$adapter_dir:/adapters/$adapter_id:ro" \
  -e NIM_HTTP_API_PORT=8000 \
  -e NIM_MODEL_PROFILE="$profile" \
  -e NIM_PEFT_SOURCE=/adapters \
  -e NIM_MODEL_NAME=nvidia/nemotron-3.5-lightning \
  -e NIM_SERVED_MODEL_NAME=nvidia/nemotron-3.5-lightning \
  -e 'NIM_PASSTHROUGH_ARGS=--enforce-eager --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder' \
  "$image"

echo "starting $container_name with adapter $adapter_id on http://localhost:$port/v1"
echo "readiness: curl -sf http://localhost:$port/v1/health/ready"
echo "stop:      docker rm -f $container_name"
