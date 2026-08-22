#!/usr/bin/env python3
"""
Download the BF16 base weights for Nemotron-3-Nano-30B-A3B.
Run via: docker compose -f docker-compose.training.yml run --rm downloader
"""
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} (BF16, ~60 GB) ...")
    path = snapshot_download(
        repo_id=MODEL_ID,
        ignore_patterns=["*.bin"],   # prefer .safetensors
    )
    print(f"Downloaded to {path}")
