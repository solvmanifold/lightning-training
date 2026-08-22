#!/usr/bin/env python3
"""Verify the isolated image exposes the TRL APIs required by train_grpo.py."""
import inspect
import json

import torch
import transformers
import trl
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from trl import GRPOConfig, GRPOTrainer
from transformers.utils.import_utils import is_mamba_2_ssm_available


sample = torch.ones(2, 4)
normalized = rmsnorm_fn(sample, torch.ones(4))


result = {
    "status": "ok",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "grpo_config": f"{GRPOConfig.__module__}.{GRPOConfig.__name__}",
    "trainer_accepts_processing_class": (
        "processing_class" in inspect.signature(GRPOTrainer.__init__).parameters
    ),
    "mamba_fast_path_available": is_mamba_2_ssm_available(),
    "portable_rmsnorm_finite": bool(torch.isfinite(normalized).all()),
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "compute_capability": (
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    ),
}
if not result["trainer_accepts_processing_class"] or not result["portable_rmsnorm_finite"]:
    result["status"] = "failed"
    result["error"] = "Unexpected GRPO API or invalid portable RMSNorm"
elif result["mamba_fast_path_available"]:
    result["status"] = "failed"
    result["error"] = "Mamba CUDA fast path must remain disabled on GB10"
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "ok" else 1)
