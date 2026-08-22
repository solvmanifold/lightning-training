#!/usr/bin/env python3
"""Exercise a real bitsandbytes NF4 forward/backward pass on the active GPU."""
import argparse
import json
import platform
import time
from pathlib import Path

import gb10_patches  # noqa: F401
import torch


def write_result(path: str, result: dict):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)


def main(args):
    result = {
        "status": "failed",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        import bitsandbytes as bnb

        result.update({
            "bitsandbytes_version": getattr(bnb, "__version__", "unknown"),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        })
        layer = bnb.nn.Linear4bit(
            32,
            16,
            bias=False,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
        ).to("cuda")
        inputs = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        started = time.perf_counter()
        output = layer(inputs)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        result.update({
            "status": "ok",
            "elapsed_seconds": time.perf_counter() - started,
            "output_is_finite": bool(torch.isfinite(output).all().item()),
        })
        if not result["output_is_finite"]:
            raise RuntimeError("NF4 preflight produced non-finite output")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_result(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    write_result(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    main(parser.parse_args())
