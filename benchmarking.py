"""Low-overhead training metrics used by configuration sweeps."""
import json
import resource
import statistics
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


def _number(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().item()
    return float(value)


class BenchmarkCallback(Callback):
    def __init__(self, output_path: str | None, token_budget: int = 0):
        self.output_path = Path(output_path) if output_path else None
        self.token_budget = token_budget
        self.started_at = None
        self.batch_started_at = None
        self.batch_tokens = 0
        self.total_tokens = 0
        self.batch_times = []
        self.grad_norms = []
        self._written = False

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_fit_start(self, trainer, pl_module):
        self._sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.started_at = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self.batch_started_at = time.perf_counter()
        mask = batch.get("attention_mask")
        self.batch_tokens = int(mask.sum().item()) if mask is not None else 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        if self.batch_started_at is not None:
            self.batch_times.append(time.perf_counter() - self.batch_started_at)
        self.total_tokens += self.batch_tokens
        accumulation = int(trainer.accumulate_grad_batches)
        # batch_idx resets at every epoch (especially relevant when
        # overfit_batches=1), while accumulation is global across the fit.
        at_optimizer_boundary = len(self.batch_times) % accumulation == 0
        if (
            self.token_budget
            and self.total_tokens >= self.token_budget
            and at_optimizer_boundary
        ):
            trainer.should_stop = True

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        squares = []
        for parameter in pl_module.parameters():
            if parameter.grad is not None:
                squares.append(parameter.grad.detach().float().norm(2).square())
        if squares:
            norm = torch.stack(squares).sum().sqrt().cpu().item()
            self.grad_norms.append(float(norm))

    def write(self, trainer, pl_module, status="ok", error=None):
        if not self.output_path or self._written:
            return
        self._sync()
        elapsed = (
            time.perf_counter() - self.started_at if self.started_at is not None else 0.0
        )
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pl_module.parameters())
        metrics = {
            "status": status,
            "error": error,
            "wall_time_seconds": elapsed,
            "optimizer_steps": int(trainer.global_step),
            "micro_batches": len(self.batch_times),
            "input_tokens": self.total_tokens,
            "tokens_per_second": (
                self.total_tokens / sum(self.batch_times) if self.batch_times else None
            ),
            "mean_batch_seconds": (
                statistics.fmean(self.batch_times) if self.batch_times else None
            ),
            "median_batch_seconds": (
                statistics.median(self.batch_times) if self.batch_times else None
            ),
            "mean_grad_norm": (
                statistics.fmean(self.grad_norms) if self.grad_norms else None
            ),
            "max_grad_norm": max(self.grad_norms) if self.grad_norms else None,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": trainable / total * 100 if total else None,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
            # Linux reports ru_maxrss in KiB. This captures CPU/unified-memory
            # pressure that nvidia-smi cannot report reliably on DGX Spark.
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "final_train_loss": _number(trainer.callback_metrics.get("train_loss")),
            "final_val_loss": _number(trainer.callback_metrics.get("val_loss")),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".partial")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(self.output_path)
        self._written = True

    def on_fit_end(self, trainer, pl_module):
        self.write(trainer, pl_module)