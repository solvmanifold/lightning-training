#!/usr/bin/env python3
"""Plan, run, and report reproducible Nano3 configuration sweeps."""
import argparse
import csv
import fnmatch
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", "docker-compose.training.yml"]


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_jobs(manifest_path: Path, tier: str, patterns: list[str]):
    manifest = yaml.safe_load(manifest_path.read_text())
    if tier not in manifest["tiers"]:
        raise SystemExit(f"Unknown tier {tier!r}; choose from {list(manifest['tiers'])}")
    base_path = ROOT / manifest["base_config"]
    base = yaml.safe_load(base_path.read_text())
    jobs = []
    for experiment in manifest["experiments"]:
        experiment_id = experiment["id"]
        if patterns and not any(fnmatch.fnmatch(experiment_id, pattern) for pattern in patterns):
            continue
        overrides = {k: v for k, v in experiment.items() if k not in {"id", "description"}}
        config = deep_merge(base, manifest["tiers"][tier])
        config = deep_merge(config, overrides)
        jobs.append({
            "id": experiment_id,
            "description": experiment.get("description", ""),
            "config": config,
            "sweep": manifest["name"],
            "tier": tier,
        })
    return jobs


def job_dir(job):
    return ROOT / "outputs" / "sweeps" / job["sweep"] / job["tier"] / job["id"]


def prepare_job(job):
    directory = job_dir(job)
    directory.mkdir(parents=True, exist_ok=True)
    config = job["config"]
    config["output"] = {
        "run_name": f"{job['sweep']}-{job['tier']}-{job['id']}",
        "checkpoint_dir": str(directory.relative_to(ROOT) / "checkpoints"),
        "log_dir": str(directory.relative_to(ROOT) / "tensorboard"),
        "save_top_k": 0,
        "save_last": False,
    }
    path = directory / "generated_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (directory / "experiment.json").write_text(json.dumps({
        "id": job["id"],
        "description": job["description"],
        "tier": job["tier"],
        "sweep": job["sweep"],
    }, indent=2) + "\n")
    return directory, path


def stream_command(command: list[str], log_path: Path) -> int:
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            raise
        return process.wait()


def failure_reason(log_path: Path, returncode: int) -> str:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    classifiers = [
        (r"out of memory|CUDA out of memory", "out_of_memory"),
        (r"no kernel image|not supported.*SM|unsupported.*compute capability", "unsupported_gpu_kernel"),
        (r"ImportError|ModuleNotFoundError", "dependency_import_error"),
        (r"metadata.*NoneType|NoneType.*metadata", "safetensors_metadata_error"),
        (r"nan|non-finite", "non_finite"),
    ]
    for pattern, label in classifiers:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:500] if lines else f"process_exit_{returncode}"


def container_name(job, prefix="sweep"):
    return re.sub(
        r"[^a-z0-9_.-]", "-", f"{prefix}-{job['tier']}-{job['id']}".lower()
    )[:63]


def container_command(job, config_path: Path, metrics_path: Path):
    name = container_name(job)
    return COMPOSE + [
        "run", "--rm", "--name", name, "smoketest",
        "python3", "train.py",
        "--config", str(config_path.relative_to(ROOT)),
        "--metrics", str(metrics_path.relative_to(ROOT)),
        "--skip-final-save",
    ]


def run_preflight(job, directory: Path) -> tuple[bool, int]:
    if not job["config"].get("model", {}).get("use_qlora", False):
        return True, 0
    output = directory / "qlora_preflight.json"
    log = directory / "qlora_preflight.log"
    name = re.sub(r"[^a-z0-9_.-]", "-", f"preflight-{job['id']}".lower())[:63]
    command = COMPOSE + [
        "run", "--rm", "--name", name, "smoketest",
        "python3", "scripts/preflight_qlora.py",
        "--output", str(output.relative_to(ROOT)),
    ]
    print(f"\nQLoRA preflight: {job['id']}")
    rc = stream_command(command, log)
    return rc == 0, rc


def available_memory_gib():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    gib = 1024 ** 3
    return values.get("MemAvailable", 0) / gib, values.get("SwapFree", 0) / gib


def run_job(job, force=False, ignore_memory_preflight=False):
    directory, config_path = prepare_job(job)
    result_path = directory / "run_result.json"
    if result_path.exists() and not force:
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "ok":
            print(f"SKIP {job['id']}: completed result exists")
            return

    available_gib, swap_free_gib = available_memory_gib()
    qlora = job["config"].get("model", {}).get("use_qlora", False)
    required_gib = 40 if qlora else 70
    if available_gib < required_gib and not ignore_memory_preflight:
        result = {
            "id": job["id"], "tier": job["tier"],
            "description": job["description"], "status": "preflight_failed",
            "returncode": None, "failure_reason": "insufficient_host_memory",
            "available_memory_gib": available_gib,
            "swap_free_gib": swap_free_gib,
            "required_memory_gib": required_gib,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"SKIP {job['id']}: {available_gib:.1f} GiB available; "
            f"benchmark preflight requires {required_gib} GiB"
        )
        return

    preflight_ok, preflight_rc = run_preflight(job, directory)
    if not preflight_ok:
        result = {
            "id": job["id"], "tier": job["tier"],
            "description": job["description"], "status": "preflight_failed",
            "returncode": preflight_rc, "failure_reason": "qlora_nf4_preflight_failed",
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return

    metrics_path = directory / "metrics.json"
    log_path = directory / "train.log"
    command = container_command(job, config_path, metrics_path)
    print(f"\n{'=' * 72}\nRUN {job['id']} ({job['tier']}): {job['description']}\n{'=' * 72}")
    started_wall = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        rc = stream_command(command, log_path)
    except KeyboardInterrupt:
        subprocess.run(
            ["docker", "stop", "--time", "5", container_name(job)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        result_path.write_text(json.dumps({
            "id": job["id"], "tier": job["tier"],
            "description": job["description"], "status": "interrupted",
            "returncode": 130, "failure_reason": "user_interrupt",
        }, indent=2) + "\n")
        raise
    result = {
        "id": job["id"],
        "tier": job["tier"],
        "description": job["description"],
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runner_wall_time_seconds": time.perf_counter() - started_wall,
        "failure_reason": None if rc == 0 else failure_reason(log_path, rc),
        "command": command,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")


REPORT_COLUMNS = [
    "id", "status", "failure_reason", "r", "target_modules", "lr", "max_seq_len",
    "grad_accumulation_steps", "optimizer_steps", "input_tokens", "tokens_per_second",
    "mean_batch_seconds", "final_train_loss", "final_val_loss", "mean_grad_norm",
    "max_grad_norm", "trainable_parameters", "trainable_percent",
    "peak_cuda_allocated_gib", "peak_cuda_reserved_gib", "peak_process_rss_gib",
    "wall_time_seconds",
]


def collect_rows(jobs):
    rows = []
    for job in jobs:
        directory = job_dir(job)
        result_path = directory / "run_result.json"
        metrics_path = directory / "metrics.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "not_run"}
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        config = job["config"]
        gib = 1024 ** 3
        rows.append({
            "id": job["id"],
            "status": result.get("status"),
            "failure_reason": result.get("failure_reason"),
            "r": config["lora"]["r"],
            "target_modules": ",".join(config["lora"]["target_modules"]),
            "lr": config["training"]["lr"],
            "max_seq_len": config["training"]["max_seq_len"],
            "grad_accumulation_steps": config["training"]["grad_accumulation_steps"],
            "optimizer_steps": metrics.get("optimizer_steps"),
            "input_tokens": metrics.get("input_tokens"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "mean_batch_seconds": metrics.get("mean_batch_seconds"),
            "final_train_loss": metrics.get("final_train_loss"),
            "final_val_loss": metrics.get("final_val_loss"),
            "mean_grad_norm": metrics.get("mean_grad_norm"),
            "max_grad_norm": metrics.get("max_grad_norm"),
            "trainable_parameters": metrics.get("trainable_parameters"),
            "trainable_percent": metrics.get("trainable_percent"),
            "peak_cuda_allocated_gib": metrics.get("peak_cuda_allocated_bytes", 0) / gib if metrics.get("peak_cuda_allocated_bytes") is not None else None,
            "peak_cuda_reserved_gib": metrics.get("peak_cuda_reserved_bytes", 0) / gib if metrics.get("peak_cuda_reserved_bytes") is not None else None,
            "peak_process_rss_gib": metrics.get("peak_process_rss_bytes", 0) / gib if metrics.get("peak_process_rss_bytes") is not None else None,
            "wall_time_seconds": metrics.get("wall_time_seconds", result.get("runner_wall_time_seconds")),
        })
    return rows


def fmt(value, digits=3):
    if value is None or value == "":
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_report(jobs):
    if not jobs:
        raise SystemExit("No experiments selected")
    directory = job_dir(jobs[0]).parent
    directory.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(jobs)
    csv_path = directory / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Nano3 configuration sweep — {jobs[0]['tier']}", "",
        "Runs use the same tier workload. The `compare` tier stops on a shared input-token budget.", "",
        "| Experiment | Status | Tok/s | Batch s | Train loss | Val loss | Grad norm | CUDA GiB | RSS GiB | Steps | Tokens |",
        "|------------|--------|------:|--------:|-----------:|---------:|----------:|---------:|--------:|------:|-------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['status']} | {fmt(row['tokens_per_second'], 1)} | "
            f"{fmt(row['mean_batch_seconds'])} | {fmt(row['final_train_loss'])} | "
            f"{fmt(row['final_val_loss'])} | {fmt(row['mean_grad_norm'])} | "
            f"{fmt(row['peak_cuda_allocated_gib'], 2)} | {fmt(row['peak_process_rss_gib'], 2)} | "
            f"{fmt(row['optimizer_steps'], 0)} | {fmt(row['input_tokens'], 0)} |"
        )
    failures = [row for row in rows if row["status"] not in {"ok", "not_run"}]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(f"- `{row['id']}`: {row['failure_reason']}")
    md_path = directory / "results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


def main(args):
    manifest_path = (ROOT / args.manifest).resolve()
    jobs = load_jobs(manifest_path, args.tier, args.experiment)
    if not jobs:
        raise SystemExit("No experiments matched")
    if args.action == "plan":
        for job in jobs:
            qlora = " qlora+preflight" if job["config"].get("model", {}).get("use_qlora") else ""
            print(f"{job['id']:<24} {job['description']}{qlora}")
        return
    if args.action == "report":
        write_report(jobs)
        return
    if args.build:
        subprocess.run(COMPOSE + ["build", "training-base"], cwd=ROOT, check=True)
    for job in jobs:
        run_job(
            job,
            force=args.force,
            ignore_memory_preflight=args.ignore_memory_preflight,
        )
    # Always refresh the full tier table after running a subset so incremental
    # runs accumulate into one stable report.
    write_report(load_jobs(manifest_path, args.tier, []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "run", "report"])
    parser.add_argument("--manifest", default="experiments/sweep.yaml")
    parser.add_argument("--tier", choices=["smoke", "overfit", "compare"], default="smoke")
    parser.add_argument("--experiment", action="append", default=[], help="ID glob; repeatable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--ignore-memory-preflight", action="store_true",
        help="Run even when unified memory is already occupied",
    )
    main(parser.parse_args())
