#!/usr/bin/env python3
"""Run the resumable HellaSwag chat-MC smoke against an OpenAI chat endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_ENDPOINT = "http://localhost:8011/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning"
STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nreceived signal {signum}; stopping after the current request", file=sys.stderr)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle]
    if limit is not None:
        cases = cases[:limit]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise RuntimeError("dataset contains duplicate case IDs")
    for case in cases:
        if case["suite"] != "hellaswag-chat-mc":
            raise RuntimeError(f"{case['id']}: unexpected suite {case['suite']}")
        if case["expected"]["label"] not in "ABCD":
            raise RuntimeError(f"{case['id']}: invalid expected label")
    return cases


def completed_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                print(f"ignoring incomplete final line {line_number} in {path}", file=sys.stderr)
                break
            row = json.loads(line)
            results[row["id"]] = row
    return results


def strict_label(content: object) -> str | None:
    if not isinstance(content, str):
        return None
    normalized = content.strip()
    return normalized if re.fullmatch(r"[ABCD]", normalized) else None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


def get_json(url: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def append_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(results: dict[str, dict[str, Any]], selected_ids: set[str]) -> dict[str, Any]:
    rows = [row for case_id, row in results.items() if case_id in selected_ids]
    successful = [row for row in rows if row["status"] == "ok"]
    valid = [row for row in successful if row["predicted_label"] is not None]
    correct = [row for row in successful if row["correct"]]
    return {
        "accuracy": len(correct) / len(successful) if successful else None,
        "completed": len(rows),
        "correct": len(correct),
        "errors": len(rows) - len(successful),
        "invalid_labels": len(successful) - len(valid),
        "successful": len(successful),
        "valid_label_rate": len(valid) / len(successful) if successful else None,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data/hellaswag/chat_mc_smoke_100.jsonl",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--run-id", default="smoke-01")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=35_003_500)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    cases = load_cases(dataset, args.limit)
    if args.validate_only:
        print(f"validated {len(cases)} cases from {dataset}")
        return 0
    if args.output is None:
        output_dir = root / "outputs/evaluations/hellaswag-chat-mc" / args.run_id
        output = output_dir / "results.jsonl"
    else:
        output = args.output.resolve()
        output_dir = output.parent
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.json"
    pid_path = output_dir / "active.pid"
    stop_path = output_dir / "STOP"

    config = {
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True,
        "max_tokens": 4,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "top_logprobs": 20,
        "top_p": 1,
    }
    signature = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "dataset_sha256": sha256(dataset),
                "endpoint": args.endpoint,
                "selected_ids": [case["id"] for case in cases],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["run_signature"] != signature:
            raise RuntimeError(f"existing run at {output_dir} has a different configuration")
    else:
        models_url = args.endpoint.rsplit("/chat/completions", 1)[0] + "/models"
        metadata = {
            "dataset": str(dataset.relative_to(root)),
            "dataset_sha256": sha256(dataset),
            "endpoint": args.endpoint,
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "evaluation_git_dirty": git_dirty(root),
            "evaluation_git_revision": git_revision(root),
            "model_config": config,
            "models_response": get_json(models_url),
            "run_id": args.run_id,
            "run_signature": signature,
            "selected_cases": len(cases),
            "started_unix_s": time.time(),
        }
        write_json(metadata_path, metadata)

    existing = completed_results(output)
    selected_ids = {case["id"] for case in cases}
    unknown = set(existing) - selected_ids
    if unknown:
        raise RuntimeError(f"output contains IDs outside this run: {sorted(unknown)[:3]}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_path.unlink(missing_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        for position, case in enumerate(cases, start=1):
            if STOP_REQUESTED or stop_path.exists():
                break
            if case["id"] in existing:
                continue
            payload = {**config, "messages": case["messages"]}
            started = time.perf_counter()
            response: dict[str, Any] | None = None
            error_message: str | None = None
            attempts = 0
            while attempts <= args.max_retries and response is None:
                attempts += 1
                try:
                    response = post_json(args.endpoint, payload, args.timeout)
                except Exception as error:  # preserve endpoint failures in the run artifact
                    error_message = str(error)
                    if attempts <= args.max_retries:
                        time.sleep(min(2**attempts, 10))
            elapsed = time.perf_counter() - started
            if response is None:
                result = {
                    "attempts": attempts,
                    "correct": False,
                    "elapsed_s": elapsed,
                    "error": error_message,
                    "expected_label": case["expected"]["label"],
                    "id": case["id"],
                    "predicted_label": None,
                    "response": None,
                    "status": "error",
                }
            else:
                choices = response.get("choices", [])
                content = choices[0].get("message", {}).get("content") if choices else None
                predicted = strict_label(content)
                result = {
                    "attempts": attempts,
                    "correct": predicted == case["expected"]["label"],
                    "elapsed_s": elapsed,
                    "error": None,
                    "expected_label": case["expected"]["label"],
                    "id": case["id"],
                    "predicted_label": predicted,
                    "response": response,
                    "status": "ok",
                }
            append_result(output, result)
            existing[case["id"]] = result
            summary = summarize(existing, selected_ids)
            write_json(summary_path, {**summary, "run_id": args.run_id, "total": len(cases)})
            print(
                f"[{position}/{len(cases)}] {case['id']} "
                f"pred={result['predicted_label']} expected={result['expected_label']} "
                f"status={result['status']} elapsed={elapsed:.2f}s",
                flush=True,
            )
    finally:
        pid_path.unlink(missing_ok=True)

    summary = summarize(existing, selected_ids)
    write_json(
        summary_path,
        {
            **summary,
            "paused": summary["completed"] < len(cases),
            "run_id": args.run_id,
            "total": len(cases),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
