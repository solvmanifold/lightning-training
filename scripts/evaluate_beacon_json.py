#!/usr/bin/env python3
"""Run resumable Beacon canonical-JSON evaluation conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from beacon_schema import field_counts, validate_beacon


DEFAULT_ENDPOINT = "http://localhost:8011/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning"
BASIC_FEWSHOT_CASE_IDS = ("beacon-train-00000", "beacon-train-00007")
OPTIMIZED_FEWSHOT_CASE_IDS = (
    "beacon-train-00000",
    "beacon-train-00001",
    "beacon-train-00007",
)
FINAL_FEWSHOT_CASE_IDS = (
    "beacon-train-00000",
    "beacon-train-00001",
    "beacon-train-00004",
    "beacon-train-00007",
)
STOP_REQUESTED = False

COMPACT_PROMPT = """Compile the request into one Beacon JSON object and output JSON only. Use exactly these fields: schema_version, action, resource_id, target_region, urgency, execution, encryption, retention_days, compression, notify.

schema_version is `beacon-job-v1`. action is replicate, snapshot, restore, or retire. target_region is one of us-north, us-south, eu-central, ap-east, and is null for snapshot or retire. urgency defaults to normal. execution defaults to apply; previews and dry runs are plan. encryption defaults to standard; sensitive or regulated data is strict. retention defaults are replicate=14, snapshot=30, restore=7, retire=90 days. Weeks are seven days. compression defaults true for replicate/snapshot and false for restore/retire. notify defaults to an empty array and addresses must be sorted. Explicit instructions and final corrections override defaults."""

MANUAL_PROMPT = """You are the Beacon canonical-job compiler. Return exactly one JSON object with no Markdown, explanation, or extra keys.

Required fields and allowed values:
- schema_version: always `beacon-job-v1`
- action: `replicate`, `snapshot`, `restore`, or `retire`
- resource_id: copy the exact res_* identifier
- target_region: `us-north`, `us-south`, `eu-central`, `ap-east`, or null
- urgency: `low`, `normal`, `high`, or `critical`
- execution: `plan` or `apply`
- encryption: `standard` or `strict`
- retention_days: integer from 1 through 365
- compression: boolean
- notify: sorted unique array of exact email addresses

Interpretation manual:
- Mirror, copy, fan out, or create a regional replica means replicate. Replicate requires the named target region.
- Checkpoint, back up, capture a point-in-time copy, or preserve current state means snapshot. Snapshot target_region is null.
- Recover, restore, roll back, or reconstitute means restore. Restore requires the named target region.
- Decommission, retire, take out of rotation, or sunset means retire. Retire target_region is null.
- No rush or whenever capacity permits means low. Ordinary scheduling means normal. Today or close of business means high. An incident, emergency, impaired production, immediately, or without delay means critical. Default is normal.
- Preview, simulate, or dry run means execution=plan. Otherwise execution=apply.
- Sensitive or regulated data requires encryption=strict. Otherwise use standard.
- Convert weeks to seven days. Defaults are replicate=14, snapshot=30, restore=7, and retire=90 days.
- Explicit compression or no-compression language overrides defaults. Defaults are true for replicate/snapshot and false for restore/retire.
- Copy notification addresses exactly, remove duplicates, and sort ascending. Default is an empty array.
- Apply all explicit clauses regardless of sentence order. If the request corrects itself, the final corrected value wins."""

OPTIMIZED_PROMPT = MANUAL_PROMPT + """

Disambiguation rules:
- A snapshot/checkpoint is a real applied action, not automatically a plan. Set execution=plan only when the request explicitly says preview, simulate, dry run, or no side effects; otherwise execution=apply for every action.
- Never infer urgency from an action being destructive, permanent, a restore, or a snapshot. `Today`, `before today ends`, and `close of business` are high. Only explicit incident, emergency, impaired-production, immediately, right-now, or without-delay language is critical."""

FINAL_PROMPT = OPTIMIZED_PROMPT + """

- When the request has no urgency phrase, urgency is normal—never low. Low requires explicit no-rush, can-wait, or whenever-capacity-permits language.
- Urgency and action language never imply encryption. Incident, emergency, restore, retirement, permanence, and critical urgency still use encryption=standard unless the request explicitly says sensitive, regulated, strict encryption, or lock encryption down."""

PROMPTS = {
    "compact": COMPACT_PROMPT,
    "manual": MANUAL_PROMPT,
    "fewshot": MANUAL_PROMPT,
    "constrained": MANUAL_PROMPT,
    "optimized": OPTIMIZED_PROMPT,
    "constrained_optimized": OPTIMIZED_PROMPT,
    "final": FINAL_PROMPT,
    "constrained_final": FINAL_PROMPT,
}


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nreceived signal {signum}; stopping after the current request", file=sys.stderr)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(root: Path, arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if limit is not None:
        rows = rows[:limit]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("dataset contains duplicate case IDs")
    if any(row["suite"] != "beacon-canonical-json" for row in rows):
        raise RuntimeError("dataset contains a non-Beacon case")
    return rows


def fewshot_messages(root: Path, case_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    training = {
        row["id"]: row
        for row in load_jsonl(root / "data/synthetic/beacon_json/train.jsonl")
    }
    messages: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = training[case_id]
        messages.extend(
            [
                *case["messages"],
                {
                    "role": "assistant",
                    "content": json.dumps(case["expected"], sort_keys=True, separators=(",", ":")),
                },
            ]
        )
    return messages


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


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def parse_content(response: dict[str, Any]) -> tuple[object | None, str | None, object]:
    choices = response.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str):
        return None, "response content is not a string", content
    try:
        return json.loads(content), None, content
    except json.JSONDecodeError as error:
        return None, str(error), content


def serving_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove keywords unsupported by NIM grammar while keeping scorer checks."""
    value = json.loads(json.dumps(schema))

    def visit(node: object) -> None:
        if isinstance(node, dict):
            node.pop("uniqueItems", None)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return value


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in (json.loads(line) for line in handle)}


def summarize(results: dict[str, dict[str, Any]], total: int) -> dict[str, Any]:
    rows = list(results.values())
    successful = [row for row in rows if row["status"] == "ok"]
    parsed = [row for row in successful if row["parsed"]]
    schema_valid = [row for row in successful if row["schema_valid"]]
    exact = [row for row in successful if row["semantic_exact"]]
    correct_fields = sum(row["field_counts"]["correct"] for row in successful)
    predicted_fields = sum(row["field_counts"]["predicted"] for row in successful)
    expected_fields = sum(row["field_counts"]["expected"] for row in successful)
    return {
        "completed": len(rows),
        "errors": len(rows) - len(successful),
        "field_precision": correct_fields / predicted_fields if predicted_fields else None,
        "field_recall": correct_fields / expected_fields if expected_fields else None,
        "json_parse_rate": len(parsed) / len(successful) if successful else None,
        "paused": len(rows) < total,
        "schema_valid_rate": len(schema_valid) / len(successful) if successful else None,
        "semantic_exact": len(exact),
        "semantic_exact_rate": len(exact) / len(successful) if successful else None,
        "successful": len(successful),
        "total": total,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data/synthetic/beacon_json/development.jsonl",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=root / "data/synthetic/beacon_json/schema.json",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-profile", choices=sorted(PROMPTS), default="compact")
    parser.add_argument("--run-id", default="compact-development-01")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=35_003_501)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    schema_path = args.schema.resolve()
    cases = load_jsonl(dataset, args.limit)
    schema = json.loads(schema_path.read_text())
    if args.prompt_profile in {"fewshot", "constrained"}:
        example_ids = BASIC_FEWSHOT_CASE_IDS
    elif args.prompt_profile in {"optimized", "constrained_optimized"}:
        example_ids = OPTIMIZED_FEWSHOT_CASE_IDS
    elif args.prompt_profile in {"final", "constrained_final"}:
        example_ids = FINAL_FEWSHOT_CASE_IDS
    else:
        example_ids = ()
    examples = fewshot_messages(root, example_ids) if example_ids else []
    if args.validate_only:
        for case in cases:
            errors = validate_beacon(case["expected"])
            if errors:
                raise RuntimeError(f"{case['id']}: invalid oracle: {errors}")
        print(f"validated {len(cases)} Beacon cases")
        return 0

    output_dir = root / "outputs/evaluations/beacon-canonical-json" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.json"
    pid_path = output_dir / "active.pid"
    stop_path = output_dir / "STOP"
    config: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 512,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "top_p": 1,
    }
    if args.prompt_profile in {"constrained", "constrained_optimized", "constrained_final"}:
        config["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "beacon_job",
                "strict": True,
                "schema": serving_schema(schema),
            },
        }
    signature = hashlib.sha256(
        json.dumps(
            {
                "case_ids": [case["id"] for case in cases],
                "config": config,
                "dataset_sha256": sha256(dataset),
                "fewshot_case_ids": example_ids,
                "prompt": PROMPTS[args.prompt_profile],
                "prompt_profile": args.prompt_profile,
                "schema_sha256": sha256(schema_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
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
            "constrained_schema_omissions": (
                ["uniqueItems"]
                if args.prompt_profile
                in {"constrained", "constrained_optimized", "constrained_final"}
                else []
            ),
            "evaluation_git_dirty": bool(git_value(root, ["status", "--porcelain"])),
            "evaluation_git_revision": git_value(root, ["rev-parse", "HEAD"]),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "fewshot_case_ids": example_ids,
            "model_config": config,
            "models_response": get_json(models_url),
            "prompt": PROMPTS[args.prompt_profile],
            "prompt_profile": args.prompt_profile,
            "run_id": args.run_id,
            "run_signature": signature,
            "schema_sha256": sha256(schema_path),
            "selected_cases": len(cases),
        }
        write_json(metadata_path, metadata)

    results = read_existing(output_path)
    selected_ids = {case["id"] for case in cases}
    if set(results) - selected_ids:
        raise RuntimeError("existing output contains IDs outside this run")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    stop_path.unlink(missing_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        for position, case in enumerate(cases, start=1):
            if STOP_REQUESTED or stop_path.exists():
                break
            if case["id"] in results:
                continue
            payload = {
                **config,
                "messages": [
                    {"role": "system", "content": PROMPTS[args.prompt_profile]},
                    *examples,
                    *case["messages"],
                ],
            }
            response = None
            error_message = None
            attempts = 0
            started = time.perf_counter()
            while attempts <= args.max_retries and response is None:
                attempts += 1
                try:
                    response = post_json(args.endpoint, payload, args.timeout)
                except Exception as error:
                    error_message = str(error)
                    if attempts <= args.max_retries:
                        time.sleep(min(2**attempts, 10))
            elapsed = time.perf_counter() - started
            if response is None:
                result = {
                    "attempts": attempts,
                    "elapsed_s": elapsed,
                    "error": error_message,
                    "field_counts": {"correct": 0, "expected": len(case["expected"]), "predicted": 0},
                    "id": case["id"],
                    "parsed": False,
                    "response": None,
                    "schema_errors": ["request failed"],
                    "schema_valid": False,
                    "semantic_exact": False,
                    "status": "error",
                    "tags": case["tags"],
                }
            else:
                predicted, parse_error, raw_content = parse_content(response)
                schema_errors = validate_beacon(predicted) if parse_error is None else [parse_error]
                correct, predicted_count, expected_count = field_counts(predicted, case["expected"])
                result = {
                    "attempts": attempts,
                    "elapsed_s": elapsed,
                    "error": parse_error,
                    "expected": case["expected"],
                    "field_counts": {
                        "correct": correct,
                        "expected": expected_count,
                        "predicted": predicted_count,
                    },
                    "id": case["id"],
                    "parsed": parse_error is None,
                    "predicted": predicted,
                    "raw_content": raw_content,
                    "response": response,
                    "schema_errors": schema_errors,
                    "schema_valid": not schema_errors,
                    "semantic_exact": predicted == case["expected"],
                    "status": "ok",
                    "tags": case["tags"],
                }
            append_jsonl(output_path, result)
            results[case["id"]] = result
            current = summarize(results, len(cases))
            write_json(summary_path, {**current, "run_id": args.run_id})
            print(
                f"[{position}/{len(cases)}] {case['id']} parsed={result['parsed']} "
                f"schema={result['schema_valid']} exact={result['semantic_exact']} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
    finally:
        pid_path.unlink(missing_ok=True)

    final = summarize(results, len(cases))
    write_json(summary_path, {**final, "run_id": args.run_id})
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
