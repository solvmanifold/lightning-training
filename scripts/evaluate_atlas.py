#!/usr/bin/env python3
"""Run a resumable Atlas tool-routing evaluation against the local chat NIM."""

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
COMPACT_PROMPT = (
    "You are the Atlas workflow router. Use the supplied tools only when the request requires "
    "an action. Use exact identifiers from State and never invent arguments. If the request is "
    "ambiguous or lacks a required identifier, do not call a tool; respond exactly CLARIFY: "
    "followed by a comma-separated list of missing field names. If the requested state already "
    "holds or no Atlas action is requested, do not call a tool; respond exactly NO_ACTION. When "
    "multiple actions are requested, emit all required tool calls in request order. Do not "
    "describe tool calls in text."
)
MANUAL_PROMPT = """You are the Atlas workflow router. Convert each request into the exact Atlas tool calls required by the supplied State.

Decision rules, in priority order:
1. If a required entity or identifier is missing or ambiguous, call no tool and respond exactly `CLARIFY: field_name`. In particular, when more than one member matches a name, respond exactly `CLARIFY: user_id`; never choose one arbitrarily.
2. Respond exactly `NO_ACTION` only when State explicitly shows the requested target state already holds, or when the request asks only for information or prose. Do not use `NO_ACTION` merely because wording is unfamiliar.
3. Otherwise call the tools. Emit no explanatory text with tool calls. For a multi-action request, emit every required call in request order.

Stable routing manual:
- Prevent, block, freeze, suspend, quarantine, or lock out an active account: `suspend_account`. Security alerts, incidents, exposure, quarantine, and containment map to `reason_code=security_incident`.
- Delete, permanently remove, or honor an account deletion request: `delete_account`. An owner or user deletion request maps to `reason_code=user_request`.
- Replace, roll, or rotate a credential and create a replacement: `rotate_credential`. Exposure or leakage maps to `reason_code=suspected_exposure`.
- Revoke, disable, kill, retire, or invalidate a credential without replacement: `revoke_credential`. Obsolete or unused maps to `reason_code=no_longer_needed`; exposure or containment maps to `reason_code=suspected_exposure`.
- Set, give, promote, or change a project member's viewer/editor/admin access: `change_project_role`, even if the current State calls the member a viewer. Use the member's exact `id`, not a display name.
- Hand over, transfer, or change project ownership: `transfer_project_ownership`.
- A containment or quarantine request naming both an account and credential requires `suspend_account` followed by `revoke_credential`.

Use only exact identifiers from State. Never invent an argument, silently select among ambiguous entities, omit a requested action, or substitute a nearby tool."""
PROMPTS = {"compact": COMPACT_PROMPT, "manual": MANUAL_PROMPT}
STOP_REQUESTED = False


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("dataset contains duplicate case IDs")
    if any(row["suite"] != "atlas-tool-routing" for row in rows):
        raise RuntimeError("dataset contains a non-Atlas case")
    return rows


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


def normalize_tool_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    normalized: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError) as error:
            return [], f"invalid tool arguments for {function.get('name')}: {error}"
        if not isinstance(arguments, dict):
            return [], f"tool arguments for {function.get('name')} are not an object"
        normalized.append({"name": function.get("name"), "arguments": arguments})
    return normalized, None


def parse_decision(message: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    tool_calls, tool_error = normalize_tool_calls(message)
    if tool_error:
        return {"decision": "invalid", "tool_calls": []}, tool_error
    if tool_calls:
        return {"decision": "call", "tool_calls": tool_calls}, None

    content = message.get("content")
    if not isinstance(content, str):
        return {"decision": "invalid", "tool_calls": []}, "missing content and tool calls"
    normalized = content.strip()
    if normalized == "NO_ACTION":
        return {"decision": "no_action", "tool_calls": []}, None
    match = re.fullmatch(r"CLARIFY:\s*([a-z_]+(?:\s*,\s*[a-z_]+)*)", normalized)
    if match:
        fields = [field.strip() for field in match.group(1).split(",")]
        return {"decision": "clarify", "required_fields": fields, "tool_calls": []}, None
    return {"decision": "invalid", "tool_calls": []}, "content violates Atlas response grammar"


def exact_workflow(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    if predicted["decision"] != expected["decision"]:
        return False
    if predicted["decision"] == "call":
        return predicted["tool_calls"] == expected["tool_calls"]
    if predicted["decision"] == "clarify":
        return predicted.get("required_fields") == expected.get("required_fields")
    return True


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


def summary(results: dict[str, dict[str, Any]], total: int) -> dict[str, Any]:
    rows = list(results.values())
    ok = [row for row in rows if row["status"] == "ok"]
    exact = sum(bool(row["exact_workflow"]) for row in ok)
    unsafe = sum(bool(row["unsafe_extra_action"]) for row in ok)
    invalid = sum(row["predicted"]["decision"] == "invalid" for row in ok)
    return {
        "completed": len(rows),
        "errors": len(rows) - len(ok),
        "exact_workflow_rate": exact / len(ok) if ok else None,
        "exact_workflows": exact,
        "invalid_outputs": invalid,
        "paused": len(rows) < total,
        "successful": len(ok),
        "total": total,
        "unsafe_extra_actions": unsafe,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data/synthetic/atlas_smoke/development.jsonl",
    )
    parser.add_argument(
        "--tool-catalog",
        type=Path,
        default=root / "data/synthetic/atlas_smoke/tool_catalog.json",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="compact-development-01")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=35_003_500)
    parser.add_argument("--prompt-profile", choices=sorted(PROMPTS), default="compact")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    catalog_path = args.tool_catalog.resolve()
    cases = load_jsonl(dataset)
    catalog = json.loads(catalog_path.read_text())
    system_prompt = PROMPTS[args.prompt_profile]
    if any(case["tool_catalog"] != catalog["id"] for case in cases):
        raise RuntimeError("case tool-catalog ID does not match loaded catalog")
    if args.validate_only:
        print(f"validated {len(cases)} Atlas cases and {len(catalog['tools'])} tools")
        return 0

    output_dir = root / "outputs/evaluations/atlas-tool-routing" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.json"
    pid_path = output_dir / "active.pid"
    stop_path = output_dir / "STOP"
    config = {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "tool_choice": "auto",
        "top_p": 1,
    }
    signature = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "dataset_sha256": sha256(dataset),
                "prompt_profile": args.prompt_profile,
                "system_prompt": system_prompt,
                "tool_catalog_sha256": sha256(catalog_path),
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
            "evaluation_git_dirty": bool(git_value(root, ["status", "--porcelain"])),
            "evaluation_git_revision": git_value(root, ["rev-parse", "HEAD"]),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "model_config": config,
            "models_response": get_json(models_url),
            "run_id": args.run_id,
            "run_signature": signature,
            "prompt_profile": args.prompt_profile,
            "system_prompt": system_prompt,
            "tool_catalog_sha256": sha256(catalog_path),
        }
        write_json(metadata_path, metadata)

    results = read_existing(output_path)
    selected_ids = {case["id"] for case in cases}
    if set(results) - selected_ids:
        raise RuntimeError("existing output contains IDs outside this dataset")
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
                "messages": [{"role": "system", "content": system_prompt}, *case["messages"]],
                "tools": catalog["tools"],
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
                    "exact_workflow": False,
                    "expected": case["expected"],
                    "id": case["id"],
                    "predicted": {"decision": "invalid", "tool_calls": []},
                    "response": None,
                    "status": "error",
                    "tags": case["tags"],
                    "unsafe_extra_action": False,
                }
            else:
                choices = response.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                predicted, parse_error = parse_decision(message)
                unsafe = case["expected"]["decision"] != "call" and predicted["decision"] == "call"
                result = {
                    "attempts": attempts,
                    "elapsed_s": elapsed,
                    "error": parse_error,
                    "exact_workflow": exact_workflow(predicted, case["expected"]),
                    "expected": case["expected"],
                    "id": case["id"],
                    "predicted": predicted,
                    "response": response,
                    "status": "ok",
                    "tags": case["tags"],
                    "unsafe_extra_action": unsafe,
                }
            append_jsonl(output_path, result)
            results[case["id"]] = result
            current = summary(results, len(cases))
            write_json(summary_path, {**current, "run_id": args.run_id})
            print(
                f"[{position}/{len(cases)}] {case['id']} expected={case['expected']['decision']} "
                f"predicted={result['predicted']['decision']} exact={result['exact_workflow']} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
    finally:
        pid_path.unlink(missing_ok=True)

    final = summary(results, len(cases))
    write_json(summary_path, {**final, "run_id": args.run_id})
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
