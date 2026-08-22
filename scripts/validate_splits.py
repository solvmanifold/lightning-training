#!/usr/bin/env python3
"""Validate that train, validation, and test JSONL partitions do not leak."""
import argparse
import json
from pathlib import Path


def load(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def user_prompt(record: dict) -> str:
    return next(
        (str(m.get("content", "")).strip() for m in record.get("messages", [])
         if m.get("role") == "user"),
        "",
    )


def source_file(record: dict) -> str | None:
    return record.get("metadata", {}).get("source_file")


def main(args):
    partitions = {
        "train": load(args.train),
        "val": load(args.val),
        "test": load(args.test),
    }
    errors = []
    prompts = {}
    sources = {}
    for name, records in partitions.items():
        values = [user_prompt(record) for record in records]
        if any(not value for value in values):
            errors.append(f"{name} contains records without a user prompt")
        if len(values) != len(set(values)):
            errors.append(f"{name} contains duplicate user prompts")
        prompts[name] = set(values)
        sources[name] = {
            source for record in records if (source := source_file(record))
        }
        print(
            f"{name}: {len(records)} records, {len(sources[name])} "
            "source groups with provenance"
        )

    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            prompt_overlap = prompts[left] & prompts[right]
            source_overlap = sources[left] & sources[right]
            if prompt_overlap:
                errors.append(
                    f"{left}/{right} share {len(prompt_overlap)} exact prompts"
                )
            if source_overlap:
                errors.append(
                    f"{left}/{right} share {len(source_overlap)} source files"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Split validation passed: no prompt or source-group leakage detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    main(parser.parse_args())
