#!/usr/bin/env python3
"""Generate the deterministic, LLM-free Beacon canonical-JSON dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


SEED = 35_003_501
VERSION = "beacon-job-v1"
SPLIT_SIZES = {"train": 2048, "development": 256, "test": 512}
REGIONS = ("us-north", "us-south", "eu-central", "ap-east")
ACTIONS = ("replicate", "snapshot", "restore", "retire")
URGENCIES = ("low", "normal", "high", "critical")
DEFAULT_RETENTION = {"replicate": 14, "snapshot": 30, "restore": 7, "retire": 90}
DEFAULT_COMPRESSION = {"replicate": True, "snapshot": True, "restore": False, "retire": False}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Beacon job",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "action",
        "resource_id",
        "target_region",
        "urgency",
        "execution",
        "encryption",
        "retention_days",
        "compression",
        "notify",
    ],
    "properties": {
        "schema_version": {"const": VERSION},
        "action": {"enum": list(ACTIONS)},
        "resource_id": {"type": "string", "pattern": "^res_[a-z]{2}_[0-9]{5}$"},
        "target_region": {"type": ["string", "null"], "enum": [*REGIONS, None]},
        "urgency": {"enum": list(URGENCIES)},
        "execution": {"enum": ["plan", "apply"]},
        "encryption": {"enum": ["standard", "strict"]},
        "retention_days": {"type": "integer", "minimum": 1, "maximum": 365},
        "compression": {"type": "boolean"},
        "notify": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[a-z0-9-]+@example\\.invalid$"},
            "uniqueItems": True,
        },
    },
}

ACTION_TEMPLATES = {
    "train": {
        "replicate": ["Mirror {resource} into {region}.", "Copy {resource} over to {region}."],
        "snapshot": ["Take a checkpoint of {resource}.", "Back up {resource}."],
        "restore": ["Recover {resource} in {region}.", "Restore {resource} into {region}."],
        "retire": ["Decommission {resource}.", "Retire {resource} from service."],
    },
    "development": {
        "replicate": ["Create a regional replica of {resource} in {region}."],
        "snapshot": ["Capture a point-in-time copy of {resource}."],
        "restore": ["Roll {resource} back in {region}."],
        "retire": ["Take {resource} permanently out of rotation."],
    },
    "test": {
        "replicate": ["Fan {resource} out to {region}."],
        "snapshot": ["Preserve the current state of {resource}."],
        "restore": ["Reconstitute {resource} within {region}."],
        "retire": ["Sunset {resource}."],
    },
}

CLAUSES = {
    "train": {
        "urgency": {
            "low": "This can wait.",
            "normal": "Use normal priority.",
            "high": "Finish it today.",
            "critical": "This is an incident; do it immediately.",
        },
        "plan": "Preview it only; do not apply changes.",
        "strict": "Treat the resource as regulated and use strict encryption.",
        "retention_days": "Keep the result for {days} days.",
        "retention_weeks": "Retain it for {weeks} weeks.",
        "compression_true": "Compress the stored data.",
        "compression_false": "Do not use compression.",
        "notify": "Notify {emails}.",
    },
    "development": {
        "urgency": {
            "low": "There is no rush.",
            "normal": "Queue it at the usual priority.",
            "high": "I need it completed before today ends.",
            "critical": "Handle this as an active emergency right now.",
        },
        "plan": "Run this as a dry run with no side effects.",
        "strict": "The data is sensitive, so lock encryption down.",
        "retention_days": "Expire the artifact after {days} days.",
        "retention_weeks": "Keep the artifact for {weeks} seven-day periods.",
        "compression_true": "Store it in compressed form.",
        "compression_false": "Leave the bytes uncompressed.",
        "notify": "Send completion notices to {emails}.",
    },
    "test": {
        "urgency": {
            "low": "Whenever capacity permits is fine.",
            "normal": "Give it ordinary scheduling precedence.",
            "high": "Make sure this lands by close of business.",
            "critical": "Production is impaired; execute without delay.",
        },
        "plan": "Simulate the job rather than committing it.",
        "strict": "Apply the regulated-data encryption posture.",
        "retention_days": "Set expiry to {days} days from completion.",
        "retention_weeks": "Use a lifetime of {weeks} weeks.",
        "compression_true": "Pack the payload to save space.",
        "compression_false": "Preserve the payload without packing it.",
        "notify": "On completion, alert {emails}.",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_case(split: str, index: int, rng: random.Random) -> dict[str, object]:
    action = ACTIONS[index % len(ACTIONS)]
    resource = f"res_{split[:2]}_{index:05d}"
    region = REGIONS[(index // len(ACTIONS)) % len(REGIONS)] if action in {"replicate", "restore"} else None
    urgency = URGENCIES[(index // 3) % len(URGENCIES)]
    execution = "plan" if index % 5 == 0 else "apply"
    encryption = "strict" if index % 6 in {0, 1} else "standard"
    explicit_retention = index % 4 != 0
    retention_days = ((index * 13) % 180) + 1 if explicit_retention else DEFAULT_RETENTION[action]
    if explicit_retention and index % 8 == 0:
        retention_days = ((index % 12) + 1) * 7
    explicit_compression = index % 3 == 0
    compression = (index // 3) % 2 == 0 if explicit_compression else DEFAULT_COMPRESSION[action]
    notify_count = index % 3
    notify = sorted(
        {
            f"team-{(index + offset * 17) % 97:02d}@example.invalid"
            for offset in range(notify_count)
        }
    )

    request_parts = [rng.choice(ACTION_TEMPLATES[split][action]).format(resource=resource, region=region)]
    clauses = CLAUSES[split]
    if urgency != "normal" or index % 7 == 0:
        request_parts.append(clauses["urgency"][urgency])
    if execution == "plan":
        request_parts.append(clauses["plan"])
    if encryption == "strict":
        request_parts.append(clauses["strict"])
    if explicit_retention:
        if retention_days % 7 == 0 and index % 8 == 0:
            request_parts.append(clauses["retention_weeks"].format(weeks=retention_days // 7))
        else:
            request_parts.append(clauses["retention_days"].format(days=retention_days))
    if explicit_compression:
        request_parts.append(clauses[f"compression_{str(compression).lower()}"])
    if notify:
        request_parts.append(clauses["notify"].format(emails=" and ".join(reversed(notify))))

    correction_tags: list[str] = []
    if index % 17 == 0:
        old_days = retention_days + 5 if retention_days <= 360 else retention_days - 5
        request_parts.append(f"Correction: use {retention_days} retention days, not {old_days}.")
        correction_tags.append("correction:retention")
    if index % 23 == 0:
        old_urgency = "high" if urgency != "high" else "low"
        request_parts.append(f"Actually, final urgency is {urgency}, not {old_urgency}.")
        correction_tags.append("correction:urgency")

    head, tail = request_parts[0], request_parts[1:]
    rng.shuffle(tail)
    request = " ".join([head, *tail])
    expected = {
        "schema_version": VERSION,
        "action": action,
        "resource_id": resource,
        "target_region": region,
        "urgency": urgency,
        "execution": execution,
        "encryption": encryption,
        "retention_days": retention_days,
        "compression": compression,
        "notify": notify,
    }
    tags = [f"split:{split}", f"action:{action}", f"execution:{execution}", *correction_tags]
    return {
        "id": f"beacon-{split}-{index:05d}",
        "suite": "beacon-canonical-json",
        "messages": [{"role": "user", "content": request}],
        "expected": expected,
        "schema": VERSION,
        "scorers": ["json_parse", "schema", "semantic_exact", "field_metrics"],
        "tags": tags,
    }


def generate(repo_root: Path) -> None:
    output_dir = repo_root / "data/synthetic/beacon_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "schema.json"
    write_json(schema_path, SCHEMA)

    artifacts: dict[str, dict[str, object]] = {}
    prompts: set[str] = set()
    for split_number, (split, size) in enumerate(SPLIT_SIZES.items()):
        rng = random.Random(SEED + split_number)
        cases = [make_case(split, index, rng) for index in range(size)]
        split_prompts = [case["messages"][0]["content"] for case in cases]
        if len(set(split_prompts)) != size or prompts.intersection(split_prompts):
            raise RuntimeError(f"duplicate Beacon prompt in {split}")
        prompts.update(split_prompts)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, cases)
        artifacts[split] = {
            "path": str(path.relative_to(repo_root)),
            "rows": size,
            "sha256": sha256(path),
        }

    manifest = {
        "artifacts": artifacts,
        "generator": {
            "path": "scripts/generate_beacon_json.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "schema": {
            "path": str(schema_path.relative_to(repo_root)),
            "sha256": sha256(schema_path),
        },
        "seed": SEED,
        "version": VERSION,
    }
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    generate(args.repo_root.resolve())


if __name__ == "__main__":
    main()
