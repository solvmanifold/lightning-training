#!/usr/bin/env python3
"""Convert frozen Beacon cases to NeMo AutoModel's OpenAI-chat JSONL format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluate_beacon_json import COMPACT_PROMPT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sft_row(case: dict[str, object]) -> dict[str, object]:
    user_message = case["messages"][0]
    assistant_content = json.dumps(
        case["expected"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return {
        "id": case["id"],
        "messages": [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": user_message["content"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "tags": case["tags"],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "data/synthetic/beacon_json"
    output_dir = root / "data/derived/beacon_sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "train": source_dir / "train.jsonl",
        "development": source_dir / "development.jsonl",
    }
    artifacts: dict[str, dict[str, object]] = {}
    converted: dict[str, list[dict[str, object]]] = {}
    for split, source in sources.items():
        rows = [sft_row(case) for case in load_jsonl(source)]
        destination = output_dir / f"{split}.jsonl"
        write_jsonl(destination, rows)
        converted[split] = rows
        artifacts[split] = {
            "path": str(destination.relative_to(root)),
            "rows": len(rows),
            "sha256": sha256(destination),
            "source_path": str(source.relative_to(root)),
            "source_sha256": sha256(source),
        }

    overfit = converted["train"][:16]
    overfit_path = output_dir / "overfit_train_16.jsonl"
    write_jsonl(overfit_path, overfit)
    artifacts["overfit_train_16"] = {
        "path": str(overfit_path.relative_to(root)),
        "rows": len(overfit),
        "sha256": sha256(overfit_path),
        "case_ids": [row["id"] for row in overfit],
    }

    manifest = {
        "artifacts": artifacts,
        "format": "openai-chat-jsonl",
        "prompt_profile": "compact",
        "prompt_sha256": hashlib.sha256(COMPACT_PROMPT.encode()).hexdigest(),
        "test_cases_included": 0,
        "version": "beacon-job-v1-sft",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(converted['train'])} training cases")
    print(f"wrote {len(converted['development'])} development cases")
    print(f"wrote {len(overfit)} overfit-smoke cases")
    print("included 0 locked-test cases")


if __name__ == "__main__":
    main()
