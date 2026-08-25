#!/usr/bin/env python3
"""Download pinned HellaSwag validation data and build chat-MC artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


SOURCE_REVISION = "a29ff8e9a04bba4bd6588223785ce105328adc57"
SOURCE_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/"
    f"{SOURCE_REVISION}/data/hellaswag_val.jsonl"
)
SOURCE_SHA256 = "0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588"
SOURCE_ROWS = 10_042
SMOKE_ROWS = 100
SYSTEM_PROMPT = (
    "Choose the most plausible continuation. Respond with exactly one capital "
    "letter: A, B, C, or D."
)
USER_TEMPLATE_ID = "hellaswag-chat-mc-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download(source_path: Path) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists() and sha256(source_path) == SOURCE_SHA256:
        return

    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "lightning-training/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tempfile.NamedTemporaryFile(dir=source_path.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
            while chunk := response.read(1024 * 1024):
                tmp.write(chunk)
    try:
        actual = sha256(temp_path)
        if actual != SOURCE_SHA256:
            raise RuntimeError(f"HellaSwag SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual}")
        os.replace(temp_path, source_path)
    finally:
        temp_path.unlink(missing_ok=True)


def prompt(row: dict[str, object]) -> str:
    labels = "ABCD"
    endings = row["endings"]
    assert isinstance(endings, list) and len(endings) == 4
    choices = "\n".join(f"{label}. {ending}" for label, ending in zip(labels, endings))
    return f"Context: {row['ctx']}\n\nContinuations:\n{choices}\n\nAnswer:"


def normalize(source: dict[str, object], source_ind_occurrence: int) -> dict[str, object]:
    label_index = int(source["label"])
    source_ind = int(source["ind"])
    duplicate_suffix = "" if source_ind_occurrence == 1 else f"-dup{source_ind_occurrence}"
    return {
        "id": f"hellaswag-val-{source_ind:05d}{duplicate_suffix}",
        "suite": "hellaswag-chat-mc",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt(source)},
        ],
        "expected": {
            "label": "ABCD"[label_index],
            "label_index": label_index,
        },
        "scorers": ["strict_mc_label"],
        "metadata": {
            "source_ind": int(source["ind"]),
            "split": source["split"],
            "split_type": source["split_type"],
            "activity_label": source["activity_label"],
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(repo_root: Path) -> None:
    source_path = repo_root / "data/raw/hellaswag/hellaswag_val.jsonl"
    smoke_path = repo_root / "data/hellaswag/chat_mc_smoke_100.jsonl"
    full_path = repo_root / "data/derived/hellaswag/chat_mc_validation_10042.jsonl"
    manifest_path = repo_root / "data/hellaswag/manifest.json"
    download(source_path)

    rows: list[dict[str, object]] = []
    source_ind_counts: Counter[int] = Counter()
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            source_ind = int(source["ind"])
            source_ind_counts[source_ind] += 1
            rows.append(normalize(source, source_ind_counts[source_ind]))

    if len(rows) != SOURCE_ROWS:
        raise RuntimeError(f"expected {SOURCE_ROWS} source rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("normalized HellaSwag case IDs are not unique")

    write_jsonl(smoke_path, rows[:SMOKE_ROWS])
    write_jsonl(full_path, rows)
    prompt_hash = hashlib.sha256(
        json.dumps(
            {"system": SYSTEM_PROMPT, "user_template": USER_TEMPLATE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_json(
        manifest_path,
        {
            "artifact": {
                "path": str(smoke_path.relative_to(repo_root)),
                "rows": SMOKE_ROWS,
                "sha256": sha256(smoke_path),
            },
            "full_artifact": {
                "path": str(full_path.relative_to(repo_root)),
                "rows": len(rows),
                "sha256": sha256(full_path),
            },
            "generator": {
                "path": "scripts/prepare_hellaswag.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "protocol": "hellaswag-chat-mc",
            "prompt_sha256": prompt_hash,
            "source": {
                "revision": SOURCE_REVISION,
                "rows": SOURCE_ROWS,
                "sha256": SOURCE_SHA256,
                "url": SOURCE_URL,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare(args.repo_root.resolve())


if __name__ == "__main__":
    main()
