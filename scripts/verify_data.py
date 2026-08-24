#!/usr/bin/env python3
"""Verify committed smoke data counts, hashes, and minimal schemas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_artifact(root: Path, artifact: dict[str, object]) -> list[dict[str, object]]:
    path = root / str(artifact["path"])
    rows = jsonl_rows(path)
    if len(rows) != int(artifact["rows"]):
        raise RuntimeError(f"{path}: expected {artifact['rows']} rows, got {len(rows)}")
    actual = sha256(path)
    if actual != artifact["sha256"]:
        raise RuntimeError(f"{path}: expected SHA-256 {artifact['sha256']}, got {actual}")
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    hellaswag_manifest = json.loads((root / "data/hellaswag/manifest.json").read_text())
    hellaswag_generator = hellaswag_manifest["generator"]
    if sha256(root / hellaswag_generator["path"]) != hellaswag_generator["sha256"]:
        raise RuntimeError("HellaSwag generator SHA-256 mismatch; regenerate its artifacts")
    hellaswag = verify_artifact(root, hellaswag_manifest["artifact"])
    if any(row["expected"]["label"] not in "ABCD" for row in hellaswag):
        raise RuntimeError("invalid HellaSwag label")

    atlas_manifest = json.loads((root / "data/synthetic/atlas_smoke/manifest.json").read_text())
    atlas_generator = atlas_manifest["generator"]
    if sha256(root / atlas_generator["path"]) != atlas_generator["sha256"]:
        raise RuntimeError("Atlas generator SHA-256 mismatch; regenerate its artifacts")
    catalog = root / atlas_manifest["tool_catalog"]["path"]
    if sha256(catalog) != atlas_manifest["tool_catalog"]["sha256"]:
        raise RuntimeError("Atlas tool catalog SHA-256 mismatch")
    catalog_value = json.loads(catalog.read_text())
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in catalog_value["tools"]
    }
    atlas: list[dict[str, object]] = []
    for artifact in atlas_manifest["artifacts"].values():
        atlas.extend(verify_artifact(root, artifact))
    valid_decisions = {"call", "clarify", "no_action"}
    if any(row["expected"]["decision"] not in valid_decisions for row in atlas):
        raise RuntimeError("invalid Atlas decision")
    if len({row["id"] for row in atlas}) != len(atlas):
        raise RuntimeError("duplicate Atlas case ID")
    for row in atlas:
        expected = row["expected"]
        tool_calls = expected["tool_calls"]
        if expected["decision"] != "call" and tool_calls:
            raise RuntimeError(f"{row['id']}: non-call decision contains a tool call")
        if expected["decision"] == "call" and not tool_calls:
            raise RuntimeError(f"{row['id']}: call decision has no tool call")
        for tool_call in tool_calls:
            name = tool_call["name"]
            if name not in schemas:
                raise RuntimeError(f"{row['id']}: unknown tool {name}")
            schema = schemas[name]
            arguments = tool_call["arguments"]
            required = set(schema["required"])
            if set(arguments) != required:
                raise RuntimeError(
                    f"{row['id']}: {name} arguments {set(arguments)} do not match {required}"
                )
            for key, value in arguments.items():
                property_schema = schema["properties"][key]
                if "enum" in property_schema and value not in property_schema["enum"]:
                    raise RuntimeError(f"{row['id']}: invalid {name}.{key} value {value}")

    print(f"verified {len(hellaswag)} HellaSwag smoke cases")
    print(f"verified {len(atlas)} Atlas smoke cases")


if __name__ == "__main__":
    main()
