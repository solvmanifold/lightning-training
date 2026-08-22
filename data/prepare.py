#!/usr/bin/env python3
"""
Convert raw data to the messages format expected by train.py.

Input  (raw.jsonl):  one JSON object per line, any of these schemas:
  {"instruction": "...", "response": "..."}           ← Alpaca-style
  {"prompt": "...", "response": "..."}                ← flat
  {"input": "...", "output": "..."}                   ← HF datasets style
  {"messages": [...]}                                 ← already in chat format

Output (train.jsonl / val.jsonl / test.jsonl):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
  python3 data/prepare.py --input data/raw.jsonl --train data/train.jsonl --val data/val.jsonl
"""
import argparse
import json
import hashlib
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Answer clearly and concisely."
)


def normalise(rec: dict) -> dict | None:
    """Return a normalised {messages: [...]} record, or None to skip."""
    if "messages" in rec:
        # Copy only the fields used downstream while retaining source
        # provenance added by generate.py.
        out = {"messages": rec["messages"]}
        if isinstance(rec.get("metadata"), dict):
            out["metadata"] = rec["metadata"]
        return out

    user_text = rec.get("instruction") or rec.get("prompt") or rec.get("input") or ""
    asst_text = rec.get("response") or rec.get("output") or ""

    if not user_text or not asst_text:
        return None

    # Optionally prepend context/input to the instruction (Alpaca convention)
    if "input" in rec and rec.get("instruction"):
        ctx = rec.get("input", "").strip()
        if ctx:
            user_text = f"{user_text}\n\n{ctx}"

    out = {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text.strip()},
            {"role": "assistant", "content": asst_text.strip()},
        ]
    }
    if isinstance(rec.get("metadata"), dict):
        out["metadata"] = rec["metadata"]
    return out


def _message_content(rec: dict, role: str) -> str:
    return next(
        (str(m.get("content", "")).strip() for m in rec["messages"]
         if m.get("role") == role),
        "",
    )


def deduplicate(records: list[dict]) -> tuple[list[dict], int, int]:
    """Deduplicate by user prompt and drop ambiguous conflicting targets.

    Exact duplicate targets add no information. More importantly, keeping one
    prompt with two different answers makes both training and reference-based
    evaluation ill-defined, so every conflicting occurrence is removed.
    """
    by_prompt: dict[str, list[dict]] = {}
    for rec in records:
        prompt = _message_content(rec, "user")
        answer = _message_content(rec, "assistant")
        if prompt and answer:
            by_prompt.setdefault(prompt, []).append(rec)

    kept: list[dict] = []
    exact_removed = conflicts_removed = 0
    for variants in by_prompt.values():
        answers = {_message_content(rec, "assistant") for rec in variants}
        if len(answers) > 1:
            conflicts_removed += len(variants)
            continue
        kept.append(variants[0])
        exact_removed += len(variants) - 1
    return kept, exact_removed, conflicts_removed


def _group_key(rec: dict, index: int) -> str:
    source = rec.get("metadata", {}).get("source_file")
    if source:
        return f"source:{source}"
    # Legacy generated data did not retain its source. Keep each deduplicated
    # record independent, while making the key stable across runs.
    prompt = _message_content(rec, "user")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    return f"record:{digest}:{index}"


def grouped_split(
    records: list[dict], val_split: float, test_split: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records while keeping every source file in exactly one split."""
    groups: dict[str, list[dict]] = {}
    for i, rec in enumerate(records):
        groups.setdefault(_group_key(rec, i), []).append(rec)

    items = list(groups.items())
    random.Random(seed).shuffle(items)
    n_total = len(records)
    targets = {
        "test": round(n_total * test_split),
        "val": round(n_total * val_split),
    }
    splits = {"train": [], "val": [], "test": []}
    for _, group in items:
        if len(splits["test"]) < targets["test"]:
            destination = "test"
        elif len(splits["val"]) < targets["val"]:
            destination = "val"
        else:
            destination = "train"
        splits[destination].extend(group)
    return splits["train"], splits["val"], splits["test"]


def write_jsonl(path: str, records: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main(args):
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = normalise(json.loads(line))
            if rec:
                records.append(rec)

    print(f"Loaded {len(records)} records after normalisation")
    if not 0 <= args.val_split < 1 or not 0 <= args.test_split < 1:
        raise SystemExit("--val-split and --test-split must be in [0, 1)")
    if args.val_split + args.test_split >= 1:
        raise SystemExit("validation and test splits must sum to less than 1")

    records, exact_removed, conflicts_removed = deduplicate(records)
    print(
        f"After deduplication: {len(records)} "
        f"(removed {exact_removed} exact duplicates and "
        f"{conflicts_removed} conflicting records)"
    )
    train_records, val_records, test_records = grouped_split(
        records, args.val_split, args.test_split, args.seed
    )
    print(
        f"Train: {len(train_records)}  Val: {len(val_records)}  "
        f"Test: {len(test_records)}"
    )

    write_jsonl(args.train, train_records)
    write_jsonl(args.val, val_records)
    write_jsonl(args.test, test_records)
    print(f"Wrote {args.train}, {args.val}, and {args.test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,            help="Raw JSONL input file")
    parser.add_argument("--train",     default="data/train.jsonl")
    parser.add_argument("--val",       default="data/val.jsonl")
    parser.add_argument("--test",      default="data/test.jsonl")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction held out for validation")
    parser.add_argument("--test-split",type=float, default=0.05, help="Fraction held out for final evaluation")
    parser.add_argument("--seed",      type=int,   default=42)
    main(parser.parse_args())
