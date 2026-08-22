#!/usr/bin/env python3
"""
Generate instruction-tuning data from local source files using a local LLM.

Walks --source-dir for code/config/markdown files, sends each to the model,
and asks it to produce Q&A pairs a developer would ask about that file.
Output is raw JSONL compatible with prepare.py (messages format).

Usage:
  python3 data/generate.py --out data/raw_generated.jsonl
  python3 data/generate.py --source-dir ~/Dev --out data/raw_generated.jsonl --pairs-per-file 4
  # Then split into train/val:
  python3 data/prepare.py --input data/raw_generated.jsonl --train data/train.jsonl --val data/val.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

# Files to include — keep small/medium files that have real content
INCLUDE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md"}
EXCLUDE_PATTERNS = [
    "__pycache__", ".git", "node_modules", ".egg-info", ".venv",
    ".ipynb_checkpoints", "outputs/", "/data/", "loss_curves", "vss-spark/data",
]
# Generic boilerplate filenames to skip regardless of project
EXCLUDE_NAMES = {
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
    "LICENSE.md", "NOTICE.md", "DOCKER-README.md", "CODE-OF-CONDUCT.md",
}
MAX_FILE_CHARS = 12000  # truncate very large files to keep prompts manageable

SYSTEM_PROMPT = (
    "You are a helpful technical assistant specializing in NVIDIA software, "
    "DGX systems, and machine learning infrastructure."
)

GENERATION_PROMPT = """\
Below is a source file from a project. Generate {n} question-and-answer pairs \
that a developer working on this codebase would genuinely ask.

Questions should be specific and practical — things like:
- What does a function/script do and how do you use it?
- Why does a particular config value or patch exist?
- How do you run or deploy something?
- What would go wrong if you changed X?

Avoid generic questions. Ground every question and answer in the actual file content.

Return ONLY a JSON array with objects of the form:
{{"q": "...", "a": "..."}}

No prose before or after the array.

FILE: {filename}
```
{content}
```"""


def should_include(path: Path) -> bool:
    p = str(path)
    if path.suffix not in INCLUDE_SUFFIXES:
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    if any(pat in p for pat in EXCLUDE_PATTERNS):
        return False
    if path.stat().st_size == 0:
        return False
    return True


def collect_files(source_dir: Path) -> list[Path]:
    files = [p for p in source_dir.rglob("*") if p.is_file() and should_include(p)]
    return sorted(files)


def _extract_pairs(raw: str, path: Path) -> list | None:
    """Try several strategies to pull a JSON array out of model output."""
    # 1. Strip markdown code fences
    stripped = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\n?```$", "", stripped.strip(), flags=re.MULTILINE)

    # 2. Try whole response as JSON (cleanest case)
    for candidate in (stripped, raw):
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Find the outermost [...] using a bracket counter (handles nested code)
    start = raw.find("[")
    if start == -1:
        print(f"  No JSON array found for {path}, skipping")
        return None
    depth, i = 0, start
    in_str, escape = False, False
    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
        elif c == "\\" and in_str:
            escape = True
        elif c == '"' and not escape:
            in_str = not in_str
        elif not in_str:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i+1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break
        i += 1

    # 4. Last resort: extract individual {"q":...,"a":...} objects
    objects = re.findall(r'\{\s*"q"\s*:\s*"(.*?)"\s*,\s*"a"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
    if objects:
        return [{"q": q, "a": a} for q, a in objects]

    print(f"  JSON parse error for {path}, skipping")
    return None


def generate_pairs(
    client: OpenAI, model: str, path: Path, source_name: str, n: int
) -> list[dict]:
    content = path.read_text(errors="replace")
    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n... (truncated)"

    prompt = GENERATION_PROMPT.format(
        n=n,
        filename=path.name,
        content=content,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
    except Exception as e:
        print(f"  API error for {path}: {e}")
        return []

    raw = resp.choices[0].message.content.strip()

    pairs = _extract_pairs(raw, path)
    if pairs is None:
        return []

    records = []
    for p in pairs:
        q = str(p.get("q", "")).strip()
        a = str(p.get("a", "")).strip()
        if q and a:
            records.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": a},
                ],
                # Keep provenance so prepare.py can hold out complete source
                # files. Splitting individual Q&A pairs from one file leaks
                # near-identical source context across train and evaluation.
                "metadata": {"source_file": source_name},
            })
    return records


def main(args):
    source_dir = Path(args.source_dir).expanduser()
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="local")

    # Resolve model name from the server
    models = client.models.list()
    model = args.model or models.data[0].id
    print(f"Using model: {model}")

    if args.retry_log:
        # Only retry files that failed in a previous run
        failed = []
        with open(args.retry_log) as f:
            for line in f:
                m = re.search(r"for (.+?): .+, skipping|for (.+?), skipping", line)
                if m:
                    p = Path((m.group(1) or m.group(2)).strip())
                    if p.exists():
                        failed.append(p)
        files = sorted(set(failed))
        print(f"Retrying {len(files)} failed files from {args.retry_log}")
        # Append to existing output
        out_f_mode = "a"
    else:
        files = collect_files(source_dir)
        print(f"Found {len(files)} files under {source_dir}")
        out_f_mode = "w"

    total = 0
    with open(out_path, out_f_mode) as out_f:
        for i, path in enumerate(files):
            rel = path.relative_to(source_dir)
            print(f"[{i+1}/{len(files)}] {rel} ...", end=" ", flush=True)
            pairs = generate_pairs(
                client, model, path, str(rel), args.pairs_per_file
            )
            for rec in pairs:
                out_f.write(json.dumps(rec) + "\n")
            total += len(pairs)
            print(f"{len(pairs)} pairs")
            time.sleep(args.delay)

    print(f"\nDone — {total} pairs written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir",    default="~/Dev",          help="Root dir to walk for source files")
    parser.add_argument("--out",           default="data/raw_generated.jsonl")
    parser.add_argument("--base-url",      default="http://localhost:8123/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--model",         default="",               help="Model ID (auto-detected if empty)")
    parser.add_argument("--pairs-per-file",type=int, default=4,      help="Q&A pairs to generate per file")
    parser.add_argument("--delay",         type=float, default=0.5,  help="Seconds to wait between files")
    parser.add_argument("--retry-log",     default="",               help="Path to a previous run's log; retry only failed files")
    main(parser.parse_args())
