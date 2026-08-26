"""Validate the Nemotron 3.5 Lightning export against the Spark brief contract."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "brief" / "dist"
TEXT_EXTENSIONS = {".html", ".css", ".js", ".json"}
ALLOWED_URLS = {
    "https://github.com/NVIDIA-NeMo/Automodel",
    "https://github.com/NVIDIA-NeMo/Automodel/blob/060cc495ac23350d4882f67ddf96ba663dd3696c/examples/llm_finetune/nemotron/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml",
    "https://github.com/solvmanifold/lightning-training/blob/main/reports/BEACON_JSON_2026-08-25.md",
    "https://github.com/solvmanifold/lightning-training/blob/main/scripts/evaluate_beacon_json.py",
    "https://github.com/solvmanifold/lightning-training/blob/main/scripts/generate_beacon_json.py",
}
FORBIDDEN = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"/home/", r"\blocalhost\b", r"\b127\.0\.0\.1\b", r"\b0\.0\.0\.0\b",
        r"\b192\.168\.", r"\b100\.\d{1,3}\.", r"MagicDNS", r"Tailscale",
    )
]
URL = re.compile(r"https?://[^\s\"'<>)]*")
AI_ROLES = {"VLA", "VLM", "LLM", "VLM/LLM", "Embedding", "Vision encoder", "Detector", "Tracker", "Detector/Tracker", "Reranker", "Speech", "Other"}


class Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inline_scripts = 0
        self.ids: set[str] = set()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and not values.get("src"):
            self.inline_scripts += 1
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            self.hrefs.append(str(values["href"]))


def fail(message: str) -> None:
    raise AssertionError(message)


def relative_path(value: str, field: str) -> Path:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or "\\" in value or any(p in {"", ".", ".."} for p in pure.parts):
        fail(f"{field} is not a normalized relative path")
    path = DIST.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        fail(f"{field} does not resolve to a regular export file")
    return path


def main() -> None:
    required = [DIST / name for name in ("index.html", "brief.json", "manifest.json")]
    if not all(path.is_file() for path in required):
        fail("required export files are missing")
    if any(path.is_symlink() for path in DIST.rglob("*")):
        fail("export must not contain symlinks")

    manifest = json.loads((DIST / "manifest.json").read_text())
    data = json.loads((DIST / "brief.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("project") != "lightning-training":
        fail("manifest identity is invalid")
    relative_path(manifest["entrypoint"], "entrypoint")
    relative_path(manifest["data"], "data")
    card = relative_path(manifest["card"]["path"], "card.path")
    with Image.open(card) as image:
        if image.format != "WEBP" or image.size != (800, 450):
            fail("card must be an 800x450 WebP")
    if card.stat().st_size != manifest["card"]["bytes"] or card.stat().st_size > 300_000:
        fail("card byte metadata or budget is invalid")
    if manifest["card"].get("width") != 800 or manifest["card"].get("height") != 450 or not manifest["card"].get("alt"):
        fail("card metadata is incomplete")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest.get("built_at", "")):
        fail("built_at must be RFC3339 UTC")
    if not manifest.get("source_revision"):
        fail("source_revision is required")
    tech = manifest.get("technologies", {})
    if set(tech) != {"application", "ai"} or not 1 <= len(tech["application"]) <= 8 or len(tech["ai"]) > 6:
        fail("technology metadata is invalid")
    if any(item.get("role") not in AI_ROLES for item in tech["ai"]):
        fail("AI technology role is invalid")

    for path in DIST.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        text = path.read_text()
        for pattern in FORBIDDEN:
            if match := pattern.search(text):
                fail(f"forbidden text {match.group(0)!r} in {path.relative_to(DIST)}")
        for match in URL.finditer(text):
            url = match.group(0).rstrip(".,;:")
            if url not in ALLOWED_URLS:
                fail(f"unapproved external URL in {path.relative_to(DIST)}: {url}")

    html = (DIST / "index.html").read_text()
    checker = Checker()
    checker.feed(html)
    if checker.inline_scripts:
        fail("inline scripts are not allowed")
    if "#main" not in checker.hrefs or "main" not in checker.ids:
        fail("keyboard skip link is missing")
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_beacon_json import (  # noqa: PLC0415
        COMPACT_PROMPT,
        FINAL_FEWSHOT_CASE_IDS,
        FINAL_PROMPT,
    )
    if data.get("winning_prompt", {}).get("system_prompt") != FINAL_PROMPT:
        fail("exported system prompt does not match the evaluator")
    if data.get("winning_prompt", {}).get("fewshot_case_ids") != list(FINAL_FEWSHOT_CASE_IDS):
        fail("exported few-shot IDs do not match the evaluator")
    if FINAL_PROMPT not in html:
        fail("exact frozen system prompt is missing from the HTML")
    if data.get("lora_prompt", {}).get("system_prompt") != COMPACT_PROMPT:
        fail("exported LoRA prompt does not match the evaluator")
    if COMPACT_PROMPT not in html:
        fail("exact LoRA system prompt is missing from the HTML")
    if len(data.get("fewshot_examples", [])) != 4:
        fail("export must contain the four complete few-shot examples")
    print("brief export validation ok")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, json.JSONDecodeError) as exc:
        print(f"brief export validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
