"""Build the self-contained Nemotron Adaptation Lab static brief."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BRIEF = ROOT / "brief"
SRC = BRIEF / "src"
DIST = BRIEF / "dist"
CARD_ALT = (
    "Nemotron Adaptation Lab decision card showing the base model at 94.92 percent "
    "exact versus LoRA at 73.24 percent, with the verdict keep the prompt."
)
TECHNOLOGIES = {
    "application": ["Python", "Docker", "NeMo AutoModel", "NVIDIA NIM"],
    "ai": [{"name": "NVIDIA Nemotron 3.5 Lightning", "role": "LLM"}],
}


def revision() -> str:
    override = os.environ.get("BRIEF_SOURCE_REVISION", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def built_at() -> str:
    override = os.environ.get("BRIEF_BUILT_AT", "").strip()
    if override:
        return override
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def make_card(path: Path) -> None:
    image = Image.new("RGB", (800, 450), "#101417")
    draw = ImageDraw.Draw(image)
    draw.ellipse((420, -240, 920, 260), fill="#1a3034")
    draw.rectangle((42, 39, 76, 73), fill="#c8ff35")
    draw.text((50, 47), "NA", font=font(11, True), fill="#101417")
    draw.text((91, 47), "NEMOTRON ADAPTATION LAB", font=font(14, True), fill="#f2f0e9")
    draw.text((42, 123), "THE FINE-TUNE", font=font(48), fill="#f2f0e9")
    draw.text((42, 177), "THAT DIDN'T SHIP.", font=font(48), fill="#c8ff35")
    draw.line((42, 258, 758, 258), fill="#364044", width=2)
    draw.text((42, 281), "BASE + PROMPT", font=font(12, True), fill="#9ba4a1")
    draw.text((42, 305), "94.92%", font=font(47), fill="#c8ff35")
    draw.text((313, 281), "LORA + COMPACT", font=font(12, True), fill="#9ba4a1")
    draw.text((313, 305), "73.24%", font=font(47), fill="#ff6c46")
    draw.rectangle((620, 286, 758, 367), fill="#f2f0e9")
    draw.text((638, 301), "DECISION", font=font(11, True), fill="#5c6465")
    draw.text((638, 326), "KEEP THE", font=font(14, True), fill="#101417")
    draw.text((638, 345), "PROMPT", font=font(14, True), fill="#101417")
    draw.text((42, 407), "NEMOTRON 3.5 LIGHTNING  /  LOCKED TEST  /  512 CASES", font=font(11), fill="#78817f")
    image.save(path, "WEBP", quality=90, method=6)


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets").mkdir(parents=True)
    shutil.copy2(SRC / "index.html", DIST / "index.html")
    shutil.copy2(SRC / "brief.css", DIST / "assets" / "brief.css")
    shutil.copy2(SRC / "brief.js", DIST / "assets" / "brief.js")

    source_revision = revision()
    timestamp = built_at()
    data = {
        "project": {
            "name": "Nemotron Adaptation Lab",
            "model": "NVIDIA Nemotron 3.5 Lightning",
            "posture": "evaluation first",
            "status": "base model with full prompt selected",
        },
        "result": {
            "verdict": "do_not_deploy_lora",
            "base_exact": 486,
            "base_exact_percent": 94.921875,
            "lora_exact": 375,
            "lora_exact_percent": 73.2421875,
            "paired_difference_percentage_points": -21.6796875,
            "paired_bootstrap_95_percentage_points": [-25.390625, -17.96875],
            "prompt_token_reduction_percent": 79.29112458388949,
            "base_mean_latency_seconds": 1.1366924148101134,
            "lora_mean_latency_seconds": 2.8099782074048107,
        },
        "paired": {
            "both_correct": 372,
            "base_only_correct": 114,
            "lora_only_correct": 3,
            "both_wrong": 23,
        },
        "dataset": {
            "generator": "deterministic and LLM-free",
            "train_cases": 2048,
            "development_cases": 256,
            "locked_test_cases": 512,
            "split_policy": "disjoint surface-language families",
        },
        "prompt_ladder": [
            {"condition": "Compact", "exact_percent": 29.69, "mean_tokens": 253},
            {"condition": "Manual", "exact_percent": 75.00, "mean_tokens": 515},
            {"condition": "Manual + fixed examples", "exact_percent": 82.81, "mean_tokens": 789},
            {"condition": "Optimized", "exact_percent": 89.06, "mean_tokens": 1025},
            {"condition": "Final", "exact_percent": 96.88, "mean_tokens": 1216},
        ],
        "failure_taxonomy": {
            "compression": 104,
            "retention_days": 41,
            "action": 39,
            "urgency": 17,
            "notify": 4,
            "execution": 2,
            "target_region": 2,
            "schema_invalid": 26,
        },
        "measurement": {
            "date": "2026-08-25",
            "built_at": timestamp,
            "source_revision": source_revision,
            "protocol": "Deterministic Beacon JSON generation; development-only prompt and checkpoint selection; one paired opening of the locked test.",
            "limitations": [
                "Prompt-ladder scores use development cases and are not independent test estimates.",
                "Latency observations are single-request measurements, not a cache-throughput benchmark.",
                "The opened test must not influence further training or prompt edits.",
            ],
        },
    }
    (DIST / "brief.json").write_text(json.dumps(data, indent=2) + "\n")
    card_path = DIST / "assets" / "card.webp"
    make_card(card_path)
    manifest = {
        "schema_version": 1,
        "project": "lightning-training",
        "entrypoint": "index.html",
        "data": "brief.json",
        "card": {
            "path": "assets/card.webp",
            "alt": CARD_ALT,
            "width": 800,
            "height": 450,
            "bytes": card_path.stat().st_size,
        },
        "source_revision": source_revision,
        "built_at": timestamp,
        "technologies": TECHNOLOGIES,
    }
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built {DIST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
