#!/usr/bin/env python3
"""
Produce the final comparison artifacts for the weekend run:
  - outputs/comparison/loss_curves.png   — train/val loss across methods
  - outputs/comparison/judge_scores.png  — bar chart of mean judge scores
  - outputs/comparison/results.md        — full writeup with comparison table

Reads:
  outputs/logs/*          — TensorBoard scalars for each run
  outputs/comparison/judge_summary.json  — from eval_judge.py

Usage:
  python3 scripts/compare_methods.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(event_dir: Path, tag: str):
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def count_scalars(event_dir: Path) -> int:
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    return sum(len(ea.Scalars(t)) for t in ea.Tags().get("scalars", []))


def find_event_dirs(logdir: Path) -> dict[str, Path]:
    """For each run_name directory under logdir, pick the version subdir with
    the most scalar events."""
    candidates: dict[str, list[Path]] = {}
    for event_file in logdir.rglob("events.out.tfevents.*"):
        d = event_file.parent
        label = d.parent.name if d.name.startswith("version_") else d.name
        candidates.setdefault(label, []).append(d)
    out: dict[str, Path] = {}
    for label, dirs in candidates.items():
        best = max(dirs, key=count_scalars)
        if count_scalars(best) > 0:
            out[label] = best
    return dict(sorted(out.items()))


def plot_loss_curves(event_dirs, out_path):
    if not event_dirs:
        print("  No event dirs — skipping loss curve plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fine-tuning methods — training loss comparison", fontsize=13)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, d) in enumerate(event_dirs.items()):
        color = colors[i % len(colors)]
        for tag, ax in [("train_loss", axes[0]), ("val_loss", axes[1])]:
            # trl uses different tag names
            steps, vals = load_scalars(d, tag)
            if not steps:
                # fall back to HF Trainer tag names
                alt = {"train_loss": "train/loss", "val_loss": "eval/loss"}.get(tag)
                if alt:
                    steps, vals = load_scalars(d, alt)
            if steps:
                ax.plot(steps, vals, label=name, color=color, linewidth=1.8)

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


def plot_judge_scores(summary, out_path):
    if not summary:
        print("  No judge summary — skipping score plot")
        return
    methods = list(summary.keys())
    scores = [
        summary[m].get("common_mean_score")
        if summary[m].get("common_mean_score") is not None
        else (summary[m].get("mean_score") or 0)
        for m in methods
    ]
    rouges = [(summary[m].get("mean_rouge") or 0) * 5 for m in methods]  # scale to 0-5 for display

    x = range(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], scores, width, label="Paired mean judge score (1-5)", color="#3b82f6")
    ax.bar([i + width/2 for i in x], rouges, width, label="Mean ROUGE-L × 5",       color="#f59e0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Judge score vs ROUGE-L across methods")
    ax.set_ylim(0, 5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")


def build_markdown_table(summary, method_order):
    lines = [
        "| Method | Judged | Paired mean | Mean ROUGE-L | Δ vs base (95% CI) | W/T/L vs base |",
        "|--------|--------|-------------|--------------|--------------------|----------------|",
    ]
    for m in method_order:
        if m not in summary:
            continue
        s = summary[m]
        def fmt(v, pct=False):
            if v is None: return "—"
            if pct: return f"{v:.1f}%"
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        paired = s.get("paired_vs_base", {})
        delta = paired.get("mean_delta")
        ci = paired.get("delta_ci_95")
        delta_text = "—"
        if delta is not None and ci:
            delta_text = f"{delta:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        record = "—"
        if paired:
            record = f"{paired.get('wins', 0)}/{paired.get('ties', 0)}/{paired.get('losses', 0)}"
        lines.append(
            f"| {m} | {s.get('n_judged', 0)}/{s['n']} | "
            f"{fmt(s.get('common_mean_score', s.get('mean_score')))} | "
            f"{fmt(s.get('mean_rouge'))} | {delta_text} | {record} |"
        )
    return "\n".join(lines)


def write_results_md(summary, event_dirs, out_path, metadata=None):
    method_order = ["base", "sft", "qlora", "dpo", "grpo"]
    ordered = [m for m in method_order if m in summary] + \
              [m for m in summary if m not in method_order]
    table = build_markdown_table(summary, ordered)

    event_list = "\n".join(f"- `{name}` → `{d}`" for name, d in event_dirs.items())
    pairwise_lines = [
        "| Comparison | N | Mean Δ | 95% CI | W/T/L |",
        "|------------|---|--------|--------|-------|",
    ]
    for comparison in metadata.get("pairwise", []):
        delta = comparison.get("mean_delta_right_minus_left")
        ci = comparison.get("delta_ci_95")
        if delta is None or not ci:
            continue
        pairwise_lines.append(
            f"| {comparison['right']} − {comparison['left']} | {comparison['n']} | "
            f"{delta:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{comparison['right_wins']}/{comparison['ties']}/{comparison['left_wins']} |"
        )
    pairwise_table = "\n".join(pairwise_lines)

    metadata = metadata or {}
    if not summary:
        completeness = "NO JUDGE SUMMARY AVAILABLE."
    elif metadata.get("complete") is True:
        completeness = "Complete: every response received a judge score."
    elif metadata:
        completeness = (
            f"INCOMPLETE: {metadata.get('missing_judgments', 'unknown')} "
            "judge scores are missing."
        )
    else:
        completeness = "LEGACY SUMMARY: completion status was not recorded."
    if metadata.get("protocol") == "legacy_validation_reuse":
        methodology = """- **Legacy pilot:** evaluated the first 30 examples from the SFT validation set, which was also used for checkpoint selection.
- The scores below are useful for pipeline debugging, not for a clean method comparison.
- Generation used temperature 0.3 and `max_new_tokens=256`."""
    else:
        methodology = """- Data split: source-grouped train / validation / test partitions produced by `data/prepare.py`, after removing duplicate and conflicting prompts.
- Test set: never used for gradient updates or checkpoint selection.
- Evaluation: each method generates one response per test prompt at temperature 0.3 and `max_new_tokens=256`. Super scores each triple on a 1–5 scale at temperature 0.0."""

    response_description = (
        "`responses/{method}.jsonl` — responses produced on the legacy validation subset."
        if metadata.get("protocol") == "legacy_validation_reuse"
        else "`responses/{method}.jsonl` — responses produced on the held-out test set."
    )

    content = f"""# Fine-Tuning Methods Comparison — Results

**Evaluation status:** {completeness}

![Loss curves](loss_curves.png)

![Judge scores](judge_scores.png)

## Results Table

{table}

- **Paired mean**: mean judge score on the same prompts successfully judged for every method.
- **Mean ROUGE-L**: reference-based F1 overlap between model response and the target answer.
- **Δ vs base**: paired mean score difference with a prompt-level bootstrap 95% confidence interval.
- **W/T/L**: wins, ties, and losses against base on prompts judged for both models.

## All pairwise comparisons

The delta and W/T/L are for the second method minus/against the first.

{pairwise_table}

## Methodology

- Dataset: 1795 Q&A pairs synthesized by Nemotron-3-Super over the local `~/Dev` codebase (`data/generate.py`).
{methodology}

## Training Budgets (not matched!)

| Method | Optimizer steps | Grad accum | Effective batch | Starting point | Notes |
|--------|-----------------|------------|-----------------|----------------|-------|
| SFT LoRA r=8 | 500 | 8 | 8 | Base model | ~22 h run |
| QLoRA | 150 | 2 | 2 | Base model (4-bit) | ~4 h run |
| DPO | 150 | 2 | 2 | SFT adapter | ~4 h run |
| GRPO | 100 | 2 | 2 (× 4 generations) | SFT adapter | ~6 h run |

The comparison is not apples-to-apples — SFT has ~3× more optimizer steps than the other methods. SFT is the reference baseline; the new methods are calibrated to fit a weekend. Interpret accordingly.

## TensorBoard event directories

{event_list}

## Files

- `loss_curves.png` — training curves for each method.
- `judge_scores.png` — bar chart of mean judge scores and ROUGE-L.
- `judge_results.jsonl` — every (method, prompt, score) record.
- `judge_summary.json` — aggregate stats.
- {response_description}
"""
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  Wrote {out_path}")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logdir = Path(args.logdir)
    event_dirs = find_event_dirs(logdir)
    print(f"Found {len(event_dirs)} event dirs: {list(event_dirs)}")

    summary = {}
    metadata = {}
    if Path(args.summary).exists():
        payload = json.load(open(args.summary))
        if "methods" in payload:
            summary = payload["methods"]
            metadata = payload
        else:
            # Backward compatibility for summaries produced before paired
            # evaluation was introduced.
            summary = payload
    else:
        print(f"WARNING: {args.summary} not found — skipping judge summary.")

    plot_loss_curves(event_dirs, out_dir / "loss_curves.png")
    plot_judge_scores(summary, out_dir / "judge_scores.png")
    write_results_md(summary, event_dirs, out_dir / "results.md", metadata)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",   default="outputs/logs")
    parser.add_argument("--summary",  default="outputs/comparison/judge_summary.json")
    parser.add_argument("--out-dir",  default="outputs/comparison")
    main(parser.parse_args())
