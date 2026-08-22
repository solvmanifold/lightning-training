#!/usr/bin/env python3
"""
Read TensorBoard event files from outputs/ and plot loss curves for all runs.
Saves a PNG next to this script (or --out path).

Usage:
  python3 scripts/plot_losses.py
  python3 scripts/plot_losses.py --logdir outputs --out outputs/loss_curves.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(event_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values


def count_scalars(event_dir: Path) -> int:
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    return sum(len(ea.Scalars(t)) for t in ea.Tags().get("scalars", []))


def find_event_dirs(logdir: Path) -> list[tuple[str, Path]]:
    """
    Recursively find TensorBoard event dirs.  Lightning saves to
    {logdir}/{run_name}/version_N/ — for each run_name, pick the version
    that has the most scalar events (i.e. the most complete run).
    Returns (run_name, event_dir) sorted by run_name.
    """
    candidates: dict[str, list[Path]] = {}
    for event_file in logdir.rglob("events.out.tfevents.*"):
        event_dir = event_file.parent
        label = event_dir.parent.name if event_dir.name.startswith("version_") else event_dir.name
        candidates.setdefault(label, []).append(event_dir)

    results = {}
    for label, dirs in candidates.items():
        best = max(dirs, key=count_scalars)
        if count_scalars(best) > 0:
            results[label] = best
    return sorted(results.items())


def main(args):
    logdir = Path(args.logdir)
    runs = find_event_dirs(logdir)

    if not runs:
        print(f"No TensorBoard event files found under {logdir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overfit smoke — Nemotron-3-Nano-30B LoRA", fontsize=13)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, edir) in enumerate(runs):
        color = colors[i % len(colors)]
        t_steps, t_vals = load_scalars(edir, "train_loss")
        v_steps, v_vals = load_scalars(edir, "val_loss")

        if t_steps:
            axes[0].plot(t_steps, t_vals, label=name, color=color, linewidth=1.8)
        if v_steps:
            axes[1].plot(v_steps, v_vals, label=name, color=color, linewidth=1.8)

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="outputs/logs")
    parser.add_argument("--out",    default="outputs/loss_curves.png")
    main(parser.parse_args())
