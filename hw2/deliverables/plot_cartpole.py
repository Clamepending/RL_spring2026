"""
Plot learning curves for CartPole small-batch and large-batch experiments.

Produces two figures:
  1. Small batch (b=1000): cartpole, cartpole_rtg, cartpole_na, cartpole_rtg_na
  2. Large batch (b=4000): cartpole_lb, cartpole_lb_rtg, cartpole_lb_na, cartpole_lb_rtg_na

X-axis: Train_EnvstepsSoFar (number of environment steps)
Y-axis: Eval_AverageReturn
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"

SMALL_BATCH_NAMES = ["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"]
LARGE_BATCH_NAMES = ["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"]

# Pattern: CartPole-v0_{exp_name}_sd{seed}_{timestamp}
DIR_PATTERN = re.compile(r"^CartPole-v0_(.+)_sd(\d+)_(\d{8}_\d{6})$")


def find_latest_runs(exp_dir: Path) -> dict[str, Path]:
    """Return a dict mapping exp_name -> path of the latest run directory."""
    runs: dict[str, list[tuple[str, Path]]] = {}
    for entry in sorted(exp_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = DIR_PATTERN.match(entry.name)
        if m is None:
            continue
        exp_name, _seed, timestamp = m.groups()
        runs.setdefault(exp_name, []).append((timestamp, entry))

    latest: dict[str, Path] = {}
    for exp_name, entries in runs.items():
        entries.sort(key=lambda x: x[0])
        latest[exp_name] = entries[-1][1]
    return latest


def load_data(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "log.csv"
    return pd.read_csv(csv_path)


def make_plot(
    exp_names: list[str],
    latest_runs: dict[str, Path],
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in exp_names:
        if name not in latest_runs:
            print(f"Warning: no run found for '{name}', skipping.")
            continue
        df = load_data(latest_runs[name])
        ax.plot(
            df["Train_EnvstepsSoFar"],
            df["Eval_AverageReturn"],
            label=name,
        )

    ax.set_xlabel("Environment Steps (Train_EnvstepsSoFar)")
    ax.set_ylabel("Average Return (Eval_AverageReturn)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    latest_runs = find_latest_runs(EXP_DIR)

    print("Latest runs found:")
    for name, path in sorted(latest_runs.items()):
        print(f"  {name}: {path.name}")

    out_dir = Path(__file__).resolve().parent
    make_plot(
        SMALL_BATCH_NAMES,
        latest_runs,
        title="CartPole — Small Batch (b=1000)",
        save_path=out_dir / "cartpole_small_batch.png",
    )
    make_plot(
        LARGE_BATCH_NAMES,
        latest_runs,
        title="CartPole — Large Batch (b=4000)",
        save_path=out_dir / "cartpole_large_batch.png",
    )


if __name__ == "__main__":
    main()
