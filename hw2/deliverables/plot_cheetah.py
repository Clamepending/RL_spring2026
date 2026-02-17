"""
Plot learning curves for HalfCheetah experiments (Experiment 2).

Produces two figures:
  1. Baseline Loss: learning curve for the critic/baseline loss (baseline run only).
  2. Eval Return: learning curves for eval average return (both runs).

X-axis: Train_EnvstepsSoFar (number of environment steps)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"

EXP_NAMES = ["cheetah", "cheetah_baseline"]

DIR_PATTERN = re.compile(r"^HalfCheetah-v4_(.+)_sd(\d+)_(\d{8}_\d{6})$")

LABEL_MAP = {
    "cheetah": "No Baseline",
    "cheetah_baseline": "With Baseline",
}


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


def plot_baseline_loss(
    latest_runs: dict[str, Path],
    save_path: Path,
):
    """Plot the baseline (critic) loss over training for the baseline run."""
    name = "cheetah_baseline"
    if name not in latest_runs:
        print(f"Warning: no run found for '{name}', skipping baseline loss plot.")
        return

    df = load_data(latest_runs[name])

    if "Baseline Loss" not in df.columns:
        print(f"Warning: 'Baseline Loss' column not found in {latest_runs[name]}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["Train_EnvstepsSoFar"],
        df["Baseline Loss"],
        color="tab:orange",
        label="Baseline Loss",
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Baseline Loss")
    ax.set_title("HalfCheetah — Baseline (Critic) Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_eval_return(
    latest_runs: dict[str, Path],
    save_path: Path,
):
    """Plot eval average return for both baseline and no-baseline runs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in EXP_NAMES:
        if name not in latest_runs:
            print(f"Warning: no run found for '{name}', skipping.")
            continue
        df = load_data(latest_runs[name])
        ax.plot(
            df["Train_EnvstepsSoFar"],
            df["Eval_AverageReturn"],
            label=LABEL_MAP.get(name, name),
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Return (Eval)")
    ax.set_title("HalfCheetah — Eval Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    latest_runs = find_latest_runs(EXP_DIR)

    print("Latest HalfCheetah runs found:")
    for name in EXP_NAMES:
        if name in latest_runs:
            print(f"  {name}: {latest_runs[name].name}")
        else:
            print(f"  {name}: NOT FOUND")

    out_dir = Path(__file__).resolve().parent

    plot_baseline_loss(
        latest_runs,
        save_path=out_dir / "cheetah_baseline_loss.png",
    )
    plot_eval_return(
        latest_runs,
        save_path=out_dir / "cheetah_eval_return.png",
    )


if __name__ == "__main__":
    main()
