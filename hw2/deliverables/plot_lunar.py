"""
Plot learning curves for LunarLander-v2 GAE-λ experiments (Experiment 3).

Produces one figure with eval average return for each λ value.

X-axis: Train_EnvstepsSoFar (number of environment steps)
Y-axis: Eval_AverageReturn
"""

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"

LAMBDAS = [0, 0.95, 0.98, 0.99, 1]
EXP_NAMES = [f"lunar_lander_lambda{lam}" for lam in LAMBDAS]

DIR_PATTERN = re.compile(r"^LunarLander-v2_(.+)_sd(\d+)_(\d{8}_\d{6})$")

LABEL_MAP = {f"lunar_lander_lambda{lam}": f"λ = {lam}" for lam in LAMBDAS}


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


def plot_eval_return(
    latest_runs: dict[str, Path],
    save_path: Path,
):
    """Plot eval average return for all λ values on a single figure."""
    fig, ax = plt.subplots(figsize=(9, 5))

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

    ax.axhline(y=150, color="gray", linestyle="--", linewidth=1, label="Target (150)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Return (Eval)")
    ax.set_title("LunarLander-v2 — GAE-λ Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def save_logs(
    exp_names: list[str],
    latest_runs: dict[str, Path],
    log_dir: Path,
):
    """Copy log.csv files for the given experiments into log_dir, renamed by exp_name."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in exp_names:
        if name not in latest_runs:
            continue
        src = latest_runs[name] / "log.csv"
        dst = log_dir / f"{name}.csv"
        shutil.copy2(src, dst)
        print(f"  Copied log: {dst}")


def main():
    latest_runs = find_latest_runs(EXP_DIR)

    print("Latest LunarLander runs found:")
    for name in EXP_NAMES:
        if name in latest_runs:
            print(f"  {name}: {latest_runs[name].name}")
        else:
            print(f"  {name}: NOT FOUND")

    out_dir = Path(__file__).resolve().parent
    log_dir = out_dir / "logs"

    save_logs(EXP_NAMES, latest_runs, log_dir)

    plot_eval_return(
        latest_runs,
        save_path=out_dir / "lunar_lander_gae_lambda.png",
    )


if __name__ == "__main__":
    main()
