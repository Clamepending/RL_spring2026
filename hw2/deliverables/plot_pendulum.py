"""
Plot learning curves for InvertedPendulum-v4: best tuned hyperparameters vs default settings.

Copies the best run's log into deliverables/logs/ and saves the comparison plot.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent / "exp"
OUT_DIR = Path(__file__).resolve().parent

BEST_RUN = "InvertedPendulum-v4_pendulum_b2407_lr2.7e-02_blr3.3e-03_bgs23_d0.9996_l1_s64_rtg_sd1_20260217_212907"
DEFAULT_RUN = "InvertedPendulum-v4_pendulum_sd1_20260217_203341"


def load_log(run_name: str) -> pd.DataFrame:
    return pd.read_csv(EXP_DIR / run_name / "log.csv")


def main():
    best_df = load_log(BEST_RUN)
    default_df = load_log(DEFAULT_RUN)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        best_df["Train_EnvstepsSoFar"],
        best_df["Eval_AverageReturn"],
        label="Tuned (b=2407, lr=2.7e-2, blr=3.3e-3, bgs=23,\n"
              "  γ=0.9996, layers=1, size=64, rtg)",
        linewidth=2,
    )
    ax.plot(
        default_df["Train_EnvstepsSoFar"],
        default_df["Eval_AverageReturn"],
        label="Default (b=5000, lr=5e-3, γ=1.0, layers=2, size=64)",
        linewidth=2,
        linestyle="--",
    )

    ax.axhline(y=1000, color="gray", linestyle=":", linewidth=1, label="Target (1000)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title("InvertedPendulum-v4 — Tuned vs Default Hyperparameters")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = OUT_DIR / "pendulum_learning_curves.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.close(fig)

    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    src = EXP_DIR / BEST_RUN / "log.csv"
    dst = log_dir / "pendulum_best.csv"
    shutil.copy2(src, dst)
    print(f"Saved log: {dst}")


if __name__ == "__main__":
    main()
