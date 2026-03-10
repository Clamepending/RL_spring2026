"""Generate deliverable plots for HW3 from wandb logs."""

from pathlib import Path

import wandb
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent

WANDB_PROJECT = "hw3"

RUNS = {
    "2.4_cartpole_dqn": "zoy20glf",
    "2.6_lunarlander_depth1": "zz1iurhc",
    "2.6_lunarlander_depth2": "uhmfuxc5",
    "2.6_lunarlander_depth4": "nznl11vt",
}


def plot_eval_return(run_id: str, title: str, filename: str):
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    hist = run.history(keys=["Eval_AverageReturn", "_step"], pandas=True)
    eval_data = hist.dropna(subset=["Eval_AverageReturn"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_data["_step"], eval_data["Eval_AverageReturn"])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_hyperparam_comparison(
    run_ids: list[tuple[str, str]],
    title: str,
    filename: str,
):
    """Plot multiple runs on the same axes for hyperparameter comparison.

    run_ids: list of (run_id, label) tuples.
    """
    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_id, label in run_ids:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        hist = run.history(keys=["Eval_AverageReturn", "_step"], pandas=True)
        eval_data = hist.dropna(subset=["Eval_AverageReturn"])
        ax.plot(eval_data["_step"], eval_data["Eval_AverageReturn"], label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    plot_eval_return(
        RUNS["2.4_cartpole_dqn"],
        "2.4 Basic Q-Learning: CartPole-v1 (DQN)",
        "2.4_cartpole_dqn.png",
    )

    plot_hyperparam_comparison(
        [
            (RUNS["2.6_lunarlander_depth1"], "1 layer"),
            (RUNS["2.6_lunarlander_depth2"], "2 layers"),
            (RUNS["2.6_lunarlander_depth4"], "4 layers"),
        ],
        "2.6 Hyperparameter Study: Network Depth on LunarLander-v2",
        "2.6_lunarlander_hyperparams.png",
    )
