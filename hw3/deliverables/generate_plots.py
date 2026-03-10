"""Generate deliverable plots for HW3 from wandb logs."""

from pathlib import Path

import wandb
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent

WANDB_PROJECT = "hw3"

RUNS = {
    "2.4_cartpole_dqn": "zoy20glf",
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


if __name__ == "__main__":
    plot_eval_return(
        RUNS["2.4_cartpole_dqn"],
        "2.4 Basic Q-Learning: CartPole-v1 (DQN)",
        "2.4_cartpole_dqn.png",
    )
