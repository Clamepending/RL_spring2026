"""Generate deliverable plots for HW3 from wandb logs."""

from pathlib import Path

import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = Path(__file__).parent

WANDB_PROJECT = "hw3"


def _step_formatter(x, _pos):
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}K"
    return f"{int(x)}"

RUNS = {
    "2.4_cartpole_dqn": "zoy20glf",
    "2.5_lunarlander_doubleq": "nznl11vt",
    "2.5_mspacman": "e3ghwz1p",
    "2.6_lunarlander_depth1": "zz1iurhc",
    "2.6_lunarlander_depth2": "uhmfuxc5",
    "2.6_lunarlander_depth4": "nznl11vt",
    "3.4_halfcheetah_sac": "2eoqbkpx",
    "3.5_halfcheetah_autotune": "liflmi52",
    "3.6_hopper_singleq": "bu0t1bbn",
    "3.6_hopper_clipq": "0d8zk185",
}


def plot_eval_return(run_id: str, title: str, filename: str):
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    hist = run.history(keys=["Eval_AverageReturn", "_step"], pandas=True)
    eval_data = hist.dropna(subset=["Eval_AverageReturn"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_data["_step"], eval_data["Eval_AverageReturn"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
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

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Average Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_two_subplots(
    run_ids_main: list[tuple[str, str]],
    run_id_secondary: str,
    secondary_key: str,
    title_main: str,
    title_secondary: str,
    ylabel_secondary: str,
    filename: str,
):
    """Two-subplot figure: (left) eval return comparison, (right) a secondary metric for one run."""
    api = wandb.Api()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for run_id, label in run_ids_main:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        hist = run.history(keys=["Eval_AverageReturn", "_step"], pandas=True)
        eval_data = hist.dropna(subset=["Eval_AverageReturn"])
        ax1.plot(eval_data["_step"], eval_data["Eval_AverageReturn"], label=label)

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Eval Average Return")
    ax1.set_title(title_main)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    run = api.run(f"{WANDB_PROJECT}/{run_id_secondary}")
    hist = run.history(keys=[secondary_key, "_step"], pandas=True, samples=10000)
    data = hist.dropna(subset=[secondary_key])
    ax2.plot(data["_step"], data[secondary_key])
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel(ylabel_secondary)
    ax2.set_title(title_secondary)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_dual_metric_comparison(
    run_ids: list[tuple[str, str]],
    metric1: str,
    metric2: str,
    title1: str,
    title2: str,
    ylabel1: str,
    ylabel2: str,
    filename: str,
):
    """Two-subplot figure comparing runs on two different metrics."""
    api = wandb.Api()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for run_id, label in run_ids:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")

        hist1 = run.history(keys=[metric1, "_step"], pandas=True)
        data1 = hist1.dropna(subset=[metric1])
        ax1.plot(data1["_step"], data1[metric1], label=label)

        hist2 = run.history(keys=[metric2, "_step"], pandas=True, samples=10000)
        data2 = hist2.dropna(subset=[metric2])
        ax2.plot(data2["_step"], data2[metric2], label=label)

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel(ylabel1)
    ax1.set_title(title1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel(ylabel2)
    ax2.set_title(title2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_train_and_eval_return(run_id: str, title: str, filename: str):
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")

    eval_hist = run.history(keys=["Eval_AverageReturn", "_step"], pandas=True)
    eval_data = eval_hist.dropna(subset=["Eval_AverageReturn"])

    train_hist = run.history(
        keys=["Train_EpisodeReturn", "_step"], pandas=True, samples=10000,
    )
    train_data = train_hist.dropna(subset=["Train_EpisodeReturn"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        train_data["_step"],
        train_data["Train_EpisodeReturn"],
        alpha=0.3,
        label="Train Episode Return",
    )
    ax.plot(
        eval_data["_step"],
        eval_data["Eval_AverageReturn"],
        label="Eval Average Return",
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_step_formatter))
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Return")
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

    plot_eval_return(
        RUNS["2.5_lunarlander_doubleq"],
        "2.5 Double-Q DQN: LunarLander-v2",
        "2.5_lunarlander_doubleq.png",
    )

    plot_train_and_eval_return(
        RUNS["2.5_mspacman"],
        "2.5 DQN: MsPacman",
        "2.5_mspacman.png",
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

    # 3.4: HalfCheetah SAC eval return
    plot_eval_return(
        RUNS["3.4_halfcheetah_sac"],
        "3.4 SAC: HalfCheetah-v4",
        "3.4_halfcheetah_sac.png",
    )

    # 3.5: Fixed vs auto-tuned temperature comparison + alpha over training
    plot_two_subplots(
        run_ids_main=[
            (RUNS["3.4_halfcheetah_sac"], "Fixed Temperature"),
            (RUNS["3.5_halfcheetah_autotune"], "Auto-tuned Temperature"),
        ],
        run_id_secondary=RUNS["3.5_halfcheetah_autotune"],
        secondary_key="temperature",
        title_main="3.5 Eval Return: Fixed vs Auto-tuned Temperature",
        title_secondary="3.5 Temperature (α) Over Training",
        ylabel_secondary="Temperature (α)",
        filename="3.5_halfcheetah_autotune.png",
    )

    # 3.6: Hopper single-Q vs clipped double-Q — eval return and q values
    plot_dual_metric_comparison(
        run_ids=[
            (RUNS["3.6_hopper_singleq"], "Single-Q"),
            (RUNS["3.6_hopper_clipq"], "Clipped Double-Q"),
        ],
        metric1="Eval_AverageReturn",
        metric2="q_values",
        title1="3.6 Eval Return: Single-Q vs Clipped Double-Q (Hopper-v4)",
        title2="3.6 Q-Values: Single-Q vs Clipped Double-Q (Hopper-v4)",
        ylabel1="Eval Average Return",
        ylabel2="Q-Values",
        filename="3.6_hopper_q_comparison.png",
    )
