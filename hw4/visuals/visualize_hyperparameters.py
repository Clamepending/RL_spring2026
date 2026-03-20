#!/usr/bin/env python3
"""Visualize GRPO hyperparameter study results."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = [
    ("eval/format_copy_fraction_predicted_number_matches_target_integer_exactly", "Eval Exact Match"),
    ("rollout/mean_total_reward_across_all_completions_in_batch_and_groups", "Rollout Mean Reward"),
    ("train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches", "Approx KL Divergence"),
    ("train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches", "PPO Clipped Fraction"),
    ("train/gradient_global_norm_after_clipping_mean_over_optimizer_steps", "Gradient Norm"),
]


def load_run(metrics_path):
    """Load one run's metrics. Returns {param: value, "steps": [...], metric_key: [...]} or None."""
    name = metrics_path.parent.name
    m = re.match(r"hp_([a-z_]+)_(.+)", name)
    if not m:
        return None
    param_name, val_str = m.groups()
    try:
        val = int(val_str)
    except ValueError:
        val = float(val_str)  # handles floats, inf, -inf

    data = {"param": param_name, "value": val, "steps": [], **{k: [] for k, _ in METRICS}}
    for line in open(metrics_path):
        r = json.loads(line.strip())
        step = r.get("step")
        if step is None:
            continue
        data["steps"].append(step)
        for k, _ in METRICS:
            data[k].append(r.get("metrics", {}).get(k, float("nan")))
    return data


def get_xy(data, metric_key):
    """(steps, values) for a metric, sorted by step, deduped."""
    pairs = [(s, v) for s, v in zip(data["steps"], data[metric_key]) if v == v]  # v==v filters NaN
    pairs = sorted(pairs, key=lambda p: p[0])
    steps, vals = [], []
    for s, v in pairs:
        if steps and steps[-1] == s:
            vals[-1] = v
        else:
            steps.append(s)
            vals.append(v)
    return steps, vals


def main():
    runs_dir = Path(__file__).parent.parent / "runs"
    out_dir = Path(__file__).parent
    if not runs_dir.exists():
        print(f"Not found: {runs_dir}")
        return

    # Group by param
    by_param = {}
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        f = d / "metrics.jsonl"
        if not f.exists():
            continue
        data = load_run(f)
        if data:
            p = data["param"]
            by_param.setdefault(p, {})[data["value"]] = data

    for param, value_data in by_param.items():
        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        for ax, (metric_key, label) in zip(axes, METRICS):
            for i, (val, data) in enumerate(sorted(value_data.items())):
                x, y = get_xy(data, metric_key)
                if x:
                    ax.plot(x, y, label=val)
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
        axes[-1].set_xlabel("Step")
        plt.suptitle(f"GRPO: {param.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(out_dir / f"grpo_{param}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: grpo_{param}.png")


if __name__ == "__main__":
    main()
