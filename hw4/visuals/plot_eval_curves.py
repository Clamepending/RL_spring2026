#!/usr/bin/env python3
"""Plot GRPO vs GR-REINFORCE eval curves on math_hard."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EVAL_KEY = "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser"

RUNS = [
    ("GR-REINFORCE", "reinforce_math_hard_metrics.jsonl", "#2196F3"),
    ("GRPO", "grpo_math_hard_metrics.jsonl", "#E91E63"),
]


def load_eval_curve(path: Path) -> tuple[list[int], list[float]]:
    """Load eval metric from metrics JSONL, taking only the last training session."""
    all_points: list[tuple[int, float]] = []
    for line in open(path):
        r = json.loads(line)
        step = r.get("step")
        val = r.get("metrics", {}).get(EVAL_KEY)
        if val is not None:
            all_points.append((step, val))

    # Split on step-0 resets; keep the last session
    sessions: list[list[tuple[int, float]]] = []
    for s, v in all_points:
        if s == 0 and sessions:
            sessions.append([])
        if not sessions:
            sessions.append([])
        sessions[-1].append((s, v))
    last = sessions[-1] if sessions else all_points

    # Deduplicate by step (keep last occurrence)
    seen: dict[int, float] = {}
    for s, v in last:
        seen[s] = v
    steps = sorted(seen.keys())
    vals = [seen[s] for s in steps]
    return steps, vals


def main():
    visuals_dir = Path(__file__).parent
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, fname, color in RUNS:
        path = visuals_dir / fname
        if not path.exists():
            print(f"Missing: {path}")
            continue
        steps, vals = load_eval_curve(path)
        ax.plot(
            steps,
            [v * 100 for v in vals],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=label,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Eval Exact Match (%)", fontsize=12)
    ax.set_title(
        "Math-Hard Eval: GRPO vs GR-REINFORCE",
        fontsize=13,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_xlim(left=-5)

    fig.tight_layout()
    out = visuals_dir / "grpo_vs_reinforce_eval.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
