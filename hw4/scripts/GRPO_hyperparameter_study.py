from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Default: batch_size=8, group_size=6 → 48 completions
BATCH_SIZE = 8
GROUP_SIZE = 6
ROLLOUT_SIZE = BATCH_SIZE * GROUP_SIZE
STEPS = 51

param_sweeps = [
    ("ppo_epochs", "--ppo_epochs", (1, 2, 4, 8, 16, 32)),
    ("minibatch_size", "--minibatch_size", (2, 4, 8, 16, 48)),
    ("kl_coef", "--kl_coef", (0, 0.02, 0.05, 0.2, 0.4, 1, 10, float('inf'))),
    ("clip_eps", "--clip_eps", (0.05, 0.1, 0.2, 0.4, 0.8, float('inf'))),
    ("grad_accum_steps", "--grad_accum_steps", (1, 3, 6, 12, 24, 48)),
]

BASE_ARGS = [
    "--task", "format_copy",
    "--algo", "grpo",
    "--batch_size", str(BATCH_SIZE),
    "--group_size", str(GROUP_SIZE),
    "--min_new_tokens", "1",
    "--max_new_tokens", "24",
    "--lr", "3e-5",
    "--ppo_epochs", "2",
    "--minibatch_size", "8",
    "--grad_accum_steps", "6",
    "--clip_eps", "0.2",
    "--kl_coef", "0.05",
    "--max_grad_norm", "0.5",
    "--steps", str(STEPS),
    "--wandb_enabled",
    "--wandb_project", "llm-rl-hw4",
    "--sample_markdown_log_interval", "1",
    "--sample_log_interval", "10",
    "--sample_log_n", "6",
    "--eval_interval", "5",
    "--save_interval", "50",
    "--warmup_steps", "10",
]


def build_args_for_sweep(param_name: str, value) -> list[str]:
    """Build CLI args, handling minibatch_size → grad_accum_steps coupling."""
    args = BASE_ARGS.copy()

    if param_name == "ppo_epochs":
        args[args.index("--ppo_epochs") + 1] = str(value)
    elif param_name == "minibatch_size":
        args[args.index("--minibatch_size") + 1] = str(value)
        # Keep effective batch = rollout_size
        grad_accum = max(1, ROLLOUT_SIZE // value)
        args[args.index("--grad_accum_steps") + 1] = str(grad_accum)
    elif param_name == "kl_coef":
        args[args.index("--kl_coef") + 1] = str(value)
    elif param_name == "clip_eps":
        args[args.index("--clip_eps") + 1] = str(value)
    elif param_name == "grad_accum_steps":
        args[args.index("--grad_accum_steps") + 1] = str(value)
    else:
        raise ValueError(f"Unknown param: {param_name}")

    return args


def run_train(args: list[str], output_dir: str, cwd: Path) -> int:
    """Run training via Modal. Returns exit code."""
    cmd = ["uv", "run", "python", "-m", "hw4.train", "--output_dir", output_dir, "--wandb_name", output_dir.split("/")[-1], *args]
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    hw4_root = Path(__file__).resolve().parent.parent
    failed = []
    for param_name, _flag, values in param_sweeps:
        for value in values:
            args = build_args_for_sweep(param_name, value)
            run_name = f"hp_{param_name}_{value}"
            output_dir = f"runs/{run_name}"

            code = run_train(args, output_dir, hw4_root)
            if code != 0:
                failed.append((param_name, value, code))

    if failed:
        print("\nFailed runs:", failed)
        sys.exit(1)
    print("\nAll runs completed.")


if __name__ == "__main__":
    main()