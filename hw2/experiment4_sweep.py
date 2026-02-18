"""Random hyperparameter search for InvertedPendulum-v4."""

import subprocess
import numpy as np

NUM_TRIALS = 20
GROUP = "pendulum_sweep_v2"
RNG = np.random.default_rng(123)


def log_uniform(low, high):
    return float(np.exp(RNG.uniform(np.log(low), np.log(high))))


def log_uniform_int(low, high):
    return int(round(log_uniform(low, high)))


def perturb_log(center, factor=3.0):
    """Sample log-uniformly within [center/factor, center*factor]."""
    return log_uniform(center / factor, center * factor)


def perturb_log_int(center, factor=3.0, low=1, high=None):
    val = int(round(perturb_log(center, factor)))
    val = max(val, low)
    if high is not None:
        val = min(val, high)
    return val


for i in range(NUM_TRIALS):
    batch_size = perturb_log_int(2653, factor=2.0, low=500)
    rtg = RNG.choice([True, False])
    blr = perturb_log(9.8e-3)
    bgs = perturb_log_int(16, factor=2.0, low=1, high=30)
    discount = 1.0 - perturb_log(1.0 - 0.9983, factor=5.0)  # perturb around 0.9983
    discount = min(discount, 0.9999)
    lr = perturb_log(2.8e-2)
    n_layers = int(RNG.integers(1, 3, endpoint=True))
    layer_size = int(RNG.choice([16, 32, 64]))

    tag = (
        f"b{batch_size}_lr{lr:.1e}_blr{blr:.1e}"
        f"_bgs{bgs}_d{discount:.4f}"
        f"_l{n_layers}_s{layer_size}"
        f"{'_rtg' if rtg else ''}"
    )
    exp_name = f"pendulum_{tag}"

    cmd = [
        "uv", "run", "src/scripts/run.py",
        "--env_name", "InvertedPendulum-v4",
        "-n", "100",
        "-b", str(batch_size),
        "-eb", "1000",
        "--learning_rate", f"{lr}",
        "--baseline_learning_rate", f"{blr}",
        "--baseline_gradient_steps", str(int(bgs)),
        "--discount", f"{discount}",
        "--n_layers", str(n_layers),
        "--layer_size", str(layer_size),
        "--exp_name", exp_name,
        "--group", GROUP,
    ]
    if rtg:
        cmd.append("--use_reward_to_go")

    print(f"\n{'='*60}")
    print(f"Trial {i+1}/{NUM_TRIALS}: {exp_name}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)
