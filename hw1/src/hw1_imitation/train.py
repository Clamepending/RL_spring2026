"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 512 # changed from 128
    lr: float = 3e-4
    weight_decay: float = 0.00
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    print(f"model dims: f{model.hidden_dims}")

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    from evaluation import evaluate_policy
    import numpy as np
    from tqdm import tqdm
    model = torch.compile(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    
    num_steps = 2000
    print(f"training for {num_steps} steps which is {num_steps*config.batch_size} datapoints, {(num_steps*config.batch_size)/len(dataset)} epochs")
    
    
    step = 0
    losses = []
    pbar = tqdm(total=num_steps)
    while step < num_steps:
        for batch_idx, (state, action_chunk) in enumerate(loader):
            if step >= num_steps:
                break
            # state, action_chunk = dataset[0]
            # state, action_chunk = state.unsqueeze(0), action_chunk.unsqueeze(0)
            state, action_chunk = next(iter(DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )))
            print(state[0,0], action_chunk[0,0,0])
            state, action_chunk = state.to(device), action_chunk.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(state, action_chunk)
            loss.backward()
            optimizer.step()
            step += 1
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix(loss=f"{losses[-1]:8.4f}")
            print(loss.item())

            # if step % 10000 == 0: # eval every 100 steps
            #     evaluate_policy(model=model, normalizer=normalizer, device=device, chunk_size=config.chunk_size, video_size=config.video_size, num_video_episodes=config.num_video_episodes, flow_num_steps=config.flow_num_steps, step=step, logger=logger)
            #     print(np.mean(losses[-10000:]))
    pbar.close()
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses)
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    
        

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
