import torch
import numpy as np
import gymnasium as gym
import gym_pusht
import imageio
from pathlib import Path
from hw1_imitation.model import FlowMatchingPolicy
from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Ensure we can import from the source directory
sys.path.append("/Users/mark/Desktop/projects/sp2026_RL_class/hw1/src")


def visualize_rollout(
    model_path, output_path, num_episodes=1, chunk_size=16, flow_num_steps=10
):
    # Load model and normalizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model directly since it was saved with torch.save(model, ...)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    data_dir = Path("/Users/mark/Desktop/projects/sp2026_RL_class/hw1/data")
    zarr_path = download_pusht(data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")

    frames = []

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx + 101)  # Another different seed
        done = False

        # Capture the static base frame BEFORE starting the flow to avoid flickering
        base_frame_for_flow = env.render()
        h, w, _ = base_frame_for_flow.shape

        step_count = 0
        while not done and step_count < 100:
            state_norm = (
                torch.from_numpy(normalizer.normalize_state(obs))
                .float()
                .to(device)
                .unsqueeze(0)
            )

            # 1. Visualize the FLOW from noise to trajectory
            action_chunk = torch.randn(1, chunk_size, 2).to(device)

            # Define colors for the chunk trajectory gradient
            colors = cm.plasma(np.linspace(0, 1, chunk_size))

            for i in range(flow_num_steps + 1):
                # Overlay current action_chunk prediction on the STATIC base frame
                scale = w / 512.0
                curr_chunk_denorm = normalizer.denormalize_action(
                    action_chunk.detach().cpu().numpy()[0]
                )
                curr_chunk_scaled = curr_chunk_denorm * scale

                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(base_frame_for_flow)
                ax.axis("off")

                # Plot the chunk with gradient
                for j in range(chunk_size):
                    ax.scatter(
                        curr_chunk_scaled[j, 0],
                        curr_chunk_scaled[j, 1],
                        color=colors[j],
                        s=15,
                        alpha=0.8,
                        edgecolors="black",
                        linewidths=0.5,
                    )
                if chunk_size > 1:
                    for j in range(chunk_size - 1):
                        ax.plot(
                            curr_chunk_scaled[j : j + 2, 0],
                            curr_chunk_scaled[j : j + 2, 1],
                            color=colors[j],
                            alpha=0.5,
                            linewidth=2,
                        )

                ax.text(
                    5,
                    15,
                    f"Flow Inference: {i}/{flow_num_steps}",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.7),
                )

                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[
                    :, :, :3
                ]
                frames.append(frame)
                plt.close(fig)

                if i < flow_num_steps:
                    with torch.no_grad():
                        pred_vel = model.forward(
                            state=state_norm,
                            action_chunk=action_chunk,
                            timestep=torch.Tensor([i / flow_num_steps]).to(device),
                        )
                        action_chunk += pred_vel * (1.0 / flow_num_steps)

            # 2. Agent follows the trajectory
            final_chunk = normalizer.denormalize_action(
                action_chunk.detach().cpu().numpy()[0]
            )
            final_chunk = np.clip(
                final_chunk, env.action_space.low, env.action_space.high
            )

            for j in range(chunk_size):
                obs, reward, terminated, truncated, info = env.step(
                    final_chunk[j].astype(np.float32)
                )
                step_count += 1

                # Update the base frame for the NEXT flow visualization and for the execution visualization
                base_frame_for_flow = env.render()

                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(base_frame_for_flow)
                ax.axis("off")

                # Show remaining trajectory
                scale = w / 512.0
                final_chunk_scaled = final_chunk * scale
                for k in range(j, chunk_size):
                    ax.scatter(
                        final_chunk_scaled[k, 0],
                        final_chunk_scaled[k, 1],
                        color=colors[k],
                        s=15,
                        alpha=0.8,
                        edgecolors="black",
                        linewidths=0.5,
                    )
                if j < chunk_size - 1:
                    for k in range(j, chunk_size - 1):
                        ax.plot(
                            final_chunk_scaled[k : k + 2, 0],
                            final_chunk_scaled[k : k + 2, 1],
                            color=colors[k],
                            alpha=0.5,
                            linewidth=2,
                        )

                ax.text(
                    5,
                    15,
                    f"Executing Chunk Step: {j + 1}/{chunk_size}",
                    color="cyan",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.7),
                )

                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[
                    :, :, :3
                ]
                frames.append(frame)
                plt.close(fig)

                if terminated or truncated:
                    done = True
                    break

            if done:
                break

    env.close()
    # Save as GIF with 10fps as requested
    imageio.mimsave(output_path, frames, fps=10)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    model_path = "/Users/mark/Desktop/projects/sp2026_RL_class/hw1/exp/flow/wandb/files/checkpoints/checkpoint_step_80000.pkl"
    output_path = (
        "/Users/mark/Desktop/projects/sp2026_RL_class/hw1/flow_visualization.gif"
    )
    visualize_rollout(model_path, output_path, num_episodes=1, chunk_size=8)
