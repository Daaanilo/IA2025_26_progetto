r"""
Run Crafter with live visual display using matplotlib.
Shows the agent acting in the environment with frame updates.
Usage (PowerShell):
  $env:PYTHONPATH = 'C:\Users\Dan98\Desktop\Progetto_IA\IA2025_26_progetto'
  .\venv\Scripts\python.exe .\scripts\run_crafter_visual.py --steps 500 --agent random
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent

# Use matplotlib for live display
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("Run Crafter with visual display")
    p.add_argument("--steps", type=int, default=500, help="Max steps to run")
    p.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "noop", "dqn"],
        help="Agent type: random, noop, or dqn (untrained)",
    )
    p.add_argument(
        "--epsilon", type=float, default=1.0, help="Epsilon for DQN (if agent=dqn)"
    )
    p.add_argument("--device", type=str, default="cpu", help="Device for DQN")
    return p.parse_args()


def main():
    args = parse_args()

    print("Creating Crafter environment...")
    env = make_crafter_env()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.n}")

    # Create agent if needed
    agent = None
    if args.agent == "dqn":
        observation_shape = (obs.shape[2], obs.shape[0], obs.shape[1])  # channels first
        num_actions = env.action_space.n
        print(f"Creating DQN agent (untrained, epsilon={args.epsilon})...")
        agent = DQNAgent(
            observation_shape=observation_shape,
            num_actions=num_actions,
            config={
                "network": {"hidden_layers": [256, 256], "dueling": True},
                "training": {},
            },
            device=args.device,
        )

    # Setup matplotlib figure for live display (larger size for better quality)
    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Crafter - Agent: {args.agent}")
    ax.axis("off")

    # Get initial frame at higher resolution (Crafter supports size parameter)
    frame = env.env.render(size=(512, 512))
    im = ax.imshow(frame, interpolation="nearest")  # nearest for pixel art style
    plt.show()

    step = 0
    episode = 0
    episode_reward = 0.0

    try:
        while step < args.steps:
            # Select action
            if args.agent == "noop":
                action = 0
            elif args.agent == "random":
                action = np.random.randint(0, env.action_space.n)
            elif args.agent == "dqn":
                state_cf = np.transpose(obs, (2, 0, 1))
                action = agent.select_action(state_cf, epsilon=args.epsilon)
            else:
                action = 0

            # Step
            res = env.step(action)
            if isinstance(res, tuple) and len(res) == 5:
                obs, reward, terminated, truncated, info = res
            elif isinstance(res, tuple) and len(res) == 4:
                obs, reward, done, info = res
                terminated = bool(done)
                truncated = False
            else:
                raise RuntimeError("Unexpected step return")

            episode_reward += reward

            # Render and update display at higher resolution
            frame = env.env.render(size=(512, 512))
            im.set_data(frame)
            ax.set_title(
                f"Crafter - Agent: {args.agent} | Step: {step} | Episode: {episode} | Reward: {episode_reward:.1f}",
                fontsize=14,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)  # small pause to allow rendering

            step += 1

            if terminated or truncated:
                print(
                    f"Episode {episode} finished at step {step} with reward {episode_reward:.2f}"
                )
                obs, info = env.reset()
                episode += 1
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        plt.ioff()
        plt.close()
        try:
            if hasattr(env, "close") and callable(env.close):
                env.close()
        except Exception:
            pass
        print("Done")


if __name__ == "__main__":
    main()
