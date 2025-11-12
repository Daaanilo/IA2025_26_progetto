"""
Step 1: Train NPC Baseline (DQN Agent) in Crafter

Obiettivo: Implementare e addestrare l'agente RL (NPC) di base sull'ambiente Crafter.
Questo script addestra un agente DQN puro senza l'integrazione dell'architettura HeRoN,
stabilendo una baseline di prestazioni per il confronto futuro.

Usage (PowerShell):
  conda activate ia2025
  python scripts/02_train_npc_baseline.py --config configs/dqn_config.yaml --episodes 1000
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent
from src.constants import NUM_ACHIEVEMENTS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline NPC (DQN) in Crafter")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn_config.yaml",
        help="Path to DQN configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (overrides config)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/baseline",
        help="Directory to save baseline models",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=100, help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--save_freq", type=int, default=500, help="Save model every N episodes"
    )
    return parser.parse_args()


def evaluate_agent(agent, env, num_episodes: int = 10):
    """
    Evaluate agent performance.

    Args:
        agent: DQN agent
        env: Crafter environment
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    all_achievements = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (greedy)
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        achievements = info.get("achievements", {})
        all_achievements.append(achievements.copy())

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    # Achievement statistics
    achievement_counts = {}
    for achievement in all_achievements[0].keys():
        counts = [ach[achievement] for ach in all_achievements]
        achievement_counts[achievement] = sum(1 for c in counts if c > 0)

    num_unlocked = sum(1 for v in achievement_counts.values() if v > 0)

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "achievements_unlocked": num_unlocked,
        "achievement_details": achievement_counts,
    }


def main():
    """Main training loop for baseline NPC."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override episodes if specified
    if args.episodes is not None:
        config["training"]["num_episodes"] = args.episodes

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize environment
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING BASELINE NPC (DQN Agent)")
    print("=" * 60)
    print("\nCreating Crafter environment...")
    env = make_crafter_env(config.get("environment"))

    # Get observation info
    obs, _ = env.reset()
    observation_shape = obs.shape
    num_actions = env.action_space.n

    print(f"Observation shape: {observation_shape}")
    print(f"Number of actions: {num_actions}")

    # Initialize agent
    print("\nInitializing DQN agent...")
    agent = DQNAgent(
        observation_shape=observation_shape, num_actions=num_actions, config=config
    )

    # Training parameters
    num_episodes = config["training"]["num_episodes"]
    max_steps = config["training"].get("max_steps_per_episode", 10000)

    print(f"\nTraining for {num_episodes} episodes")
    print(f"Max steps per episode: {max_steps}")

    # Training loop
    training_log = []
    best_mean_reward = -float("inf")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            # Select action
            action = agent.select_action(obs)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)

            # Train
            if agent.can_train():
                # train_step may return a loss; we don't need to keep it here
                _ = agent.train_step()

            obs = next_obs
            episode_reward += reward
            episode_length += 1

        # Update target network
        agent.update_target_network()

        # Logging
        achievements = info.get("achievements", {})
        num_achievements = sum(1 for v in achievements.values() if v > 0)
        log_entry = {
            "episode": episode + 1,
            "reward": episode_reward,
            "length": episode_length,
            "achievements": num_achievements,
            "epsilon": agent.epsilon,
        }
        training_log.append(log_entry)

        if (episode + 1) % 10 == 0:
            recent_rewards = [entry["reward"] for entry in training_log[-10:]]
            mean_recent = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Mean(10): {mean_recent:.2f} | "
                f"Length: {episode_length} | "
                f"Achievements: {num_achievements}/{NUM_ACHIEVEMENTS} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            print(f"\n--- Evaluation at episode {episode + 1} ---")
            eval_metrics = evaluate_agent(agent, env, num_episodes=10)
            print(
                f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}"
            )
            print(f"Mean Length: {eval_metrics['mean_length']:.2f}")
            print(
                f"Achievements Unlocked: {eval_metrics['achievements_unlocked']}/{NUM_ACHIEVEMENTS}"
            )

            # Save best model
            if eval_metrics["mean_reward"] > best_mean_reward:
                best_mean_reward = eval_metrics["mean_reward"]
                best_path = os.path.join(args.save_dir, "npc_baseline_best.pth")
                agent.save(best_path)
                print(f"New best model saved: {best_path}")

            print()

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"npc_baseline_ep{episode+1}.pth"
            )
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_metrics = evaluate_agent(agent, env, num_episodes=100)
    print(
        f"\nFinal Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}"
    )
    print(f"Final Mean Length: {final_metrics['mean_length']:.2f}")
    print(
        f"Final Achievements Unlocked: {final_metrics['achievements_unlocked']}/{NUM_ACHIEVEMENTS}"
    )
    print("\nAchievement Details:")
    for ach, count in final_metrics["achievement_details"].items():
        if count > 0:
            print(f"  {ach}: {count}/100 episodes")

    # Save final model
    final_path = os.path.join(args.save_dir, "npc_baseline_final.pth")
    agent.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Save training log
    log_path = os.path.join(args.save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "training_log": training_log,
                "final_metrics": final_metrics,
                "config": config,
            },
            f,
            indent=2,
        )
    print(f"Training log saved: {log_path}")

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
