"""
Step 6: Final Evaluation and Comparison

Obiettivo: Valutare le prestazioni finali del sistema HeRoN e confrontare
con la baseline NPC. Misurare l'abilità del NPC nello svolgere i task di Crafter,
lo score ottenuto, e gli obiettivi sbloccati.

Questo script:
1. Valuta il modello HeRoN addestrato
2. Confronta con la baseline NPC (se disponibile)
3. Genera report dettagliati sulle prestazioni
4. Analizza i 22 obiettivi di Crafter
5. Produce visualizzazioni e statistiche

Usage (PowerShell):
  conda activate ia2025
  python scripts/07_evaluate_and_compare.py --heron_model models/heron/heron_best.pth --baseline_model models/baseline/npc_baseline_best.pth --episodes 100
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent
from src.llm import Helper, Reviewer
from src.constants import NUM_ACHIEVEMENTS


def _as_int_count(x):
    """Normalize various possible representations of a count to an int.

    Accepts: int, float, set, list, dict (returns len), or numeric-string.
    Falls back to 0 for unexpected values.
    """
    if isinstance(x, (set, list, tuple, dict)):
        return len(x)
    try:
        return int(x)
    except Exception:
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare HeRoN performance"
    )
    parser.add_argument(
        "--heron_model",
        type=str,
        default="models/heron/heron_best.pth",
        help="Path to trained HeRoN model (default: models/heron/heron_best.pth)",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="models/baseline/npc_baseline_best.pth",
        help="Path to baseline NPC model (for comparison) (default: models/baseline/npc_baseline_best.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn_config.yaml",
        help="Path to DQN configuration file",
    )
    parser.add_argument(
        "--heron_config",
        type=str,
        default="configs/heron_config.yaml",
        help="Path to HeRoN configuration file",
    )
    parser.add_argument(
        "--helper_config",
        type=str,
        default="configs/helper_config.yaml",
        help="Path to Helper configuration file",
    )
    parser.add_argument(
        "--reviewer_config",
        type=str,
        default="configs/reviewer_config.yaml",
        help="Path to Reviewer configuration file",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--use_heron_inference",
        action="store_true",
        help="Use full HeRoN architecture during evaluation",
    )
    return parser.parse_args()


class EvaluationResults:
    """Container for evaluation results."""

    def __init__(self, name: str):
        self.name = name
        self.episode_rewards = []
        self.episode_lengths = []
        self.all_achievements = []

    def add_episode(self, reward: float, length: int, achievements: Dict):
        """Add episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.all_achievements.append(achievements.copy())

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        stats = {
            "name": self.name,
            "num_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
        }

        # Achievement statistics
        if self.all_achievements:
            achievement_counts = {}
            for achievement in self.all_achievements[0].keys():
                counts = [ach[achievement] for ach in self.all_achievements]
                achievement_counts[achievement] = {
                    "total_unlocks": sum(1 for c in counts if c > 0),
                    "success_rate": sum(1 for c in counts if c > 0) / len(counts) * 100,
                    "total_count": sum(counts),
                }

            stats["achievements"] = achievement_counts
            stats["unique_achievements_unlocked"] = sum(
                1
                for ach_stats in achievement_counts.values()
                if ach_stats["total_unlocks"] > 0
            )
            stats["avg_achievements_per_episode"] = np.mean(
                [sum(1 for v in ach.values() if v > 0) for ach in self.all_achievements]
            )

        return stats


def evaluate_agent(
    agent: DQNAgent,
    env,
    num_episodes: int,
    render: bool = False,
    name: str = "Agent",
    use_heron: bool = False,
    helper=None,
    reviewer=None,
    heron_config: dict = None,
) -> EvaluationResults:
    """
    Evaluate agent performance.

    Args:
        agent: DQN agent to evaluate
        env: Crafter environment
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        name: Name for this evaluation
        use_heron: Whether to use HeRoN architecture
        helper: Helper instance (if using HeRoN)
        reviewer: Reviewer instance (if using HeRoN)
        heron_config: HeRoN configuration (if using HeRoN)

    Returns:
        EvaluationResults object
    """
    results = EvaluationResults(name)

    # HeRoN settings
    if use_heron and heron_config:
        interaction_config = heron_config.get("heron", {}).get("interaction", {})
        helper_query_freq = interaction_config.get("helper_query_freq", 50)
        actions_per_query = interaction_config.get("actions_per_query", 5)
        use_reviewer_flag = interaction_config.get("use_reviewer", True)
        acceptance_threshold = interaction_config.get("acceptance_threshold", 6.0)

    print(f"\nEvaluating {name}...")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # HeRoN state
        current_action_plan = []
        action_history = []
        last_helper_query_step = 0

        while not done:
            if use_heron and helper:
                # HeRoN decision making
                if episode_length == 0 or (
                    episode_length - last_helper_query_step >= helper_query_freq
                ):
                    state_info = env.get_state_description()
                    recent_actions = [
                        env.action_id_to_name(a) for a in action_history[-10:]
                    ]

                    suggested_actions = helper.suggest_actions(
                        state_info=state_info,
                        recent_actions=recent_actions,
                        num_actions=actions_per_query,
                    )

                    if use_reviewer_flag and reviewer:
                        review = reviewer.review_actions(state_info, suggested_actions)
                        rating = review.get("rating", 5)
                        if rating >= acceptance_threshold:
                            current_action_plan = review.get(
                                "improved_actions", suggested_actions
                            )
                        else:
                            current_action_plan = []
                    else:
                        current_action_plan = suggested_actions

                    last_helper_query_step = episode_length

                # Execute action from plan or use policy
                if current_action_plan:
                    action_name = current_action_plan.pop(0)
                    action = env.action_name_to_id(action_name)
                else:
                    action = agent.select_action(obs, epsilon=0.0)

                action_history.append(action)
            else:
                # Standard DQN policy
                action = agent.select_action(obs, epsilon=0.0)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Store results
        achievements = info.get("achievements", {})
        results.add_episode(episode_reward, episode_length, achievements)

        num_achievements = sum(1 for v in achievements.values() if v > 0)

        if (episode + 1) % 10 == 0:
            print(
                f"  Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Length: {episode_length} | "
                f"Achievements: {num_achievements}/{NUM_ACHIEVEMENTS}"
            )

    return results


def print_comparison(heron_stats: Dict, baseline_stats: Dict = None):
    """
    Print comparison between HeRoN and baseline.

    Args:
        heron_stats: HeRoN evaluation statistics
        baseline_stats: Baseline evaluation statistics (optional)
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    def _as_int_count(x):
        """Normalize various possible representations of a count to an int.

        Accepts: int, float, set, list, dict (returns len), or numeric-string.
        Falls back to 0 for unexpected values.
        """
        if isinstance(x, (set, list, tuple, dict)):
            return len(x)
        try:
            return int(x)
        except Exception:
            return 0

    # Print HeRoN results
    print(f"\n{heron_stats['name']}:")
    print(f"  Episodes: {heron_stats['num_episodes']}")
    print(
        f"  Mean Reward: {heron_stats['mean_reward']:.2f} ± {heron_stats['std_reward']:.2f}"
    )
    print(
        f"  Reward Range: [{heron_stats['min_reward']:.2f}, {heron_stats['max_reward']:.2f}]"
    )
    print(
        f"  Mean Episode Length: {heron_stats['mean_length']:.2f} ± {heron_stats['std_length']:.2f}"
    )
    heron_unique = _as_int_count(heron_stats.get("unique_achievements_unlocked", 0))
    print(f"  Unique Achievements Unlocked: {heron_unique}/{NUM_ACHIEVEMENTS}")
    print(
        f"  Avg Achievements per Episode: {heron_stats['avg_achievements_per_episode']:.2f}"
    )

    # Print baseline results if available
    if baseline_stats:
        print(f"\n{baseline_stats['name']}:")
        print(f"  Episodes: {baseline_stats['num_episodes']}")
        print(
            f"  Mean Reward: {baseline_stats['mean_reward']:.2f} ± {baseline_stats['std_reward']:.2f}"
        )
        print(
            f"  Reward Range: [{baseline_stats['min_reward']:.2f}, {baseline_stats['max_reward']:.2f}]"
        )
        print(
            f"  Mean Episode Length: {baseline_stats['mean_length']:.2f} ± {baseline_stats['std_length']:.2f}"
        )
        baseline_unique = _as_int_count(
            baseline_stats.get("unique_achievements_unlocked", 0)
        )
        print(f"  Unique Achievements Unlocked: {baseline_unique}/{NUM_ACHIEVEMENTS}")
        print(
            f"  Avg Achievements per Episode: {baseline_stats['avg_achievements_per_episode']:.2f}"
        )

        # Compute improvements
        print("\n" + "-" * 80)
        print("IMPROVEMENTS (HeRoN vs Baseline):")
        print("-" * 80)

        reward_improvement = (
            (heron_stats["mean_reward"] - baseline_stats["mean_reward"])
            / abs(baseline_stats["mean_reward"])
            * 100
            if baseline_stats["mean_reward"] != 0
            else 0
        )
        achievement_improvement = heron_unique - baseline_unique
        avg_achievement_improvement = (
            heron_stats["avg_achievements_per_episode"]
            - baseline_stats["avg_achievements_per_episode"]
        )

        print(f"  Reward: {reward_improvement:+.2f}%")
        percent = (
            (achievement_improvement / NUM_ACHIEVEMENTS * 100)
            if NUM_ACHIEVEMENTS
            else 0
        )
        print(
            f"  Unique Achievements: {achievement_improvement:+d} ({percent:+.1f}%)"
        )
        print(f"  Avg Achievements/Episode: {avg_achievement_improvement:+.2f}")

    # Detailed achievement breakdown
    print("\n" + "-" * 80)
    print("ACHIEVEMENT BREAKDOWN")
    print("-" * 80)

    print(f"\n{heron_stats['name']} - Achievement Success Rates:")
    achievements = heron_stats.get("achievements", {})
    sorted_achievements = sorted(
        achievements.items(), key=lambda x: x[1]["success_rate"], reverse=True
    )

    for ach_name, ach_stats in sorted_achievements:
        if ach_stats["total_unlocks"] > 0:
            print(
                f"  {ach_name:25s}: {ach_stats['success_rate']:5.1f}% "
                f"({ach_stats['total_unlocks']:3d}/{heron_stats['num_episodes']} episodes, "
                f"total: {ach_stats['total_count']:3d})"
            )

    if baseline_stats:
        print(f"\n{baseline_stats['name']} - Achievement Success Rates:")
        achievements = baseline_stats.get("achievements", {})
        sorted_achievements = sorted(
            achievements.items(), key=lambda x: x[1]["success_rate"], reverse=True
        )

        for ach_name, ach_stats in sorted_achievements:
            if ach_stats["total_unlocks"] > 0:
                print(
                    f"  {ach_name:25s}: {ach_stats['success_rate']:5.1f}% "
                    f"({ach_stats['total_unlocks']:3d}/{baseline_stats['num_episodes']} episodes, "
                    f"total: {ach_stats['total_count']:3d})"
                )


def save_results(heron_stats: Dict, baseline_stats: Dict, output_dir: str):
    """
    Save evaluation results to files.

    Args:
        heron_stats: HeRoN evaluation statistics
        baseline_stats: Baseline evaluation statistics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "heron": heron_stats,
                "baseline": baseline_stats,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\n✓ Results saved to: {json_path}")

    # Create visualizations
    try:
        create_visualizations(heron_stats, baseline_stats, output_dir, timestamp)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")


def create_visualizations(
    heron_stats: Dict, baseline_stats: Dict, output_dir: str, timestamp: str
):
    """
    Create visualization plots.

    Args:
        heron_stats: HeRoN evaluation statistics
        baseline_stats: Baseline evaluation statistics
        output_dir: Output directory
        timestamp: Timestamp string
    """
    sns.set_style("whitegrid")

    # Figure 1: Reward comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    names = [heron_stats["name"]]
    rewards = [heron_stats["mean_reward"]]
    errors = [heron_stats["std_reward"]]

    if baseline_stats:
        names.append(baseline_stats["name"])
        rewards.append(baseline_stats["mean_reward"])
        errors.append(baseline_stats["std_reward"])

    axes[0].bar(
        names, rewards, yerr=errors, capsize=5, alpha=0.7, color=["#2ecc71", "#e74c3c"]
    )
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Mean Reward Comparison")
    axes[0].grid(axis="y", alpha=0.3)

    # Achievement comparison
    heron_achievements = _as_int_count(
        heron_stats.get("unique_achievements_unlocked", 0)
    )
    baseline_achievements = _as_int_count(
        baseline_stats.get("unique_achievements_unlocked", 0) if baseline_stats else 0
    )

    axes[1].bar(
        names,
        (
            [heron_achievements, baseline_achievements]
            if baseline_stats
            else [heron_achievements]
        ),
        alpha=0.7,
        color=["#2ecc71", "#e74c3c"],
    )
    axes[1].set_ylabel("Unique Achievements Unlocked")
    axes[1].set_ylim(0, NUM_ACHIEVEMENTS)
    axes[1].set_title(f"Achievement Unlocking (out of {NUM_ACHIEVEMENTS})")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved to: {fig_path}")
    plt.close()

    # Figure 2: Achievement heatmap
    if baseline_stats:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Prepare data
        heron_ach = heron_stats.get("achievements", {})
        baseline_ach = baseline_stats.get("achievements", {})

        achievement_names = list(heron_ach.keys())
        heron_rates = [heron_ach[name]["success_rate"] for name in achievement_names]
        baseline_rates = [
            baseline_ach[name]["success_rate"] for name in achievement_names
        ]

        data = np.array([heron_rates, baseline_rates])

        sns.heatmap(
            data,
            xticklabels=achievement_names,
            yticklabels=["HeRoN", "Baseline"],
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            ax=ax,
            cbar_kws={"label": "Success Rate (%)"},
        )

        plt.xticks(rotation=45, ha="right")
        plt.title("Achievement Success Rates Comparison (%)")
        plt.tight_layout()

        heatmap_path = os.path.join(output_dir, f"achievement_heatmap_{timestamp}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        print(f"✓ Heatmap saved to: {heatmap_path}")
        plt.close()


def main():
    """Main evaluation process."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("STEP 6: FINAL EVALUATION AND COMPARISON")
    print("=" * 80)

    # Load configurations
    print("\nLoading configurations...")
    with open(args.config, "r") as f:
        dqn_config = yaml.safe_load(f)

    # Initialize environment
    print("Creating Crafter environment...")
    env = make_crafter_env(dqn_config.get("environment"))

    # Get observation info
    obs, _ = env.reset()
    observation_shape = obs.shape
    num_actions = env.action_space.n

    # Evaluate HeRoN
    print("\n" + "-" * 80)
    print("EVALUATING HERON MODEL")
    print("-" * 80)

    heron_agent = DQNAgent(observation_shape, num_actions, dqn_config)
    heron_agent.load(args.heron_model)
    print(f"✓ HeRoN model loaded from: {args.heron_model}")

    # Load HeRoN components if using full inference
    helper = None
    reviewer = None
    heron_config = None

    if args.use_heron_inference:
        print("\nLoading HeRoN components for inference...")
        with open(args.heron_config, "r") as f:
            heron_config = yaml.safe_load(f)
        with open(args.helper_config, "r") as f:
            helper_config = yaml.safe_load(f)
        with open(args.reviewer_config, "r") as f:
            reviewer_config = yaml.safe_load(f)

        helper = Helper(helper_config)
        reviewer = Reviewer(reviewer_config)
        try:
            reviewer.load_model(use_fine_tuned=True)
            print("✓ HeRoN components loaded")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned Reviewer: {e}")

    heron_results = evaluate_agent(
        agent=heron_agent,
        env=env,
        num_episodes=args.episodes,
        render=args.render,
        name="HeRoN",
        use_heron=args.use_heron_inference,
        helper=helper,
        reviewer=reviewer,
        heron_config=heron_config,
    )
    heron_stats = heron_results.compute_statistics()

    # Evaluate Baseline (if provided)
    baseline_stats = None
    if args.baseline_model:
        print("\n" + "-" * 80)
        print("EVALUATING BASELINE MODEL")
        print("-" * 80)

        baseline_agent = DQNAgent(observation_shape, num_actions, dqn_config)
        baseline_agent.load(args.baseline_model)
        print(f"✓ Baseline model loaded from: {args.baseline_model}")

        baseline_results = evaluate_agent(
            agent=baseline_agent,
            env=env,
            num_episodes=args.episodes,
            render=args.render,
            name="Baseline NPC",
        )
        baseline_stats = baseline_results.compute_statistics()

    # Print comparison
    print_comparison(heron_stats, baseline_stats)

    # Save results
    save_results(heron_stats, baseline_stats, args.output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)

    # Summary
    print("\n📊 Summary:")
    print(f"  HeRoN Mean Reward: {heron_stats['mean_reward']:.2f}")
    heron_summary_ach = _as_int_count(heron_stats.get("unique_achievements_unlocked", 0))
    print(f"  HeRoN Achievements: {heron_summary_ach}/{NUM_ACHIEVEMENTS}")

    if baseline_stats:
        reward_diff = heron_stats["mean_reward"] - baseline_stats["mean_reward"]
        baseline_summary_ach = _as_int_count(
            baseline_stats.get("unique_achievements_unlocked", 0)
        )
        ach_diff = heron_summary_ach - baseline_summary_ach

        print("\n  Improvement over Baseline:")
        print(
            f"    Reward: {reward_diff:+.2f} ({reward_diff/abs(baseline_stats['mean_reward'])*100:+.1f}%)"
        )
        print(f"    Achievements: {ach_diff:+d}")

        if reward_diff > 0 and ach_diff >= 0:
            print("\n  ✅ HeRoN shows positive improvements!")
        elif reward_diff > 0 or ach_diff > 0:
            print("\n  ⚠️  HeRoN shows mixed results")
        else:
            print("\n  ❌ HeRoN needs further tuning")

    print(f"\n  Full results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
