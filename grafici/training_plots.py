import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "training"
OUTPUT_DIR = Path(__file__).parent / "training"

# Variant configurations with their file paths
VARIANTS = {
    "DQN Base": {
        "metrics": TRAINING_DIR / "dqn_base_output" / "dqn_base_metrics.jsonl",
        "achievements": TRAINING_DIR / "dqn_base_output" / "dqn_base_achievement_statistics.json",
        "color": "#1f77b4"
    },
    "DQN + Helper": {
        "metrics": TRAINING_DIR / "dqn_helper_output" / "dqn_helper_metrics.jsonl",
        "achievements": TRAINING_DIR / "dqn_helper_output" / "dqn_helper_achievement_statistics.json",
        "color": "#ff7f0e"
    },
    "HeRoN Initial": {
        "metrics": TRAINING_DIR / "heron_initial_output" / "heron_initial_metrics.jsonl",
        "achievements": TRAINING_DIR / "heron_initial_output" / "heron_initial_achievement_statistics.json",
        "color": "#2ca02c"
    },
    "HeRoN Final": {
        "metrics": TRAINING_DIR / "heron_final_output" / "heron_final_metrics.jsonl",
        "achievements": TRAINING_DIR / "heron_final_output" / "heron_final_achievement_statistics.json",
        "color": "#d62728"
    },
    "HeRoN Random": {
        "metrics": TRAINING_DIR / "heron_random_output" / "heron_random_metrics.jsonl",
        "achievements": TRAINING_DIR / "heron_random_output" / "heron_random_achievement_statistics.json",
        "color": "#9467bd"
    }
}

# Filter VARIANTS to only include those that have existing metrics files
VARIANTS = {name: config for name, config in VARIANTS.items() 
            if (config["metrics"].exists() and config["achievements"].exists())}

# All 22 achievements in Crafter
ALL_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up"
]


def load_jsonl(filepath):
    """Load a JSONL file and return list of dictionaries."""
    data = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def load_json(filepath):
    """Load a JSON file and return dictionary."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def moving_average(data, window=10):
    """Calculate moving average for smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def setup_plot_style():
    """Setup matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10


def plot_learning_curves():
    """Plot shaped reward learning curves for all variants."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, config in VARIANTS.items():
        data = load_jsonl(config["metrics"])
        if data:
            rewards = [d["shaped_reward"] for d in data]
            episodes = range(len(rewards))

            # Plot raw data with transparency
            ax.plot(episodes, rewards, alpha=0.2, color=config["color"])

            # Plot smoothed line
            if len(rewards) > 10:
                smoothed = moving_average(rewards, window=10)
                ax.plot(range(9, len(rewards)), smoothed,
                       label=f"{name} (avg: {np.mean(rewards):.2f})",
                       color=config["color"], linewidth=2)
            else:
                ax.plot(episodes, rewards, label=name,
                       color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Shaped Reward")
    ax.set_title("Learning Curves - Shaped Reward Over Training")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_learning_curves.png", dpi=150)
    plt.close()
    print("Generated: 01_learning_curves.png")


def plot_cumulative_achievements():
    """Plot cumulative achievements over episodes."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, config in VARIANTS.items():
        data = load_jsonl(config["metrics"])
        if data:
            achievements = [d["achievements_unlocked"] for d in data]
            cumulative = np.cumsum(achievements)
            episodes = range(len(cumulative))

            ax.plot(episodes, cumulative,
                   label=f"{name} (total: {cumulative[-1]})",
                   color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Achievements")
    ax.set_title("Cumulative Achievement Unlocks Over Training")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_cumulative_achievements.png", dpi=150)
    plt.close()
    print("Generated: 02_cumulative_achievements.png")


def plot_unique_achievements_bar():
    """Bar chart of unique achievements unlocked per variant."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    unique_counts = []
    colors = []

    for name, config in VARIANTS.items():
        stats = load_json(config["achievements"])
        if stats:
            names.append(name)
            unique_counts.append(stats["unique_achievements_unlocked"])
            colors.append(config["color"])

    bars = ax.barh(names, unique_counts, color=colors, edgecolor='black')

    # Add value labels
    for bar, count in zip(bars, unique_counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
               f'{count}/22 ({count/22*100:.1f}%)',
               va='center', fontsize=10)

    ax.set_xlabel("Unique Achievements Unlocked")
    ax.set_title("Unique Achievements Unlocked per Variant (out of 22)")
    ax.set_xlim(0, 25)
    ax.axvline(x=22, color='red', linestyle='--', alpha=0.5, label='Max (22)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_unique_achievements_bar.png", dpi=150)
    plt.close()
    print("Generated: 03_unique_achievements_bar.png")


def plot_achievement_heatmap():
    """Heatmap showing which achievements each variant unlocked."""
    # Create matrix: variants x achievements
    matrix = []
    variant_names = []

    for name, config in VARIANTS.items():
        stats = load_json(config["achievements"])
        if stats:
            variant_names.append(name)
            row = []
            unlocked = set(stats.get("unique_achievement_names", []))
            for ach in ALL_ACHIEVEMENTS:
                row.append(1 if ach in unlocked else 0)
            matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Create heatmap
    cmap = sns.color_palette(["#f0f0f0", "#2ca02c"], as_cmap=True)
    sns.heatmap(matrix, annot=False, cmap=cmap,
                xticklabels=[a.replace('_', '\n') for a in ALL_ACHIEVEMENTS],
                yticklabels=variant_names,
                cbar_kws={'label': 'Unlocked'},
                linewidths=0.5, linecolor='white',
                ax=ax)

    ax.set_title("Achievement Unlock Matrix (Green = Unlocked)")
    ax.set_xlabel("Achievement")
    ax.set_ylabel("Variant")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_achievement_heatmap.png", dpi=150)
    plt.close()
    print("Generated: 04_achievement_heatmap.png")


def plot_helper_calls():
    """Plot helper calls evolution for HeRoN variants."""
    fig, ax = plt.subplots(figsize=(14, 7))

    heron_variants = ["DQN + Helper", "HeRoN Initial", "HeRoN Final", "HeRoN Random"]

    for name in heron_variants:
        config = VARIANTS[name]
        data = load_jsonl(config["metrics"])
        if data:
            calls = [d["helper_calls"] for d in data]
            episodes = range(len(calls))

            # Plot raw data with transparency
            ax.plot(episodes, calls, alpha=0.2, color=config["color"])

            # Plot smoothed line
            if len(calls) > 10:
                smoothed = moving_average(calls, window=10)
                ax.plot(range(9, len(calls)), smoothed,
                       label=f"{name} (avg: {np.mean(calls):.1f})",
                       color=config["color"], linewidth=2)
            else:
                ax.plot(episodes, calls, label=name,
                       color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Helper Calls")
    ax.set_title("LLM Helper Calls per Episode")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_helper_calls.png", dpi=150)
    plt.close()
    print("Generated: 05_helper_calls.png")


def plot_hallucination_rate():
    """Plot hallucination rate evolution."""
    fig, ax = plt.subplots(figsize=(14, 7))

    heron_variants = ["DQN + Helper", "HeRoN Initial", "HeRoN Final", "HeRoN Random"]

    for name in heron_variants:
        config = VARIANTS[name]
        data = load_jsonl(config["metrics"])
        if data:
            rates = [d.get("hallucination_rate", 0) * 100 for d in data]  # Convert to percentage
            episodes = range(len(rates))

            if len(rates) > 10:
                smoothed = moving_average(rates, window=10)
                ax.plot(range(9, len(rates)), smoothed,
                       label=f"{name} (avg: {np.mean(rates):.2f}%)",
                       color=config["color"], linewidth=2)
            else:
                ax.plot(episodes, rates, label=name,
                       color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("LLM Hallucination Rate Over Training")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_hallucination_rate.png", dpi=150)
    plt.close()
    print("Generated: 06_hallucination_rate.png")


def plot_reward_per_helper_call():
    """Plot reward per helper call efficiency."""
    fig, ax = plt.subplots(figsize=(14, 7))

    heron_variants = ["DQN + Helper", "HeRoN Initial", "HeRoN Final", "HeRoN Random"]

    for name in heron_variants:
        config = VARIANTS[name]
        data = load_jsonl(config["metrics"])
        if data:
            # Filter out episodes with 0 helper calls
            rewards = [d["reward_per_helper_call"] for d in data if d["helper_calls"] > 0]
            episodes = [d["episode"] for d in data if d["helper_calls"] > 0]

            if rewards:
                if len(rewards) > 10:
                    smoothed = moving_average(rewards, window=10)
                    ax.plot(range(9, len(rewards)), smoothed,
                           label=f"{name} (avg: {np.mean(rewards):.4f})",
                           color=config["color"], linewidth=2)
                else:
                    ax.plot(episodes, rewards, label=name,
                           color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward per Helper Call")
    ax.set_title("LLM Efficiency - Reward per Helper Call")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_reward_per_helper_call.png", dpi=150)
    plt.close()
    print("Generated: 07_reward_per_helper_call.png")


def plot_moves_per_episode():
    """Plot episode length (moves) over training."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, config in VARIANTS.items():
        data = load_jsonl(config["metrics"])
        if data:
            moves = [d["moves"] for d in data]
            episodes = range(len(moves))

            if len(moves) > 10:
                smoothed = moving_average(moves, window=10)
                ax.plot(range(9, len(moves)), smoothed,
                       label=f"{name} (avg: {np.mean(moves):.1f})",
                       color=config["color"], linewidth=2)
            else:
                ax.plot(episodes, moves, label=name,
                       color=config["color"], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Moves per Episode")
    ax.set_title("Episode Length (Agent Survival)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_moves_per_episode.png", dpi=150)
    plt.close()
    print("Generated: 08_moves_per_episode.png")


def plot_native_vs_shaped_reward():
    """Compare native reward vs shaped reward."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, config in VARIANTS.items():
        data = load_jsonl(config["metrics"])
        if data:
            native = [d["native_reward"] for d in data]
            shaped = [d["shaped_reward"] for d in data]
            episodes = range(len(native))

            if len(native) > 10:
                native_smooth = moving_average(native, window=10)
                shaped_smooth = moving_average(shaped, window=10)

                axes[0].plot(range(9, len(native)), native_smooth,
                            label=name, color=config["color"], linewidth=2)
                axes[1].plot(range(9, len(shaped)), shaped_smooth,
                            label=name, color=config["color"], linewidth=2)

    axes[0].set_ylabel("Native Reward")
    axes[0].set_title("Native Reward (Environment)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Shaped Reward")
    axes[1].set_title("Shaped Reward (with Bonus)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_native_vs_shaped_reward.png", dpi=150)
    plt.close()
    print("Generated: 09_native_vs_shaped_reward.png")


def plot_summary_statistics():
    """Create a summary bar chart with multiple metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    variant_names = []
    avg_rewards = []
    total_achievements = []
    avg_moves = []
    unique_achievements = []
    colors = []

    for name, config in VARIANTS.items():
        data = load_jsonl(config["metrics"])
        stats = load_json(config["achievements"])

        if data and stats:
            variant_names.append(name)
            avg_rewards.append(np.mean([d["shaped_reward"] for d in data]))
            total_achievements.append(sum([d["achievements_unlocked"] for d in data]))
            avg_moves.append(np.mean([d["moves"] for d in data]))
            unique_achievements.append(stats["unique_achievements_unlocked"])
            colors.append(config["color"])

    # Average Shaped Reward
    axes[0, 0].barh(variant_names, avg_rewards, color=colors, edgecolor='black')
    axes[0, 0].set_xlabel("Average Shaped Reward")
    axes[0, 0].set_title("Average Shaped Reward")

    # Total Achievements
    axes[0, 1].barh(variant_names, total_achievements, color=colors, edgecolor='black')
    axes[0, 1].set_xlabel("Total Achievement Unlocks")
    axes[0, 1].set_title("Total Achievement Unlocks")

    # Average Moves
    axes[1, 0].barh(variant_names, avg_moves, color=colors, edgecolor='black')
    axes[1, 0].set_xlabel("Average Moves per Episode")
    axes[1, 0].set_title("Average Episode Length")

    # Unique Achievements
    axes[1, 1].barh(variant_names, unique_achievements, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel("Unique Achievements (out of 22)")
    axes[1, 1].set_title("Unique Achievements Unlocked")
    axes[1, 1].axvline(x=22, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_summary_statistics.png", dpi=150)
    plt.close()
    print("Generated: 10_summary_statistics.png")


def main():
    """Generate all training plots."""
    print("=" * 50)
    print("Training Plots Generator")
    print("=" * 50)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup plot style
    setup_plot_style()

    # Check which data files exist
    print("\nChecking data files...")
    for name, config in VARIANTS.items():
        metrics_exists = config["metrics"].exists()
        achievements_exists = config["achievements"].exists()
        print(f"  {name}: metrics={metrics_exists}, achievements={achievements_exists}")

    print("\nGenerating plots...")

    # Generate all plots
    plot_learning_curves()
    plot_cumulative_achievements()
    plot_unique_achievements_bar()
    plot_achievement_heatmap()
    plot_helper_calls()
    plot_hallucination_rate()
    plot_reward_per_helper_call()
    plot_moves_per_episode()
    plot_native_vs_shaped_reward()
    plot_summary_statistics()

    print("\n" + "=" * 50)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
