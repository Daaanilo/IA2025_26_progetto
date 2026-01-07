"""
Testing Plots Generator for HeRoN in Crafter
Generates visualization graphs for test evaluation data.

Authors: Danilo Gisolfi & Vincenzo Maiellaro
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
TESTING_DIR = BASE_DIR / "testing" / "test_output"
OUTPUT_DIR = Path(__file__).parent / "testing"

# Test configurations with their file paths
TEST_MODELS = {
    "HeRoN Final": {
        "metrics": TESTING_DIR / "heron_crafter_final" / "test_metrics.jsonl",
        "achievements": TESTING_DIR / "heron_crafter_final" / "test_achievement_statistics.json",
        "color": "#d62728"
    },
    "DQN + Helper": {
        "metrics": TESTING_DIR / "dqn_helper_crafter_final" / "test_metrics.jsonl",
        "achievements": TESTING_DIR / "dqn_helper_crafter_final" / "test_achievement_statistics.json",
        "color": "#ff7f0e"
    },
    "Nuovo DQN": {
        "metrics": TESTING_DIR / "nuovo_crafter_dqn_final" / "test_metrics.jsonl",
        "achievements": TESTING_DIR / "nuovo_crafter_dqn_final" / "test_achievement_statistics.json",
        "color": "#1f77b4"
    }
}

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


def setup_plot_style():
    """Setup matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10


def plot_reward_boxplot():
    """Box plot comparing reward distributions across test models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect data
    model_names = []
    shaped_rewards_data = []
    native_rewards_data = []
    colors = []

    for name, config in TEST_MODELS.items():
        data = load_jsonl(config["metrics"])
        if data:
            model_names.append(name)
            shaped_rewards_data.append([d["shaped_reward"] for d in data])
            native_rewards_data.append([d["native_reward"] for d in data])
            colors.append(config["color"])

    # Shaped Reward Box Plot
    bp1 = axes[0].boxplot(shaped_rewards_data, tick_labels=model_names, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Shaped Reward")
    axes[0].set_title("Shaped Reward Distribution (Test)")
    axes[0].grid(True, alpha=0.3)

    # Native Reward Box Plot
    bp2 = axes[1].boxplot(native_rewards_data, tick_labels=model_names, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Native Reward")
    axes[1].set_title("Native Reward Distribution (Test)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_reward_boxplot.png", dpi=150)
    plt.close()
    print("Generated: 01_reward_boxplot.png")


def plot_achievements_bar():
    """Bar chart comparing achievement statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = []
    unique_counts = []
    unlock_ratios = []
    total_instances = []
    colors = []

    for name, config in TEST_MODELS.items():
        stats = load_json(config["achievements"])
        if stats:
            model_names.append(name)
            unique_counts.append(stats["unique_achievements_unlocked"])
            unlock_ratios.append(stats["unlock_ratio"] * 100)
            total_instances.append(stats["total_unlock_instances"])
            colors.append(config["color"])

    # Unique Achievements
    bars1 = axes[0].bar(model_names, unique_counts, color=colors, edgecolor='black')
    axes[0].set_ylabel("Count")
    axes[0].set_title("Unique Achievements Unlocked (out of 22)")
    axes[0].axhline(y=22, color='red', linestyle='--', alpha=0.5, label='Max (22)')
    for bar, count, ratio in zip(bars1, unique_counts, unlock_ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{count}\n({ratio:.1f}%)', ha='center', va='bottom', fontsize=10)

    # Total Unlock Instances
    bars2 = axes[1].bar(model_names, total_instances, color=colors, edgecolor='black')
    axes[1].set_ylabel("Total Instances")
    axes[1].set_title("Total Achievement Unlock Instances")
    for bar, count in zip(bars2, total_instances):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_achievements_bar.png", dpi=150)
    plt.close()
    print("Generated: 02_achievements_bar.png")


def plot_achievement_radar():
    """Radar/spider chart showing achievement profile for each model."""
    # Get achievement unlock counts for each model
    model_data = {}

    for name, config in TEST_MODELS.items():
        stats = load_json(config["achievements"])
        if stats:
            # Create a dict mapping achievement name to unlock count
            ach_counts = {}
            for ach_stat in stats["per_achievement_stats"]:
                ach_counts[ach_stat["achievement_name"]] = ach_stat["unlock_count"]
            model_data[name] = ach_counts

    if not model_data:
        print("No data available for radar chart")
        return

    # Select achievements that were unlocked by at least one model
    unlocked_achievements = set()
    for data in model_data.values():
        for ach, count in data.items():
            if count > 0:
                unlocked_achievements.add(ach)

    # Sort for consistent ordering
    achievements = sorted(list(unlocked_achievements))

    if len(achievements) < 3:
        print("Not enough achievements unlocked for radar chart")
        return

    # Setup radar chart
    num_vars = len(achievements)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for name, config in TEST_MODELS.items():
        if name in model_data:
            values = [model_data[name].get(ach, 0) for ach in achievements]
            # Normalize to 0-1 scale
            max_val = max(max(values), 1)
            values_normalized = [v / max_val for v in values]
            values_normalized += values_normalized[:1]  # Complete the loop

            ax.plot(angles, values_normalized, 'o-', linewidth=2,
                   label=name, color=config["color"])
            ax.fill(angles, values_normalized, alpha=0.25, color=config["color"])

    # Format the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a.replace('_', '\n') for a in achievements], size=8)
    ax.set_title("Achievement Profile Comparison (Normalized)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_achievement_radar.png", dpi=150)
    plt.close()
    print("Generated: 03_achievement_radar.png")


def plot_metrics_comparison():
    """Comprehensive metrics comparison across test models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = []
    avg_shaped = []
    avg_native = []
    avg_moves = []
    avg_achievements = []
    colors = []

    for name, config in TEST_MODELS.items():
        data = load_jsonl(config["metrics"])
        if data:
            model_names.append(name)
            avg_shaped.append(np.mean([d["shaped_reward"] for d in data]))
            avg_native.append(np.mean([d["native_reward"] for d in data]))
            avg_moves.append(np.mean([d["moves"] for d in data]))
            avg_achievements.append(np.mean([d["achievements_unlocked"] for d in data]))
            colors.append(config["color"])

    # Average Shaped Reward
    bars1 = axes[0, 0].bar(model_names, avg_shaped, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].set_title("Average Shaped Reward")
    for bar, val in zip(bars1, avg_shaped):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom')

    # Average Native Reward
    bars2 = axes[0, 1].bar(model_names, avg_native, color=colors, edgecolor='black')
    axes[0, 1].set_ylabel("Average Reward")
    axes[0, 1].set_title("Average Native Reward")
    for bar, val in zip(bars2, avg_native):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom')

    # Average Moves
    bars3 = axes[1, 0].bar(model_names, avg_moves, color=colors, edgecolor='black')
    axes[1, 0].set_ylabel("Average Moves")
    axes[1, 0].set_title("Average Episode Length (Survival)")
    for bar, val in zip(bars3, avg_moves):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom')

    # Average Achievements per Episode
    bars4 = axes[1, 1].bar(model_names, avg_achievements, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel("Average Achievements")
    axes[1, 1].set_title("Average Achievements per Episode")
    for bar, val in zip(bars4, avg_achievements):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_metrics_comparison.png", dpi=150)
    plt.close()
    print("Generated: 04_metrics_comparison.png")


def plot_episode_performance():
    """Line plot showing performance over test episodes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for name, config in TEST_MODELS.items():
        data = load_jsonl(config["metrics"])
        if data:
            episodes = [d["episode"] for d in data]
            rewards = [d["shaped_reward"] for d in data]
            achievements = [d["achievements_unlocked"] for d in data]

            axes[0].plot(episodes, rewards, 'o-', label=name,
                        color=config["color"], linewidth=1.5, markersize=4, alpha=0.7)
            axes[1].plot(episodes, achievements, 'o-', label=name,
                        color=config["color"], linewidth=1.5, markersize=4, alpha=0.7)

    axes[0].set_ylabel("Shaped Reward")
    axes[0].set_title("Shaped Reward per Test Episode")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Achievements")
    axes[1].set_title("Achievements per Test Episode")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_episode_performance.png", dpi=150)
    plt.close()
    print("Generated: 05_episode_performance.png")


def plot_achievement_heatmap():
    """Heatmap showing which achievements each model unlocked during testing."""
    matrix = []
    model_names = []

    for name, config in TEST_MODELS.items():
        stats = load_json(config["achievements"])
        if stats:
            model_names.append(name)
            row = []
            unlocked = set(stats.get("unique_achievement_names", []))
            for ach in ALL_ACHIEVEMENTS:
                row.append(1 if ach in unlocked else 0)
            matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 5))

    cmap = sns.color_palette(["#f0f0f0", "#2ca02c"], as_cmap=True)
    sns.heatmap(matrix, annot=False, cmap=cmap,
                xticklabels=[a.replace('_', '\n') for a in ALL_ACHIEVEMENTS],
                yticklabels=model_names,
                cbar_kws={'label': 'Unlocked'},
                linewidths=0.5, linecolor='white',
                ax=ax)

    ax.set_title("Achievement Unlock Matrix - Test Evaluation (Green = Unlocked)")
    ax.set_xlabel("Achievement")
    ax.set_ylabel("Model")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_achievement_heatmap.png", dpi=150)
    plt.close()
    print("Generated: 06_achievement_heatmap.png")


def plot_summary_table():
    """Generate a summary statistics table as an image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Collect statistics
    rows = []
    for name, config in TEST_MODELS.items():
        data = load_jsonl(config["metrics"])
        stats = load_json(config["achievements"])

        if data and stats:
            row = [
                name,
                f"{np.mean([d['shaped_reward'] for d in data]):.2f}",
                f"{np.std([d['shaped_reward'] for d in data]):.2f}",
                f"{np.mean([d['native_reward'] for d in data]):.2f}",
                f"{np.mean([d['moves'] for d in data]):.1f}",
                f"{np.mean([d['achievements_unlocked'] for d in data]):.2f}",
                f"{stats['unique_achievements_unlocked']}/22",
                f"{stats['unlock_ratio']*100:.1f}%"
            ]
            rows.append(row)

    columns = ['Model', 'Avg Shaped', 'Std Shaped', 'Avg Native',
               'Avg Moves', 'Avg Ach/Ep', 'Unique Ach', 'Unlock %']

    # Create table
    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                    cellLoc='center', colColours=['#4472C4']*len(columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title("Test Evaluation Summary Statistics", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: 07_summary_table.png")


def main():
    """Generate all testing plots."""
    print("=" * 50)
    print("Testing Plots Generator")
    print("=" * 50)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup plot style
    setup_plot_style()

    # Check which data files exist
    print("\nChecking data files...")
    for name, config in TEST_MODELS.items():
        metrics_exists = config["metrics"].exists()
        achievements_exists = config["achievements"].exists()
        print(f"  {name}: metrics={metrics_exists}, achievements={achievements_exists}")

    print("\nGenerating plots...")

    # Generate all plots
    plot_reward_boxplot()
    plot_achievements_bar()
    plot_achievement_radar()
    plot_metrics_comparison()
    plot_episode_performance()
    plot_achievement_heatmap()
    plot_summary_table()

    print("\n" + "=" * 50)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
