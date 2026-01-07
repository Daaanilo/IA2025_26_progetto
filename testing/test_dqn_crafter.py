"""
Test Script for Trained DQN Models on Crafter Environment

Tests trained DQN models and generates evaluation metrics using the existing
EvaluationSystem infrastructure.

Usage:
    python testing/test_dqn_crafter.py --model training/heron_final_output/models/heron_crafter_final
    python testing/test_dqn_crafter.py --model training/dqn_base_output/models/nuovo_crafter_dqn_final -e 50
    python testing/test_dqn_crafter.py --model <path> --seed 42 --output-dir results/test_output
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from evaluation.evaluation_system import EvaluationSystem
from evaluation.evaluation_plots import generate_all_plots


# Achievement name-to-ID mapping (same as training scripts)
ACHIEVEMENT_NAME_TO_ID = {
    'collect_coal': 0,
    'collect_diamond': 1,
    'collect_drink': 2,
    'collect_iron': 3,
    'collect_sapling': 4,
    'collect_stone': 5,
    'collect_wood': 6,
    'defeat_skeleton': 7,
    'defeat_zombie': 8,
    'eat_cow': 9,
    'eat_plant': 10,
    'make_iron_pickaxe': 11,
    'make_iron_sword': 12,
    'make_stone_pickaxe': 13,
    'make_stone_sword': 14,
    'make_wood_pickaxe': 15,
    'make_wood_sword': 16,
    'place_furnace': 17,
    'place_plant': 18,
    'place_stone': 19,
    'place_table': 20,
    'wake_up': 21
}


def export_achievement_statistics_json(evaluation_system, output_file):
    """
    Export achievement statistics to JSON format.
    """
    achievement_stats = evaluation_system.get_achievement_statistics()

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    serializable_stats = convert_to_serializable(achievement_stats)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    print(f"[Test] Exported achievement statistics to {output_file}")


def test_dqn(model_path: str, episodes: int = 300, seed: int = None, output_dir: str = None):
    """
    Test a trained DQN model on Crafter environment.

    Args:
        model_path: Path to model checkpoint (without .pth extension)
        episodes: Number of test episodes (default: 300)
        seed: Random seed for reproducibility (optional)
        output_dir: Output directory for results (optional)

    Returns:
        EvaluationSystem with test results
    """
    # Initialize environment
    env = CrafterEnv(seed=seed, reward=True, length=10000)

    # Load trained agent
    agent = DQNAgent(state_size=43, action_size=17)

    # Check if model exists
    model_file = f"{model_path}.pth"
    if not os.path.exists(model_file):
        print(f"[Error] Model file not found: {model_file}")
        print(f"[Info] Available models:")
        model_dirs = [
            "training/heron_final_output/models",
            "training/heron_initial_output/models",
            "training/heron_random_output/models",
            "training/dqn_base_output/models",
            "training/dqn_helper_output/models"
        ]
        for dir_path in model_dirs:
            full_path = Path(__file__).parent.parent / dir_path
            if full_path.exists():
                pth_files = list(full_path.glob("*.pth"))
                for pth in pth_files:
                    if not pth.name.endswith("_target.pth"):
                        print(f"  - {dir_path}/{pth.stem}")
        return None

    agent.load(model_path)
    agent.epsilon = 0.0  # Pure exploitation for testing

    # Initialize evaluation system
    evaluation_system = EvaluationSystem(num_achievements=22)

    print(f"\n{'='*60}")
    print(f"TESTING DQN MODEL ON CRAFTER")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed if seed else 'None (random)'}")
    print(f"Epsilon: {agent.epsilon} (pure exploitation)")
    print(f"{'='*60}\n")

    # Track overall statistics
    total_achievements_all = []
    survival_count = 0

    for e in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, 43])
        done = False
        total_reward = 0
        moves = 0

        # Track achievements for this episode
        episode_achievement_names = set()
        previous_achievements_set = set()

        while not done:
            action = agent.act(state, env)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            moves += 1

            # Track newly unlocked achievements
            achievements_dict = info.get('achievements', {})
            current_achievements_set = {name for name, count in achievements_dict.items() if count > 0}
            newly_unlocked_names = current_achievements_set - previous_achievements_set

            if newly_unlocked_names:
                episode_achievement_names.update(newly_unlocked_names)
                # Track in evaluation system
                newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names
                                     if name in ACHIEVEMENT_NAME_TO_ID}
                evaluation_system.add_episode_achievements(e, newly_unlocked_ids, moves)

            previous_achievements_set = current_achievements_set
            next_state = np.reshape(next_state, [1, 43])
            state = next_state

        # Check survival (discount > 0 means agent is alive)
        survived = info.get('discount', 0) > 0
        if survived:
            survival_count += 1

        # Record episode metrics via EvaluationSystem
        evaluation_system.add_episode(
            episode=e,
            shaped_reward=total_reward,
            native_reward=total_reward,  # Same for test (no shaping bonuses)
            shaped_bonus=0,
            achievements_unlocked=len(episode_achievement_names),
            moves=moves,
            helper_calls=0,  # No helper in test
            hallucinations=0,
            valid_actions_percentage=100.0
        )

        total_achievements_all.append(len(episode_achievement_names))

        # Progress output
        if (e + 1) % 10 == 0 or e == 0:
            survival_rate = (survival_count / (e + 1)) * 100
            print(f"Episode {e+1:3d}/{episodes} | Reward: {total_reward:8.2f} | "
                  f"Achievements: {len(episode_achievement_names):2d} | Moves: {moves:4d} | "
                  f"Survived: {'Yes' if survived else 'No'} | Survival Rate: {survival_rate:.1f}%")

    # Close environment (wrapped in try-except as Crafter Env may not have close())
    try:
        env.close()
    except AttributeError:
        pass  # Crafter Env doesn't have close() method

    # Finalize evaluation system
    evaluation_system.finalize()

    # Setup output directory
    if output_dir is None:
        model_name = Path(model_path).stem
        output_dir = f"testing/test_output/{model_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export all results using existing infrastructure
    evaluation_system.export_to_jsonl(str(output_path / "test_metrics.jsonl"))
    export_achievement_statistics_json(evaluation_system, str(output_path / "test_achievement_statistics.json"))
    evaluation_system.export_summary_json(str(output_path / "test_evaluation.json"))

    # Generate plots
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generate_all_plots(evaluation_system, output_dir=str(plots_dir))

    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    evaluation_system.print_summary_report()

    # Additional test-specific statistics
    print(f"\n[Test-Specific Metrics]")
    print(f"  Final Survival Rate: {(survival_count / episodes) * 100:.2f}%")
    print(f"  Mean Achievements per Episode: {np.mean(total_achievements_all):.2f}")
    print(f"  Max Achievements in Single Episode: {np.max(total_achievements_all)}")

    print(f"\n[Output Files]")
    print(f"  Results saved to: {output_path}")
    print(f"  - test_metrics.jsonl")
    print(f"  - test_achievement_statistics.json")
    print(f"  - test_evaluation.json")
    print(f"  - plots/")
    print(f"{'='*60}\n")

    return evaluation_system


def main():
    parser = argparse.ArgumentParser(
        description="Test trained DQN models on Crafter environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  training/heron_final_output/models/heron_crafter_final     - HeRoN Final (full system)
  training/heron_initial_output/models/heron_crafter_final   - HeRoN Initial (fixed 100-step LLM)
  training/heron_random_output/models/heron_crafter_final    - HeRoN Random (random LLM activation)
  training/dqn_base_output/models/nuovo_crafter_dqn_final    - Pure DQN baseline
  training/dqn_helper_output/models/dqn_helper_crafter_final - DQN + Helper

Examples:
  python testing/test_dqn_crafter.py --model training/heron_final_output/models/heron_crafter_final
  python testing/test_dqn_crafter.py --model training/dqn_base_output/models/nuovo_crafter_dqn_final -e 50
  python testing/test_dqn_crafter.py --model training/heron_random_output/models/heron_crafter_final -s 42
        """
    )

    parser.add_argument("--model", "-m", required=True,
                        help="Path to model checkpoint (without .pth extension)")
    parser.add_argument("--episodes", "-e", type=int, default=300,
                        help="Number of test episodes (default: 300)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: testing/test_output/<model_name>)")

    args = parser.parse_args()

    result = test_dqn(args.model, args.episodes, args.seed, args.output_dir)

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
