import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent


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




def test_dqn(model_path: str, episodes: int = 300, seed: int = None, output_dir: str = None):
    env = CrafterEnv(seed=seed, reward=True, length=10000)

    agent = DQNAgent(state_size=43, action_size=17)

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
    agent.epsilon = 0.0

    print(f"\n{'='*60}")
    print(f"TESTING DQN MODEL ON CRAFTER")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed if seed else 'None (random)'}")
    print(f"Epsilon: {agent.epsilon} (pure exploitation)")
    print(f"{'='*60}\n")

    total_achievements_all = []
    total_rewards = []
    total_moves = []
    survival_count = 0
    episode_results = []

    for e in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, 43])
        done = False
        total_reward = 0
        moves = 0

        episode_achievement_names = set()
        previous_achievements_set = set()

        while not done:
            action = agent.act(state, env)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            moves += 1

            achievements_dict = info.get('achievements', {})
            current_achievements_set = {name for name, count in achievements_dict.items() if count > 0}
            newly_unlocked_names = current_achievements_set - previous_achievements_set

            if newly_unlocked_names:
                episode_achievement_names.update(newly_unlocked_names)

            previous_achievements_set = current_achievements_set
            next_state = np.reshape(next_state, [1, 43])
            state = next_state

        survived = info.get('discount', 0) > 0
        if survived:
            survival_count += 1

        total_achievements_all.append(len(episode_achievement_names))
        total_rewards.append(total_reward)
        total_moves.append(moves)

        episode_results.append({
            'episode': e,
            'reward': float(total_reward),
            'achievements': len(episode_achievement_names),
            'achievement_names': list(episode_achievement_names),
            'moves': moves,
            'survived': survived
        })

        if (e + 1) % 10 == 0 or e == 0:
            survival_rate = (survival_count / (e + 1)) * 100
            print(f"Episode {e+1:3d}/{episodes} | Reward: {total_reward:8.2f} | "
                  f"Achievements: {len(episode_achievement_names):2d} | Moves: {moves:4d} | "
                  f"Survived: {'Yes' if survived else 'No'} | Survival Rate: {survival_rate:.1f}%")

    try:
        env.close()
    except AttributeError:
        pass

    if output_dir is None:
        model_name = Path(model_path).stem
        output_dir = f"testing/test_output/{model_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        'model_path': model_path,
        'episodes': episodes,
        'seed': seed,
        'survival_rate': float((survival_count / episodes) * 100),
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_achievements': float(np.mean(total_achievements_all)),
        'max_achievements': int(np.max(total_achievements_all)),
        'mean_moves': float(np.mean(total_moves))
    }

    with open(output_path / "test_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'episodes': episode_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Episodes: {episodes}")
    print(f"  Survival Rate: {summary['survival_rate']:.2f}%")
    print(f"  Mean Reward: {summary['mean_reward']:.2f} (+/- {summary['std_reward']:.2f})")
    print(f"  Mean Achievements: {summary['mean_achievements']:.2f}")
    print(f"  Max Achievements: {summary['max_achievements']}")
    print(f"  Mean Moves: {summary['mean_moves']:.2f}")

    print(f"\n[Output Files]")
    print(f"  Results saved to: {output_path}")
    print(f"  - test_results.json")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test trained DQN models on Crafter environment",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
