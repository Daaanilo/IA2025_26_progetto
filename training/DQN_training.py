"""Training DQN baseline senza LLM per Crafter."""

import numpy as np
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from training.reward_shaper import CrafterRewardShaper
, ACHIEVEMENT_NAME_TO_ID

from evaluation.achievement_learning_curves import AchievementLearningCurvePlotter

ACHIEVEMENT_ID_TO_NAME = {v: k for k, v in ACHIEVEMENT_NAME_TO_ID.items()}


def export_achievement_statistics_json_old(evaluation_system, output_file="crafter_achievement_statistics.json"):
    """Esporta statistiche achievement in formato JSON."""
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
    
    serializable_stats['achievement_id_to_name'] = ACHIEVEMENT_ID_TO_NAME
    
    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"[Export] Saved achievement statistics to: {output_file}")


def train_dqn_baseline(episodes=300, batch_size=64, episode_length=1000, load_model_path=None):
    print("[Baseline DQN] Initializing Crafter environment...")
    env = CrafterEnv(length=episode_length)
    
    print(f"[Baseline DQN] State size: {env.state_size}, Action size: {env.action_size}")
    
    agent = DQNAgent(env.state_size, env.action_size, epsilon=1.0, load_model_path=load_model_path)
    print(f"[Baseline DQN] DQN Agent initialized (epsilon={agent.epsilon:.4f})")

    reward_shaper = CrafterRewardShaper()
    
    rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    
    best_achievement_count = 0
    best_episode = 0
    
    print(f"\n[Baseline DQN] Starting training for {episodes} episodes...")
    print("="*70)
    
    for episode in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, env.state_size])
        
        episode_reward = 0
        episode_native_reward = 0
        episode_shaped_bonus = 0
        episode_achievements = 0
        episode_moves = 0
        previous_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
        previous_info = info.copy()
        reward_shaper.reset_episode()
        
        for step in range(episode_length):
            # Seleziona azione con epsilon-greedy
            action = agent.act(state, env)
            
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            current_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
            newly_unlocked_names = current_achievements_set - previous_achievements_set
            achievements_this_step = len(newly_unlocked_names)
            
            if newly_unlocked_names:

            previous_achievements_set = current_achievements_set
            
            # Calcola reward shaped (base + bonus risorse/health/tools - penalità)
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, info, previous_info
            )
            shaped_bonus = sum(bonus_components.values())
            
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # Training con batch da memoria (se abbastanza campioni)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            episode_reward += shaped_reward
            episode_native_reward += native_reward
            episode_shaped_bonus += shaped_bonus
            episode_achievements += achievements_this_step
            episode_moves += 1
            
            previous_info = info.copy()
            state = next_state
            
            if done:
                break
        
        rewards_per_episode.append(episode_reward)
        native_rewards_per_episode.append(episode_native_reward)
        shaped_bonus_per_episode.append(episode_shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(episode_moves)
        
        valid_actions_percentage = 0.0


        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = episode
            checkpoint_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"nuovo_dqn_best_ep{episode}_ach{episode_achievements}"
            agent.save(str(checkpoint_path))
            print(f"[Baseline DQN] ✓ New best model saved: {episode_achievements} achievements (episode {episode})")
        
        agent.decay_epsilon_linear(episode, total_episodes=episodes)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            avg_achievements = np.mean(achievements_per_episode[-10:])
            print(f"Episode {episode+1:3d}/{episodes} | Avg Reward (last 10): {avg_reward:6.2f} | "
                  f"Avg Achievements: {avg_achievements:4.1f} | Epsilon: {agent.epsilon:.4f}")
    
    print("="*70)
    print("[Baseline DQN] Training completed.")
    print(f"[Baseline DQN] Best performance: {best_achievement_count} achievements at episode {best_episode}")

    print("[Baseline DQN] Saving final model...")
    models_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "nuovo_crafter_dqn_final"
    agent.save(str(final_model_path))
    print(f"[Baseline DQN] ✓ Final model saved: {final_model_path}.*")
    
    checkpoint_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'checkpoints'
    best_checkpoint = checkpoint_dir / f"nuovo_dqn_best_ep{best_episode}_ach{best_achievement_count}"
    print(f"[Baseline DQN] ✓ Best model saved: {best_checkpoint}.*")

    return (rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
            achievements_per_episode, moves_per_episode)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN baseline on Crafter environment")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN batch size")
    parser.add_argument("--episode-length", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model")
    
    args = parser.parse_args()
    
    (rewards, native_rewards, shaped_bonus, achievements, moves) = train_dqn_baseline(
        episodes=args.episodes,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        load_model_path=args.load_model
    )

    print("\n[Training Complete] Final Statistics:")
    print(f"  Average Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"  Average Native Reward: {sum(native_rewards)/len(native_rewards):.2f}")
    print(f"  Average Achievements: {sum(achievements)/len(achievements):.2f}")
