import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from training.reward_shaper import CrafterRewardShaper
from evaluation.evaluation_system import EvaluationSystem
from evaluation.evaluation_plots import generate_all_plots, ACHIEVEMENT_NAME_TO_ID


def train_dqn_baseline(episodes=300, batch_size=32, episode_length=1000, load_model_path=None):
    """
    Train pure DQN agent on Crafter environment.
    
    Args:
        episodes: Number of training episodes
        batch_size: DQN replay buffer batch size
        episode_length: Steps per episode
        load_model_path: Optional path to pre-trained model
    """
    
    print("[Baseline DQN] Initializing Crafter environment...")
    env = CrafterEnv(length=episode_length)
    
    print(f"[Baseline DQN] State size: {env.state_size}, Action size: {env.action_size}")
    
    # Initialize DQN agent
    agent = DQNAgent(env.state_size, env.action_size, epsilon=1.0, load_model_path=load_model_path)
    print(f"[Baseline DQN] DQN Agent initialized (epsilon={agent.epsilon:.4f})")
    
    # Initialize evaluation system
    evaluation_system = EvaluationSystem()
    
    # Initialize Reward Shaper
    reward_shaper = CrafterRewardShaper()
    
    # Metrics tracking
    rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    
    # Best model tracking
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
        # Track achievements by name for proper ID mapping
        previous_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
        previous_info = info.copy()  # Per reward shaping avanzato
        reward_shaper.reset_episode()
        
        for step in range(episode_length):
            # DQN selects action (no LLM)
            action = agent.act(state, env)
            
            # Execute action in environment
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # Track achievements unlocked this step
            current_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
            newly_unlocked_names = current_achievements_set - previous_achievements_set
            achievements_this_step = len(newly_unlocked_names)
            
            # Add achievement IDs to evaluation system
            if newly_unlocked_names:
                newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names 
                                     if name in ACHIEVEMENT_NAME_TO_ID}
                evaluation_system.add_episode_achievements(episode, newly_unlocked_ids, step)
            
            previous_achievements_set = current_achievements_set
            
            # Use centralized reward shaper
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, info, previous_info
            )
            shaped_bonus = sum(bonus_components.values())
            
            # Remember experience
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # Train DQN when sufficient memory samples available
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            # Update metrics
            episode_reward += shaped_reward
            episode_native_reward += native_reward
            episode_shaped_bonus += shaped_bonus
            episode_achievements += achievements_this_step
            episode_moves += 1
            
            previous_info = info.copy()
            state = next_state
            
            if done:
                break
        
        # Record episode metrics
        rewards_per_episode.append(episode_reward)
        native_rewards_per_episode.append(episode_native_reward)
        shaped_bonus_per_episode.append(episode_shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(episode_moves)
        
        # Add to evaluation system
        evaluation_system.add_episode(
            episode=episode,
            shaped_reward=episode_reward,
            native_reward=episode_native_reward,
            shaped_bonus=episode_shaped_bonus,
            achievements_unlocked=episode_achievements,
            moves=episode_moves,
            helper_calls=0,  # No LLM in baseline
            hallucinations=0  # No LLM in baseline
        )
        
        # Save best model checkpoint
        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = episode
            checkpoint_dir = Path(__file__).parent / 'baseline_dqn_output' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"baseline_dqn_best_ep{episode}_ach{episode_achievements}"
            agent.save(str(checkpoint_path))
            print(f"[Baseline DQN] ✓ New best model saved: {episode_achievements} achievements (episode {episode})")
        
        # Progress logging
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            avg_achievements = np.mean(achievements_per_episode[-10:])
            print(f"Episode {episode+1:3d}/{episodes} | Avg Reward (last 10): {avg_reward:6.2f} | "
                  f"Avg Achievements: {avg_achievements:4.1f} | Epsilon: {agent.epsilon:.4f}")
    
    print("="*70)
    print("[Baseline DQN] Training completed.")
    print(f"[Baseline DQN] Best performance: {best_achievement_count} achievements at episode {best_episode}")
    
    # Finalize evaluation
    evaluation_system.finalize()
    
    # Save final model
    print("[Baseline DQN] Saving final model...")
    models_dir = Path(__file__).parent / 'baseline_dqn_output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "baseline_crafter_dqn_final"
    agent.save(str(final_model_path))
    print(f"[Baseline DQN] ✓ Final model saved: {final_model_path}.*")
    
    checkpoint_dir = Path(__file__).parent / 'baseline_dqn_output' / 'checkpoints'
    best_checkpoint = checkpoint_dir / f"baseline_dqn_best_ep{best_episode}_ach{best_achievement_count}"
    print(f"[Baseline DQN] ✓ Best model saved: {best_checkpoint}.*")
    
    # Export metrics
    print("[Baseline DQN] Exporting metrics...")
    output_dir = Path(__file__).parent / 'baseline_dqn_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "baseline_crafter_dqn_metrics.jsonl"
    evaluation_system.export_to_jsonl(str(jsonl_path))
    
    # Generate plots
    print("[Baseline DQN] Generating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    generate_all_plots(evaluation_system, output_dir=str(plot_dir), title_prefix="DQN Baseline")
    
    # Print evaluation report
    print("\n[Baseline DQN] Final Evaluation Report:")
    evaluation_system.print_summary_report()
    
    # Export evaluation summary
    evaluation_json = output_dir / "baseline_crafter_dqn_evaluation.json"
    evaluation_system.export_summary_json(str(evaluation_json))
    
    return evaluation_system


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN baseline on Crafter environment")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="DQN batch size")
    parser.add_argument("--episode-length", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model")
    
    args = parser.parse_args()
    
    # Run baseline training
    eval_system = train_dqn_baseline(
        episodes=args.episodes,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        load_model_path=args.load_model
    )
