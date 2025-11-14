"""
F10: Baseline Crafter DQN - Pure Reinforcement Learning (No LLM)

Pure DQN agent training on Crafter environment without LLM assistance.
Serves as baseline comparison for HeRoN three-agent system.

Same metrics collection as heron_crafter.py for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from evaluation.evaluation_system import EvaluationSystem, ACHIEVEMENT_NAME_TO_ID
import os


def plot_baseline_metrics(rewards, native_rewards, shaped_bonus, achievements, moves, 
                         output_prefix="baseline_crafter_dqn"):
    """Generate 5 baseline evaluation plots (identical to heron_crafter.py)."""
    
    episodes = range(1, len(rewards) + 1)
    
    # 1. Rewards over time
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, rewards, label='Shaped Reward (native + bonus)', linewidth=2, marker='o', markersize=4)
    plt.plot(episodes, native_rewards, label='Native Reward (sparse)', linewidth=2, marker='s', markersize=4)
    plt.plot(episodes, shaped_bonus, label='Shaped Bonus Total', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Reward', fontsize=12, fontweight='bold')
    plt.title('DQN Baseline - Reward Trends', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / 'baseline_dqn_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{output_prefix}_rewards.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline DQN] Saved {output_file}")
    
    # 2. Cumulative achievements
    cumulative_achievements = np.cumsum(achievements)
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, cumulative_achievements, linewidth=2.5, marker='o', markersize=5, color='green')
    plt.fill_between(episodes, cumulative_achievements, alpha=0.3, color='green')
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Achievements', fontsize=12, fontweight='bold')
    plt.title('DQN Baseline - Cumulative Achievement Unlocks', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = output_dir / f'{output_prefix}_achievements.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline DQN] Saved {output_file}")
    
    # 3. Moves per episode
    plt.figure(figsize=(14, 6))
    plt.bar(episodes, moves, color='steelblue', alpha=0.7, edgecolor='black')
    plt.plot(episodes, moves, color='darkblue', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Moves per Episode', fontsize=12, fontweight='bold')
    plt.title('DQN Baseline - Episode Length (Moves)', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = output_dir / f'{output_prefix}_moves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline DQN] Saved {output_file}")
    
    # 4. Efficiency metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Reward per move
    reward_per_move = [r / m if m > 0 else 0 for r, m in zip(rewards, moves)]
    ax1.plot(episodes, reward_per_move, linewidth=2.5, marker='o', markersize=4, color='steelblue')
    ax1.fill_between(episodes, reward_per_move, alpha=0.3, color='steelblue')
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Reward per Move', fontsize=11, fontweight='bold')
    ax1.set_title('Efficiency: Reward per Move', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Achievements per move
    ach_per_move = [a / m if m > 0 else 0 for a, m in zip(achievements, moves)]
    ax2.plot(episodes, ach_per_move, linewidth=2.5, marker='s', markersize=4, color='green')
    ax2.fill_between(episodes, ach_per_move, alpha=0.3, color='green')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Achievements per Move', fontsize=11, fontweight='bold')
    ax2.set_title('Efficiency: Achievements per Move', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f'{output_prefix}_efficiency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline DQN] Saved {output_file}")
    
    # 5. Multi-metric dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rewards
    axes[0, 0].plot(episodes, rewards, linewidth=2, marker='o', markersize=3, color='steelblue', label='Shaped')
    axes[0, 0].plot(episodes, native_rewards, linewidth=2, marker='s', markersize=3, color='coral', label='Native')
    axes[0, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Reward', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('A. Reward Trends', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Achievements
    axes[0, 1].plot(episodes, cumulative_achievements, linewidth=2.5, marker='o', markersize=3, color='green')
    axes[0, 1].fill_between(episodes, cumulative_achievements, alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Achievements', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('B. Achievement Unlocks', fontsize=11, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Moves
    axes[1, 0].bar(episodes, moves, color='steelblue', alpha=0.5, edgecolor='black')
    axes[1, 0].plot(episodes, moves, color='darkblue', linewidth=2, marker='o', markersize=3)
    axes[1, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Moves per Episode', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('C. Episode Length', fontsize=11, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Efficiency scatter
    axes[1, 1].scatter(moves, achievements, c=rewards, s=80, cmap='viridis', alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Moves per Episode', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Achievements Unlocked', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('D. Efficiency Trade-off', fontsize=11, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('DQN Baseline - Multi-Metric Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'{output_prefix}_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline DQN] Saved {output_file}")


def export_baseline_metrics_jsonl(rewards, native_rewards, shaped_bonus, achievements, moves,
                               output_file="baseline_crafter_dqn_metrics.jsonl"):
    """Export baseline metrics to JSONL."""
    import json
    
    # Ensure output directory exists
    output_dir = Path(__file__).parent / 'baseline_dqn_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(rewards)):
            record = {
                'episode': i + 1,
                'shaped_reward': rewards[i],
                'native_reward': native_rewards[i],
                'shaped_bonus': shaped_bonus[i],
                'achievements_unlocked': achievements[i],
                'moves': moves[i]
            }
            f.write(json.dumps(record) + "\n")
    print(f"[Baseline DQN] Exported metrics to {output_path}")


def calculate_shaped_reward(native_reward, info, previous_info, achievements_this_step):
    """
    OTTIMIZZATO: Reward shaping avanzato per accelerare apprendimento.
    
    Bonus per:
    - Achievements sbloccati (+1.0 ciascuno)
    - Risorse raccolte (+0.1 per ogni risorsa)
    - Crafting tools/weapons (+0.3)
    - Sopravvivenza (health, food, drink) (+0.05)
    - Esplorazione (+0.01 per nuove posizioni)
    """
    shaped_bonus = 0.0
    
    # Base achievement bonus
    shaped_bonus += achievements_this_step * 1.0
    
    if previous_info is None:
        return native_reward + shaped_bonus, shaped_bonus
    
    prev_inv = previous_info.get('inventory', {})
    curr_inv = info.get('inventory', {})
    
    # Bonus risorse raccolte
    resources = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
    for resource in resources:
        prev_val = prev_inv.get(resource, 0)
        curr_val = curr_inv.get(resource, 0)
        if curr_val > prev_val:
            shaped_bonus += 0.1 * (curr_val - prev_val)
    
    # Bonus crafting tools/weapons
    tools = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
             'wood_sword', 'stone_sword', 'iron_sword']
    for tool in tools:
        prev_val = prev_inv.get(tool, 0)
        curr_val = curr_inv.get(tool, 0)
        if curr_val > prev_val:
            shaped_bonus += 0.3
    
    # Bonus sopravvivenza (health, food, drink positivi)
    health = curr_inv.get('health', 0)
    food = curr_inv.get('food', 0)
    drink = curr_inv.get('drink', 0)
    
    if health > 5:  # Bonus se health sopra soglia
        shaped_bonus += 0.02
    if food > 5:
        shaped_bonus += 0.02
    if drink > 5:
        shaped_bonus += 0.02
    
    # Penalizzazione morte (health a zero)
    if health == 0 and prev_inv.get('health', 0) > 0:
        shaped_bonus -= 1.0
    
    shaped_reward = native_reward + shaped_bonus
    return shaped_reward, shaped_bonus


def train_dqn_baseline(episodes=200, batch_size=32, episode_length=500, load_model_path=None):
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
            
            # OTTIMIZZATO: Reward shaping avanzato
            shaped_reward, shaped_bonus = calculate_shaped_reward(
                native_reward, info, previous_info, achievements_this_step
            )
            
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
    export_baseline_metrics_jsonl(
        rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
        achievements_per_episode, moves_per_episode,
        output_file="baseline_crafter_dqn_metrics.jsonl"
    )
    
    # Generate plots
    print("[Baseline DQN] Generating plots...")
    plot_baseline_metrics(
        rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
        achievements_per_episode, moves_per_episode,
        output_prefix="baseline_crafter_dqn"
    )
    
    # Print evaluation report
    print("\n[Baseline DQN] Final Evaluation Report:")
    evaluation_system.print_summary_report()
    
    # Export evaluation summary
    output_dir = Path(__file__).parent / 'baseline_dqn_output'
    evaluation_json = output_dir / "baseline_crafter_dqn_evaluation.json"
    evaluation_system.export_summary_json(str(evaluation_json))
    
    return evaluation_system


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN baseline on Crafter environment")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="DQN batch size")
    parser.add_argument("--episode-length", type=int, default=500, help="Steps per episode")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model")
    
    args = parser.parse_args()
    
    # Run baseline training
    eval_system = train_dqn_baseline(
        episodes=args.episodes,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        load_model_path=args.load_model
    )
