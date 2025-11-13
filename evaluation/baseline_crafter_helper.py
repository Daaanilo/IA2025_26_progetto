"""
F10: Baseline Crafter Helper - LLM Always-On (No Threshold Decay)

Helper LLM always enabled without threshold decay strategy.
DQN disabled entirely - only uses Helper suggestions.
Serves as baseline comparison for HeRoN three-agent system.

Same metrics collection as heron_crafter.py for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import lmstudio as lms
from classes.crafter_environment import CrafterEnv
from classes.crafter_helper import CrafterHelper, SequenceExecutor
from evaluation_system import EvaluationSystem

# LM Studio configuration
SERVER_API_HOST = "http://127.0.0.1:1234"
lms.get_default_client(SERVER_API_HOST)


def plot_baseline_metrics(rewards, native_rewards, shaped_bonus, achievements, moves, 
                         helper_calls, hallucinations,
                         output_prefix="baseline_crafter_helper"):
    """Generate 5 baseline evaluation plots."""
    
    episodes = range(1, len(rewards) + 1)
    
    # 1. Rewards over time
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, rewards, label='Shaped Reward (native + bonus)', linewidth=2, marker='o', markersize=4)
    plt.plot(episodes, native_rewards, label='Native Reward (sparse)', linewidth=2, marker='s', markersize=4)
    plt.plot(episodes, shaped_bonus, label='Shaped Bonus Total', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Reward', fontsize=12, fontweight='bold')
    plt.title('Helper-Only Baseline - Reward Trends', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline Helper] Saved {output_prefix}_rewards.png")
    
    # 2. Cumulative achievements
    cumulative_achievements = np.cumsum(achievements)
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, cumulative_achievements, linewidth=2.5, marker='o', markersize=5, color='green')
    plt.fill_between(episodes, cumulative_achievements, alpha=0.3, color='green')
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Achievements', fontsize=12, fontweight='bold')
    plt.title('Helper-Only Baseline - Cumulative Achievement Unlocks', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_achievements.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline Helper] Saved {output_prefix}_achievements.png")
    
    # 3. Moves per episode
    plt.figure(figsize=(14, 6))
    plt.bar(episodes, moves, color='steelblue', alpha=0.7, edgecolor='black')
    plt.plot(episodes, moves, color='darkblue', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Moves per Episode', fontsize=12, fontweight='bold')
    plt.title('Helper-Only Baseline - Episode Length (Moves)', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_moves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline Helper] Saved {output_prefix}_moves.png")
    
    # 4. Helper stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Helper calls
    ax1.bar(episodes, helper_calls, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.plot(episodes, helper_calls, color='darkblue', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Helper Calls per Episode', fontsize=11, fontweight='bold')
    ax1.set_title('Helper Call Frequency (Always Enabled)', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Hallucination rate
    hallucination_rates = [h / max(1, c) for h, c in zip(hallucinations, helper_calls)]
    ax2.scatter(episodes, hallucination_rates, color='coral', s=50, alpha=0.7, edgecolor='black')
    ax2.plot(episodes, hallucination_rates, color='darkred', linewidth=2, marker='o', markersize=4)
    ax2.axhline(np.mean(hallucination_rates), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(hallucination_rates):.1%}')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Hallucination Rate', fontsize=11, fontweight='bold')
    ax2.set_title('LLM Hallucination Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(hallucination_rates) * 1.1])
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_helper_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline Helper] Saved {output_prefix}_helper_stats.png")
    
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
    
    # Helper stats
    ax_helper = axes[1, 0]
    ax_helper_twin = ax_helper.twinx()
    ax_helper.bar(episodes, helper_calls, alpha=0.5, color='steelblue', label='Helper Calls')
    ax_helper_twin.plot(episodes, hallucination_rates, color='red', linewidth=2, marker='o', markersize=4, label='Hallucination Rate')
    ax_helper.set_xlabel('Episode', fontsize=10, fontweight='bold')
    ax_helper.set_ylabel('Helper Calls', fontsize=10, fontweight='bold', color='steelblue')
    ax_helper_twin.set_ylabel('Hallucination Rate', fontsize=10, fontweight='bold', color='red')
    ax_helper.set_title('C. Helper LLM Activity', fontsize=11, fontweight='bold')
    ax_helper.grid(alpha=0.3, axis='y')
    
    # Efficiency scatter
    axes[1, 1].scatter(moves, achievements, c=rewards, s=80, cmap='viridis', alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Moves per Episode', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Achievements Unlocked', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('D. Efficiency Trade-off', fontsize=11, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Helper-Only Baseline - Multi-Metric Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Baseline Helper] Saved {output_prefix}_dashboard.png")


def export_baseline_metrics_csv(rewards, native_rewards, shaped_bonus, achievements, moves,
                               helper_calls, hallucinations,
                               output_file="baseline_crafter_helper_metrics.csv"):
    """Export baseline metrics to CSV."""
    hallucination_rates = [h / max(1, c) for h, c in zip(hallucinations, helper_calls)]
    
    df = pd.DataFrame({
        'episode': range(1, len(rewards) + 1),
        'shaped_reward': rewards,
        'native_reward': native_rewards,
        'shaped_bonus': shaped_bonus,
        'achievements_unlocked': achievements,
        'moves': moves,
        'helper_calls': helper_calls,
        'hallucinations': hallucinations,
        'hallucination_rate': hallucination_rates
    })
    df.to_csv(output_file, index=False)
    print(f"[Baseline Helper] Exported metrics to {output_file}")


def train_helper_baseline(episodes=50, episode_length=500):
    """
    Train Helper-only baseline on Crafter environment.
    Helper LLM always enabled, no DQN, no threshold decay.
    
    Args:
        episodes: Number of training episodes
        episode_length: Steps per episode
    """
    
    print("[Baseline Helper] Initializing Crafter environment...")
    env = CrafterEnv(length=episode_length)
    
    print(f"[Baseline Helper] State size: {env.state_size}, Action size: {env.action_size}")
    
    # Initialize Helper
    helper = CrafterHelper(env=env)
    print(f"[Baseline Helper] CrafterHelper initialized")
    
    # Initialize sequence executor
    executor = SequenceExecutor(env=env, helper=helper)
    print(f"[Baseline Helper] SequenceExecutor initialized")
    
    # Initialize evaluation system
    evaluation_system = EvaluationSystem()
    
    # Metrics tracking
    rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    helper_calls_per_episode = []
    hallucinations_per_episode = []
    
    print(f"\n[Baseline Helper] Starting training for {episodes} episodes...")
    print("="*70)
    
    for episode in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, env.state_size])
        
        episode_reward = 0
        episode_native_reward = 0
        episode_shaped_bonus = 0
        episode_achievements = 0
        episode_moves = 0
        episode_helper_calls = 0
        episode_hallucinations = 0
        previous_achievements = len(info.get('achievements', {}))
        
        for step in range(episode_length):
            # Helper always generates action sequence (no threshold)
            try:
                action_sequence = helper.generate_action_sequence(state)
                episode_helper_calls += 1
                
                if action_sequence is None:
                    episode_hallucinations += 1
                    # Fallback: random action
                    action = np.random.randint(0, env.action_size)
                else:
                    # Execute first action from sequence
                    action = action_sequence[0]
            except Exception as e:
                print(f"[Baseline Helper] Helper error: {e}")
                episode_hallucinations += 1
                action = np.random.randint(0, env.action_size)
            
            # Execute action in environment
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # Count achievements unlocked this step
            current_achievements = len(info.get('achievements', {}))
            achievements_this_step = max(0, current_achievements - previous_achievements)
            previous_achievements = current_achievements
            
            # Simple reward shaping: bonus for achievements
            shaped_reward = native_reward + (0.5 * achievements_this_step)
            shaped_bonus = shaped_reward - native_reward
            
            # Update metrics
            episode_reward += shaped_reward
            episode_native_reward += native_reward
            episode_shaped_bonus += shaped_bonus
            episode_achievements += achievements_this_step
            episode_moves += 1
            
            state = next_state
            
            if done:
                break
        
        # Record episode metrics
        rewards_per_episode.append(episode_reward)
        native_rewards_per_episode.append(episode_native_reward)
        shaped_bonus_per_episode.append(episode_shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(episode_moves)
        helper_calls_per_episode.append(episode_helper_calls)
        hallucinations_per_episode.append(episode_hallucinations)
        
        # Add to evaluation system
        evaluation_system.add_episode(
            episode=episode,
            shaped_reward=episode_reward,
            native_reward=episode_native_reward,
            shaped_bonus=episode_shaped_bonus,
            achievements_unlocked=episode_achievements,
            moves=episode_moves,
            helper_calls=episode_helper_calls,
            hallucinations=episode_hallucinations
        )
        
        # Progress logging
        if (episode + 1) % max(1, episodes // 10) == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            avg_achievements = np.mean(achievements_per_episode[-10:])
            avg_helper_calls = np.mean(helper_calls_per_episode[-10:])
            print(f"Episode {episode+1:3d}/{episodes} | Avg Reward (last 10): {avg_reward:6.2f} | "
                  f"Avg Achievements: {avg_achievements:4.1f} | Avg Helper Calls: {avg_helper_calls:4.1f}")
    
    print("="*70)
    print("[Baseline Helper] Training completed.")
    
    # Finalize evaluation
    evaluation_system.finalize()
    
    # Export metrics
    print("[Baseline Helper] Exporting metrics...")
    export_baseline_metrics_csv(
        rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
        achievements_per_episode, moves_per_episode, helper_calls_per_episode,
        hallucinations_per_episode,
        output_file="baseline_crafter_helper_metrics.csv"
    )
    
    # Generate plots
    print("[Baseline Helper] Generating plots...")
    plot_baseline_metrics(
        rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
        achievements_per_episode, moves_per_episode, helper_calls_per_episode,
        hallucinations_per_episode,
        output_prefix="baseline_crafter_helper"
    )
    
    # Print evaluation report
    print("\n[Baseline Helper] Final Evaluation Report:")
    evaluation_system.print_summary_report()
    
    # Export evaluation summary
    evaluation_system.export_summary_json("baseline_crafter_helper_evaluation.json")
    
    return evaluation_system


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Helper-only baseline on Crafter environment")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--episode-length", type=int, default=500, help="Steps per episode")
    
    args = parser.parse_args()
    
    # Run baseline training
    eval_system = train_helper_baseline(
        episodes=args.episodes,
        episode_length=args.episode_length
    )
