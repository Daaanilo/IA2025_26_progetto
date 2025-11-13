"""
F09: Main Training Orchestration Script
Multi-phase training with curriculum learning, dynamic hyperparameters, and early stopping
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lmstudio as lms
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from classes.crafter_helper import CrafterHelper, SequenceExecutor
from classes.instructor_agent import InstructorAgent
from evaluation.evaluation_system import EvaluationSystem
from evaluation.evaluation_plots import AdvancedPlotter, generate_all_plots

# F09: Import curriculum and scheduling components
from HeRoN.curriculum_manager import (
    CurriculumManager, HyperparameterScheduler, EarlyStoppingManager
)

# LM Studio configuration
SERVER_API_HOST = "http://127.0.0.1:1234"
# Note: LM Studio client will be created with context manager when needed

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")


class CrafterRewardShaper:
    """Augments sparse Crafter rewards with intrinsic bonuses - with adaptive weights (F09)."""
    
    ACTION_NAMES = {
        0: 'move_up', 1: 'move_down', 2: 'move_left', 3: 'move_right',
        4: 'do', 5: 'sleep',
        6: 'place_stone', 7: 'place_table', 8: 'place_furnace', 9: 'place_plant',
        10: 'make_wood_pickaxe', 11: 'make_stone_pickaxe', 12: 'make_iron_pickaxe',
        13: 'make_wood_sword', 14: 'make_stone_sword', 15: 'make_iron_sword',
        16: 'noop'
    }
    
    def __init__(self, initial_weights=None):
        # F09: Adaptive reward shaping weights
        self.weights = initial_weights or {
            'resource_collection': 0.1,
            'health_management': 0.05,
            'tier_progression': 0.05,
            'tool_usage': 0.02
        }
        self.bonus_tracker = {key: [] for key in self.weights.keys()}
    
    def update_weights(self, new_weights):
        """F09: Update reward shaping weights dynamically."""
        self.weights.update(new_weights)
        print(f"[Reward Shaping] Updated weights: {self.weights}")
    
    def calculate_shaped_reward(self, native_reward, state, info, previous_info, action):
        """Calculate total shaped reward = native + weighted intrinsic bonuses."""
        shaped_reward = native_reward
        bonuses = {key: 0.0 for key in self.weights.keys()}
        
        if previous_info is None:
            return shaped_reward, bonuses
        
        # Calculate base bonuses (same logic as F08)
        bonuses['resource_collection'] = self._calculate_resource_bonus(info, previous_info, action)
        bonuses['health_management'] = self._calculate_health_bonus(info, previous_info, action)
        bonuses['tier_progression'] = self._calculate_tier_bonus(info, previous_info)
        bonuses['tool_usage'] = self._calculate_tool_bonus(info, previous_info, action)
        
        # F09: Apply adaptive weights
        total_bonus = sum(bonuses[key] * self.weights[key] for key in bonuses.keys())
        shaped_reward += total_bonus
        
        # Track
        for key in bonuses:
            self.bonus_tracker[key].append(bonuses[key])
        
        return shaped_reward, bonuses
    
    def _calculate_resource_bonus(self, info, previous_info, action):
        if action != 4:
            return 0.0
        bonus = 0.0
        resources = ['wood', 'stone', 'iron', 'coal', 'diamond']
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        for resource in resources:
            if curr_inv.get(resource, 0) > prev_inv.get(resource, 0):
                bonus += 1.0  # Return normalized bonus (will be weighted later)
        return min(bonus, 1.0)
    
    def _calculate_health_bonus(self, info, previous_info, action):
        bonus = 0.0
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        curr_health = curr_inv.get('health', 10)
        prev_health = prev_inv.get('health', 10)
        if curr_health > prev_health:
            bonus += 1.0
        if prev_inv.get('food', 0) > curr_inv.get('food', 0):
            bonus += 0.6
        if prev_inv.get('drink', 0) > curr_inv.get('drink', 0):
            bonus += 0.6
        return min(bonus, 1.0)
    
    def _calculate_tier_bonus(self, info, previous_info):
        curr_achievements = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
        prev_achievements = set(k for k, v in previous_info.get('achievements', {}).items() if v >= 1)
        if curr_achievements - prev_achievements:
            return 1.0
        return 0.0
    
    def _calculate_tool_bonus(self, info, previous_info, action):
        if action in [10, 11, 12]:  # Tool crafting actions
            return 1.0
        return 0.0
    
    def get_statistics(self):
        stats = {}
        for key, values in self.bonus_tracker.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'total': np.sum(values),
                    'count': len(values)
                }
            else:
                stats[key] = {'mean': 0.0, 'total': 0.0, 'count': 0}
        return stats
    
    def reset_episode(self):
        for key in self.bonus_tracker:
            self.bonus_tracker[key] = []


def train_heron_with_curriculum(
    episodes=500,
    batch_size=32,
    checkpoint_interval=10,
    reviewer_model_path="reviewer_retrained",
    output_dir="./",
    lr_strategy='step_decay',
    epsilon_strategy='linear_decay',
    threshold_strategy='linear_decay',
    enable_curriculum=True,
    enable_early_stopping=True
):
    """
    F09: Enhanced HeRoN training with curriculum learning and dynamic hyperparameters.
    
    Args:
        episodes: Total training episodes
        batch_size: DQN replay batch size
        checkpoint_interval: Save checkpoint every N episodes
        reviewer_model_path: Path to fine-tuned Reviewer model
        output_dir: Directory for outputs
        lr_strategy: Learning rate schedule ('constant', 'step_decay', 'exponential', 'cosine')
        epsilon_strategy: Epsilon decay strategy ('linear_decay', 'exponential_decay', 'staged')
        threshold_strategy: Threshold decay strategy ('linear_decay', 'staged')
        enable_curriculum: Enable curriculum learning
        enable_early_stopping: Enable early stopping
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Save training configuration
    config = {
        'episodes': episodes,
        'batch_size': batch_size,
        'checkpoint_interval': checkpoint_interval,
        'lr_strategy': lr_strategy,
        'epsilon_strategy': epsilon_strategy,
        'threshold_strategy': threshold_strategy,
        'enable_curriculum': enable_curriculum,
        'enable_early_stopping': enable_early_stopping,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize environment
    print("\n[Init] Initializing Crafter environment...")
    env = CrafterEnv(area=(64, 64), view=(9, 9), size=(64, 64), reward=True, 
                     length=500, seed=None)  # Will use curriculum length
    
    # F09: Initialize curriculum manager
    if enable_curriculum:
        curriculum = CurriculumManager(total_episodes=episodes)
        print(f"[Curriculum] Enabled with {episodes} episodes")
        print(f"[Curriculum] Stage thresholds: Early={curriculum.stage_thresholds['early']}, Mid={curriculum.stage_thresholds['mid']}")
    else:
        curriculum = None
        print("[Curriculum] Disabled")
    
    # F09: Initialize hyperparameter scheduler
    scheduler = HyperparameterScheduler(
        initial_lr=0.001,
        initial_epsilon=1.0,
        initial_threshold=1.0
    )
    
    # F09: Initialize early stopping manager
    if enable_early_stopping:
        early_stopper = EarlyStoppingManager(patience=100, min_episodes=200)
        print(f"[Early Stopping] Enabled (patience={early_stopper.patience}, min_episodes={early_stopper.min_episodes})")
    else:
        early_stopper = None
        print("[Early Stopping] Disabled")
    
    # Initialize DQN Agent
    print("[Init] Initializing DQN Agent...")
    initial_lr = scheduler.get_learning_rate(0, episodes, strategy=lr_strategy)
    initial_epsilon = scheduler.get_epsilon(0, episodes, strategy=epsilon_strategy)
    agent = DQNAgent(env.state_size, env.action_size, load_model_path=None, 
                     learning_rate=initial_lr, epsilon=initial_epsilon)
    print(f"[Init] DQN initialized: lr={initial_lr:.6f}, epsilon={initial_epsilon:.4f}")
    
    # Initialize Helper
    print("[Init] Initializing CrafterHelper...")
    helper = CrafterHelper(server_host=SERVER_API_HOST, model_name="llama-3.2-3b-instruct")
    
    # Initialize Reviewer
    print(f"[Init] Loading Reviewer from: {reviewer_model_path}")
    try:
        tokenizer_reviewer = AutoTokenizer.from_pretrained(reviewer_model_path)
        model_reviewer = T5ForConditionalGeneration.from_pretrained(reviewer_model_path).to(device)
        instructor = InstructorAgent(model_reviewer, tokenizer_reviewer, device)
        use_reviewer = True
        print("[Init] Reviewer loaded successfully")
    except Exception as e:
        print(f"[WARNING] Failed to load Reviewer: {e}")
        print("[WARNING] Training without Reviewer refinement")
        instructor = None
        use_reviewer = False
    
    # Initialize components
    executor = SequenceExecutor(agent, env)
    reward_shaper = CrafterRewardShaper()
    evaluation_system = EvaluationSystem(num_achievements=22)
    
    # Metrics tracking
    rewards_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    helper_calls = []
    hallucinations = []
    shaped_rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    epsilon_history = []
    lr_history = []
    threshold_history = []
    
    # Performance tracking
    best_achievement_count = 0
    best_episode = -1
    all_achievements_unlocked = set()
    
    threshold_episodes = 600
    
    print(f"\n[Training] Starting multi-phase training for up to {episodes} episodes...")
    print(f"[Training] Checkpoint interval: {checkpoint_interval} episodes")
    
    # ===== MAIN TRAINING LOOP =====
    for e in range(episodes):
        
        # F09: Update hyperparameters based on schedule
        current_lr = scheduler.get_learning_rate(e, episodes, strategy=lr_strategy)
        current_epsilon = scheduler.get_epsilon(e, episodes, strategy=epsilon_strategy)
        current_threshold = scheduler.get_threshold(e, threshold_episodes, strategy=threshold_strategy)
        
        # Update agent hyperparameters
        if abs(current_lr - agent.learning_rate) > 1e-7:
            agent.update_learning_rate(current_lr)
        agent.epsilon = current_epsilon
        
        lr_history.append(current_lr)
        epsilon_history.append(current_epsilon)
        threshold_history.append(current_threshold)
        
        # F09: Get curriculum-based episode length
        if enable_curriculum:
            episode_length = curriculum.get_episode_length(e)
            env._length = episode_length
        else:
            episode_length = 500
        
        # F09: Update reward shaping weights based on curriculum
        if enable_curriculum:
            achievement_rate = np.mean(achievements_per_episode[-10:]) if len(achievements_per_episode) >= 10 else 0.0
            adaptive_weights = curriculum.get_reward_shaping_weights(e, achievement_rate)
            reward_shaper.update_weights(adaptive_weights)
        
        # Reset environment
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        total_shaped_reward = 0
        total_native_reward = 0
        moves = 0
        episode_achievements = 0
        episode_helper_calls = 0
        episode_hallucinations = 0
        
        previous_info = None
        reward_shaper.reset_episode()
        executor.current_sequence = []
        executor.current_sequence_index = 0
        
        # Print episode header
        if enable_curriculum:
            stage = curriculum.get_stage(e)
            print(f"\n[Episode {e}/{episodes}] Stage: {stage.upper()}, Length: {episode_length}, "
                  f"LR: {current_lr:.6f}, Epsilon: {current_epsilon:.4f}, Threshold: {current_threshold:.4f}")
        else:
            print(f"\n[Episode {e}/{episodes}] Length: {episode_length}, "
                  f"LR: {current_lr:.6f}, Epsilon: {current_epsilon:.4f}, Threshold: {current_threshold:.4f}")
        
        # ===== STEP LOOP =====
        while not done and moves < episode_length:
            p = np.random.rand()
            use_llm = (p > current_threshold) and (e < threshold_episodes)
            
            if use_llm:
                episode_helper_calls += 1
                try:
                    current_info = env._last_info if hasattr(env, '_last_info') else {}
                    
                    should_replan = (
                        executor.current_sequence and 
                        previous_info is not None and
                        helper.should_replan(state, current_info, previous_info, executor.current_sequence)
                    )
                    
                    if should_replan:
                        executor.interrupt_sequence()
                    
                    if not executor.current_sequence or executor.current_sequence_index >= len(executor.current_sequence):
                        action_sequence, helper_response = helper.generate_action_sequence(
                            state, current_info, previous_info
                        )
                        
                        if action_sequence is None:
                            episode_hallucinations += 1
                            action = agent.act(state, env)
                        else:
                            if use_reviewer and instructor is not None:
                                game_description = helper.describe_crafter_state(state, current_info, previous_info)
                                reviewer_feedback = instructor.generate_suggestion(game_description, helper_response)
                                
                                refined_prompt = (
                                    f"Game state: {game_description}\n"
                                    f"Your initial response: {helper_response}\n"
                                    f"Reviewer feedback: {reviewer_feedback}\n"
                                    f"Please refine your action sequence considering this feedback.\n"
                                    f"Generate 3-5 actions in square brackets."
                                )
                                
                                try:
                                    with lms.Client() as client:
                                        model = client.llm.model("llama-3.2-3b-instruct")
                                        refined_response = model.respond(refined_prompt)
                                        refined_response = str(refined_response)
                                        refined_response = re.sub(r"<think>.*?</think>", "", refined_response, flags=re.DOTALL).strip()
                                        action_sequence = helper.parse_action_sequence(refined_response)
                                        
                                        if action_sequence is None:
                                            episode_hallucinations += 1
                                            action = agent.act(state, env)
                                        else:
                                            executor.current_sequence = action_sequence
                                            executor.current_sequence_index = 0
                                            action = executor.current_sequence[executor.current_sequence_index]
                                            executor.current_sequence_index += 1
                                except Exception:
                                    action = agent.act(state, env)
                            else:
                                executor.current_sequence = action_sequence
                                executor.current_sequence_index = 0
                                action = executor.current_sequence[executor.current_sequence_index]
                                executor.current_sequence_index += 1
                    else:
                        action = executor.current_sequence[executor.current_sequence_index]
                        executor.current_sequence_index += 1
                
                except Exception:
                    episode_hallucinations += 1
                    action = agent.act(state, env)
            else:
                action = agent.act(state, env)
            
            # Execute action
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # Reward shaping
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, next_state, info, previous_info, action
            )
            
            total_native_reward += native_reward
            total_shaped_reward += shaped_reward
            total_reward += shaped_reward
            
            # Track achievements
            if previous_info is not None:
                curr_achievements = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
                prev_achievements = set(k for k, v in previous_info.get('achievements', {}).items() if v >= 1)
                new_achievements = curr_achievements - prev_achievements
                episode_achievements += len(new_achievements)
                all_achievements_unlocked.update(new_achievements)
            
            # Store and replay
            agent.remember(state, action, shaped_reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            state = next_state
            previous_info = info
            moves += 1
        
        # ===== EPISODE END =====
        shaped_bonus = total_shaped_reward - total_native_reward
        
        rewards_per_episode.append(total_shaped_reward)
        native_rewards_per_episode.append(total_native_reward)
        shaped_bonus_per_episode.append(shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(moves)
        helper_calls.append(episode_helper_calls)
        hallucinations.append(episode_hallucinations)
        
        evaluation_system.add_episode(
            episode=e,
            shaped_reward=total_shaped_reward,
            native_reward=total_native_reward,
            shaped_bonus=shaped_bonus,
            achievements_unlocked=episode_achievements,
            moves=moves,
            helper_calls=episode_helper_calls,
            hallucinations=episode_hallucinations
        )
        
        # F09: Performance-based checkpointing
        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = e
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"best_model_ep{e}_ach{episode_achievements}")
            agent.save(checkpoint_path)
            print(f"  [Checkpoint] New best model saved (achievements: {episode_achievements})")
        
        # Periodic checkpoints
        if (e + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_ep{e}")
            agent.save(checkpoint_path)
            print(f"  [Checkpoint] Periodic checkpoint saved")
        
        print(f"  Reward (Shaped/Native/Bonus): {total_shaped_reward:.2f}/{total_native_reward:.2f}/{shaped_bonus:.2f}")
        print(f"  Achievements: {episode_achievements}, Moves: {moves}, Helper Calls: {episode_helper_calls}, Hallucinations: {episode_hallucinations}")
        print(f"  Unique Achievements Unlocked: {len(all_achievements_unlocked)}/22")
        
        # F09: Early stopping check
        if enable_early_stopping and early_stopper.should_stop(e, achievements_per_episode):
            print(f"\n[Early Stopping] Training terminated at episode {e}")
            break
    
    # ===== TRAINING COMPLETE =====
    print(f"\n[Training] Complete!")
    print(f"[Training] Total Episodes: {len(rewards_per_episode)}")
    print(f"[Training] Average Shaped Reward: {np.mean(rewards_per_episode):.2f}")
    print(f"[Training] Average Native Reward: {np.mean(native_rewards_per_episode):.2f}")
    print(f"[Training] Total Unique Achievements: {len(all_achievements_unlocked)}/22")
    print(f"[Training] Best Model: Episode {best_episode}, Achievements: {best_achievement_count}")
    
    # Save final model
    final_path = os.path.join(output_dir, "models", "crafter_heron_final")
    agent.save(final_path)
    print(f"[Save] Final model saved to: {final_path}")
    
    # Export results
    evaluation_system.finalize()
    evaluation_system.export_to_csv(os.path.join(output_dir, "heron_crafter_extended_metrics.csv"))
    evaluation_system.export_summary_json(os.path.join(output_dir, "heron_crafter_evaluation.json"))
    
    # Generate plots
    generate_all_plots(evaluation_system, output_dir=os.path.join(output_dir, "plots"))
    
    # F09: Save hyperparameter histories
    hp_df = pd.DataFrame({
        'episode': list(range(len(lr_history))),
        'learning_rate': lr_history,
        'epsilon': epsilon_history,
        'threshold': threshold_history
    })
    hp_df.to_csv(os.path.join(output_dir, "hyperparameter_history.csv"), index=False)
    
    return evaluation_system


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F09: HeRoN Iterative Training with Curriculum Learning")
    parser.add_argument("--episodes", type=int, default=500, help="Total training episodes")
    parser.add_argument("--batch_size", type=int, default=32, help="DQN replay batch size")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Checkpoint every N episodes")
    parser.add_argument("--reviewer_model", type=str, default="reviewer_retrained", help="Reviewer model path")
    parser.add_argument("--output_dir", type=str, default="./training_output", help="Output directory")
    parser.add_argument("--lr_strategy", type=str, default="step_decay", 
                       choices=['constant', 'step_decay', 'exponential', 'cosine'],
                       help="Learning rate schedule")
    parser.add_argument("--epsilon_strategy", type=str, default="linear_decay",
                       choices=['linear_decay', 'exponential_decay', 'staged'],
                       help="Epsilon decay strategy")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decay",
                       choices=['linear_decay', 'staged'],
                       help="Threshold decay strategy")
    parser.add_argument("--disable_curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("F09: HeRoN Iterative Training with Curriculum Learning")
    print(f"{'='*80}\n")
    
    train_heron_with_curriculum(
        episodes=args.episodes,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        reviewer_model_path=args.reviewer_model,
        output_dir=args.output_dir,
        lr_strategy=args.lr_strategy,
        epsilon_strategy=args.epsilon_strategy,
        threshold_strategy=args.threshold_strategy,
        enable_curriculum=not args.disable_curriculum,
        enable_early_stopping=not args.disable_early_stopping
    )
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}\n")
