"""
F08: HeRoN Integration for Crafter Environment
Three-agent system: DQNAgent (NPC) + CrafterHelper (LLM) + InstructorAgent (Reviewer)

Training loop:
1. Probability threshold decay controls LLM involvement (starts at 1.0, decays by 0.1 per episode)
2. On each step: if p > threshold and e < 600, invoke Helper→Reviewer→Helper workflow
3. Otherwise, use DQN directly or via SequenceExecutor fallback
4. Intrinsic reward shaping enhances sparse Crafter rewards (+0.1 resources, +0.05 health/tier, +0.02 tools)
5. Track achievements, helper calls, hallucinations, moves, and shaped rewards separately

Note: Reviewer model path is a placeholder - update after F06 fine-tuning completion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
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

# Achievement name-to-ID mapping for Crafter's 22 achievements
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

# Reverse mapping for plotting labels
ACHIEVEMENT_ID_TO_NAME = {v: k for k, v in ACHIEVEMENT_NAME_TO_ID.items()}

# LM Studio configuration
# Note: lmstudio v1.5.0 uses context manager syntax - connection handled in CrafterHelper
# No need to initialize global client here

# Device selection: CUDA (NVIDIA) or CPU only (NO MPS - Crafter incompatible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")
print(f"[Config] NOTE: MPS (Apple Silicon) not supported by Crafter environment")

# Reviewer model configuration (PPO fine-tuned model)
REVIEWER_MODEL_PATH = "reviewer_retrained_ppo"
REVIEWER_TOKENIZER_PATH = "reviewer_retrained_ppo"

print(f"[Config] Reviewer model path: {REVIEWER_MODEL_PATH}")
print(f"[Config] Using PPO fine-tuned reviewer model")

try:
    tokenizer_reviewer = AutoTokenizer.from_pretrained(REVIEWER_TOKENIZER_PATH)
    model_reviewer = T5ForConditionalGeneration.from_pretrained(REVIEWER_MODEL_PATH).to(device)
    print("[Config] Reviewer model loaded successfully")
except Exception as e:
    print(f"[WARNING] Failed to load Reviewer model: {e}")
    print("[WARNING] Training will proceed without Reviewer refinement")
    tokenizer_reviewer = None
    model_reviewer = None


# ============================================================================
# Reward Shaping for Sparse Crafter Rewards
# ============================================================================

class CrafterRewardShaper:
    """
    Augments sparse Crafter rewards with intrinsic bonuses for learning signal.
    
    Bonuses:
    - Resource collection: +0.1
    - Health management: +0.05
    - Tier progression: +0.05
    - Tool crafting: +0.3 (aligned with baseline DQN)
    - Death penalty: -1.0 (to encourage survival)
    """
    
    # Crafter action ID mapping
    # OFFICIAL CRAFTER ACTION MAPPING (corrected)
    ACTION_NAMES = {
        0: 'noop',
        1: 'move_left', 2: 'move_right', 3: 'move_up', 4: 'move_down',
        5: 'do', 6: 'sleep',
        7: 'place_stone', 8: 'place_table', 9: 'place_furnace', 10: 'place_plant',
        11: 'make_wood_pickaxe', 12: 'make_stone_pickaxe', 13: 'make_iron_pickaxe',
        14: 'make_wood_sword', 15: 'make_stone_sword', 16: 'make_iron_sword'
    }
    
    def __init__(self):
        self.bonus_tracker = {
            'resource_collection': [],
            'health_management': [],
            'tier_progression': [],
            'tool_usage': []
        }
    
    def calculate_shaped_reward(self, native_reward, state, info, previous_info, action):
        """
        Calculate total shaped reward = native + intrinsic bonuses.
        
        Args:
            native_reward: Sparse reward from Crafter (+1 for achievement, 0 otherwise)
            state: Current 41-dim state vector
            info: Current info dict (inventory, achievements, etc)
            previous_info: Previous info dict for state change detection
            action: Action ID (0-16)
        
        Returns:
            tuple: (shaped_reward, bonus_components_dict)
        """
        shaped_reward = native_reward
        bonuses = {
            'resource_collection': 0.0,
            'health_management': 0.0,
            'tier_progression': 0.0,
            'tool_usage': 0.0,
            'death_penalty': 0.0
        }
        
        if previous_info is None:
            return shaped_reward, bonuses
        
        # ===== DEATH PENALTY (-1.0) =====
        # Penalize agent for dying to encourage survival
        curr_health = info.get('inventory', {}).get('health', 10)
        prev_health = previous_info.get('inventory', {}).get('health', 10)
        if curr_health == 0 and prev_health > 0:
            bonuses['death_penalty'] = -1.0
        
        if previous_info is None:
            return shaped_reward, bonuses
        
        # ===== 1. Resource Collection Bonus (+0.1) =====
        # Reward agent for collecting resources (wood, stone, iron, coal, diamond)
        bonuses['resource_collection'] = self._calculate_resource_bonus(
            info, previous_info, action
        )
        
        # ===== 2. Health Management Bonus (+0.05) =====
        # Reward agent for maintaining health through food/drink
        bonuses['health_management'] = self._calculate_health_bonus(
            info, previous_info, action
        )
        
        # ===== 3. Tier Progression Bonus (+0.05) =====
        # Reward agent for advancing achievement tiers (collect→place→craft→interact)
        bonuses['tier_progression'] = self._calculate_tier_bonus(
            info, previous_info
        )
        
        # ===== 4. Tool Usage Bonus (+0.02) =====
        # Reward agent for using correct tool on resource type
        bonuses['tool_usage'] = self._calculate_tool_bonus(
            info, previous_info, action
        )
        
        # Sum all bonuses
        total_bonus = sum(bonuses.values())
        shaped_reward += total_bonus
        
        # Track for statistics
        for key in bonuses:
            self.bonus_tracker[key].append(bonuses[key])
        
        return shaped_reward, bonuses
    
    def _calculate_resource_bonus(self, info, previous_info, action):
        """
        +0.1 for successful resource collection.
        Detect: inventory increase for wood, stone, iron, coal, diamond after [do] action
        """
        if action != 5:  # [do] action (ID 5, not 4!)
            return 0.0
        
        bonus = 0.0
        resources = ['wood', 'stone', 'iron', 'coal', 'diamond']
        
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        
        for resource in resources:
            curr_count = curr_inv.get(resource, 0)
            prev_count = prev_inv.get(resource, 0)
            if curr_count > prev_count:
                bonus += 0.1
        
        return min(bonus, 0.1)  # Cap at 0.1 per step (even if multiple resources collected)
    
    def _calculate_health_bonus(self, info, previous_info, action):
        """
        +0.05 for maintaining/restoring health through food/drink consumption.
        Detect: health increase or food/drink decrease
        """
        bonus = 0.0
        
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        
        curr_health = curr_inv.get('health', 10)
        prev_health = prev_inv.get('health', 10)
        
        # Health increased (via food/drink/healing)
        if curr_health > prev_health:
            bonus += 0.05
        
        # Food or drink consumed (inventory decreased)
        if prev_inv.get('food', 0) > curr_inv.get('food', 0):
            bonus += 0.03
        if prev_inv.get('drink', 0) > curr_inv.get('drink', 0):
            bonus += 0.03
        
        return min(bonus, 0.05)  # Cap at 0.05 per step
    
    def _calculate_tier_bonus(self, info, previous_info):
        """
        +0.05 for advancing through achievement tier chains.
        E.g., collecting resources→placing structures→crafting tools→defeating enemies
        """
        bonus = 0.0
        
        curr_achievements = set(
            k for k, v in info.get('achievements', {}).items() if v >= 1
        )
        prev_achievements = set(
            k for k, v in previous_info.get('achievements', {}).items() if v >= 1
        )
        
        # Define tier chains
        tiers = {
            'collect': ['collect_wood', 'collect_stone', 'collect_iron', 'collect_coal', 'collect_diamond'],
            'place': ['place_stone', 'place_table', 'place_furnace', 'place_plant'],
            'make': ['make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe'],
            'interact': ['eat_plant', 'eat_cow', 'defeat_zombie', 'defeat_skeleton']
        }
        
        # Check for progression from collect → place → make → interact
        tier_order = ['collect', 'place', 'make', 'interact']
        for i, current_tier in enumerate(tier_order):
            current_tier_achievements = set(tiers[current_tier])
            if current_tier_achievements & curr_achievements and \
               not (current_tier_achievements & prev_achievements):
                # This tier was unlocked in this step
                bonus += 0.05 * (i + 1) / 4  # Scale bonus by tier level
                break
        
        return min(bonus, 0.05)
    
    def _calculate_tool_bonus(self, info, previous_info, action):
        """
        +0.3 for crafting tools/weapons (aligned with baseline DQN).
        E.g., crafting pickaxes or swords to improve survival and resource gathering.
        """
        bonus = 0.0
        
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        
        # Bonus for crafting any tool or weapon
        tools = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                 'wood_sword', 'stone_sword', 'iron_sword']
        
        for tool in tools:
            prev_val = prev_inv.get(tool, 0)
            curr_val = curr_inv.get(tool, 0)
            if curr_val > prev_val:
                bonus += 0.3  # Increased from 0.02 to match baseline DQN
        
        return min(bonus, 0.3)  # Cap at 0.3 per step
    
    def get_statistics(self):
        """Return average bonuses across tracked steps."""
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
        """Reset statistics tracker for new episode."""
        for key in self.bonus_tracker:
            self.bonus_tracker[key] = []


# ============================================================================
# Main Training Loop
# ============================================================================

def train_dqn_crafter(episodes=100, batch_size=32, episode_length=1000, threshold_episodes=600):
    """
    Train DQNAgent in Crafter environment with Helper and Reviewer integration.
    
    Args:
        episodes: Number of training episodes
        batch_size: DQN replay batch size
        episode_length: Steps per episode (reduced from 10000 for faster testing)
        threshold_episodes: Episodes after which to stop using LLM (disable threshold decay)
    """
    
    # Initialize environment and agents
    print("\n[Init] Initializing Crafter environment...")
    env = CrafterEnv(area=(64, 64), view=(9, 9), size=(64, 64), reward=True, 
                     length=episode_length, seed=None)
    
    print(f"[Init] State size: {env.state_size}, Action size: {env.action_size}")
    print(f"[Init] Using device: {device}")
    
    # Initialize DQN Agent with consistent device
    print("[Init] Initializing DQN Agent...")
    agent = DQNAgent(env.state_size, env.action_size, load_model_path=None)
    
    # Verify agent device matches global device
    print(f"[Init] DQN Agent device: {agent.device}")
    if agent.device != device:
        print(f"[WARNING] Device mismatch: Agent={agent.device}, Global={device}")
        print(f"[WARNING] This may cause issues with Reviewer interaction")
    
    # Initialize CrafterHelper (LLM)
    print("[Init] Initializing CrafterHelper...")
    # Note: lmstudio v1.5.0 uses context manager - server_host parameter ignored
    helper = CrafterHelper(model_name="qwen2.5-1.5b-instruct")
    
    # Initialize Reviewer (fine-tuned model)
    print("[Init] Initializing InstructorAgent (Reviewer)...")
    if model_reviewer is not None and tokenizer_reviewer is not None:
        instructor = InstructorAgent(model_reviewer, tokenizer_reviewer, device)
        use_reviewer = True
    else:
        instructor = None
        use_reviewer = False
        print("[Init] WARNING: Reviewer not available - training without refinement")
    
    # Initialize Sequence Executor for action sequence management
    executor = SequenceExecutor(agent, env)
    
    # Initialize Reward Shaper
    reward_shaper = CrafterRewardShaper()
    
    # Metrics tracking
    rewards_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    helper_calls = []
    hallucinations = []
    shaped_rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    
    # F09: Performance tracking for checkpointing
    best_achievement_count = 0
    best_episode = -1
    
    # Initialize F10 Evaluation System
    evaluation_system = EvaluationSystem(num_achievements=22)
    
    # F09: Threshold decay per EPISODE (not per step)
    threshold = 1.0
    threshold_decay_per_episode = 0.01  # Decays from 1.0 to 0.0 over 100 episodes
    
    print(f"\n[Training] Starting training for {episodes} episodes...")
    print(f"[Training] Threshold decay: {threshold_decay_per_episode} per episode (stops at episode {threshold_episodes})")
    print(f"[Training] Episode length: {episode_length} steps")
    print(f"[Training] Initial epsilon: {agent.epsilon:.4f}")
    
    # ===== EPISODE LOOP =====
    for e in range(episodes):
        state, initial_info = env.reset()  # Unpack tuple (state, info)
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        total_shaped_reward = 0
        total_native_reward = 0
        moves = 0
        episode_achievements = 0
        episode_helper_calls = 0
        episode_hallucinations = 0
        
        previous_info = initial_info  # Use info from reset
        reward_shaper.reset_episode()
        executor.current_sequence = []  # Reset sequence for new episode
        executor.current_sequence_index = 0
        
        # CRITICAL: Clear LLM conversation history to prevent context overflow (4096 token limit)
        helper.reset_conversation()
        
        # Monitor conversation length during episode
        conversation_reset_threshold = 10  # Reset after N Helper calls to prevent overflow
        
        # ===== STEP LOOP =====
        while not done and moves < episode_length:
            p = np.random.rand()
            
            # Decide: LLM or DQN
            use_llm = (p > threshold) and (e < threshold_episodes)
            
            if use_llm:
                # ===== LLM WORKFLOW: Helper → Reviewer → Helper =====
                episode_helper_calls += 1
                
                # Periodic conversation reset to prevent overflow (every N calls)
                if episode_helper_calls > 0 and episode_helper_calls % conversation_reset_threshold == 0:
                    helper.reset_conversation()
                    print(f"[Helper] Periodic conversation reset (call #{episode_helper_calls}) to prevent overflow")
                
                try:
                    # Get current game state and info
                    current_info = env._last_info if hasattr(env, '_last_info') else {}
                    
                    # Check if we should re-plan current sequence (Strategy B)
                    should_replan = (
                        executor.current_sequence and 
                        previous_info is not None and
                        helper.should_replan(state, current_info, previous_info, executor.current_sequence)
                    )
                    
                    if should_replan:
                        print(f"\n[Episode {e}, Step {moves}] Re-planning triggered - interrupting sequence")
                        executor.interrupt_sequence()
                    
                    # If sequence exhausted or interrupted, get new one
                    if not executor.current_sequence or executor.current_sequence_index >= len(executor.current_sequence):
                        print(f"\n[Episode {e}, Step {moves}] Helper generating new sequence...")
                        action_sequence, helper_response = helper.generate_action_sequence(
                            state, current_info, previous_info
                        )
                        
                        if action_sequence is None:
                            # Hallucination: LLM failed to generate valid sequence
                            episode_hallucinations += 1
                            print(f"[Helper] Hallucination detected - falling back to DQN")
                            action = agent.act(state, env)
                        else:
                            # 2. Reviewer refines suggestion (if available)
                            if use_reviewer and instructor is not None:
                                print(f"[Reviewer] Refining Helper suggestion...")
                                game_description = helper.describe_crafter_state(state, current_info, previous_info)
                                reviewer_feedback = instructor.generate_suggestion(game_description, helper_response)
                                print(f"[Reviewer] Feedback: {reviewer_feedback}\n")
                                
                                # 3. Helper reprompts WITHIN ITS CONVERSATION CONTEXT (FIX #3)
                                refined_prompt = (
                                    f"Reviewer feedback on your suggestion: {reviewer_feedback}\n"
                                    f"Please refine your action sequence based on this feedback.\n"
                                    f"Generate 3-5 actions in square brackets."
                                )
                                
                                try:
                                    # Use Helper's generate_action_sequence to preserve conversation context
                                    action_sequence, refined_response = helper.generate_action_sequence(
                                        state, current_info, previous_info, override_prompt=refined_prompt
                                    )
                                    print(f"[Helper] Refined response: {refined_response}\n")
                                    
                                    if action_sequence is None:
                                        episode_hallucinations += 1
                                        action = agent.act(state, env)
                                    else:
                                        # Store sequence in executor
                                        executor.current_sequence = action_sequence
                                        executor.current_sequence_index = 0
                                        action = executor.current_sequence[executor.current_sequence_index]
                                        executor.current_sequence_index += 1
                                        
                                        # CRITICAL: Clear conversation after Reviewer cycle to prevent overflow
                                        # This resets context while keeping the refined sequence
                                        helper.reset_conversation()
                                        print(f"[Helper] Conversation reset after Reviewer refinement (context overflow prevention)")
                                        
                                except Exception as e:
                                    print(f"[Helper] Error during refinement: {e}")
                                    episode_hallucinations += 1
                                    action = agent.act(state, env)
                            else:
                                # Store sequence and get first action
                                executor.current_sequence = action_sequence
                                executor.current_sequence_index = 0
                                action = executor.current_sequence[executor.current_sequence_index]
                                executor.current_sequence_index += 1
                    else:
                        # Continue with current sequence
                        action = executor.current_sequence[executor.current_sequence_index]
                        executor.current_sequence_index += 1
                
                except Exception as e:
                    print(f"[LLM] Error: {e}")
                    episode_hallucinations += 1
                    action = agent.act(state, env)
            
            else:
                # ===== DQN DIRECT OR FALLBACK =====
                action = agent.act(state, env)
            
            # ===== EXECUTE ACTION =====
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # ===== REWARD SHAPING =====
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, next_state, info, previous_info, action
            )
            
            total_native_reward += native_reward
            total_shaped_reward += shaped_reward
            total_reward += shaped_reward  # DQN trains on shaped reward
            
            # ===== UPDATE ACHIEVEMENTS COUNT =====
            if previous_info is not None:
                curr_achievements = set(
                    k for k, v in info.get('achievements', {}).items() if v >= 1
                )
                prev_achievements = set(
                    k for k, v in previous_info.get('achievements', {}).items() if v >= 1
                )
                newly_unlocked_names = curr_achievements - prev_achievements
                episode_achievements += len(newly_unlocked_names)
                
                # Map achievement names to IDs and track in evaluation system
                if newly_unlocked_names:
                    newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names 
                                         if name in ACHIEVEMENT_NAME_TO_ID}
                    evaluation_system.add_episode_achievements(e, newly_unlocked_ids, moves)
            
            # ===== STORE EXPERIENCE =====
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # ===== REPLAY =====
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            # ===== UPDATE STATE =====
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
        
        # Add to F10 EvaluationSystem
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
            checkpoint_path = os.path.join("checkpoints", f"best_model_ep{e}_ach{episode_achievements}")
            agent.save(checkpoint_path)
            print(f"\n[Checkpoint] New best model saved: {checkpoint_path}")
        
        # F09: Periodic checkpoints every 10 episodes
        if (e + 1) % 10 == 0:
            checkpoint_path = os.path.join("checkpoints", f"model_ep{e}")
            agent.save(checkpoint_path)
            print(f"[Checkpoint] Periodic checkpoint saved: {checkpoint_path}")
        
        print(f"\n[Episode {e}] Done!")
        print(f"  Total Reward (Shaped): {total_shaped_reward:.2f}")
        print(f"  Native Reward: {total_native_reward:.2f}")
        print(f"  Shaped Bonus: {shaped_bonus:.2f}")
        print(f"  Achievements Unlocked: {episode_achievements}")
        print(f"  Moves: {moves}")
        print(f"  Helper Calls: {episode_helper_calls}, Hallucinations: {episode_hallucinations}")
        print(f"  Epsilon: {agent.epsilon:.4f}, Threshold: {threshold:.4f}")
        print(f"  Helper Stats: {helper.get_statistics()}")
        
        # F09: Decay threshold PER EPISODE (not per step) - FIX #1
        if e < threshold_episodes:
            print(f"  Current Threshold Used: {threshold:.4f}")
            threshold = max(0, threshold - threshold_decay_per_episode)
            print(f"  Next Episode Threshold: {threshold:.4f}")
        else:
            print(f"  Threshold Decay Disabled (episode >= {threshold_episodes})")
    
    # ===== TRAINING COMPLETE =====
    print(f"\n[Training] Complete!")
    print(f"[Training] Average Shaped Reward: {np.mean(rewards_per_episode):.2f}")
    print(f"[Training] Average Native Reward: {np.mean(native_rewards_per_episode):.2f}")
    print(f"[Training] Average Shaped Bonus: {np.mean(shaped_rewards_per_episode):.2f}")
    print(f"[Training] Average Achievements: {np.mean(achievements_per_episode):.2f}")
    print(f"[Training] Average Moves: {np.mean(moves_per_episode):.2f}")
    print(f"[Training] Total Helper Calls: {sum(helper_calls)}")
    print(f"[Training] Total Hallucinations: {sum(hallucinations)}")
    print(f"[Training] Reward Shaping Stats: {reward_shaper.get_statistics()}")
    print(f"[Training] Best Model: Episode {best_episode}, Achievements: {best_achievement_count}")
    
    # Save final model
    print(f"\n[Save] Saving final trained agent...")
    final_path = os.path.join("models", "crafter_heron_final")
    agent.save(final_path)
    print(f"[Save] Final model saved to: {final_path}")
    
    # ===== F10: EVALUATION SYSTEM FINALIZATION =====
    print(f"\n[F10 Evaluation] Finalizing evaluation system...")
    evaluation_system.finalize()
    
    # Export metrics and summaries
    print(f"[F10 Evaluation] Exporting metrics...")
    evaluation_system.export_to_jsonl("heron_crafter_extended_metrics.jsonl")
    evaluation_system.export_summary_json("heron_crafter_evaluation.json")
    
    # Export per-achievement statistics to JSON
    print(f"[F10 Evaluation] Exporting per-achievement statistics...")
    export_achievement_statistics_json(evaluation_system, "heron_crafter_achievement_statistics.json")
    
    # Generate advanced plots
    print(f"[F10 Evaluation] Generating advanced evaluation plots...")
    generate_all_plots(evaluation_system, output_dir="./evaluation_plots")
    
    # Generate per-achievement plots
    print(f"[F10 Evaluation] Generating per-achievement progression curves...")
    plot_achievement_curves(evaluation_system, moves_per_episode, "heron_crafter_achievement_curves.png")
    
    print(f"[F10 Evaluation] Generating achievement heatmap...")
    plot_achievement_heatmap(evaluation_system, "heron_crafter_achievement_heatmap.png")
    
    print(f"[F10 Evaluation] Generating achievement timeline...")
    plot_achievement_timeline(evaluation_system, "heron_crafter_achievement_timeline.png")
    
    # Print comprehensive report
    print(f"[F10 Evaluation] Printing summary report...")
    evaluation_system.print_summary_report()
    
    return (rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
            achievements_per_episode, moves_per_episode, helper_calls, hallucinations,
            helper.get_statistics(), reward_shaper.get_statistics(), evaluation_system)


# ============================================================================
# Visualization and Export
# ============================================================================

def plot_training(rewards, native_rewards, shaped_bonus, achievements, moves, 
                  helper_calls, hallucinations):
    """Create comprehensive training visualization plots (same as DQN baseline)."""
    
    episodes = range(1, len(rewards) + 1)
    output_prefix = "heron_crafter"
    
    # 1. Rewards over time
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, rewards, label='Shaped Reward (native + bonus)', linewidth=2, marker='o', markersize=4)
    plt.plot(episodes, native_rewards, label='Native Reward (sparse)', linewidth=2, marker='s', markersize=4)
    plt.plot(episodes, shaped_bonus, label='Shaped Bonus Total', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Reward', fontsize=12, fontweight='bold')
    plt.title('HeRoN Training - Reward Trends', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_rewards.png")
    
    # 2. Cumulative achievements
    cumulative_achievements = np.cumsum(achievements)
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, cumulative_achievements, linewidth=2.5, marker='o', markersize=5, color='green')
    plt.fill_between(episodes, cumulative_achievements, alpha=0.3, color='green')
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Achievements', fontsize=12, fontweight='bold')
    plt.title('HeRoN Training - Cumulative Achievement Unlocks', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_achievements.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_achievements.png")
    
    # 3. Moves per episode
    plt.figure(figsize=(14, 6))
    plt.bar(episodes, moves, color='steelblue', alpha=0.7, edgecolor='black')
    plt.plot(episodes, moves, color='darkblue', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Moves per Episode', fontsize=12, fontweight='bold')
    plt.title('HeRoN Training - Episode Length (Moves)', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_moves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_moves.png")
    
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
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_efficiency.png")
    
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
    
    plt.suptitle('HeRoN Training - Multi-Metric Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_dashboard.png")
    
    # 6. Helper Usage and Hallucinations (HeRoN-specific)
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(episodes, helper_calls, color='blue', alpha=0.7, label='Helper Calls', linewidth=2, marker='o', markersize=3)
    ax1.set_title('Helper Calls per Episode', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Calls', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(1, 2, 2)
    hallucination_rate = [h / max(1, c) for h, c in zip(hallucinations, helper_calls)]
    ax2.plot(episodes, hallucination_rate, color='red', alpha=0.7, label='Hallucination Rate', linewidth=2, marker='s', markersize=3)
    ax2.set_title('LLM Hallucination Rate per Episode', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Hallucination Rate', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_helper_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_prefix}_helper_stats.png")


def export_metrics_csv(rewards, native_rewards, shaped_bonus, achievements, moves,
                      helper_calls, hallucinations, output_file="heron_crafter_metrics.csv"):
    """Export training metrics to CSV."""
    
    df = pd.DataFrame({
        'episode': list(range(len(rewards))),
        'shaped_reward': rewards,
        'native_reward': native_rewards,
        'shaped_bonus': shaped_bonus,
        'achievements_unlocked': achievements,
        'moves': moves,
        'helper_calls': helper_calls,
        'hallucinations': hallucinations,
        'hallucination_rate': [h / max(1, c) for h, c in zip(hallucinations, helper_calls)]
    })
    
    df.to_csv(output_file, index=False)
    print(f"[Export] Saved metrics to: {output_file}")


# ============================================================================
# Per-Achievement Visualization Functions
# ============================================================================

def plot_achievement_curves(evaluation_system, moves_per_episode, output_file="heron_crafter_achievement_curves.png"):
    """
    Create per-achievement progression curves similar to PPO reference image.
    Shows 22 subplots (4x6 grid) with unlock count over cumulative training steps.
    Includes shaded min/max bands.
    
    Args:
        evaluation_system: EvaluationSystem instance with achievement tracking
        moves_per_episode: List of moves per episode to compute cumulative steps
        output_file: Path to save the plot
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    cumulative_matrix = achievement_stats.get('cumulative_achievement_matrix', [])
    
    if not cumulative_matrix:
        print("[Warning] No achievement data available for plotting curves")
        return
    
    # Compute cumulative steps (x-axis)
    cumulative_steps = np.cumsum([0] + moves_per_episode)  # Prepend 0 for episode 0
    cumulative_steps = cumulative_steps[:len(cumulative_matrix)]  # Match matrix length
    
    # Create 4x6 subplot grid for 22 achievements
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    axes = axes.flatten()
    
    for ach_id in range(22):
        ax = axes[ach_id]
        
        # Extract trajectory for this achievement
        trajectory = [cumulative_matrix[ep][ach_id] for ep in range(len(cumulative_matrix))]
        
        # Plot line
        ax.plot(cumulative_steps, trajectory, color='green', linewidth=1.5, alpha=0.8)
        
        # Add shaded region (min/max) - using same trajectory for now
        # In multi-run experiments, this would show variance across runs
        min_trajectory = np.maximum(0, np.array(trajectory) - 0.1)  # Placeholder
        max_trajectory = np.array(trajectory) + 0.1  # Placeholder
        ax.fill_between(cumulative_steps, min_trajectory, max_trajectory, 
                        color='green', alpha=0.2)
        
        # Formatting
        achievement_name = ACHIEVEMENT_ID_TO_NAME.get(ach_id, f"Achievement {ach_id}")
        ax.set_title(achievement_name.replace('_', ' ').title(), fontsize=9)
        ax.set_xlabel('Steps', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
    
    # Hide extra subplots (22 achievements, 24 subplot positions)
    for idx in range(22, 24):
        axes[idx].axis('off')
    
    plt.suptitle('Achievement Curves of HeRoN Training', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved per-achievement curves to: {output_file}")
    plt.close()


def plot_achievement_heatmap(evaluation_system, output_file="heron_crafter_achievement_heatmap.png"):
    """
    Create achievement unlock frequency heatmap (22 achievements × episode bins).
    Shows when achievements are being unlocked during training.
    
    Args:
        evaluation_system: EvaluationSystem instance with achievement tracking
        output_file: Path to save the heatmap
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    episode_matrix = achievement_stats.get('episode_achievement_matrix', [])
    
    if not episode_matrix:
        print("[Warning] No achievement data available for heatmap")
        return
    
    # Transpose matrix: rows = achievements, columns = episodes
    num_episodes = len(episode_matrix)
    num_achievements = 22
    
    heatmap_data = np.zeros((num_achievements, num_episodes))
    for ep in range(num_episodes):
        for ach_id in range(num_achievements):
            heatmap_data[ach_id, ep] = episode_matrix[ep][ach_id]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(0, num_episodes, max(1, num_episodes // 20)))
    ax.set_yticks(np.arange(num_achievements))
    
    # Set labels
    achievement_labels = [ACHIEVEMENT_ID_TO_NAME.get(i, f"Ach{i}").replace('_', ' ').title() 
                         for i in range(num_achievements)]
    ax.set_yticklabels(achievement_labels, fontsize=9)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Achievement', fontsize=12)
    ax.set_title('Achievement Unlock Frequency Heatmap (Per Episode)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Unlock Count', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved achievement heatmap to: {output_file}")
    plt.close()


def plot_achievement_timeline(evaluation_system, output_file="heron_crafter_achievement_timeline.png"):
    """
    Create bar chart showing first unlock episode for each achievement.
    Visualizes progression through achievement unlocking timeline.
    
    Args:
        evaluation_system: EvaluationSystem instance with achievement tracking
        output_file: Path to save the timeline
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    per_ach_stats = achievement_stats.get('per_achievement_stats', [])
    
    if not per_ach_stats:
        print("[Warning] No achievement data available for timeline")
        return
    
    # Extract first unlock episodes
    achievement_ids = []
    first_unlock_episodes = []
    achievement_labels = []
    
    for stat in per_ach_stats:
        ach_id = stat['achievement_id']
        first_ep = stat.get('first_unlock_episode')
        
        if first_ep is not None:  # Only plot unlocked achievements
            achievement_ids.append(ach_id)
            first_unlock_episodes.append(first_ep)
            achievement_labels.append(ACHIEVEMENT_ID_TO_NAME.get(ach_id, f"Ach{ach_id}"))
    
    if not achievement_ids:
        print("[Warning] No achievements unlocked yet")
        return
    
    # Sort by first unlock episode
    sorted_indices = np.argsort(first_unlock_episodes)
    sorted_labels = [achievement_labels[i] for i in sorted_indices]
    sorted_episodes = [first_unlock_episodes[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_labels) * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
    bars = ax.barh(range(len(sorted_labels)), sorted_episodes, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels([label.replace('_', ' ').title() for label in sorted_labels], fontsize=10)
    ax.set_xlabel('First Unlock Episode', fontsize=12)
    ax.set_title('Achievement First Unlock Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, episode) in enumerate(zip(bars, sorted_episodes)):
        ax.text(episode + 0.5, i, f'{episode}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved achievement timeline to: {output_file}")
    plt.close()


def export_achievement_statistics_json(evaluation_system, output_file="heron_crafter_achievement_statistics.json"):
    """
    Export comprehensive per-achievement statistics to JSON file.
    Includes episode-achievement matrix, cumulative trajectories, and per-achievement stats.
    
    Args:
        evaluation_system: EvaluationSystem instance with achievement tracking
        output_file: Path to save JSON file
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    
    # Convert numpy arrays to lists for JSON serialization
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
    
    # Add achievement name mapping
    serializable_stats['achievement_id_to_name'] = ACHIEVEMENT_ID_TO_NAME
    
    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"[Export] Saved achievement statistics to: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HeRoN F08: Crafter Environment Integration")
    print("Three-Agent System: DQNAgent + CrafterHelper + InstructorAgent")
    print("="*80)
    
    # Train
    (rewards, native_rewards, shaped_bonus, achievements, moves, 
     helper_calls, hallucinations, helper_stats, reward_shaper_stats, eval_system) = train_dqn_crafter(
        episodes=300,  # Start with 50 for testing, increase for full training
        batch_size=32,
        episode_length=1000,  # Reduced from 1000 for testing
        threshold_episodes=600
    )
    
    # Visualize
    plot_training(rewards, native_rewards, shaped_bonus, achievements, moves,
                  helper_calls, hallucinations)
    
    # Export
    export_metrics_csv(rewards, native_rewards, shaped_bonus, achievements, moves,
                      helper_calls, hallucinations)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
