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

# LM Studio configuration
SERVER_API_HOST = "http://127.0.0.1:1234"
lms.get_default_client(SERVER_API_HOST)

# Device selection: MPS (Apple Silicon), CUDA (NVIDIA), or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Reviewer model configuration (PLACEHOLDER - update after F06 completion)
REVIEWER_MODEL_PATH = "reviewer_retrained"  # TODO: Update after F06
REVIEWER_TOKENIZER_PATH = "reviewer_retrained"  # TODO: Update after F06

print(f"[Config] Reviewer model path (PLACEHOLDER): {REVIEWER_MODEL_PATH}")
print(f"[Config] Note: Update paths after F06 fine-tuning completion")

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
    """Augments sparse Crafter rewards with intrinsic bonuses for learning signal."""
    
    # Crafter action ID mapping
    ACTION_NAMES = {
        0: 'move_up', 1: 'move_down', 2: 'move_left', 3: 'move_right',
        4: 'do', 5: 'sleep',
        6: 'place_stone', 7: 'place_table', 8: 'place_furnace', 9: 'place_plant',
        10: 'make_wood_pickaxe', 11: 'make_stone_pickaxe', 12: 'make_iron_pickaxe',
        13: 'make_wood_sword', 14: 'make_stone_sword', 15: 'make_iron_sword',
        16: 'noop'
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
            'tool_usage': 0.0
        }
        
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
        if action != 4:  # [do] action
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
        +0.02 for using correct tool on resource type.
        E.g., using pickaxe on stone/iron/coal/diamond (not wood with pickaxe)
        """
        bonus = 0.0
        
        # Map action IDs to tool usage scenarios
        # 10-12: make pickaxes; 13-15: make swords (tool creation)
        # Bonus: if tool was used to collect harder resource
        
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        
        # Check if better-tier pickaxe was created/used
        tool_actions = {
            10: ['wood_pickaxe'],  # make_wood_pickaxe
            11: ['stone_pickaxe'],  # make_stone_pickaxe
            12: ['iron_pickaxe'],   # make_iron_pickaxe
        }
        
        if action in tool_actions:
            # Tier progression in tools
            bonus += 0.02
        
        return bonus
    
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
    
    # Initialize DQN Agent
    print("[Init] Initializing DQN Agent...")
    agent = DQNAgent(env.state_size, env.action_size, load_model_path=None)
    
    # Initialize CrafterHelper (LLM)
    print("[Init] Initializing CrafterHelper...")
    helper = CrafterHelper(server_host=SERVER_API_HOST, model_name="llama-3.2-3b-instruct")
    
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
        executor.current_sequence = []  # Reset sequence for new episode
        executor.current_sequence_index = 0
        
        # ===== STEP LOOP =====
        while not done and moves < episode_length:
            p = np.random.rand()
            
            # Decide: LLM or DQN
            use_llm = (p > threshold) and (e < threshold_episodes)
            
            if use_llm:
                # ===== LLM WORKFLOW: Helper → Reviewer → Helper =====
                episode_helper_calls += 1
                
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
                                
                                # 3. Helper reprompts based on Reviewer feedback
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
                                        print(f"[Helper] Refined response: {refined_response}\n")
                                        
                                        # Re-parse with refined response
                                        action_sequence = helper.parse_action_sequence(refined_response)
                                        if action_sequence is None:
                                            episode_hallucinations += 1
                                            action = agent.act(state, env)
                                        else:
                                            # Store sequence in executor
                                            executor.current_sequence = action_sequence
                                            executor.current_sequence_index = 0
                                            action = executor.current_sequence[executor.current_sequence_index]
                                            executor.current_sequence_index += 1
                                except Exception as e:
                                    print(f"[Helper] Error during refinement: {e}")
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
                episode_achievements += len(curr_achievements - prev_achievements)
            
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
        
        # F09: Decay threshold PER EPISODE (not per step)
        if e < threshold_episodes:
            threshold = max(0, threshold - threshold_decay_per_episode)
            print(f"  Next Episode Threshold: {threshold:.4f}")
    
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
    evaluation_system.export_metrics_dataframe().to_csv("heron_crafter_extended_metrics.csv", index=False)
    evaluation_system.export_to_csv("heron_crafter_extended_metrics.csv")
    evaluation_system.export_summary_json("heron_crafter_evaluation.json")
    
    # Generate advanced plots
    print(f"[F10 Evaluation] Generating advanced evaluation plots...")
    generate_all_plots(evaluation_system, output_dir="./evaluation_plots")
    
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
    """Create comprehensive training visualization plots."""
    
    # Plot 1: Shaped Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Shaped Reward', color='blue', alpha=0.7)
    plt.plot(native_rewards, label='Native Reward', color='green', alpha=0.7)
    plt.plot(shaped_bonus, label='Shaped Bonus', color='orange', alpha=0.7)
    plt.title('Rewards per Episode (Crafter + HeRoN Integration)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("heron_crafter_rewards.png", dpi=150, bbox_inches='tight')
    print("[Plot] Saved: heron_crafter_rewards.png")
    
    # Plot 2: Achievements
    plt.figure(figsize=(10, 6))
    cumulative_achievements = np.cumsum(achievements)
    plt.plot(cumulative_achievements, color='purple', linewidth=2)
    plt.title('Cumulative Achievements Unlocked per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Achievements')
    plt.grid(True, alpha=0.3)
    plt.savefig("heron_crafter_achievements.png", dpi=150, bbox_inches='tight')
    print("[Plot] Saved: heron_crafter_achievements.png")
    
    # Plot 3: Moves per Episode
    plt.figure(figsize=(10, 6))
    plt.plot(moves, color='teal', alpha=0.7)
    plt.title('Moves per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')
    plt.grid(True, alpha=0.3)
    plt.savefig("heron_crafter_moves.png", dpi=150, bbox_inches='tight')
    print("[Plot] Saved: heron_crafter_moves.png")
    
    # Plot 4: Helper Usage and Hallucinations
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(helper_calls, color='blue', alpha=0.7, label='Helper Calls')
    ax1.set_title('Helper Calls per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Number of Calls')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(1, 2, 2)
    hallucination_rate = [h / max(1, c) for h, c in zip(hallucinations, helper_calls)]
    ax2.plot(hallucination_rate, color='red', alpha=0.7, label='Hallucination Rate')
    ax2.set_title('LLM Hallucination Rate per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Hallucination Rate')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("heron_crafter_helper_stats.png", dpi=150, bbox_inches='tight')
    print("[Plot] Saved: heron_crafter_helper_stats.png")


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
        episodes=50,  # Start with 50 for testing, increase for full training
        batch_size=32,
        episode_length=500,  # Reduced from 10000 for testing
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
