import numpy as np
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
from training.reward_shaper import CrafterRewardShaper
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
# Main Training Loop
# ============================================================================

def train_dqn_crafter(episodes=100, batch_size=64, episode_length=1000, threshold_episodes=100):
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
    helper = CrafterHelper(model_name="qwen/qwen3-4b-2507")
    
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
    
    # Performance tracking for checkpointing
    best_achievement_count = 0
    best_episode = -1
    
    # Initialize Evaluation System
    evaluation_system = EvaluationSystem(num_achievements=22)
    
    # Threshold decay per EPISODE (not per step)
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
        
        # Track achievements unlocked this episode (for Helper's episode summary)
        episode_achievements_list = []
        
        # CRITICAL: Clear LLM conversation history ONLY at episode start
        helper.reset_conversation()
        print(f"[Episode {e}] Conversation history cleared for new episode")
        
        # ===== STEP LOOP =====
        while not done and moves < episode_length:
            p = np.random.rand()
            
            # Decide: LLM or DQN
            # Threshold logic: p < threshold means use LLM
            # threshold=1.0 → always LLM, threshold=0.0 → never LLM
            use_llm = (p < threshold) and (e < threshold_episodes)
            
            if use_llm:
                # ===== LLM WORKFLOW: Helper → Reviewer → Helper =====
                episode_helper_calls += 1
                
                # Update Helper's episode progress for smart context summarization
                helper.update_episode_progress(
                    achievements=episode_achievements_list,
                    step_count=moves,
                    reward=total_shaped_reward
                )
                
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
                                
                                # Store feedback for potential episode summary (if context reset occurs)
                                helper.update_episode_progress(reviewer_feedback=reviewer_feedback)
                                
                                # 3. Helper reprompts WITHIN ITS CONVERSATION CONTEXT (FIX #3)
                                refined_prompt = (
                                    f"REVIEWER FEEDBACK: {reviewer_feedback}\n\n"
                                    f"CRITICAL: Refine your action sequence to address the feedback above.\n"
                                    f"You MUST:\n"
                                    f"1. Use ONLY these 17 valid actions: move_up, move_down, move_left, move_right, do, sleep, "
                                    f"place_stone, place_table, place_furnace, place_plant, "
                                    f"make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, "
                                    f"make_wood_sword, make_stone_sword, make_iron_sword, noop\n"
                                    f"2. Generate EXACTLY 3-5 actions in square brackets: [action1], [action2], [action3]\n"
                                    f"3. Follow the Reviewer's strategic advice (e.g., prioritize resource gathering if suggested)\n"
                                    f"4. NO placeholders, NO invented actions, NO parentheses!\n\n"
                                    f"Format: [move_right], [do], [move_left], [noop]\n"
                                    f"One-line reason.\n\n"
                                    f"Generate your REFINED sequence now:"
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
                                        
                                        # NOTE: Conversation reset removed - let periodic reset handle it
                                        # This preserves context for better learning from Reviewer feedback
                                        
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
                native_reward, info, previous_info
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
                
                # Track achievement names for Helper's episode summary
                if newly_unlocked_names:
                    episode_achievements_list.extend(newly_unlocked_names)
                
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
        
        # Calculate valid actions percentage
        # Formula: (Valid Actions / Total Actions Generated) * 100%
        # Where: Valid Actions = Total Helper Calls - Hallucinations
        if episode_helper_calls > 0:
            valid_actions = episode_helper_calls - episode_hallucinations
            valid_actions_percentage = (valid_actions / episode_helper_calls) * 100.0
        else:
            valid_actions_percentage = 0.0
        
        print(f"  Valid Actions Percentage: {valid_actions_percentage:.2f}% ({episode_helper_calls - episode_hallucinations}/{episode_helper_calls})")
        
        # Add to F10 EvaluationSystem
        evaluation_system.add_episode(
            episode=e,
            shaped_reward=total_shaped_reward,
            native_reward=total_native_reward,
            shaped_bonus=shaped_bonus,
            achievements_unlocked=episode_achievements,
            moves=moves,
            helper_calls=episode_helper_calls,
            hallucinations=episode_hallucinations,
            valid_actions_percentage=valid_actions_percentage
        )
        
        # F09: Performance-based checkpointing
        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = e
            checkpoint_dir = Path(__file__).parent / 'heron_output' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"heron_best_ep{e}_ach{episode_achievements}"
            agent.save(str(checkpoint_path))
            print(f"\n[Checkpoint] New best model saved: {checkpoint_path}.*")
        
        # F09: Periodic checkpoints every 10 episodes
        if (e + 1) % 10 == 0:
            checkpoint_dir = Path(__file__).parent / 'heron_output' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"heron_ep{e}"
            agent.save(str(checkpoint_path))
            print(f"[Checkpoint] Periodic checkpoint saved: {checkpoint_path}.*")
        
        print(f"\n[Episode {e}] Done!")
        print(f"  Total Reward (Shaped): {total_shaped_reward:.2f}")
        print(f"  Native Reward: {total_native_reward:.2f}")
        print(f"  Shaped Bonus: {shaped_bonus:.2f}")
        print(f"  Achievements Unlocked: {episode_achievements}")
        print(f"  Moves: {moves}")
        print(f"  Helper Calls: {episode_helper_calls}, Hallucinations: {episode_hallucinations}")
        print(f"  Epsilon: {agent.epsilon:.4f}, Threshold: {threshold:.4f}")
        print(f"  Helper Stats: {helper.get_statistics()}")
        
        # Decay threshold PER EPISODE (not per step) - FIX #1
        if e < threshold_episodes:
            print(f"  Current Threshold Used: {threshold:.4f}")
            threshold = max(0, threshold - threshold_decay_per_episode)
            print(f"  Next Episode Threshold: {threshold:.4f}")
        else:
            print(f"  Threshold Decay Disabled (episode >= {threshold_episodes})")

        # Epsilon decay lineare per episodio (come da documentazione: 1.0 -> 0.05 in 300 episodi)
        agent.decay_epsilon_linear(e, total_episodes=episodes)
    
    # ===== TRAINING COMPLETE =====
    print(f"\n[Training] Complete!")
    print(f"[Training] Average Shaped Reward: {np.mean(rewards_per_episode):.2f}")
    print(f"[Training] Average Native Reward: {np.mean(native_rewards_per_episode):.2f}")
    print(f"[Training] Average Shaped Bonus: {np.mean(shaped_bonus_per_episode):.2f}")
    print(f"[Training] Average Achievements: {np.mean(achievements_per_episode):.2f}")
    print(f"[Training] Average Moves: {np.mean(moves_per_episode):.2f}")
    print(f"[Training] Total Helper Calls: {sum(helper_calls)}")
    print(f"[Training] Total Hallucinations: {sum(hallucinations)}")
    print(f"[Training] Reward Shaping Stats: {reward_shaper.get_statistics()}")
    print(f"[Training] Best Model: Episode {best_episode}, Achievements: {best_achievement_count}")
    
    # Finalize evaluation
    evaluation_system.finalize()
    
    # Save final model
    print("\n[HeRoN] Saving final model...")
    models_dir = Path(__file__).parent / 'heron_output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "heron_crafter_final"
    agent.save(str(final_model_path))
    print(f"[HeRoN] ✓ Final model saved: {final_model_path}.*")
    
    checkpoint_dir = Path(__file__).parent / 'heron_output' / 'checkpoints'
    best_checkpoint = checkpoint_dir / f"heron_best_ep{best_episode}_ach{best_achievement_count}"
    print(f"[HeRoN] ✓ Best model saved: {best_checkpoint}.*")
    
    # Export metrics
    print("[HeRoN] Exporting metrics...")
    output_dir = Path(__file__).parent / 'heron_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "heron_crafter_metrics.jsonl"
    evaluation_system.export_to_jsonl(str(jsonl_path))
    
    # Export per-achievement statistics to JSON
    print("[HeRoN] Exporting per-achievement statistics...")
    achievement_stats_path = output_dir / "heron_achievement_statistics.json"
    export_achievement_statistics_json(evaluation_system, str(achievement_stats_path))
    
    # Generate plots
    print("[HeRoN] Generating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    generate_all_plots(evaluation_system, output_dir=str(plot_dir), title_prefix="HeRoN")
    
    # Print evaluation report
    print("\n[HeRoN] Final Evaluation Report:")
    evaluation_system.print_summary_report()
    
    # Export evaluation summary
    evaluation_json = output_dir / "heron_evaluation.json"
    evaluation_system.export_summary_json(str(evaluation_json))
    print(f"[HeRoN] ✓ Evaluation summary saved: {evaluation_json}")
    
    return (rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
            achievements_per_episode, moves_per_episode, helper_calls, hallucinations,
            helper.get_statistics(), reward_shaper.get_statistics(), evaluation_system)


# ============================================================================
# Visualization and Export
# ============================================================================

# ============================================================================
# Per-Achievement Visualization Functions
# ============================================================================


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
        episodes=300,  # 300 episodi come da documentazione
        batch_size=64,  # 64 come da documentazione
        episode_length=1000,  # 1000 steps per episodio come da documentazione
        threshold_episodes=100  # LLM attivo per primi 100 episodi come da documentazione
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
