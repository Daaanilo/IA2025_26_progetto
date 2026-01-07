import numpy as np
import re
import os
import sys
import json
from pathlib import Path

# Aggiungo parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import lmstudio as lms
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from classes.crafter_helper import CrafterHelper, SequenceExecutor
from evaluation.evaluation_system import EvaluationSystem
from training.reward_shaper import CrafterRewardShaper
from evaluation.evaluation_plots import AdvancedPlotter, generate_all_plots

# Mappa nomi-ID per gli achievement di Crafter
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

# Mappa inversa
ACHIEVEMENT_ID_TO_NAME = {v: k for k, v in ACHIEVEMENT_NAME_TO_ID.items()}

# Scelta device: CUDA o CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")
print(f"[Config] NOTE: MPS (Apple Silicon) not supported by Crafter environment")

# ============================================================================
# DQN + Helper Configuration (same as HERON_initial, without Reviewer)
# ============================================================================
ASSISTED_STEPS = 100  # LLM attivo solo nei primi 100 step di ogni episodio

# ============================================================================
# Main Training Loop
# ============================================================================

def train_dqn_helper(episodes=300, batch_size=64, episode_length=1000, threshold_episodes=100):
    """
    Allena l'agente DQN aiutato dall'Helper (senza Reviewer).
    
    Logica (come HERON_initial):
    - Se moves < ASSISTED_STEPS e episodio < threshold_episodes: usa LLM
    - Altrimenti: usa DQN
    """
    
    # Inizializza ambiente
    print("\n[Init] Initializing Crafter environment...")
    env = CrafterEnv(area=(64, 64), view=(9, 9), size=(64, 64), reward=True, 
                     length=episode_length, seed=None)
    
    print(f"[Init] State size: {env.state_size}, Action size: {env.action_size}")
    print(f"[Init] Using device: {device}")
    
    # Inizializza DQN Agent
    print("[Init] Initializing DQN Agent...")
    agent = DQNAgent(env.state_size, env.action_size, load_model_path=None)
    
    # Controllo device
    print(f"[Init] DQN Agent device: {agent.device}")
    if agent.device != device:
        print(f"[WARNING] Device mismatch: Agent={agent.device}, Global={device}")
    
    # Inizializza Helper (LLM)
    print("[Init] Initializing CrafterHelper...")
    helper = CrafterHelper(model_name="qwen/qwen3-4b-2507")
    
    # Niente Reviewer qui
    use_reviewer = False
    print("[Init] Config: DQN + Helper (Reviewer DISABLED)")
    
    # Esecutore di sequenze
    executor = SequenceExecutor(agent, env)
    
    # Reward Shaper
    reward_shaper = CrafterRewardShaper()
    
    # Metriche
    rewards_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    helper_calls = []
    hallucinations = []
    shaped_rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    
    # Tracking performance
    best_achievement_count = 0
    best_episode = -1
    
    # Sistema di valutazione
    evaluation_system = EvaluationSystem(num_achievements=22)
    
    print("\n" + "="*80)
    print("DQN + Helper Configuration (same as HERON_initial, without Reviewer)")
    print(f"LLM attivo nei primi {ASSISTED_STEPS} step di ogni episodio")
    print(f"LLM attivo solo per i primi {threshold_episodes} episodi")
    print("="*80)
    
    print(f"\n[Training] Starting training for {episodes} episodes...")
    print(f"[Training] ASSISTED_STEPS: {ASSISTED_STEPS} (LLM attivo nei primi {ASSISTED_STEPS} step)")
    print(f"[Training] threshold_episodes: {threshold_episodes} (LLM attivo fino all'episodio {threshold_episodes-1})")
    print(f"[Training] Episode length: {episode_length} steps")
    print(f"[Training] Initial epsilon: {agent.epsilon:.4f}")
    
    # ===== EPISODE LOOP =====
    for e in range(episodes):
        state, initial_info = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        total_shaped_reward = 0
        total_native_reward = 0
        moves = 0
        episode_achievements = 0
        episode_helper_calls = 0
        episode_hallucinations = 0
        
        previous_info = initial_info
        reward_shaper.reset_episode()
        executor.current_sequence = []  # Reset sequenza
        executor.current_sequence_index = 0
        
        # Lista achievement sbloccati in questo episodio
        episode_achievements_list = []
        
        # Pulisce la memoria dell'LLM a inizio episodio
        helper.reset_conversation()
        print(f"[Episode {e}] Conversation history cleared for new episode")
        
        # ===== STEP LOOP =====
        while not done and moves < episode_length:
            
            # ===== DQN + Helper: LLM solo nei primi ASSISTED_STEPS =====
            use_llm = (moves < ASSISTED_STEPS) and (e < threshold_episodes)
            
            if use_llm:
                # ===== WORKFLOW LLM: Solo Helper =====
                episode_helper_calls += 1
                
                # Aggiorna progressi per l'helper
                helper.update_episode_progress(
                    achievements=episode_achievements_list,
                    step_count=moves,
                    reward=total_shaped_reward
                )
                
                try:
                    # Stato attuale
                    current_info = env._last_info if hasattr(env, '_last_info') else {}
                    
                    # Controlla se serve ripianificare
                    should_replan = (
                        executor.current_sequence and 
                        previous_info is not None and
                        helper.should_replan(state, current_info, previous_info, executor.current_sequence)
                    )
                    
                    if should_replan:
                        print(f"\n[Episode {e}, Step {moves}] Re-planning triggered - interrupting sequence")
                        executor.interrupt_sequence()
                    
                    # Se sequenza finita o interrotta, fanne una nuova
                    if not executor.current_sequence or executor.current_sequence_index >= len(executor.current_sequence):
                        print(f"\n[Episode {e}, Step {moves}] Helper generating new sequence...")
                        action_sequence, helper_response = helper.generate_action_sequence(
                            state, current_info, previous_info
                        )
                        
                        if action_sequence is None:
                            # Allucinazione
                            episode_hallucinations += 1
                            print(f"[Helper] Hallucination detected - falling back to DQN")
                            action = agent.act(state, env)
                        else:
                            # Nuova sequenza
                            executor.current_sequence = action_sequence
                            executor.current_sequence_index = 0
                            action = executor.current_sequence[executor.current_sequence_index]
                            executor.current_sequence_index += 1
                    else:
                        # Continua sequenza corrente
                        action = executor.current_sequence[executor.current_sequence_index]
                        executor.current_sequence_index += 1
                
                except Exception as e:
                    print(f"[LLM] Error: {e}")
                    episode_hallucinations += 1
                    action = agent.act(state, env)
            
            else:
                # ===== DQN DIRETTA =====
                action = agent.act(state, env)
            
            # ===== ESEGUE AZIONE =====
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # ===== REWARD SHAPING =====
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, info, previous_info
            )
            
            total_native_reward += native_reward
            total_shaped_reward += shaped_reward
            total_reward += shaped_reward  
            
            # ===== UPDATE ACHIEVEMENTS =====
            if previous_info is not None:
                curr_achievements = set(
                    k for k, v in info.get('achievements', {}).items() if v >= 1
                )
                prev_achievements = set(
                    k for k, v in previous_info.get('achievements', {}).items() if v >= 1
                )
                newly_unlocked_names = curr_achievements - prev_achievements
                episode_achievements += len(newly_unlocked_names)
                
                # Tiene traccia dei nomi
                if newly_unlocked_names:
                    episode_achievements_list.extend(newly_unlocked_names)
                
                # Aggiunge a evaluation system
                if newly_unlocked_names:
                    newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names 
                                         if name in ACHIEVEMENT_NAME_TO_ID}
                    evaluation_system.add_episode_achievements(e, newly_unlocked_ids, moves)
            
            # ===== SALVA ESPERIENZA =====
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # ===== REPLAY =====
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            # ===== AGGIORNA STATO =====
            state = next_state
            previous_info = info
            moves += 1
        
        # ===== FINE EPISODIO =====
        shaped_bonus = total_shaped_reward - total_native_reward
        
        rewards_per_episode.append(total_shaped_reward)
        native_rewards_per_episode.append(total_native_reward)
        shaped_bonus_per_episode.append(shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(moves)
        helper_calls.append(episode_helper_calls)
        hallucinations.append(episode_hallucinations)
        
        # Calcolo % azioni valide
        if episode_helper_calls > 0:
            valid_actions = episode_helper_calls - episode_hallucinations
            valid_actions_percentage = (valid_actions / episode_helper_calls) * 100.0
        else:
            valid_actions_percentage = 0.0
        
        print(f"  Valid Actions Percentage: {valid_actions_percentage:.2f}% ({episode_helper_calls - episode_hallucinations}/{episode_helper_calls})")
        
        # Aggiungi a EvaluationSystem
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
        
        # Salva checkpoint se record
        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = e
            checkpoint_dir = Path(__file__).parent / 'dqn_helper_output' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"dqn_helper_best_ep{e}_ach{episode_achievements}"
            agent.save(str(checkpoint_path))
            print(f"\n[Checkpoint] New best model saved: {checkpoint_path}.*")
        
        # Checkpoint periodico (ogni 10)
        if (e + 1) % 10 == 0:
            checkpoint_dir = Path(__file__).parent / 'dqn_helper_output' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"dqn_helper_ep{e}"
            agent.save(str(checkpoint_path))
            print(f"[Checkpoint] Periodic checkpoint saved: {checkpoint_path}.*")
        
        print(f"\n[Episode {e}] Done! (DQN + Helper)")
        print(f"  Total Reward (Shaped): {total_shaped_reward:.2f}")
        print(f"  Native Reward: {total_native_reward:.2f}")
        print(f"  Shaped Bonus: {shaped_bonus:.2f}")
        print(f"  Achievements Unlocked: {episode_achievements}")
        print(f"  Moves: {moves}")
        print(f"  Helper Calls: {episode_helper_calls}, Hallucinations: {episode_hallucinations}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print(f"  LLM Active: {e < threshold_episodes} (Episode {e} < {threshold_episodes})")
        print(f"  Helper Stats: {helper.get_statistics()}")
        
        # Epsilon decay
        agent.decay_epsilon_linear(e, total_episodes=episodes)
    
    # ===== TRAINING COMPLETATO =====
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
    
    # Finalizza valutazione
    evaluation_system.finalize()
    
    # Salva tutto
    print("\n[DQN+Helper] Saving final model...")
    models_dir = Path(__file__).parent / 'dqn_helper_output' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "dqn_helper_crafter_final"
    agent.save(str(final_model_path))
    print(f"[DQN+Helper] ✓ Final model saved: {final_model_path}.*")
    
    checkpoint_dir = Path(__file__).parent / 'dqn_helper_output' / 'checkpoints'
    best_checkpoint = checkpoint_dir / f"dqn_helper_best_ep{best_episode}_ach{best_achievement_count}"
    print(f"[DQN+Helper] ✓ Best model saved: {best_checkpoint}.*")
    
    # Esporta metriche
    print("[DQN+Helper] Exporting metrics...")
    output_dir = Path(__file__).parent / 'dqn_helper_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dqn_helper_crafter_metrics.jsonl"
    evaluation_system.export_to_jsonl(str(jsonl_path))
    
    # Esporta statistiche achievement
    print("[DQN+Helper] Exporting per-achievement statistics...")
    achievement_stats_path = output_dir / "dqn_helper_achievement_statistics.json"
    export_achievement_statistics_json(evaluation_system, str(achievement_stats_path))
    
    # Grafici
    print("[DQN+Helper] Generating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    generate_all_plots(evaluation_system, output_dir=str(plot_dir))
    
    # Report finale
    print("\n[DQN+Helper] Final Evaluation Report:")
    evaluation_system.print_summary_report()
    
    # Export summary
    evaluation_json = output_dir / "dqn_helper_evaluation.json"
    evaluation_system.export_summary_json(str(evaluation_json))
    print(f"[DQN+Helper] ✓ Evaluation summary saved: {evaluation_json}")
    
    return (rewards_per_episode, native_rewards_per_episode, shaped_bonus_per_episode,
            achievements_per_episode, moves_per_episode, helper_calls, hallucinations,
            helper.get_statistics(), reward_shaper.get_statistics(), evaluation_system)


# ============================================================================
# Visualization and Export
# ============================================================================

def export_achievement_statistics_json(evaluation_system, output_file="dqn_helper_achievement_statistics.json"):
    """
    Esporta statistiche complete sugli achievement in JSON.
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    
    # Converte numpy array in liste
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
    
    # Aggiunge mapping nomi
    serializable_stats['achievement_id_to_name'] = ACHIEVEMENT_ID_TO_NAME
    
    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"[Export] Saved achievement statistics to: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN + Helper (No Reviewer) on Crafter - HERON_initial style")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN batch size")
    parser.add_argument("--episode-length", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--threshold-episodes", type=int, default=100, help="Episodes with LLM active")
    
    args = parser.parse_args()
    
    print("="*80)
    print("DQN + Helper Training (No Reviewer) - HERON_initial Style")
    print(f"Configuration: LLM active in first {ASSISTED_STEPS} steps of each episode")
    print("Two-Agent System: DQNAgent + CrafterHelper")
    print("="*80)
    
    # Via al training
    (rewards, native_rewards, shaped_bonus, achievements, moves, 
     helper_calls, hallucinations, helper_stats, reward_shaper_stats, eval_system) = train_dqn_helper(
        episodes=args.episodes,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        threshold_episodes=args.threshold_episodes
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
