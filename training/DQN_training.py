import numpy as np
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.crafter_environment import CrafterEnv
from classes.agent import DQNAgent
from training.reward_shaper import CrafterRewardShaper
from evaluation.evaluation_system import EvaluationSystem, ACHIEVEMENT_NAME_TO_ID
from evaluation.evaluation_plots import generate_all_plots
from evaluation.achievement_learning_curves import AchievementLearningCurvePlotter

# Achievement name-to-ID mapping (needed for export function)
ACHIEVEMENT_ID_TO_NAME = {v: k for k, v in ACHIEVEMENT_NAME_TO_ID.items()}


def export_achievement_statistics_json(evaluation_system, output_file="crafter_achievement_statistics.json"):
    """
    Salva le statistiche degli achievement in un JSON bello grosso.
    Serve per vedere chi ha sbloccato cosa e quando.
    """
    achievement_stats = evaluation_system.get_achievement_statistics()
    
    # Converto numpy in liste altrimenti json esplode
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
    
    # Aggiungo mappa nomi
    serializable_stats['achievement_id_to_name'] = ACHIEVEMENT_ID_TO_NAME
    
    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"[Export] Saved achievement statistics to: {output_file}")


def train_dqn_baseline(episodes=300, batch_size=64, episode_length=1000, load_model_path=None):
    """
    Addestra una DQN classica su Crafter.
    """
    
    print("[Baseline DQN] Initializing Crafter environment...")
    env = CrafterEnv(length=episode_length)
    
    print(f"[Baseline DQN] State size: {env.state_size}, Action size: {env.action_size}")
    
    # Inizializza l'agente
    agent = DQNAgent(env.state_size, env.action_size, epsilon=1.0, load_model_path=load_model_path)
    print(f"[Baseline DQN] DQN Agent initialized (epsilon={agent.epsilon:.4f})")
    
    # Sistema di valutazione
    evaluation_system = EvaluationSystem()
    
    # Reward Shaper per aiutare l'agente
    reward_shaper = CrafterRewardShaper()
    
    # Tracking delle metriche
    rewards_per_episode = []
    native_rewards_per_episode = []
    shaped_bonus_per_episode = []
    achievements_per_episode = []
    moves_per_episode = []
    
    # Per salvare il modello migliore
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
        # Tengo traccia degli achievement per mapparli
        previous_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
        previous_info = info.copy()  # Per reward shaping
        reward_shaper.reset_episode()
        
        for step in range(episode_length):
            # La DQN sceglie l'azione
            action = agent.act(state, env)
            
            # Esegue l'azione
            next_state, native_reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # Achievement sbloccati in questo step
            current_achievements_set = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
            newly_unlocked_names = current_achievements_set - previous_achievements_set
            achievements_this_step = len(newly_unlocked_names)
            
            # Aggiunge ID al sistema di valutazione
            if newly_unlocked_names:
                newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names 
                                     if name in ACHIEVEMENT_NAME_TO_ID}
                evaluation_system.add_episode_achievements(episode, newly_unlocked_ids, step)
            
            previous_achievements_set = current_achievements_set
            
            # Calcola reward shapato
            shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
                native_reward, info, previous_info
            )
            shaped_bonus = sum(bonus_components.values())
            
            # Salva in memoria
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # Allena la rete se ha abbastanza dati
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)
            
            # Aggiorna metriche
            episode_reward += shaped_reward
            episode_native_reward += native_reward
            episode_shaped_bonus += shaped_bonus
            episode_achievements += achievements_this_step
            episode_moves += 1
            
            previous_info = info.copy()
            state = next_state
            
            if done:
                break
        
        # Salva metriche episodio
        rewards_per_episode.append(episode_reward)
        native_rewards_per_episode.append(episode_native_reward)
        shaped_bonus_per_episode.append(episode_shaped_bonus)
        achievements_per_episode.append(episode_achievements)
        moves_per_episode.append(episode_moves)
        
        # Niente LLM qui
        valid_actions_percentage = 0.0
        
        # Aggiunge a evaluation system
        evaluation_system.add_episode(
            episode=episode,
            shaped_reward=episode_reward,
            native_reward=episode_native_reward,
            shaped_bonus=episode_shaped_bonus,
            achievements_unlocked=episode_achievements,
            moves=episode_moves,
            helper_calls=0,
            hallucinations=0,
            valid_actions_percentage=valid_actions_percentage
        )
        
        # Salva checkpoint se è il migliore
        if episode_achievements > best_achievement_count:
            best_achievement_count = episode_achievements
            best_episode = episode
            checkpoint_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"nuovo_dqn_best_ep{episode}_ach{episode_achievements}"
            agent.save(str(checkpoint_path))
            print(f"[Baseline DQN] ✓ New best model saved: {episode_achievements} achievements (episode {episode})")
        
        # Epsilon decay (lineare su 300 ep)
        agent.decay_epsilon_linear(episode, total_episodes=episodes)

        # Log
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
    
    # Salva il modello finale
    print("[Baseline DQN] Saving final model...")
    models_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "nuovo_crafter_dqn_final"
    agent.save(str(final_model_path))
    print(f"[Baseline DQN] ✓ Final model saved: {final_model_path}.*")
    
    checkpoint_dir = Path(__file__).parent / 'DQN_nuovo_training' / 'checkpoints'
    best_checkpoint = checkpoint_dir / f"nuovo_dqn_best_ep{best_episode}_ach{best_achievement_count}"
    print(f"[Baseline DQN] ✓ Best model saved: {best_checkpoint}.*")
    
    # Esporta metriche
    print("[Baseline DQN] Exporting metrics...")
    output_dir = Path(__file__).parent / 'DQN_nuovo_training'
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "nuovo_crafter_dqn_metrics.jsonl"
    evaluation_system.export_to_jsonl(str(jsonl_path))
    
    # Esporta statistiche achievement (serve per i grafici)
    print("[Baseline DQN] Exporting per-achievement statistics...")
    achievement_stats_path = output_dir / "nuovo_crafter_dqn_achievement_statistics.json"
    export_achievement_statistics_json(evaluation_system, str(achievement_stats_path))
    
    # Genera i grafici
    print("[Baseline DQN] Generating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    generate_all_plots(evaluation_system, output_dir=str(plot_dir))
    
    # Curve di apprendimento degli achievement
    print("[Baseline DQN] Generating achievement learning curves...")
    curves_dir = plot_dir / "achievement_curves"
    curves_dir.mkdir(exist_ok=True)
    
    try:
        # Uso AchievementLearningCurvePlotter (lo stesso di HeRoN)
        plotter = AchievementLearningCurvePlotter(str(achievement_stats_path))
        plotter.plot_all_achievements(
            output_dir=str(curves_dir),
            window=10,
            use_steps=False,
            only_unlocked=True
        )
        print(f"[Baseline DQN] ✓ Achievement learning curves generated in {curves_dir}")
    except Exception as e:
        print(f"[Baseline DQN] ⚠ Warning: Could not generate achievement curves: {e}")
        print("[Baseline DQN]   This is normal if no achievements were unlocked during training.")
    
    # Report finale
    print("\n[Baseline DQN] Final Evaluation Report:")
    try:
        evaluation_system.print_summary_report()
    except (KeyError, IndexError) as e:
        print(f"[Baseline DQN] ⚠ Note: Could not generate full summary report ({e})")
        print("[Baseline DQN]   All metrics have been saved to JSON and JSONL files.")
    
    # Esporta riassunto valutazione
    evaluation_json = output_dir / "nuovo_crafter_dqn_evaluation.json"
    evaluation_system.export_summary_json(str(evaluation_json))
    
    return evaluation_system


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN baseline on Crafter environment")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN batch size")
    parser.add_argument("--episode-length", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pre-trained model")
    
    args = parser.parse_args()
    
    # Via col training
    eval_system = train_dqn_baseline(
        episodes=args.episodes,
        batch_size=args.batch_size,
        episode_length=args.episode_length,
        load_model_path=args.load_model
    )
