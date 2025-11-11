"""
Step 5: Train Integrated HeRoN Architecture

Obiettivo: Addestrare il framework completo HeRoN con il ciclo iterativo RL-LLM.
Questo combina NPC (agente RL), Helper (LLM zero-shot), e Reviewer (LLM fine-tuned)
in un ciclo di apprendimento integrato.

Flusso HeRoN:
1. NPC percepisce lo stato dell'ambiente
2. Helper propone una sequenza di azioni (strategia)
3. Reviewer valuta e affina/corregge la strategia
4. La critica del Reviewer guida la politica del NPC
5. Il NPC impara attraverso l'esperienza e il feedback

Usage (PowerShell):
  conda activate ia2025
  python scripts/5_train_heron.py --config configs/heron_config.yaml --episodes 2000
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent
from src.llm import Helper, Reviewer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train integrated HeRoN architecture')
    parser.add_argument('--config', type=str, default='configs/heron_config.yaml',
                       help='Path to HeRoN configuration file')
    parser.add_argument('--dqn_config', type=str, default='configs/dqn_config.yaml',
                       help='Path to DQN configuration file')
    parser.add_argument('--helper_config', type=str, default='configs/helper_config.yaml',
                       help='Path to Helper configuration file')
    parser.add_argument('--reviewer_config', type=str, default='configs/reviewer_config.yaml',
                       help='Path to Reviewer configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--save_dir', type=str, default='models/heron',
                       help='Directory to save models')
    parser.add_argument('--load_npc', type=str, default=None,
                       help='Path to pre-trained NPC model (e.g., baseline)')
    parser.add_argument('--eval_freq', type=int, default=100,
                       help='Evaluate every N episodes')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Save model every N episodes')
    parser.add_argument('--log_interactions', action='store_true',
                       help='Log Helper-Reviewer interactions to file')
    return parser.parse_args()


class HeRoNTrainer:
    """Trainer for integrated HeRoN architecture."""
    
    def __init__(self, heron_config: dict, dqn_config: dict, 
                 helper_config: dict, reviewer_config: dict, save_dir: str):
        """
        Initialize HeRoN trainer.
        
        Args:
            heron_config: HeRoN architecture configuration
            dqn_config: DQN agent configuration
            helper_config: Helper LLM configuration
            reviewer_config: Reviewer LLM configuration
            save_dir: Directory to save models and logs
        """
        self.heron_config = heron_config
        self.dqn_config = dqn_config
        self.helper_config = helper_config
        self.reviewer_config = reviewer_config
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize environment
        print("Creating Crafter environment...")
        self.env = make_crafter_env(dqn_config.get('environment'))
        
        # Get observation info
        obs, _ = self.env.reset()
        self.observation_shape = obs.shape
        self.num_actions = self.env.action_space.n
        
        # Initialize components
        self._initialize_components()
        
        # HeRoN interaction configuration
        interaction_config = heron_config.get('heron', {}).get('interaction', {})
        self.helper_query_freq = interaction_config.get('helper_query_freq', 50)
        self.actions_per_query = interaction_config.get('actions_per_query', 5)
        self.use_reviewer = interaction_config.get('use_reviewer', True)
        self.acceptance_threshold = interaction_config.get('acceptance_threshold', 6.0)
        self.reward_shaping_weight = interaction_config.get('reward_shaping_weight', 0.3)
        
        # State tracking
        self.current_action_plan = []
        self.last_helper_query_step = 0
        self.action_history = []
        
        # Statistics
        self.helper_calls = 0
        self.reviewer_calls = 0
        self.plan_acceptances = 0
        self.plan_rejections = 0
        
    def _initialize_components(self):
        """Initialize NPC, Helper, and Reviewer."""
        print("\nInitializing HeRoN components...")
        
        # Initialize NPC (DQN Agent)
        print("  - NPC (DQN Agent)")
        self.npc = DQNAgent(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            config=self.dqn_config
        )
        
        # Initialize Helper
        print("  - Helper (LLM zero-shot)")
        self.helper = Helper(self.helper_config)
        
        # Initialize Reviewer
        print("  - Reviewer (LLM fine-tuned)")
        self.reviewer = Reviewer(
            self.reviewer_config,
            device=self.reviewer_config.get('device', 'cuda')
        )
        
        try:
            self.reviewer.load_model(use_fine_tuned=True)
            print("    ✓ Fine-tuned Reviewer loaded")
        except Exception as e:
            print(f"    Warning: Could not load fine-tuned Reviewer: {e}")
            print("    Using base Reviewer model")
            try:
                self.reviewer.load_model(use_fine_tuned=False)
            except Exception:
                print("    Warning: Reviewer will operate in mock mode")
    
    def get_helper_strategy(self, state_info: Dict, recent_actions: List[str]) -> List[str]:
        """
        Query Helper for action strategy.
        
        Args:
            state_info: Current state information
            recent_actions: Recent action history
            
        Returns:
            List of suggested action names
        """
        self.helper_calls += 1
        suggested_actions = self.helper.suggest_actions(
            state_info=state_info,
            recent_actions=recent_actions,
            num_actions=self.actions_per_query
        )
        return suggested_actions
    
    def review_strategy(self, state_info: Dict, suggested_actions: List[str]) -> Dict:
        """
        Review Helper's strategy with Reviewer.
        
        Args:
            state_info: Current state information
            suggested_actions: Actions suggested by Helper
            
        Returns:
            Review feedback dictionary
        """
        self.reviewer_calls += 1
        feedback = self.reviewer.review_actions(
            state_info=state_info,
            suggested_actions=suggested_actions
        )
        return feedback
    
    def select_action(self, obs, step: int, state_info: Dict) -> int:
        """
        Select action using HeRoN architecture.
        
        Args:
            obs: Current observation
            step: Current step in episode
            state_info: Current state information
            
        Returns:
            Action ID to execute
        """
        # Check if we need to query Helper for new plan
        if step == 0 or (step - self.last_helper_query_step >= self.helper_query_freq):
            # Get recent actions for context
            recent_actions = [self.env.action_id_to_name(a) for a in self.action_history[-10:]]
            
            # Get Helper's strategy
            suggested_actions = self.get_helper_strategy(state_info, recent_actions)
            
            # Review strategy if Reviewer is enabled
            if self.use_reviewer:
                review = self.review_strategy(state_info, suggested_actions)
                rating = review.get('rating', 5)
                
                # Accept or reject based on threshold
                if rating >= self.acceptance_threshold:
                    self.current_action_plan = review.get('improved_actions', suggested_actions)
                    self.plan_acceptances += 1
                else:
                    # Rejected - use NPC's own policy
                    self.current_action_plan = []
                    self.plan_rejections += 1
            else:
                self.current_action_plan = suggested_actions
            
            self.last_helper_query_step = step
        
        # Execute action from plan or use NPC policy
        if self.current_action_plan:
            # Follow Helper's plan
            action_name = self.current_action_plan.pop(0)
            action = self.env.action_name_to_id(action_name)
        else:
            # Use NPC's learned policy
            action = self.npc.select_action(obs)
        
        return action
    
    def compute_reward_shaping(self, review_feedback: Dict) -> float:
        """
        Compute additional reward based on Reviewer feedback.
        
        Args:
            review_feedback: Feedback from Reviewer
            
        Returns:
            Shaped reward component
        """
        rating = review_feedback.get('rating', 5)
        # Normalize rating (1-10) to reward (-1 to +1)
        shaped_reward = (rating - 5.5) / 4.5
        return shaped_reward * self.reward_shaping_weight
    
    def train_episode(self, episode: int, max_steps: int, log_interactions: bool = False) -> Dict:
        """
        Train one episode with HeRoN architecture.
        
        Args:
            episode: Episode number
            max_steps: Maximum steps per episode
            log_interactions: Whether to log interactions
            
        Returns:
            Dictionary with episode metrics
        """
        obs, _ = self.env.reset()
        step = 0
        episode_reward = 0
        self.action_history = []
        self.current_action_plan = []
        
        interactions_log = []
        
        while step < max_steps:
            state_info = self.env.get_state_description()
            state_info['step'] = step
            
            # Select action using HeRoN
            action = self.select_action(obs, step, state_info)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.npc.store_transition(obs, action, reward, next_obs, done)
            
            # Train NPC
            if self.npc.can_train():
                loss = self.npc.train_step()
            
            # Track action
            self.action_history.append(action)
            
            # Update
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Update target network
        self.npc.update_target_network()
        
        # Get achievements
        achievements = info.get('achievements', {})
        num_achievements = sum(1 for v in achievements.values() if v > 0)
        
        return {
            'episode': episode,
            'reward': episode_reward,
            'length': step,
            'achievements': num_achievements,
            'epsilon': self.npc.epsilon,
            'helper_calls': self.helper_calls,
            'reviewer_calls': self.reviewer_calls,
            'plan_acceptances': self.plan_acceptances,
            'plan_rejections': self.plan_rejections
        }


def evaluate_heron(trainer: HeRoNTrainer, num_episodes: int = 10) -> Dict:
    """
    Evaluate HeRoN performance.
    
    Args:
        trainer: HeRoN trainer instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    all_achievements = []
    
    # Temporarily disable exploration
    old_epsilon = trainer.npc.epsilon
    trainer.npc.epsilon = 0.0
    
    for episode in range(num_episodes):
        obs, _ = trainer.env.reset()
        step = 0
        episode_reward = 0
        trainer.action_history = []
        trainer.current_action_plan = []
        
        while step < 10000:
            state_info = trainer.env.get_state_description()
            state_info['step'] = step
            
            action = trainer.select_action(obs, step, state_info)
            obs, reward, terminated, truncated, info = trainer.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        achievements = info.get('achievements', {})
        all_achievements.append(achievements.copy())
    
    # Restore exploration
    trainer.npc.epsilon = old_epsilon
    
    # Calculate statistics
    achievement_counts = {}
    for achievement in all_achievements[0].keys():
        counts = [ach[achievement] for ach in all_achievements]
        achievement_counts[achievement] = sum(1 for c in counts if c > 0)
    
    num_unlocked = sum(1 for v in achievement_counts.values() if v > 0)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'achievements_unlocked': num_unlocked,
        'achievement_details': achievement_counts
    }


def main():
    """Main training loop."""
    args = parse_args()
    
    # Load configurations
    print("Loading configurations...")
    with open(args.config, 'r') as f:
        heron_config = yaml.safe_load(f)
    with open(args.dqn_config, 'r') as f:
        dqn_config = yaml.safe_load(f)
    with open(args.helper_config, 'r') as f:
        helper_config = yaml.safe_load(f)
    with open(args.reviewer_config, 'r') as f:
        reviewer_config = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("STEP 5: TRAIN INTEGRATED HERON ARCHITECTURE")
    print("="*60)
    
    # Initialize trainer
    trainer = HeRoNTrainer(
        heron_config=heron_config,
        dqn_config=dqn_config,
        helper_config=helper_config,
        reviewer_config=reviewer_config,
        save_dir=args.save_dir
    )
    
    # Load pre-trained NPC if provided
    if args.load_npc:
        print(f"\nLoading pre-trained NPC from: {args.load_npc}")
        trainer.npc.load(args.load_npc)
        print("✓ NPC loaded successfully")
    
    # Training parameters
    num_episodes = args.episodes or heron_config['training']['num_episodes']
    max_steps = heron_config['training'].get('max_steps_per_episode', 10000)
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Helper query frequency: {trainer.helper_query_freq}")
    print(f"  Actions per query: {trainer.actions_per_query}")
    print(f"  Use Reviewer: {trainer.use_reviewer}")
    print(f"  Acceptance threshold: {trainer.acceptance_threshold}")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    training_log = []
    best_mean_reward = -float('inf')
    
    for episode in range(num_episodes):
        metrics = trainer.train_episode(
            episode=episode,
            max_steps=max_steps,
            log_interactions=args.log_interactions
        )
        
        training_log.append(metrics)
        
        # Logging
        if (episode + 1) % 10 == 0:
            recent_rewards = [m['reward'] for m in training_log[-10:]]
            mean_recent = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {metrics['reward']:.2f} | "
                  f"Mean(10): {mean_recent:.2f} | "
                  f"Achievements: {metrics['achievements']}/22 | "
                  f"ε: {metrics['epsilon']:.3f} | "
                  f"Accept/Reject: {trainer.plan_acceptances}/{trainer.plan_rejections}")
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            print(f"\n--- Evaluation at episode {episode + 1} ---")
            eval_metrics = evaluate_heron(trainer, num_episodes=10)
            print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"Achievements Unlocked: {eval_metrics['achievements_unlocked']}/22")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_mean_reward:
                best_mean_reward = eval_metrics['mean_reward']
                best_path = os.path.join(args.save_dir, 'heron_best.pth')
                trainer.npc.save(best_path)
                print(f"✓ New best model saved: {best_path}")
            print()
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'heron_ep{episode+1}.pth')
            trainer.npc.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_metrics = evaluate_heron(trainer, num_episodes=100)
    print(f"\nFinal Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"Final Achievements Unlocked: {final_metrics['achievements_unlocked']}/22")
    print("\nAchievement Details:")
    for ach, count in final_metrics['achievement_details'].items():
        if count > 0:
            print(f"  {ach}: {count}/100 episodes")
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'heron_final.pth')
    trainer.npc.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training log
    log_path = os.path.join(args.save_dir, 'heron_training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'training_log': training_log,
            'final_metrics': final_metrics,
            'config': heron_config
        }, f, indent=2)
    print(f"Training log saved: {log_path}")
    
    print("\n" + "="*60)
    print("HERON TRAINING COMPLETED")
    print("="*60)
    print("\nNext step: Run script 6 to evaluate and compare results")


if __name__ == '__main__':
    main()
