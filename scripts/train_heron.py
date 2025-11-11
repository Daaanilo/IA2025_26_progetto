"""
Integrated HeRoN training script - combines NPC, Helper, and Reviewer
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent
from src.llm import Helper, Reviewer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train HeRoN in Crafter')
    parser.add_argument('--config', type=str, default='configs/heron_config.yaml',
                       help='Path to HeRoN configuration file')
    parser.add_argument('--dqn_config', type=str, default='configs/dqn_config.yaml',
                       help='Path to DQN configuration file')
    parser.add_argument('--helper_config', type=str, default='configs/helper_config.yaml',
                       help='Path to Helper configuration file')
    parser.add_argument('--reviewer_config', type=str, default='configs/reviewer_config.yaml',
                       help='Path to Reviewer configuration file')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--load_npc', type=str, default=None,
                       help='Path to pre-trained NPC model')
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
        
        # Configuration
        interaction_config = heron_config.get('heron', {}).get('interaction', {})
        self.helper_query_freq = interaction_config.get('helper_query_freq', 50)
        self.actions_per_query = interaction_config.get('actions_per_query', 5)
        self.use_reviewer = interaction_config.get('use_reviewer', True)
        self.acceptance_threshold = interaction_config.get('acceptance_threshold', 6.0)
        
        # State tracking
        self.current_action_plan = []
        self.last_helper_query_step = 0
        self.action_history = []
        
    def _initialize_components(self):
        """Initialize NPC, Helper, and Reviewer."""
        device = self.heron_config.get('general', {}).get('device', 'cuda')
        
        # Initialize NPC (DQN Agent)
        print("Initializing NPC (DQN Agent)...")
        self.npc = DQNAgent(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            config=self.dqn_config.get('agent', {}),
            device=device
        )
        
        # Initialize Helper
        components = self.heron_config.get('heron', {}).get('components', {})
        if components.get('helper_enabled', True):
            print("Initializing Helper LLM...")
            try:
                self.helper = Helper(self.helper_config)
                print("Helper initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Helper: {e}")
                self.helper = None
        else:
            self.helper = None
        
        # Initialize Reviewer
        if components.get('reviewer_enabled', True) and self.use_reviewer:
            print("Initializing Reviewer LLM...")
            try:
                self.reviewer = Reviewer(self.reviewer_config, device=device)
                # Try to load fine-tuned model (will use base model if not available)
                self.reviewer.load_model(use_fine_tuned=False)
                print("Reviewer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Reviewer: {e}")
                self.reviewer = None
        else:
            self.reviewer = None
    
    def should_query_helper(self, step: int) -> bool:
        """Determine if Helper should be queried."""
        if self.helper is None:
            return False
        
        # Query when action plan is exhausted or at regular intervals
        return (len(self.current_action_plan) == 0 and 
                step - self.last_helper_query_step >= self.helper_query_freq)
    
    def get_action(self, obs: np.ndarray, step: int, state_info: Dict[str, Any]) -> int:
        """
        Get next action using HeRoN architecture.
        
        Args:
            obs: Current observation
            step: Current step number
            state_info: Current state information
            
        Returns:
            Action to take
        """
        # Check if we should query Helper
        if self.should_query_helper(step):
            self._query_helper_for_actions(state_info)
        
        # Use Helper's action plan if available
        if len(self.current_action_plan) > 0:
            action_name = self.current_action_plan.pop(0)
            try:
                action = self.env.action_name_to_id(action_name)
                return action
            except ValueError:
                print(f"Warning: Invalid action '{action_name}' from Helper")
        
        # Fallback to NPC policy
        return self.npc.select_action(obs)
    
    def _query_helper_for_actions(self, state_info: Dict[str, Any]):
        """Query Helper for action suggestions."""
        print("\n--- Querying Helper for actions ---")
        
        # Get recent action names
        recent_actions = [
            self.env.action_id_to_name(a) for a in self.action_history[-10:]
        ] if self.action_history else []
        
        # Get suggestions from Helper
        suggested_actions = self.helper.suggest_actions(
            state_info=state_info,
            recent_actions=recent_actions,
            num_actions=self.actions_per_query
        )
        
        print(f"Helper suggested: {suggested_actions}")
        
        # Review actions if Reviewer is available
        if self.reviewer is not None and self.use_reviewer:
            print("--- Reviewer evaluating suggestions ---")
            feedback = self.reviewer.review_actions(state_info, suggested_actions)
            rating = feedback['rating']
            print(f"Reviewer rating: {rating}/10")
            print(f"Feedback: {feedback['reasoning'][:100]}...")
            
            # Use improved actions if rating is low
            if rating < self.acceptance_threshold and feedback['improved_actions']:
                print("Using Reviewer's improved actions")
                self.current_action_plan = feedback['improved_actions']
            else:
                self.current_action_plan = suggested_actions
        else:
            self.current_action_plan = suggested_actions
        
        self.last_helper_query_step = state_info.get('step', 0)
        print(f"Action plan set: {self.current_action_plan}\n")
    
    def train(self, num_episodes: int = 1000):
        """
        Train HeRoN architecture.
        
        Args:
            num_episodes: Number of training episodes
        """
        print(f"\n{'='*60}")
        print(f"Starting HeRoN Training for {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Reset action plan
            self.current_action_plan = []
            self.action_history = []
            
            while not done:
                # Get state information
                state_info = self.env.get_state_description()
                state_info['step'] = episode_steps
                
                # Get action
                action = self.get_action(obs, episode_steps, state_info)
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition in NPC
                self.npc.store_transition(obs, action, reward, next_obs, done)
                
                # Update NPC
                loss = self.npc.update()
                
                # Update tracking
                self.action_history.append(action)
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                self.npc.step()
            
            # Episode summary
            achievements = info.get('achievements', {})
            num_achievements = sum(1 for v in achievements.values() if v > 0)
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes} Complete")
            print(f"Steps: {episode_steps}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Achievements: {num_achievements}/22")
            print(f"Epsilon: {self.npc.epsilon:.3f}")
            
            if self.helper:
                helper_stats = self.helper.get_statistics()
                print(f"Helper Calls: {helper_stats['total_calls']}")
            
            if self.reviewer:
                reviewer_stats = self.reviewer.get_statistics()
                print(f"Reviewer Avg Rating: {reviewer_stats['average_rating']}")
            
            print(f"{'='*60}\n")
            
            # Save checkpoint periodically
            if (episode + 1) % 100 == 0:
                model_path = os.path.join(self.save_dir, f'heron_npc_episode_{episode+1}.pth')
                self.npc.save(model_path)
                print(f"Checkpoint saved to {model_path}")
        
        # Save final model
        final_path = os.path.join(self.save_dir, 'heron_npc_final.pth')
        self.npc.save(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")


def main():
    """Main function."""
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
    
    # Create trainer
    trainer = HeRoNTrainer(
        heron_config=heron_config,
        dqn_config=dqn_config,
        helper_config=helper_config,
        reviewer_config=reviewer_config,
        save_dir=args.save_dir
    )
    
    # Load pre-trained NPC if specified
    if args.load_npc:
        print(f"Loading pre-trained NPC from {args.load_npc}")
        trainer.npc.load(args.load_npc)
    
    # Train
    num_episodes = heron_config.get('training', {}).get('num_episodes', 1000)
    trainer.train(num_episodes=num_episodes)


if __name__ == '__main__':
    main()
