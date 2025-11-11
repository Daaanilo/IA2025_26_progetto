"""
Training script for DQN NPC in Crafter
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.agents import DQNAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN NPC for Crafter')
    parser.add_argument('--config', type=str, default='configs/dqn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: cuda or cpu (overrides config)')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_dqn(config: dict, save_dir: str):
    """
    Train DQN agent.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save models
    """
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create environment
    print("Creating Crafter environment...")
    env = make_crafter_env(config.get('environment'))
    
    # Get observation shape
    obs, _ = env.reset()
    observation_shape = obs.shape if len(obs.shape) == 3 else (obs.shape[0], obs.shape[1], obs.shape[2])
    num_actions = env.action_space.n
    
    print(f"Observation shape: {observation_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create agent
    device = config.get('device', 'cuda')
    agent_config = config.get('agent', {})
    
    print(f"Creating DQN agent (device: {device})...")
    agent = DQNAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        config=agent_config,
        device=device
    )
    
    # Training configuration
    training_config = config.get('training', {})
    total_timesteps = training_config.get('total_timesteps', 5000000)
    eval_freq = training_config.get('eval_freq', 50000)
    eval_episodes = training_config.get('eval_episodes', 10)
    save_freq = training_config.get('save_freq', 100000)
    log_freq = training_config.get('log_freq', 1000)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    
    episode = 0
    episode_reward = 0
    episode_steps = 0
    episode_rewards = []
    
    obs, _ = env.reset()
    
    for step in range(total_timesteps):
        # Select action
        action = agent.select_action(obs)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Update agent
        loss = agent.update()
        
        # Update state
        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        agent.step()
        
        # Log progress
        if step % log_freq == 0 and loss is not None:
            print(f"Step: {step:,} | Episode: {episode} | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {loss:.4f} | "
                  f"Episode Reward: {episode_reward:.2f}")
        
        # Handle episode end
        if done:
            episode_rewards.append(episode_reward)
            
            # Print episode summary
            achievements = info.get('achievements', {})
            num_achievements = sum(1 for v in achievements.values() if v > 0)
            
            print(f"\n=== Episode {episode} Complete ===")
            print(f"Steps: {episode_steps}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Achievements: {num_achievements}/22")
            print("=" * 40 + "\n")
            
            # Reset episode
            obs, _ = env.reset()
            episode += 1
            episode_reward = 0
            episode_steps = 0
        
        # Evaluation
        if step % eval_freq == 0 and step > 0:
            print(f"\n{'='*50}")
            print(f"EVALUATION at step {step:,}")
            print(f"{'='*50}")
            eval_rewards = evaluate_agent(agent, env, eval_episodes)
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            print(f"Evaluation Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"{'='*50}\n")
        
        # Save model
        if step % save_freq == 0 and step > 0:
            model_path = os.path.join(save_dir, f'dqn_agent_step_{step}.pth')
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'dqn_agent_final.pth')
    agent.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    env.close()


def evaluate_agent(agent, env, num_episodes: int):
    """
    Evaluate agent performance.
    
    Args:
        agent: DQN agent
        env: Environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (no exploration)
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"  Eval Episode {ep+1}/{num_episodes}: {episode_reward:.2f}")
    
    return episode_rewards


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    
    # Train agent
    train_dqn(config, args.save_dir)


if __name__ == '__main__':
    main()
