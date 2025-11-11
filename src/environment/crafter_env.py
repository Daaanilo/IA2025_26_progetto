"""
Crafter Environment Wrapper for HeRoN Architecture
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
import crafter


class CrafterWrapper:
    """
    Wrapper for Crafter environment to integrate with HeRoN architecture.
    Provides additional state information and achievement tracking.
    """
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Crafter environment wrapper.
        
        Args:
            env_config: Configuration dictionary for the environment
        """
        # Create base Crafter environment
        self.env = crafter.Env()

        # Expose common attributes expected from gym environments
        self.action_space = getattr(self.env, 'action_space', None)
        self.observation_space = getattr(self.env, 'observation_space', None)

        self.env_config = env_config or {}
        self.max_episode_steps = self.env_config.get('max_episode_steps', 10000)
        
        # Action space information
        self.action_names = [
            'noop', 'move_left', 'move_right', 'move_up', 'move_down',
            'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
            'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe',
            'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword',
            'make_iron_sword'
        ]
        
        # Achievement tracking
        self.achievements = {
            'collect_coal': 0, 'collect_diamond': 0, 'collect_drink': 0,
            'collect_iron': 0, 'collect_sapling': 0, 'collect_stone': 0,
            'collect_wood': 0, 'defeat_skeleton': 0, 'defeat_zombie': 0,
            'eat_cow': 0, 'eat_plant': 0, 'make_iron_pickaxe': 0,
            'make_iron_sword': 0, 'make_stone_pickaxe': 0, 'make_stone_sword': 0,
            'make_wood_pickaxe': 0, 'make_wood_sword': 0, 'place_furnace': 0,
            'place_plant': 0, 'place_stone': 0, 'place_table': 0, 'wake_up': 0
        }
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.action_history = []
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and tracking variables."""
        # Crafter.Env.reset may return observation only or (obs, info)
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs = res
            info = {}
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.action_history = []
        self.achievements = {k: 0 for k in self.achievements}
        
        # Add additional info
        info['achievements'] = self.achievements.copy()
        info['step'] = self.current_step
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Crafter.Env.step may follow gym signature or a simpler (obs, reward, done, info)
        res = self.env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
        elif isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("Unexpected return signature from crafter.Env.step")
        
        # Update tracking
        self.current_step += 1
        self.episode_reward += reward
        self.action_history.append(action)
        
        # Update achievements if present in info
        if 'achievements' in info:
            for achievement, count in info['achievements'].items():
                if achievement in self.achievements:
                    self.achievements[achievement] = count
        
        # Add additional info
        info['step'] = self.current_step
        info['episode_reward'] = self.episode_reward
        info['action_name'] = self.action_names[action]
        info['achievements'] = self.achievements.copy()
        
        # Check for max steps
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def get_state_description(self) -> Dict[str, Any]:
        """
        Get a human-readable description of the current state.
        Useful for LLM Helper and Reviewer.
        
        Returns:
            Dictionary with state information
        """
        # Note: Crafter doesn't expose internal state directly
        # We track what we can from observations and info
        state_desc = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'achievements': self.achievements.copy(),
            'recent_actions': [
                self.action_names[a] 
                for a in self.action_history[-10:]
            ] if self.action_history else []
        }
        
        return state_desc
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get a mask of valid actions in the current state.
        
        Returns:
            Binary mask where 1 indicates a valid action
        """
        # In Crafter, all actions are always valid
        # But some might not be useful (e.g., crafting without resources)
        return np.ones(self.action_space.n, dtype=np.float32)
    
    def action_name_to_id(self, action_name: str) -> int:
        """Convert action name to action ID."""
        try:
            return self.action_names.index(action_name)
        except ValueError:
            raise ValueError(f"Unknown action: {action_name}")
    
    def action_id_to_name(self, action_id: int) -> str:
        """Convert action ID to action name."""
        return self.action_names[action_id]

    def close(self) -> None:
        """
        Safely close the wrapped environment if it provides a close method.
        Some versions of the Crafter environment do not implement "close";
        calling this method will not raise in that case.
        """
        try:
            if hasattr(self.env, "close") and callable(self.env.close):
                try:
                    self.env.close()
                except Exception:
                    # ignore errors from underlying env close
                    pass
        except Exception:
            pass
        # No parent to call — done
        return


def make_crafter_env(config: Optional[Dict[str, Any]] = None) -> CrafterWrapper:
    """
    Factory function to create a Crafter environment.
    
    Args:
        config: Environment configuration
        
    Returns:
        Wrapped Crafter environment
    """
    return CrafterWrapper(env_config=config)
