"""
Utility functions for HeRoN project
"""

import os
import yaml
import json
import torch
import numpy as np
from typing import Dict, Any, List
from .constants import NUM_ACHIEVEMENTS
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_trajectory(trajectory: List[Dict[str, Any]], save_path: str):
    """
    Save episode trajectory to JSON file.

    Args:
        trajectory: List of state-action-reward tuples
        save_path: Path to save JSON file
    """
    with open(save_path, "w") as f:
        json.dump(trajectory, f, indent=2)


def load_trajectory(load_path: str) -> List[Dict[str, Any]]:
    """
    Load trajectory from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        Trajectory data
    """
    with open(load_path, "r") as f:
        return json.load(f)


def calculate_achievement_score(achievements: Dict[str, int]) -> float:
    """
    Calculate a normalized achievement score (0-100).

    Args:
        achievements: Dictionary of achievement counts

    Returns:
        Normalized score
    """
    # Simple scoring: 1 point per achievement unlocked
    num_unlocked = sum(1 for count in achievements.values() if count > 0)
    max_achievements = NUM_ACHIEVEMENTS
    return (num_unlocked / max_achievements) * 100


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    Useful for normalizing rewards or observations.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Initialize running statistics.

        Args:
            epsilon: Small value to avoid division by zero
            shape: Shape of the values to track
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update statistics with new batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def print_system_info():
    """Print system and environment information."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    # Test utilities
    print_system_info()
