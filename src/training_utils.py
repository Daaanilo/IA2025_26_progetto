"""
Logger and Checkpoint Manager for HeRoN Training
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class TrainingLogger:
    """Logs training metrics and progress."""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"

        self.metrics = []
        self.start_time = datetime.now()

    def log(self, metrics: Dict[str, Any], step: int = None) -> None:
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metrics
            step: Training step (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": (datetime.now() - self.start_time).total_seconds(),
            "step": step,
            **metrics,
        }

        self.metrics.append(log_entry)

        # Append to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_episode(
        self, episode: int, reward: float, length: int, achievements: int, **kwargs
    ) -> None:
        """
        Log episode metrics.

        Args:
            episode: Episode number
            reward: Episode reward
            length: Episode length
            achievements: Number of achievements unlocked
            **kwargs: Additional metrics
        """
        metrics = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "achievements": achievements,
            **kwargs,
        }
        self.log(metrics, step=episode)

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all logged metrics."""
        return self.metrics

    def get_latest_metrics(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last n metrics."""
        return self.metrics[-n:]

    def compute_statistics(
        self, metric_name: str, window: int = 100
    ) -> Dict[str, float]:
        """
        Compute statistics for a metric over a window.

        Args:
            metric_name: Name of the metric
            window: Window size

        Returns:
            Dictionary with mean, std, min, max
        """
        values = [m[metric_name] for m in self.metrics[-window:] if metric_name in m]

        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    def save_summary(self, additional_info: Dict[str, Any] = None) -> None:
        """
        Save training summary.

        Args:
            additional_info: Additional information to include
        """
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_episodes": len(self.metrics),
            "total_time": (datetime.now() - self.start_time).total_seconds(),
        }

        # Add metric statistics
        if self.metrics:
            for metric_name in ["reward", "length", "achievements"]:
                if any(metric_name in m for m in self.metrics):
                    summary[f"{metric_name}_stats"] = self.compute_statistics(
                        metric_name, len(self.metrics)
                    )

        if additional_info:
            summary.update(additional_info)

        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved to: {summary_file}")


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

        self.checkpoints = []
        self.best_metric = float("-inf")
        self.best_checkpoint = None

    def save_checkpoint(
        self, agent, episode: int, metrics: Dict[str, Any], is_best: bool = False
    ) -> str:
        """
        Save a checkpoint.

        Args:
            agent: Agent to save
            episode: Current episode
            metrics: Current metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_ep{episode}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint (compatible with DQNAgent interface)
        # Expect agent to expose: policy_net, target_net, optimizer, epsilon
        checkpoint = {
            "episode": episode,
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "total_steps": getattr(agent, "total_steps", 0),
            "training_steps": getattr(agent, "training_steps", 0),
            "epsilon": getattr(agent, "epsilon", None),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self.checkpoints.append(
            {
                "path": checkpoint_path,
                "episode": episode,
                "metric": metrics.get("reward", 0),
            }
        )

        # Save best separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = str(best_path)
            print(f"  💾 Saved best model: {best_path}")

        # Remove old checkpoints
        self._cleanup_checkpoints()

        return str(checkpoint_path)

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by episode and remove oldest
            self.checkpoints.sort(key=lambda x: x["episode"])
            to_remove = self.checkpoints[: -self.max_checkpoints]

            for checkpoint in to_remove:
                try:
                    os.remove(checkpoint["path"])
                except Exception as e:
                    print(f"[CheckpointCleanup] Warning: could not remove {checkpoint['path']}: {e}")

            self.checkpoints = self.checkpoints[-self.max_checkpoints :]

    def load_checkpoint(self, agent, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            agent: Agent to load into
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=getattr(agent, "device", None)
        )

        # Load into expected DQNAgent attributes if present
        if hasattr(agent, "policy_net") and "policy_state_dict" in checkpoint:
            agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        elif hasattr(agent, "q_network") and "agent_state_dict" in checkpoint:
            # legacy support
            agent.q_network.load_state_dict(checkpoint["agent_state_dict"])

        if hasattr(agent, "target_net") and "target_state_dict" in checkpoint:
            agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        elif hasattr(agent, "target_network") and "target_state_dict" in checkpoint:
            agent.target_network.load_state_dict(checkpoint["target_state_dict"])

        if hasattr(agent, "optimizer") and "optimizer_state_dict" in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                # optimizer state may be incompatible across device types
                pass

        # Restore counters if present
        agent.total_steps = checkpoint.get(
            "total_steps", getattr(agent, "total_steps", 0)
        )
        agent.training_steps = checkpoint.get(
            "training_steps", getattr(agent, "training_steps", 0)
        )
        if "epsilon" in checkpoint:
            agent.epsilon = checkpoint["epsilon"]

        print(f"Loaded checkpoint from episode {checkpoint.get('episode', '?')}")

        return checkpoint.get("metrics", {})

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        return self.best_checkpoint

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return self.checkpoints


class MetricsTracker:
    """Track and analyze training metrics over time."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            "rewards": [],
            "lengths": [],
            "achievements": [],
            "losses": [],
            "epsilon": [],
        }

    def update(
        self,
        reward: float = None,
        length: int = None,
        achievements: int = None,
        loss: float = None,
        epsilon: float = None,
    ) -> None:
        """
        Update metrics.

        Args:
            reward: Episode reward
            length: Episode length
            achievements: Number of achievements
            loss: Training loss
            epsilon: Exploration rate
        """
        if reward is not None:
            self.history["rewards"].append(reward)
        if length is not None:
            self.history["lengths"].append(length)
        if achievements is not None:
            self.history["achievements"].append(achievements)
        if loss is not None:
            self.history["losses"].append(loss)
        if epsilon is not None:
            self.history["epsilon"].append(epsilon)

    def get_moving_average(self, metric: str, window: int = 100) -> List[float]:
        """
        Get moving average of a metric.

        Args:
            metric: Metric name
            window: Window size

        Returns:
            List of moving averages
        """
        values = self.history.get(metric, [])
        if not values or len(values) < window:
            return values

        moving_avg = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(values[start : i + 1]))

        return moving_avg

    def get_statistics(self, metric: str, window: int = None) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            metric: Metric name
            window: Window size (None for all history)

        Returns:
            Dictionary with statistics
        """
        values = self.history.get(metric, [])
        if window:
            values = values[-window:]

        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values),
        }

    def is_improving(
        self, metric: str, window: int = 100, threshold: float = 0.05
    ) -> bool:
        """
        Check if a metric is improving.

        Args:
            metric: Metric name
            window: Window size for comparison
            threshold: Minimum improvement threshold

        Returns:
            True if improving
        """
        values = self.history.get(metric, [])
        if len(values) < window * 2:
            return False

        recent_mean = np.mean(values[-window:])
        previous_mean = np.mean(values[-2 * window : -window])

        improvement = (recent_mean - previous_mean) / abs(previous_mean)
        return improvement > threshold

    def export_history(self, filepath: str) -> None:
        """
        Export metrics history to file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def import_history(self, filepath: str) -> None:
        """
        Import metrics history from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, "r") as f:
            self.history = json.load(f)
