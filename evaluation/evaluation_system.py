"""
F10: Sistema di Valutazione - Core Evaluation System

Comprehensive evaluation framework for HeRoN Crafter training:
- Aggregates per-episode metrics into learning curves
- Computes per-achievement unlock statistics
- Calculates efficiency ratios (reward/move, reward/helper_call)
- Detects convergence patterns and learning trends
- Provides statistical summaries (mean, std, min, max, moving averages)
- Enables baseline comparisons
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class EpisodeMetrics:
    """Per-episode metrics snapshot."""
    episode: int
    shaped_reward: float
    native_reward: float
    shaped_bonus: float
    achievements_unlocked: int
    moves: int
    helper_calls: int
    hallucinations: int
    hallucination_rate: float
    
    # Optional extended metrics (populated by EvaluationSystem)
    reward_per_move: Optional[float] = None
    reward_per_helper_call: Optional[float] = None
    achievements_per_move: Optional[float] = None


@dataclass
class EpisodeAchievementData:
    """Achievement unlock details for an episode."""
    episode: int
    achievements_unlocked: set  # Set of achievement IDs unlocked
    achievements_count: int
    new_total_achievements: int


class AchievementTracker:
    """Tracks per-achievement unlock statistics across training."""
    
    def __init__(self, num_achievements: int = 22):
        self.num_achievements = num_achievements
        self.unlock_episodes = [[] for _ in range(num_achievements)]  # Episodes when each achievement unlocked
        self.first_unlock_episode = [None] * num_achievements  # First episode each was unlocked
        self.unlock_count = [0] * num_achievements  # Total times each was unlocked
        self.total_unlocks = 0
        self.unique_achievements_unlocked = set()
        self.episode_achievement_matrix = []  # List of [num_achievements] arrays per episode
        self.current_episode_count = -1  # Track current episode for matrix building
    
    def track_episode(self, episode: int, achievements_unlocked: set):
        """Record achievement unlocks for an episode."""
        # Initialize episode achievement array if new episode
        if episode > self.current_episode_count:
            # Fill missing episodes with zeros
            while len(self.episode_achievement_matrix) <= episode:
                self.episode_achievement_matrix.append([0] * self.num_achievements)
            self.current_episode_count = episode
        
        # Track achievements for this episode
        for ach_id in achievements_unlocked:
            if 0 <= ach_id < self.num_achievements:
                self.unlock_episodes[ach_id].append(episode)
                self.unlock_count[ach_id] += 1
                self.total_unlocks += 1
                self.unique_achievements_unlocked.add(ach_id)
                
                # Increment episode achievement matrix
                self.episode_achievement_matrix[episode][ach_id] += 1
                
                if self.first_unlock_episode[ach_id] is None:
                    self.first_unlock_episode[ach_id] = episode
    
    def get_statistics(self) -> Dict:
        """Return achievement-level statistics."""
        total_possible = self.num_achievements
        unique_unlocked = len(self.unique_achievements_unlocked)
        
        # Achievement distribution
        unlocked_counts = [c for c in self.unlock_count if c > 0]
        
        # Calculate cumulative unlock matrix (for plotting)
        cumulative_matrix = self._compute_cumulative_matrix()
        
        # Compute per-achievement statistics across episodes
        per_ach_stats = self._compute_per_achievement_episode_stats(cumulative_matrix)
        
        return {
            'total_possible_achievements': total_possible,
            'unique_achievements_unlocked': unique_unlocked,
            'unlock_ratio': unique_unlocked / total_possible if total_possible > 0 else 0,
            'total_unlock_instances': self.total_unlocks,
            'avg_unlocks_per_achievement': np.mean(unlocked_counts) if unlocked_counts else 0,
            'max_unlocks_per_achievement': max(self.unlock_count) if self.unlock_count else 0,
            'min_unlocks_per_achievement': min(c for c in self.unlock_count if c > 0) if unlocked_counts else 0,
            'never_unlocked': total_possible - unique_unlocked,
            'per_achievement_stats': self._get_per_achievement_stats(),
            'episode_achievement_matrix': self.episode_achievement_matrix,  # Raw matrix
            'cumulative_achievement_matrix': cumulative_matrix,  # Cumulative for plotting
            'per_achievement_episode_stats': per_ach_stats  # Min/max/mean per achievement
        }
    
    def _get_per_achievement_stats(self) -> List[Dict]:
        """Detailed stats per achievement."""
        stats = []
        for ach_id in range(self.num_achievements):
            stats.append({
                'achievement_id': ach_id,
                'unlock_count': self.unlock_count[ach_id],
                'first_unlock_episode': self.first_unlock_episode[ach_id],
                'first_unlock_move': None  # Populated in EvaluationSystem
            })
        return stats
    
    def _compute_cumulative_matrix(self) -> List[List[int]]:
        """Compute cumulative achievement matrix for plotting progression."""
        if not self.episode_achievement_matrix:
            return []
        
        num_episodes = len(self.episode_achievement_matrix)
        cumulative = [[0] * self.num_achievements for _ in range(num_episodes)]
        
        for ep in range(num_episodes):
            for ach_id in range(self.num_achievements):
                prev_count = cumulative[ep-1][ach_id] if ep > 0 else 0
                cumulative[ep][ach_id] = prev_count + self.episode_achievement_matrix[ep][ach_id]
        
        return cumulative
    
    def _compute_per_achievement_episode_stats(self, cumulative_matrix: List[List[int]]) -> List[Dict]:
        """Compute min/max/mean unlock counts per achievement across episodes for shaded plotting."""
        if not cumulative_matrix:
            return []
        
        stats = []
        num_episodes = len(cumulative_matrix)
        
        for ach_id in range(self.num_achievements):
            # Extract this achievement's trajectory across episodes
            trajectory = [cumulative_matrix[ep][ach_id] for ep in range(num_episodes)]
            
            stats.append({
                'achievement_id': ach_id,
                'min_unlock_count': min(trajectory) if trajectory else 0,
                'max_unlock_count': max(trajectory) if trajectory else 0,
                'mean_unlock_count': np.mean(trajectory) if trajectory else 0,
                'final_unlock_count': trajectory[-1] if trajectory else 0
            })
        
        return stats


class EfficiencyAnalyzer:
    """Analyzes learning efficiency metrics."""
    
    def __init__(self, metrics_list: List[EpisodeMetrics]):
        self.metrics = metrics_list
        self._compute_efficiency_metrics()
    
    def _compute_efficiency_metrics(self):
        """Calculate reward/move, reward/helper_call, achievements/move ratios."""
        for metric in self.metrics:
            # Avoid division by zero
            metric.reward_per_move = metric.shaped_reward / metric.moves if metric.moves > 0 else 0
            metric.reward_per_helper_call = (metric.shaped_reward / metric.helper_calls 
                                           if metric.helper_calls > 0 else metric.shaped_reward)
            metric.achievements_per_move = metric.achievements_unlocked / metric.moves if metric.moves > 0 else 0
    
    def get_efficiency_statistics(self) -> Dict:
        """Compute efficiency summaries."""
        reward_per_move = [m.reward_per_move for m in self.metrics if m.reward_per_move is not None]
        reward_per_helper = [m.reward_per_helper_call for m in self.metrics if m.reward_per_helper_call is not None]
        ach_per_move = [m.achievements_per_move for m in self.metrics if m.achievements_per_move is not None]
        
        return {
            'reward_per_move': {
                'mean': np.mean(reward_per_move) if reward_per_move else 0,
                'std': np.std(reward_per_move) if reward_per_move else 0,
                'min': np.min(reward_per_move) if reward_per_move else 0,
                'max': np.max(reward_per_move) if reward_per_move else 0
            },
            'reward_per_helper_call': {
                'mean': np.mean(reward_per_helper) if reward_per_helper else 0,
                'std': np.std(reward_per_helper) if reward_per_helper else 0,
                'min': np.min(reward_per_helper) if reward_per_helper else 0,
                'max': np.max(reward_per_helper) if reward_per_helper else 0
            },
            'achievements_per_move': {
                'mean': np.mean(ach_per_move) if ach_per_move else 0,
                'std': np.std(ach_per_move) if ach_per_move else 0,
                'min': np.min(ach_per_move) if ach_per_move else 0,
                'max': np.max(ach_per_move) if ach_per_move else 0
            }
        }


class ConvergenceDetector:
    """Detects learning convergence and trend patterns."""
    
    def __init__(self, metrics_list: List[EpisodeMetrics], window_size: int = 10):
        self.metrics = metrics_list
        self.window_size = window_size
    
    def detect_convergence(self, metric_name: str = 'shaped_reward', threshold: float = 0.05) -> Dict:
        """
        Detect if metric has converged using moving average stability.
        
        Args:
            metric_name: 'shaped_reward', 'native_reward', 'achievements_unlocked', etc.
            threshold: Convergence if moving std < threshold
        
        Returns:
            Dict with convergence status, episode, and moving statistics
        """
        if len(self.metrics) < self.window_size:
            return {'converged': False, 'episode': None, 'reason': 'insufficient_data'}
        
        values = [getattr(m, metric_name) for m in self.metrics]
        moving_mean = self._moving_average(values, self.window_size)
        moving_std = self._moving_std(values, self.window_size)
        
        # Find convergence point
        converged_episode = None
        for i in range(self.window_size, len(moving_std)):
            if moving_std[i] < threshold:
                converged_episode = self.metrics[i].episode
                break
        
        return {
            'converged': converged_episode is not None,
            'converged_at_episode': converged_episode,
            'threshold': threshold,
            'final_moving_std': moving_std[-1] if moving_std else None,
            'moving_mean': moving_mean,
            'moving_std': moving_std
        }
    
    def get_trend(self, metric_name: str = 'shaped_reward', window_size: int = None) -> Dict:
        """Compute trend (improving/stable/declining) over last N episodes."""
        if window_size is None:
            window_size = self.window_size
        
        if len(self.metrics) < window_size * 2:
            return {'trend': 'insufficient_data', 'episodes_analyzed': len(self.metrics)}
        
        values = [getattr(m, metric_name) for m in self.metrics]
        
        # Compare recent vs earlier windows
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        improvement = (second_half - first_half) / (abs(first_half) + 1e-6)
        
        if improvement > 0.1:
            trend = 'improving'
        elif improvement < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'improvement_ratio': improvement,
            'first_half_mean': first_half,
            'second_half_mean': second_half,
            'episodes_analyzed': len(values)
        }
    
    @staticmethod
    def _moving_average(values: List[float], window: int) -> List[float]:
        """Compute moving average."""
        if len(values) < window:
            return values
        return [np.mean(values[max(0, i-window):i+1]) for i in range(len(values))]
    
    @staticmethod
    def _moving_std(values: List[float], window: int) -> List[float]:
        """Compute moving standard deviation."""
        if len(values) < window:
            return [np.std(values[:i+1]) for i in range(len(values))]
        return [np.std(values[max(0, i-window):i+1]) for i in range(len(values))]


class EvaluationSystem:
    """
    Comprehensive evaluation framework orchestrating metrics aggregation, 
    per-achievement analysis, efficiency computation, and convergence detection.
    """
    
    def __init__(self, num_achievements: int = 22):
        self.num_achievements = num_achievements
        self.metrics_list: List[EpisodeMetrics] = []
        self.achievement_tracker = AchievementTracker(num_achievements)
        self.efficiency_analyzer: Optional[EfficiencyAnalyzer] = None
        self.convergence_detector: Optional[ConvergenceDetector] = None
    
    def add_episode(self, episode: int, shaped_reward: float, native_reward: float, 
                    shaped_bonus: float, achievements_unlocked: int, moves: int,
                    helper_calls: int, hallucinations: int):
        """Add metrics from a completed training episode."""
        hallucination_rate = hallucinations / max(1, helper_calls)
        
        metric = EpisodeMetrics(
            episode=episode,
            shaped_reward=shaped_reward,
            native_reward=native_reward,
            shaped_bonus=shaped_bonus,
            achievements_unlocked=achievements_unlocked,
            moves=moves,
            helper_calls=helper_calls,
            hallucinations=hallucinations,
            hallucination_rate=hallucination_rate
        )
        
        self.metrics_list.append(metric)
    
    def add_episode_achievements(self, episode: int, achievements_unlocked: set):
        """Add detailed achievement unlock information."""
        self.achievement_tracker.track_episode(episode, achievements_unlocked)
    
    def finalize(self):
        """Compute all statistics after training completes."""
        if not self.metrics_list:
            raise ValueError("No metrics added. Cannot finalize.")
        
        self.efficiency_analyzer = EfficiencyAnalyzer(self.metrics_list)
        self.convergence_detector = ConvergenceDetector(self.metrics_list)
    
    def get_summary_statistics(self) -> Dict:
        """Get comprehensive summary statistics."""
        if not self.metrics_list:
            return {}
        
        shaped_rewards = [m.shaped_reward for m in self.metrics_list]
        native_rewards = [m.native_reward for m in self.metrics_list]
        achievements = [m.achievements_unlocked for m in self.metrics_list]
        moves = [m.moves for m in self.metrics_list]
        helper_calls = [m.helper_calls for m in self.metrics_list]
        hallucination_rates = [m.hallucination_rate for m in self.metrics_list]
        
        return {
            'total_episodes': len(self.metrics_list),
            'shaped_reward': {
                'mean': np.mean(shaped_rewards),
                'std': np.std(shaped_rewards),
                'min': np.min(shaped_rewards),
                'max': np.max(shaped_rewards),
                'final': shaped_rewards[-1] if shaped_rewards else None
            },
            'native_reward': {
                'mean': np.mean(native_rewards),
                'std': np.std(native_rewards),
                'min': np.min(native_rewards),
                'max': np.max(native_rewards),
                'final': native_rewards[-1] if native_rewards else None
            },
            'achievements_unlocked': {
                'mean': np.mean(achievements),
                'std': np.std(achievements),
                'total': sum(achievements),
                'max_per_episode': np.max(achievements),
                'final': achievements[-1] if achievements else None
            },
            'moves': {
                'mean': np.mean(moves),
                'std': np.std(moves),
                'total': sum(moves),
                'min': np.min(moves),
                'max': np.max(moves)
            },
            'helper_calls': {
                'mean': np.mean(helper_calls),
                'std': np.std(helper_calls),
                'total': sum(helper_calls),
                'final_episode': helper_calls[-1] if helper_calls else None
            },
            'hallucinations': {
                'total': sum(m.hallucinations for m in self.metrics_list),
                'mean_rate': np.mean(hallucination_rates),
                'std_rate': np.std(hallucination_rates),
                'final_rate': hallucination_rates[-1] if hallucination_rates else None
            }
        }
    
    def get_achievement_statistics(self) -> Dict:
        """Get per-achievement unlock statistics."""
        return self.achievement_tracker.get_statistics()
    
    def get_efficiency_statistics(self) -> Dict:
        """Get efficiency ratio statistics."""
        if not self.efficiency_analyzer:
            raise ValueError("Must call finalize() first")
        return self.efficiency_analyzer.get_efficiency_statistics()
    
    def get_convergence_report(self) -> Dict:
        """Get convergence analysis across key metrics."""
        if not self.convergence_detector:
            raise ValueError("Must call finalize() first")
        
        return {
            'shaped_reward_convergence': self.convergence_detector.detect_convergence('shaped_reward'),
            'shaped_reward_trend': self.convergence_detector.get_trend('shaped_reward'),
            'achievements_convergence': self.convergence_detector.detect_convergence('achievements_unlocked'),
            'achievements_trend': self.convergence_detector.get_trend('achievements_unlocked'),
            'helper_calls_trend': self.convergence_detector.get_trend('helper_calls')
        }
    
    def export_metrics_dataframe(self) -> pd.DataFrame:
        """Export all metrics as pandas DataFrame for external analysis."""
        data = [asdict(m) for m in self.metrics_list]
        return pd.DataFrame(data)
    
    def export_to_jsonl(self, filepath: str):
        """Export metrics to JSONL file (one JSON object per line)."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for metric in self.metrics_list:
                record = asdict(metric)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[EvaluationSystem] Exported metrics to {filepath}")
    
    def export_summary_json(self, filepath: str):
        """Export summary statistics to JSON file."""
        summary = {
            'summary_statistics': self.get_summary_statistics(),
            'achievement_statistics': self.get_achievement_statistics(),
            'efficiency_statistics': self.get_efficiency_statistics(),
            'convergence_report': self.get_convergence_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[EvaluationSystem] Exported summary to {filepath}")
    
    def print_summary_report(self):
        """Print human-readable summary report."""
        summary = self.get_summary_statistics()
        achievements = self.get_achievement_statistics()
        efficiency = self.get_efficiency_statistics()
        convergence = self.get_convergence_report()
        
        print("\n" + "="*70)
        print("EVALUATION SYSTEM - TRAINING SUMMARY REPORT")
        print("="*70)
        
        print(f"\n[Training Overview]")
        print(f"  Total Episodes: {summary['total_episodes']}")
        
        print(f"\n[Shaped Reward (Native + Bonuses)]")
        print(f"  Mean: {summary['shaped_reward']['mean']:.3f} ± {summary['shaped_reward']['std']:.3f}")
        print(f"  Range: [{summary['shaped_reward']['min']:.3f}, {summary['shaped_reward']['max']:.3f}]")
        print(f"  Final: {summary['shaped_reward']['final']:.3f}")
        
        print(f"\n[Native Reward (Sparse)]")
        print(f"  Mean: {summary['native_reward']['mean']:.3f} ± {summary['native_reward']['std']:.3f}")
        print(f"  Final: {summary['native_reward']['final']:.3f}")
        
        print(f"\n[Achievements Unlocked]")
        print(f"  Mean per episode: {summary['achievements_unlocked']['mean']:.2f}")
        print(f"  Total unlocked: {summary['achievements_unlocked']['total']}")
        print(f"  Max in single episode: {summary['achievements_unlocked']['max_per_episode']}")
        
        print(f"\n[Achievement Statistics]")
        print(f"  Unique achievements unlocked: {achievements['unique_achievements_unlocked']}/{achievements['total_possible_achievements']}")
        print(f"  Unlock ratio: {achievements['unlock_ratio']:.1%}")
        print(f"  Total unlock instances: {achievements['total_unlock_instances']}")
        print(f"  Never unlocked: {achievements['never_unlocked']}")
        
        print(f"\n[Efficiency Metrics]")
        print(f"  Reward per move: {efficiency['reward_per_move']['mean']:.4f} ± {efficiency['reward_per_move']['std']:.4f}")
        print(f"  Reward per helper call: {efficiency['reward_per_helper_call']['mean']:.4f} ± {efficiency['reward_per_helper_call']['std']:.4f}")
        print(f"  Achievements per move: {efficiency['achievements_per_move']['mean']:.4f}")
        
        print(f"\n[Helper LLM Metrics]")
        print(f"  Total helper calls: {summary['helper_calls']['total']}")
        print(f"  Mean calls per episode: {summary['helper_calls']['mean']:.2f}")
        print(f"  Hallucination rate: {summary['hallucinations']['mean_rate']:.1%}")
        
        print(f"\n[Convergence Analysis]")
        if convergence['shaped_reward_convergence']['converged']:
            print(f"  ✓ Shaped reward converged at episode {convergence['shaped_reward_convergence']['converged_at_episode']}")
        else:
            print(f"  ✗ Shaped reward not converged (moving std: {convergence['shaped_reward_convergence']['final_moving_std']:.4f})")
        
        print(f"  Reward trend: {convergence['shaped_reward_trend']['trend'].upper()}")
        print(f"  Achievement trend: {convergence['achievements_trend']['trend'].upper()}")
        
        print("\n" + "="*70 + "\n")


class BaselineComparator:
    """Compare evaluation results across different configurations."""
    
    def __init__(self):
        self.configurations: Dict[str, Dict] = {}
    
    def add_configuration(self, name: str, summary: Dict, achievements: Dict, efficiency: Dict):
        """Add evaluation results from a configuration."""
        self.configurations[name] = {
            'summary': summary,
            'achievements': achievements,
            'efficiency': efficiency
        }
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table across configurations."""
        rows = []
        
        for config_name, data in self.configurations.items():
            summary = data['summary']
            rows.append({
                'Configuration': config_name,
                'Episodes': summary['total_episodes'],
                'Avg Reward': f"{summary['shaped_reward']['mean']:.3f}",
                'Reward Std': f"{summary['shaped_reward']['std']:.3f}",
                'Total Achievements': summary['achievements_unlocked']['total'],
                'Avg Achievements/Ep': f"{summary['achievements_unlocked']['mean']:.2f}",
                'Total Helper Calls': summary['helper_calls']['total'],
                'Hallucination Rate': f"{summary['hallucinations']['mean_rate']:.1%}",
                'Reward/Move': f"{data['efficiency']['reward_per_move']['mean']:.4f}",
                'Reward/Helper': f"{data['efficiency']['reward_per_helper_call']['mean']:.4f}"
            })
        
        return pd.DataFrame(rows)
    
    def print_comparison(self):
        """Print human-readable comparison."""
        df = self.generate_comparison_table()
        print("\n" + "="*120)
        print("BASELINE COMPARISON TABLE")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")
    
    def export_comparison_jsonl(self, filepath: str):
        """Export comparison to JSONL."""
        df = self.generate_comparison_table()
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"[BaselineComparator] Exported comparison to {filepath}")
