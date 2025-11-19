"""
F10: Sistema di Valutazione - Advanced Visualizations

Comprehensive plotting utilities for evaluation analysis:
- Achievement unlock heatmaps (episode vs achievement)
- Reward distribution histograms
- Moving average trends with confidence bands
- Helper dependency decay curves
- Efficiency scatter plots (moves vs achievements)
- Multi-metric dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from evaluation.evaluation_system import EpisodeMetrics, EvaluationSystem


class AdvancedPlotter:
    """Advanced visualization utilities for F10 evaluation."""
    
    def __init__(self, output_dir: str = "./evaluation_plots", figsize: tuple = (15, 10)):
        self.output_dir = output_dir
        self.figsize = figsize
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure matplotlib style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_achievement_heatmap(self, metrics_list: List[EpisodeMetrics], 
                                 output_file: str = "achievement_heatmap.png"):
        """
        Create heatmap: episodes (x-axis) vs achievements (y-axis).
        Shows cumulative unlock pattern over training.
        """
        num_episodes = len(metrics_list)
        
        # Create figure with cumulative achievements over episodes
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Use a simple visualization: show cumulative achievements over time
        episodes = np.arange(num_episodes)
        achievements_cumsum = np.cumsum([m.achievements_unlocked for m in metrics_list])
        
        # Plot as area chart
        ax.fill_between(episodes, achievements_cumsum, alpha=0.3, color='green')
        ax.plot(episodes, achievements_cumsum, color='darkgreen', linewidth=2.5, marker='o', markersize=3)
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Achievements', fontsize=12, fontweight='bold')
        ax.set_title('Achievement Unlock Progress Over Training', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved achievement heatmap to {filepath}")
    
    def plot_reward_distribution(self, metrics_list: List[EpisodeMetrics], 
                                output_file: str = "reward_distribution.png"):
        """
        Distribution histograms of shaped/native rewards.
        Shows reward variance across episodes.
        """
        shaped_rewards = [m.shaped_reward for m in metrics_list]
        native_rewards = [m.native_reward for m in metrics_list]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Shaped reward histogram
        axes[0].hist(shaped_rewards, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(shaped_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(shaped_rewards):.2f}')
        axes[0].set_xlabel('Shaped Reward', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Shaped Reward Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Native reward histogram
        axes[1].hist(native_rewards, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(native_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(native_rewards):.2f}')
        axes[1].set_xlabel('Native Reward', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Native Reward Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved reward distribution to {filepath}")
    
    def plot_moving_average_trends(self, metrics_list: List[EpisodeMetrics], 
                                   window_size: int = 10,
                                   output_file: str = "moving_averages.png"):
        """
        Plot reward trends with moving averages and confidence bands.
        """
        episodes = [m.episode for m in metrics_list]
        shaped_rewards = [m.shaped_reward for m in metrics_list]
        native_rewards = [m.native_reward for m in metrics_list]
        
        # Compute moving averages
        ma_shaped = self._moving_average(shaped_rewards, window_size)
        ma_native = self._moving_average(native_rewards, window_size)
        
        # Compute confidence bands (std over window)
        std_shaped = self._moving_std(shaped_rewards, window_size)
        std_native = self._moving_std(native_rewards, window_size)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot raw data (light)
        ax.scatter(episodes, shaped_rewards, alpha=0.3, s=30, color='steelblue', label='Shaped Reward (per episode)')
        ax.scatter(episodes, native_rewards, alpha=0.3, s=30, color='coral', label='Native Reward (per episode)')
        
        # Plot moving averages
        ax.plot(episodes, ma_shaped, color='steelblue', linewidth=2.5, label=f'Shaped MA (window={window_size})')
        ax.plot(episodes, ma_native, color='coral', linewidth=2.5, label=f'Native MA (window={window_size})')
        
        # Add confidence bands
        ax.fill_between(episodes, 
                        np.array(ma_shaped) - np.array(std_shaped),
                        np.array(ma_shaped) + np.array(std_shaped),
                        alpha=0.2, color='steelblue')
        ax.fill_between(episodes,
                        np.array(ma_native) - np.array(std_native),
                        np.array(ma_native) + np.array(std_native),
                        alpha=0.2, color='coral')
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Trends with Moving Averages and Confidence Bands', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved moving average trends to {filepath}")
    
    def plot_helper_dependency_decay(self, metrics_list: List[EpisodeMetrics],
                                    output_file: str = "helper_dependency.png"):
        """
        Show how helper dependency decays over training.
        Plot: helper_calls per episode over time.
        """
        episodes = [m.episode for m in metrics_list]
        helper_calls = [m.helper_calls for m in metrics_list]
        hallucination_rates = [m.hallucination_rate for m in metrics_list]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Helper calls trend
        axes[0].bar(episodes, helper_calls, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].plot(episodes, helper_calls, color='darkblue', linewidth=2, marker='o', markersize=4, label='Helper Calls')
        axes[0].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Helper Calls per Episode', fontsize=11, fontweight='bold')
        axes[0].set_title('Helper LLM Dependency Decay', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].legend()
        
        # Hallucination rate trend
        axes[1].scatter(episodes, hallucination_rates, color='coral', s=50, alpha=0.7, edgecolor='black', label='Hallucination Rate')
        axes[1].plot(episodes, hallucination_rates, color='darkred', linewidth=2, marker='s', markersize=4)
        axes[1].axhline(np.mean(hallucination_rates), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(hallucination_rates):.1%}')
        axes[1].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Hallucination Rate', fontsize=11, fontweight='bold')
        axes[1].set_title('LLM Hallucination Rate Over Training', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, max(hallucination_rates) * 1.1])
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved helper dependency decay to {filepath}")
    
    def plot_efficiency_scatter(self, metrics_list: List[EpisodeMetrics],
                               output_file: str = "efficiency_scatter.png"):
        """
        Scatter plot: achievements vs moves, sized by reward.
        Shows efficiency trade-offs.
        """
        achievements = [m.achievements_unlocked for m in metrics_list]
        moves = [m.moves for m in metrics_list]
        rewards = [m.shaped_reward for m in metrics_list]
        episodes = [m.episode for m in metrics_list]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(moves, achievements, c=rewards, s=100, 
                            cmap='viridis', alpha=0.6, edgecolor='black', linewidth=1)
        
        # Annotate some interesting points
        for i in [0, len(metrics_list)//4, len(metrics_list)//2, len(metrics_list)-1]:
            ax.annotate(f"E{episodes[i]}", 
                       (moves[i], achievements[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Moves (Steps per Episode)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Achievements Unlocked', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Trade-off: Achievements vs Moves\n(Color: Shaped Reward)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Shaped Reward')
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved efficiency scatter to {filepath}")
    
    def plot_multi_metric_dashboard(self, evaluation_system: EvaluationSystem,
                                    output_file: str = "multi_metric_dashboard.png"):
        """
        4-subplot dashboard with key metrics.
        """
        metrics_list = evaluation_system.metrics_list
        
        episodes = [m.episode for m in metrics_list]
        shaped_rewards = [m.shaped_reward for m in metrics_list]
        achievements = [m.achievements_unlocked for m in metrics_list]
        helper_calls = [m.helper_calls for m in metrics_list]
        hallucination_rates = [m.hallucination_rate for m in metrics_list]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Shaped reward trend
        axes[0, 0].plot(episodes, shaped_rewards, color='steelblue', linewidth=2, marker='o', markersize=3)
        axes[0, 0].fill_between(episodes, shaped_rewards, alpha=0.3, color='steelblue')
        axes[0, 0].axhline(np.mean(shaped_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(shaped_rewards):.2f}')
        axes[0, 0].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('Shaped Reward', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('A. Shaped Reward Trend', fontsize=11, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Cumulative achievements
        cumsum_achievements = np.cumsum(achievements)
        axes[0, 1].plot(episodes, cumsum_achievements, color='green', linewidth=2.5, marker='s', markersize=3)
        axes[0, 1].fill_between(episodes, cumsum_achievements, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Episode', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Cumulative Achievements', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('B. Cumulative Achievement Unlocks', fontsize=11, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Helper calls + hallucination rate
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.bar(episodes, helper_calls, alpha=0.5, color='steelblue', label='Helper Calls')
        line2 = ax3_twin.plot(episodes, hallucination_rates, color='red', linewidth=2, marker='o', 
                             markersize=4, label='Hallucination Rate')
        
        ax3.set_xlabel('Episode', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Helper Calls', fontsize=10, fontweight='bold', color='steelblue')
        ax3_twin.set_ylabel('Hallucination Rate', fontsize=10, fontweight='bold', color='red')
        ax3.set_title('C. Helper LLM Activity & Hallucinations', fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. Efficiency: moves vs achievements
        moves = [m.moves for m in metrics_list]
        scatter = axes[1, 1].scatter(moves, achievements, c=shaped_rewards, s=80,
                                    cmap='viridis', alpha=0.6, edgecolor='black')
        axes[1, 1].set_xlabel('Moves per Episode', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Achievements Unlocked', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('D. Efficiency: Achievements vs Moves', fontsize=11, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 1], label='Shaped Reward')
        
        plt.suptitle('F10: Multi-Metric Evaluation Dashboard', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = f"{self.output_dir}/{output_file}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[AdvancedPlotter] Saved multi-metric dashboard to {filepath}")
    
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


def generate_all_plots(evaluation_system: EvaluationSystem, output_dir: str = "./evaluation_plots"):
    """
    Convenience function: generate all evaluation plots.
    """
    plotter = AdvancedPlotter(output_dir=output_dir)
    
    plotter.plot_reward_distribution(evaluation_system.metrics_list)
    plotter.plot_moving_average_trends(evaluation_system.metrics_list)
    plotter.plot_helper_dependency_decay(evaluation_system.metrics_list)
    plotter.plot_efficiency_scatter(evaluation_system.metrics_list)
    plotter.plot_achievement_heatmap(evaluation_system.metrics_list)
    plotter.plot_multi_metric_dashboard(evaluation_system)
    
    print(f"\n[AdvancedPlotter] All plots generated in {output_dir}/")
