#!/usr/bin/env python3
"""
Script per generare plots dai dati di training HeRoN
Legge i dati JSONL e JSON per creare visualizzazioni
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.evaluation_plots import AdvancedPlotter
from evaluation.evaluation_system import EvaluationSystem, EpisodeMetrics


def load_metrics_from_jsonl(jsonl_file: str) -> list:
    """Load metrics from JSONL file."""
    metrics_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            
            # Create EpisodeMetrics object
            metric = EpisodeMetrics(
                episode=data.get('episode', idx),
                shaped_reward=data.get('shaped_reward', 0),
                native_reward=data.get('native_reward', 0),
                shaped_bonus=data.get('shaped_bonus', 0),
                achievements_unlocked=data.get('achievements_unlocked', 0),
                moves=data.get('moves', 0),
                helper_calls=data.get('helper_calls', 0),
                hallucinations=data.get('hallucinations', 0),
                hallucination_rate=data.get('hallucination_rate', 0),
                reward_per_move=data.get('reward_per_move', 0),
                reward_per_helper_call=data.get('reward_per_helper_call', 0),
                achievements_per_move=data.get('achievements_per_move', 0)
            )
            metrics_list.append(metric)
    
    print(f"[Data Loader] Loaded {len(metrics_list)} episodes from {jsonl_file}")
    return metrics_list


def load_achievement_stats(json_file: str) -> dict:
    """Load achievement statistics from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[Data Loader] Loaded achievement statistics from {json_file}")
    return data


def main():
    # Paths
    heron_metrics_file = "training/heron_output/heron_crafter_metrics.jsonl"
    heron_achievements_file = "training/heron_output/heron_achievement_statistics.json"
    
    baseline_metrics_file = "training/baseline_dqn_output/baseline_crafter_dqn_metrics.jsonl"
    baseline_achievements_file = "training/baseline_dqn_output/baseline_crafter_dqn_evaluation.json"
    
    output_dir = "evaluation_plots"
    
    print("=" * 80)
    print("HeRoN Training Evaluation - Plot Generation")
    print("=" * 80)
    
    # Load HeRoN metrics
    print("\n[1/2] Loading HeRoN metrics...")
    try:
        heron_metrics = load_metrics_from_jsonl(heron_metrics_file)
        heron_stats = load_achievement_stats(heron_achievements_file)
        
        # Create EvaluationSystem object for HeRoN
        heron_eval = EvaluationSystem()
        heron_eval.metrics_list = heron_metrics
        heron_eval.achievement_stats = heron_stats
        
        print(f"✓ Loaded {len(heron_metrics)} HeRoN episodes")
        
    except FileNotFoundError as e:
        print(f"⚠️  HeRoN files not found: {e}")
        heron_eval = None
    
    # Load Baseline metrics (if available)
    print("\n[2/3] Loading Baseline metrics...")
    try:
        baseline_metrics = load_metrics_from_jsonl(baseline_metrics_file)
        
        # Create EvaluationSystem object for Baseline
        baseline_eval = EvaluationSystem()
        baseline_eval.metrics_list = baseline_metrics
        
        print(f"✓ Loaded {len(baseline_metrics)} Baseline episodes")
        
    except FileNotFoundError as e:
        print(f"⚠️  Baseline files not found: {e}")
        baseline_eval = None

    # Load DQN + Helper metrics
    print("\n[3/3] Loading DQN + Helper metrics...")
    dqn_helper_metrics_file = "training/dqn_helper_output/dqn_helper_crafter_metrics.jsonl"
    dqn_helper_achievements_file = "training/dqn_helper_output/dqn_helper_achievement_statistics.json"
    
    try:
        dqn_helper_metrics = load_metrics_from_jsonl(dqn_helper_metrics_file)
        dqn_helper_stats = load_achievement_stats(dqn_helper_achievements_file)
        
        # Create EvaluationSystem object for DQN + Helper
        dqn_helper_eval = EvaluationSystem()
        dqn_helper_eval.metrics_list = dqn_helper_metrics
        dqn_helper_eval.achievement_stats = dqn_helper_stats
        
        print(f"✓ Loaded {len(dqn_helper_metrics)} DQN + Helper episodes")
        
    except FileNotFoundError as e:
        print(f"⚠️  DQN + Helper files not found: {e}")
        dqn_helper_eval = None
    
    # Generate plots for HeRoN
    if heron_eval:
        print(f"\n{'='*80}")
        print("Generating HeRoN plots...")
        print(f"{'='*80}")
        
        plotter = AdvancedPlotter(output_dir=f"{output_dir}/heron")
        
        try:
            print("\n→ Generating reward distribution...")
            plotter.plot_reward_distribution(heron_eval.metrics_list)
            
            print("→ Generating moving average trends...")
            plotter.plot_moving_average_trends(heron_eval.metrics_list, window_size=10)
            
            print("→ Generating helper dependency decay...")
            plotter.plot_helper_dependency_decay(heron_eval.metrics_list)
            
            print("→ Generating efficiency scatter...")
            plotter.plot_efficiency_scatter(heron_eval.metrics_list)
            
            print("→ Generating achievement heatmap...")
            plotter.plot_achievement_heatmap(heron_eval.metrics_list)
            
            print("→ Generating multi-metric dashboard...")
            plotter.plot_multi_metric_dashboard(heron_eval)
            
            print(f"\n✓ All HeRoN plots generated in {output_dir}/heron/")
            
        except Exception as e:
            print(f"❌ Error generating HeRoN plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate plots for Baseline
    if baseline_eval:
        print(f"\n{'='*80}")
        print("Generating Baseline plots...")
        print(f"{'='*80}")
        
        plotter = AdvancedPlotter(output_dir=f"{output_dir}/baseline")
        
        try:
            print("\n→ Generating reward distribution...")
            plotter.plot_reward_distribution(baseline_eval.metrics_list)
            
            print("→ Generating moving average trends...")
            plotter.plot_moving_average_trends(baseline_eval.metrics_list, window_size=10)
            
            print("→ Generating helper dependency decay...")
            plotter.plot_helper_dependency_decay(baseline_eval.metrics_list)
            
            print("→ Generating efficiency scatter...")
            plotter.plot_efficiency_scatter(baseline_eval.metrics_list)
            
            print("→ Generating achievement heatmap...")
            plotter.plot_achievement_heatmap(baseline_eval.metrics_list)
            
            print("→ Generating multi-metric dashboard...")
            plotter.plot_multi_metric_dashboard(baseline_eval)
            
            print(f"\n✓ All Baseline plots generated in {output_dir}/baseline/")
            
        except Exception as e:
            print(f"❌ Error generating Baseline plots: {e}")
            import traceback
            traceback.print_exc()

    # Generate plots for DQN + Helper
    if dqn_helper_eval:
        print(f"\n{'='*80}")
        print("Generating DQN + Helper plots...")
        print(f"{'='*80}")
        
        # Output to training directory as requested
        dqn_helper_output_dir = "training/dqn_helper_output/plots"
        plotter = AdvancedPlotter(output_dir=dqn_helper_output_dir)
        
        try:
            print("\n→ Generating reward distribution...")
            plotter.plot_reward_distribution(dqn_helper_eval.metrics_list)
            
            print("→ Generating moving average trends...")
            plotter.plot_moving_average_trends(dqn_helper_eval.metrics_list, window_size=10)
            
            print("→ Generating helper dependency decay...")
            plotter.plot_helper_dependency_decay(dqn_helper_eval.metrics_list)
            
            print("→ Generating efficiency scatter...")
            plotter.plot_efficiency_scatter(dqn_helper_eval.metrics_list)
            
            print("→ Generating achievement heatmap...")
            plotter.plot_achievement_heatmap(dqn_helper_eval.metrics_list)
            
            print("→ Generating multi-metric dashboard...")
            plotter.plot_multi_metric_dashboard(dqn_helper_eval)
            
            print(f"\n✓ All DQN + Helper plots generated in {dqn_helper_output_dir}/")
            
        except Exception as e:
            print(f"❌ Error generating DQN + Helper plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    if heron_eval and baseline_eval:
        print(f"\nHeRoN vs Baseline Comparison:")
        print(f"  HeRoN Episodes:        {len(heron_eval.metrics_list)}")
        print(f"  Baseline Episodes:     {len(baseline_eval.metrics_list)}")
        print(f"  HeRoN Avg Reward:      {np.mean([m.shaped_reward for m in heron_eval.metrics_list]):.3f}")
        print(f"  Baseline Avg Reward:   {np.mean([m.shaped_reward for m in baseline_eval.metrics_list]):.3f}")
        print(f"  HeRoN Total Achievements:  {sum([m.achievements_unlocked for m in heron_eval.metrics_list])}")
        print(f"  Baseline Total Achievements: {sum([m.achievements_unlocked for m in baseline_eval.metrics_list])}")
    elif heron_eval:
        print(f"\nHeRoN Metrics:")
        print(f"  Episodes:           {len(heron_eval.metrics_list)}")
        print(f"  Avg Reward:         {np.mean([m.shaped_reward for m in heron_eval.metrics_list]):.3f}")
        print(f"  Total Achievements: {sum([m.achievements_unlocked for m in heron_eval.metrics_list])}")
    
    print(f"\nPlots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
