#!/usr/bin/env python3
"""
Script per generare learning curves individuali per ogni achievement sbloccato
Genera plot per sia HeRoN che Baseline
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.evaluation_system import ACHIEVEMENT_ID_TO_NAME


class AchievementCurvesGenerator:
    """Genera learning curves per gli achievement da file JSON."""
    
    def __init__(self):
        self.achievement_stats = {}
        self.summary_stats = {}
    
    def load_achievement_data(self, json_file: str):
        """Carica dati da file JSON."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both nested and flat JSON structures
        if 'achievement_statistics' in data:
            # Nested structure (old format)
            self.achievement_stats = data.get('achievement_statistics', {})
            self.summary_stats = data.get('summary_statistics', {})
        else:
            # Flat structure (new format from export_achievement_statistics_json)
            self.achievement_stats = data
            self.summary_stats = {'total_episodes': len(data.get('cumulative_achievement_matrix', []))}
        
        print(f"[Loader] Loaded data from {json_file}")
        return data
    
    def generate_achievement_curve(self, achievement_id: int, output_dir: Path, 
                                   config_name: str, window: int = 10):
        """Genera learning curve per un singolo achievement."""
        
        # Verifica se achievement è stato sbloccato
        per_achievement_stats = self.achievement_stats.get('per_achievement_stats', [])
        ach_stats = next((s for s in per_achievement_stats 
                         if s.get('achievement_id') == achievement_id), None)
        
        if not ach_stats or not ach_stats.get('unlocked', False):
            return False
        
        achievement_name = ACHIEVEMENT_ID_TO_NAME.get(achievement_id, f"Unknown_{achievement_id}")
        total_unlocks = ach_stats.get('unlock_count', 0)
        first_unlock_ep = ach_stats.get('first_unlock_episode')
        
        if total_unlocks == 0:
            return False
        
        # Estrai dati cumulativi
        cumulative_matrix = self.achievement_stats.get('cumulative_achievement_matrix', [])
        if not cumulative_matrix or achievement_id >= len(cumulative_matrix[0]) if cumulative_matrix else True:
            return False
        
        cumulative_unlocks = [episode_data[achievement_id] for episode_data in cumulative_matrix]
        
        # Calcola media mobile
        episodes = np.arange(len(cumulative_unlocks))
        moving_avg = self._moving_average(cumulative_unlocks, window)
        moving_std = self._moving_std(cumulative_unlocks, window)
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot area di confidenza e media mobile
        if len(episodes) >= window:
            # Sufficient episodes for moving average with full window
            episodes_adjusted = episodes[window-1:]
            moving_avg_adjusted = moving_avg[window-1:]
            moving_std_adjusted = moving_std[window-1:]
            ax.fill_between(episodes_adjusted, 
                           np.array(moving_avg_adjusted) - np.array(moving_std_adjusted),
                           np.array(moving_avg_adjusted) + np.array(moving_std_adjusted),
                           alpha=0.3, color='green', label='Variability (±1 std)')
            
            # Plot linea media mobile
            ax.plot(episodes_adjusted, moving_avg_adjusted, 
                   color='darkgreen', linewidth=2.5, label='Moving Average')
        else:
            # Not enough episodes - plot simple moving average over all data
            ax.fill_between(episodes, 
                           np.array(moving_avg) - np.array(moving_std),
                           np.array(moving_avg) + np.array(moving_std),
                           alpha=0.3, color='green', label='Variability (±1 std)')
            ax.plot(episodes, moving_avg, 
                   color='darkgreen', linewidth=2.5, label=f'Moving Average (window={len(episodes)})')
        
        # Plot dati raw
        ax.plot(episodes, cumulative_unlocks, 
               color='lightgreen', linewidth=0.8, alpha=0.5, label='Raw Data')
        
        # Marker per primo unlock
        if first_unlock_ep is not None and first_unlock_ep < len(episodes):
            ax.plot(first_unlock_ep, cumulative_unlocks[first_unlock_ep], 
                   marker='*', markersize=15, color='gold',
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'First Unlock (Ep {first_unlock_ep})', zorder=5)
        
        # Formattazione
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Unlocks', fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curve: {achievement_name.replace("_", " ").title()} ({config_name})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        
        # Aggiungi statistiche
        total_episodes = self.summary_stats.get('total_episodes', 0)
        stats_text = f'Total Unlocks: {total_unlocks}\n'
        stats_text += f'Episodes: {total_episodes}\n'
        stats_text += f'First Unlock: Ep {first_unlock_ep if first_unlock_ep is not None else "Never"}'
        
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Salva
        output_file = output_dir / f'{achievement_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {achievement_name:<30s} (Unlocks: {total_unlocks:>3d})")
        return True
    
    def generate_all_achievement_curves(self, json_file: str, output_dir: Path, config_name: str):
        """Genera curve per tutti gli achievement sbloccati."""
        
        print(f"\n{'='*80}")
        print(f"Generating Achievement Learning Curves: {config_name}")
        print(f"{'='*80}\n")
        
        self.load_achievement_data(json_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plotted_count = 0
        
        for achievement_id in range(22):
            if self.generate_achievement_curve(achievement_id, output_dir, config_name):
                plotted_count += 1
        
        print(f"\n{'='*80}")
        print(f"Achievement Curves Generated: {plotted_count}/22")
        print(f"{'='*80}\n")
        
        return plotted_count
    
    @staticmethod
    def _moving_average(values, window: int):
        """Calcola media mobile."""
        return [np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]
    
    @staticmethod
    def _moving_std(values, window: int):
        """Calcola deviazione standard mobile."""
        return [np.std(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def main():
    print("=" * 80)
    print("Achievement Learning Curves Generator")
    print("=" * 80)
    
    # HeRoN
    print("\n[1/2] Generating HeRoN achievement curves...")
    generator_heron = AchievementCurvesGenerator()
    heron_count = generator_heron.generate_all_achievement_curves(
        json_file="training/heron_output/heron_achievement_statistics.json",
        output_dir=Path("evaluation_plots/heron/achievement_curves"),
        config_name="HeRoN (DQN + Helper + Threshold Decay)"
    )
    
    # Baseline
    print("\n[2/2] Generating Baseline achievement curves...")
    generator_baseline = AchievementCurvesGenerator()
    baseline_count = generator_baseline.generate_all_achievement_curves(
        json_file="training/baseline_dqn_output/baseline_crafter_dqn_evaluation.json",
        output_dir=Path("evaluation_plots/baseline/achievement_curves"),
        config_name="Baseline DQN (No LLM)"
    )
    
    # Resoconto finale
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"HeRoN Achievement Curves:       {heron_count} generated")
    print(f"Baseline Achievement Curves:    {baseline_count} generated")
    print(f"Total Curves:                   {heron_count + baseline_count}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
