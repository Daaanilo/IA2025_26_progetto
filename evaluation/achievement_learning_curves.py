"""
Achievement Learning Curves Plotter

Genera grafici di learning curve per ogni achievement di Crafter, 
mostrando il progresso dell'apprendimento nel tempo.

Usage:
    python achievement_learning_curves.py <evaluation_json_file>
    python achievement_learning_curves.py baseline_crafter_dqn_evaluation.json --output-dir plots/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.evaluation_system import ACHIEVEMENT_ID_TO_NAME


class AchievementLearningCurvePlotter:
    """Genera learning curves per achievements di Crafter."""
    
    def __init__(self, json_path: str):
        """Load evaluation JSON file."""
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Support both nested and flat JSON structures
        if 'achievement_statistics' in self.data:
            # Nested structure (old format)
            self.achievement_stats = self.data.get('achievement_statistics', {})
            self.summary_stats = self.data.get('summary_statistics', {})
        else:
            # Flat structure (new format from export_achievement_statistics_json)
            self.achievement_stats = self.data
            # Calculate total moves from cumulative matrix length (approximate)
            num_episodes = len(self.data.get('cumulative_achievement_matrix', []))
            self.summary_stats = {
                'total_episodes': num_episodes,
                'moves': {'total': num_episodes * 1000}  # Approximate
            }
        
        # Estrai dati per plotting
        self.cumulative_matrix = self.achievement_stats.get('cumulative_achievement_matrix', [])
        self.episode_matrix = self.achievement_stats.get('episode_achievement_matrix', [])
        self.per_achievement_stats = self.achievement_stats.get('per_achievement_stats', [])
        
        # Calcola steps cumulativi (assumendo moves per episode)
        self.total_episodes = self.summary_stats.get('total_episodes', 0)
        self.total_steps = self.summary_stats.get('moves', {}).get('total', 0)
    
    def _calculate_moving_average(self, data: List[float], window: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola media mobile e intervallo di confidenza.
        
        Returns:
            (steps, moving_avg, std_bounds_upper, std_bounds_lower)
        """
        if not data or len(data) < window:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        data = np.array(data)
        steps = np.arange(len(data))
        
        # Calcola media mobile
        moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Calcola deviazione standard mobile
        moving_std = np.array([
            np.std(data[max(0, i-window):i+1]) 
            for i in range(len(data))
        ])[window-1:]
        
        # Intervalli di confidenza (±1 std)
        std_upper = moving_avg + moving_std
        std_lower = moving_avg - moving_std
        
        # Aggiusta steps per allinearsi con moving average
        steps_adjusted = steps[window-1:]
        
        return steps_adjusted, moving_avg, std_upper, std_lower
    
    def plot_achievement_learning_curve(self, achievement_id: int, output_dir: Path = None, 
                                       window: int = 10, use_steps: bool = False):
        """
        Genera learning curve per un singolo achievement.
        
        Args:
            achievement_id: ID dell'achievement (0-21)
            output_dir: Directory di output per salvare il grafico
            window: Finestra per media mobile
            use_steps: Se True usa steps invece di episodes sull'asse X
        """
        if not self.cumulative_matrix or achievement_id >= len(self.cumulative_matrix[0]):
            print(f"⚠️  No data for achievement ID {achievement_id}")
            return
        
        # Estrai dati per questo achievement
        achievement_name = ACHIEVEMENT_ID_TO_NAME.get(achievement_id, f"Unknown_{achievement_id}")
        
        # Ottieni conteggi cumulativi per questo achievement attraverso gli episodi
        cumulative_unlocks = [episode_data[achievement_id] for episode_data in self.cumulative_matrix]
        
        # Trova stats dettagliate
        ach_stats = next((s for s in self.per_achievement_stats 
                         if s.get('achievement_id') == achievement_id), {})
        
        total_unlocks = ach_stats.get('unlock_count', 0)
        first_unlock_ep = ach_stats.get('first_unlock_episode')
        
        if total_unlocks == 0:
            print(f"⚠️  Achievement '{achievement_name}' was never unlocked - skipping plot")
            return
        
        # Calcola media mobile e intervalli
        episodes = np.arange(len(cumulative_unlocks))
        
        # Converti in steps se richiesto
        if use_steps and self.total_steps > 0:
            steps_per_episode = self.total_steps / self.total_episodes
            x_values = episodes * steps_per_episode
            x_label = 'Training Steps'
            x_format = '1.0e'
        else:
            x_values = episodes
            x_label = 'Episode'
            x_format = 'd'
        
        # Calcola media mobile
        x_smooth, y_smooth, y_upper, y_lower = self._calculate_moving_average(
            cumulative_unlocks, window=window
        )
        
        if use_steps and self.total_steps > 0:
            steps_per_episode = self.total_steps / self.total_episodes
            x_smooth = x_smooth * steps_per_episode
        
        # Crea il grafico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot area di confidenza (variabilità)
        if len(x_smooth) > 0:
            ax.fill_between(x_smooth, y_lower, y_upper, 
                           alpha=0.3, color='green', 
                           label='Variability (±1 std)')
        
        # Plot linea media
        if len(x_smooth) > 0:
            ax.plot(x_smooth, y_smooth, 
                   color='darkgreen', linewidth=2.5, 
                   label='Moving Average')
        
        # Plot dati raw (opzionale, più leggero)
        ax.plot(x_values, cumulative_unlocks, 
               color='lightgreen', linewidth=0.8, alpha=0.4,
               label='Raw Data')
        
        # Marker per primo unlock
        if first_unlock_ep is not None and first_unlock_ep < len(x_values):
            first_unlock_x = x_values[first_unlock_ep]
            first_unlock_y = cumulative_unlocks[first_unlock_ep]
            ax.plot(first_unlock_x, first_unlock_y, 
                   marker='*', markersize=15, color='gold',
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'First Unlock (Ep {first_unlock_ep})')
        
        # Formattazione
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Unlocks', fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curve: {achievement_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        
        # Aggiungi statistiche in alto a destra
        stats_text = f'Total Unlocks: {total_unlocks}\n'
        stats_text += f'Episodes: {self.total_episodes}\n'
        if use_steps:
            stats_text += f'Steps: {self.total_steps:,}'
        
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Salva
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'learning_curve_{achievement_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_achievements(self, output_dir: str = None, window: int = 10, 
                            use_steps: bool = False, only_unlocked: bool = True):
        """
        Genera learning curves per tutti gli achievements.
        
        Args:
            output_dir: Directory di output
            window: Finestra per media mobile
            use_steps: Se True usa steps invece di episodes
            only_unlocked: Se True plotta solo achievements sbloccati
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(self.json_path.stem + "_learning_curves")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Generating Achievement Learning Curves")
        print(f"{'='*80}")
        print(f"Output directory: {output_path}")
        print(f"Window size: {window}")
        print(f"X-axis: {'Steps' if use_steps else 'Episodes'}")
        print(f"Filter: {'Unlocked only' if only_unlocked else 'All achievements'}")
        print(f"{'='*80}\n")
        
        plotted_count = 0
        skipped_count = 0
        
        for achievement_id in range(22):
            achievement_name = ACHIEVEMENT_ID_TO_NAME.get(achievement_id, f"Unknown_{achievement_id}")
            
            # Check se achievement è stato sbloccato
            ach_stats = next((s for s in self.per_achievement_stats 
                             if s.get('achievement_id') == achievement_id), {})
            is_unlocked = ach_stats.get('unlocked', False)
            
            if only_unlocked and not is_unlocked:
                skipped_count += 1
                continue
            
            print(f"Plotting {achievement_id+1:2d}/22: {achievement_name:<25s} ", end='')
            
            try:
                self.plot_achievement_learning_curve(
                    achievement_id, 
                    output_dir=output_path,
                    window=window,
                    use_steps=use_steps
                )
                plotted_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n{'='*80}")
        print(f"Plotting Complete!")
        print(f"  Plotted: {plotted_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total:   {plotted_count + skipped_count}")
        print(f"{'='*80}\n")
    
    def plot_achievement_grid(self, output_file: str = None, only_unlocked: bool = True,
                             figsize: Tuple[int, int] = (20, 16)):
        """
        Genera una griglia con tutte le learning curves in un unico grafico.
        
        Args:
            output_file: Path del file di output
            only_unlocked: Se True mostra solo achievements sbloccati
            figsize: Dimensioni della figura
        """
        # Filtra achievements
        if only_unlocked:
            achievements_to_plot = [
                ach for ach in self.per_achievement_stats 
                if ach.get('unlocked', False)
            ]
        else:
            achievements_to_plot = self.per_achievement_stats
        
        if not achievements_to_plot:
            print("⚠️  No achievements to plot!")
            return
        
        n_achievements = len(achievements_to_plot)
        
        # Calcola layout griglia
        n_cols = 4
        n_rows = (n_achievements + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, ach_stats in enumerate(achievements_to_plot):
            achievement_id = ach_stats['achievement_id']
            achievement_name = ach_stats['achievement_name']
            
            ax = axes[idx]
            
            # Estrai dati
            cumulative_unlocks = [ep[achievement_id] for ep in self.cumulative_matrix]
            episodes = np.arange(len(cumulative_unlocks))
            
            # Plot
            ax.plot(episodes, cumulative_unlocks, 
                   color='green', linewidth=1.5, alpha=0.7)
            ax.fill_between(episodes, cumulative_unlocks, 
                           alpha=0.3, color='green')
            
            # Formattazione
            ax.set_title(achievement_name.replace('_', ' ').title(), 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('Episode', fontsize=8)
            ax.set_ylabel('Cumulative', fontsize=8)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)
            
            # Aggiungi totale
            total = ach_stats.get('unlock_count', 0)
            ax.text(0.95, 0.95, f'Total: {total}',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Nascondi assi inutilizzati
        for idx in range(n_achievements, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Achievement Learning Curves - {self.json_path.stem}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Salva
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved grid: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate learning curves for Crafter achievements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate curves for all unlocked achievements
  python achievement_learning_curves.py baseline_crafter_dqn_evaluation.json
  
  # Specify output directory
  python achievement_learning_curves.py heron_evaluation.json --output-dir plots/heron/
  
  # Use training steps instead of episodes
  python achievement_learning_curves.py data.json --use-steps
  
  # Generate grid view
  python achievement_learning_curves.py data.json --grid --grid-output grid.png
  
  # Plot all achievements (including never unlocked)
  python achievement_learning_curves.py data.json --all
        """
    )
    
    parser.add_argument('json_file', help='Path to evaluation JSON file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--window', '-w', type=int, default=10, 
                       help='Window size for moving average (default: 10)')
    parser.add_argument('--use-steps', action='store_true',
                       help='Use training steps instead of episodes on X-axis')
    parser.add_argument('--all', action='store_true',
                       help='Plot all achievements (including never unlocked)')
    parser.add_argument('--grid', action='store_true',
                       help='Generate grid view with all achievements')
    parser.add_argument('--grid-output', help='Output file for grid view')
    
    args = parser.parse_args()
    
    try:
        plotter = AchievementLearningCurvePlotter(args.json_file)
        
        if args.grid:
            grid_output = args.grid_output or f"{Path(args.json_file).stem}_achievement_grid.png"
            plotter.plot_achievement_grid(
                output_file=grid_output,
                only_unlocked=not args.all
            )
        else:
            plotter.plot_all_achievements(
                output_dir=args.output_dir,
                window=args.window,
                use_steps=args.use_steps,
                only_unlocked=not args.all
            )
    
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


if __name__ == "__main__":
    main()
