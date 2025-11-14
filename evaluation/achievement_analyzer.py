"""
Achievement Analyzer - Tool to visualize and analyze achievements from training runs

Usage:
    python achievement_analyzer.py <evaluation_json_file>
    python achievement_analyzer.py baseline_crafter_dqn_evaluation.json
    python achievement_analyzer.py heron_crafter_evaluation.json --compare baseline_crafter_dqn_evaluation.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class AchievementAnalyzer:
    """Analyzes achievement data from evaluation JSON files."""
    
    def __init__(self, json_path: str):
        """Load evaluation JSON file."""
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.achievement_stats = self.data.get('achievement_statistics', {})
        self.summary_stats = self.data.get('summary_statistics', {})
    
    def print_summary(self):
        """Print a comprehensive summary of achievements."""
        print("\n" + "="*80)
        print(f"ACHIEVEMENT ANALYSIS: {self.json_path.name}")
        print("="*80)
        
        # Overall statistics
        total_possible = self.achievement_stats.get('total_possible_achievements', 22)
        unique_unlocked = self.achievement_stats.get('unique_achievements_unlocked', 0)
        unlock_ratio = self.achievement_stats.get('unlock_ratio', 0)
        total_unlocks = self.achievement_stats.get('total_unlock_instances', 0)
        
        print(f"\n[OVERALL STATISTICS]")
        print(f"  Total Achievements Unlocked: {unique_unlocked}/{total_possible} ({unlock_ratio:.1%})")
        print(f"  Total Unlock Instances: {total_unlocks}")
        print(f"  Never Unlocked: {self.achievement_stats.get('never_unlocked', total_possible)}")
        
        # List of unlocked achievements
        unlocked_names = self.achievement_stats.get('unique_achievement_names', [])
        never_unlocked_names = self.achievement_stats.get('never_unlocked_names', [])
        
        if unlocked_names:
            print(f"\n[UNLOCKED ACHIEVEMENTS] ({len(unlocked_names)} total)")
            print("  " + "-"*76)
            for i, name in enumerate(unlocked_names, 1):
                # Find detailed stats for this achievement
                per_ach = self.achievement_stats.get('per_achievement_stats', [])
                ach_data = next((a for a in per_ach if a.get('achievement_name') == name), {})
                
                unlock_count = ach_data.get('unlock_count', 0)
                first_episode = ach_data.get('first_unlock_episode', 'N/A')
                first_move = ach_data.get('first_unlock_move', 'N/A')
                
                print(f"  {i:2d}. {name:25s} | Unlocks: {unlock_count:3d} | "
                      f"First: Ep {first_episode}, Move {first_move}")
        else:
            print(f"\n[UNLOCKED ACHIEVEMENTS]")
            print("  ⚠️  No achievements were unlocked during training!")
        
        # List of never unlocked achievements
        if never_unlocked_names:
            print(f"\n[NEVER UNLOCKED ACHIEVEMENTS] ({len(never_unlocked_names)} total)")
            print("  " + "-"*76)
            
            # Group by category for better readability
            collect_achs = [n for n in never_unlocked_names if n.startswith('collect_')]
            make_achs = [n for n in never_unlocked_names if n.startswith('make_')]
            place_achs = [n for n in never_unlocked_names if n.startswith('place_')]
            defeat_achs = [n for n in never_unlocked_names if n.startswith('defeat_')]
            eat_achs = [n for n in never_unlocked_names if n.startswith('eat_')]
            other_achs = [n for n in never_unlocked_names 
                         if not any(n.startswith(p) for p in ['collect_', 'make_', 'place_', 'defeat_', 'eat_'])]
            
            if collect_achs:
                print(f"  📦 Collection: {', '.join(collect_achs)}")
            if make_achs:
                print(f"  🔨 Crafting:   {', '.join(make_achs)}")
            if place_achs:
                print(f"  🏗️  Placement:  {', '.join(place_achs)}")
            if defeat_achs:
                print(f"  ⚔️  Combat:     {', '.join(defeat_achs)}")
            if eat_achs:
                print(f"  🍖 Survival:   {', '.join(eat_achs)}")
            if other_achs:
                print(f"  ✨ Other:      {', '.join(other_achs)}")
        
        # Episode statistics
        total_episodes = self.summary_stats.get('total_episodes', 0)
        avg_achievements = self.summary_stats.get('achievements_unlocked', {}).get('mean', 0)
        
        print(f"\n[TRAINING PERFORMANCE]")
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Avg Achievements per Episode: {avg_achievements:.2f}")
        print(f"  Total Achievement Unlocks: {self.summary_stats.get('achievements_unlocked', {}).get('total', 0)}")
        
        print("\n" + "="*80 + "\n")
    
    def export_detailed_report(self, output_path: str = None):
        """Export detailed achievement report to text file."""
        if output_path is None:
            output_path = self.json_path.stem + "_achievement_report.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Redirect print to file
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_summary()
            
            # Add per-episode breakdown if available
            achievements_per_episode = self.achievement_stats.get('achievements_per_episode', [])
            if achievements_per_episode:
                print("[PER-EPISODE ACHIEVEMENT BREAKDOWN]")
                print("-"*80)
                for ep_idx, achievements in enumerate(achievements_per_episode):
                    if achievements:
                        print(f"  Episode {ep_idx}: {', '.join(achievements)}")
                print("-"*80 + "\n")
            
            sys.stdout = original_stdout
        
        print(f"✓ Detailed report saved to: {output_path}")
    
    def get_unlocked_names(self) -> List[str]:
        """Return list of unlocked achievement names."""
        return self.achievement_stats.get('unique_achievement_names', [])
    
    def get_never_unlocked_names(self) -> List[str]:
        """Return list of never unlocked achievement names."""
        return self.achievement_stats.get('never_unlocked_names', [])


def compare_achievements(file1: str, file2: str):
    """Compare achievements between two training runs."""
    analyzer1 = AchievementAnalyzer(file1)
    analyzer2 = AchievementAnalyzer(file2)
    
    unlocked1 = set(analyzer1.get_unlocked_names())
    unlocked2 = set(analyzer2.get_unlocked_names())
    
    print("\n" + "="*80)
    print(f"ACHIEVEMENT COMPARISON")
    print("="*80)
    print(f"File 1: {Path(file1).name}")
    print(f"File 2: {Path(file2).name}")
    print("-"*80)
    
    # Statistics comparison
    stats1 = analyzer1.achievement_stats
    stats2 = analyzer2.achievement_stats
    
    print(f"\n{'Metric':<40} {'File 1':>15} {'File 2':>15} {'Diff':>10}")
    print("-"*80)
    print(f"{'Unique Achievements Unlocked':<40} {stats1.get('unique_achievements_unlocked', 0):>15d} "
          f"{stats2.get('unique_achievements_unlocked', 0):>15d} "
          f"{stats2.get('unique_achievements_unlocked', 0) - stats1.get('unique_achievements_unlocked', 0):>+10d}")
    print(f"{'Total Unlock Instances':<40} {stats1.get('total_unlock_instances', 0):>15d} "
          f"{stats2.get('total_unlock_instances', 0):>15d} "
          f"{stats2.get('total_unlock_instances', 0) - stats1.get('total_unlock_instances', 0):>+10d}")
    print(f"{'Unlock Ratio':<40} {stats1.get('unlock_ratio', 0):>14.1%} "
          f"{stats2.get('unlock_ratio', 0):>14.1%} "
          f"{(stats2.get('unlock_ratio', 0) - stats1.get('unlock_ratio', 0))*100:>+9.1f}%")
    
    # Set comparisons
    only_in_1 = unlocked1 - unlocked2
    only_in_2 = unlocked2 - unlocked1
    in_both = unlocked1 & unlocked2
    
    print(f"\n[ACHIEVEMENT OVERLAP]")
    print(f"  Unlocked in both: {len(in_both)}")
    print(f"  Only in File 1:   {len(only_in_1)}")
    print(f"  Only in File 2:   {len(only_in_2)}")
    
    if in_both:
        print(f"\n  Common achievements: {', '.join(sorted(in_both))}")
    
    if only_in_1:
        print(f"\n  Only in {Path(file1).name}:")
        print(f"    {', '.join(sorted(only_in_1))}")
    
    if only_in_2:
        print(f"\n  Only in {Path(file2).name}:")
        print(f"    {', '.join(sorted(only_in_2))}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze achievements from Crafter training evaluation files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python achievement_analyzer.py baseline_crafter_dqn_evaluation.json
  python achievement_analyzer.py heron_crafter_evaluation.json --export report.txt
  python achievement_analyzer.py heron.json --compare baseline.json
        """
    )
    
    parser.add_argument('json_file', help='Path to evaluation JSON file')
    parser.add_argument('--export', metavar='FILE', help='Export detailed report to text file')
    parser.add_argument('--compare', metavar='JSON', help='Compare with another evaluation JSON')
    
    args = parser.parse_args()
    
    try:
        # Load and analyze first file
        analyzer = AchievementAnalyzer(args.json_file)
        analyzer.print_summary()
        
        # Export if requested
        if args.export:
            analyzer.export_detailed_report(args.export)
        
        # Compare if requested
        if args.compare:
            compare_achievements(args.json_file, args.compare)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
