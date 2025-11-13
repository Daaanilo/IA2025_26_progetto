"""
F10: Sistema di Valutazione - Report Generator

Synthesizes evaluation metrics into comprehensive markdown/text reports.
Generates:
- Summary tables (HeRoN vs baselines)
- Correlation analysis (helper_calls vs achievements)
- Per-configuration statistics
- Actionable recommendations based on convergence/efficiency patterns
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from evaluation_system import EvaluationSystem, BaselineComparator


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "./evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configurations: Dict[str, Dict] = {}
    
    def load_configuration(self, name: str, json_path: str):
        """Load evaluation results from JSON file."""
        with open(json_path, 'r') as f:
            self.configurations[name] = json.load(f)
        print(f"[ReportGenerator] Loaded configuration: {name}")
    
    def load_csv_metrics(self, name: str, csv_path: str) -> pd.DataFrame:
        """Load raw metrics from CSV file."""
        return pd.read_csv(csv_path)
    
    def generate_summary_report(self, eval_system: EvaluationSystem, 
                               config_name: str,
                               output_file: Optional[str] = None) -> str:
        """Generate summary report for single configuration."""
        
        if output_file is None:
            output_file = f"{self.output_dir}/{config_name}_summary_report.md"
        
        summary = eval_system.get_summary_statistics()
        achievements = eval_system.get_achievement_statistics()
        efficiency = eval_system.get_efficiency_statistics()
        convergence = eval_system.get_convergence_report()
        
        report = f"""# F10 Evaluation Report: {config_name}

## Executive Summary

**Training Configuration**: {config_name}
**Total Episodes**: {summary['total_episodes']}

### Performance Highlights

- **Final Shaped Reward**: {summary['shaped_reward']['final']:.3f}
- **Total Achievements Unlocked**: {summary['achievements_unlocked']['total']} / {achievements['total_possible_achievements']}
- **Achievement Unlock Ratio**: {achievements['unlock_ratio']:.1%}
- **Average Reward per Episode**: {summary['shaped_reward']['mean']:.3f} ± {summary['shaped_reward']['std']:.3f}
- **Average Achievements per Episode**: {summary['achievements_unlocked']['mean']:.2f}

---

## Detailed Metrics

### Reward Statistics

| Metric | Mean | Std Dev | Min | Max | Final |
|--------|------|---------|-----|-----|-------|
| Shaped Reward | {summary['shaped_reward']['mean']:.3f} | {summary['shaped_reward']['std']:.3f} | {summary['shaped_reward']['min']:.3f} | {summary['shaped_reward']['max']:.3f} | {summary['shaped_reward']['final']:.3f} |
| Native Reward | {summary['native_reward']['mean']:.3f} | {summary['native_reward']['std']:.3f} | {summary['native_reward']['min']:.3f} | {summary['native_reward']['max']:.3f} | {summary['native_reward']['final']:.3f} |

**Interpretation**: 
- Shaped reward includes bonus components for learning signal
- Native reward is sparse Crafter reward (+1 per achievement)
- Difference (shaped - native) shows impact of reward shaping

### Achievement Statistics

| Metric | Value |
|--------|-------|
| Unique Achievements Unlocked | {achievements['unique_achievements_unlocked']} / {achievements['total_possible_achievements']} |
| Unlock Ratio | {achievements['unlock_ratio']:.1%} |
| Total Unlock Instances | {achievements['total_unlock_instances']} |
| Average Unlocks per Achievement | {achievements['avg_unlocks_per_achievement']:.2f} |
| Never Unlocked | {achievements['never_unlocked']} |

**Interpretation**: Shows achievement diversity and coverage across training

### Efficiency Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Reward per Move | {efficiency['reward_per_move']['mean']:.4f} | {efficiency['reward_per_move']['std']:.4f} | {efficiency['reward_per_move']['min']:.4f} | {efficiency['reward_per_move']['max']:.4f} |
| Reward per Helper Call | {efficiency['reward_per_helper_call']['mean']:.4f} | {efficiency['reward_per_helper_call']['std']:.4f} | {efficiency['reward_per_helper_call']['min']:.4f} | {efficiency['reward_per_helper_call']['max']:.4f} |
| Achievements per Move | {efficiency['achievements_per_move']['mean']:.4f} | {efficiency['achievements_per_move']['std']:.4f} | {efficiency['achievements_per_move']['min']:.4f} | {efficiency['achievements_per_move']['max']:.4f} |

**Interpretation**: Measures learning efficiency and action quality

### Helper LLM Metrics

| Metric | Value |
|--------|-------|
| Total Helper Calls | {summary['helper_calls']['total']} |
| Average Calls per Episode | {summary['helper_calls']['mean']:.2f} |
| Total Hallucinations | {summary['hallucinations']['total']} |
| Mean Hallucination Rate | {summary['hallucinations']['mean_rate']:.1%} |

**Interpretation**: LLM reliability and usage intensity

### Movement Efficiency

| Metric | Mean | Std Dev | Total |
|--------|------|---------|-------|
| Moves per Episode | {summary['moves']['mean']:.1f} | {summary['moves']['std']:.1f} | {summary['moves']['total']} |

---

## Convergence Analysis

### Shaped Reward Convergence
- **Status**: {"✓ CONVERGED" if convergence['shaped_reward_convergence']['converged'] else "✗ NOT CONVERGED"}
- **Convergence Episode**: {convergence['shaped_reward_convergence']['converged_at_episode'] if convergence['shaped_reward_convergence']['converged'] else "N/A"}
- **Final Moving Std**: {convergence['shaped_reward_convergence']['final_moving_std']:.4f}

### Learning Trends

| Metric | Trend | Improvement Ratio |
|--------|-------|-------------------|
| Shaped Reward | {convergence['shaped_reward_trend']['trend'].upper()} | {convergence['shaped_reward_trend']['improvement_ratio']:+.2%} |
| Achievements | {convergence['achievements_trend']['trend'].upper()} | {convergence['achievements_trend']['improvement_ratio']:+.2%} |
| Helper Calls | {convergence['helper_calls_trend']['trend'].upper()} | {convergence['helper_calls_trend']['improvement_ratio']:+.2%} |

**Interpretation**:
- **Improving**: Configuration is learning; reward/achievements increasing over time
- **Stable**: Performance plateaued; further learning unlikely
- **Declining**: Performance degrading; potential training instability

---

## Recommendations

### Based on Convergence Status
"""
        
        if convergence['shaped_reward_convergence']['converged']:
            report += f"\n✓ **Shaped reward converged at episode {convergence['shaped_reward_convergence']['converged_at_episode']}**\n"
            report += "  - Training has reached stability\n"
            report += "  - Further episodes may provide diminishing returns\n"
        else:
            report += "\n⚠ **Shaped reward has not converged**\n"
            report += "  - Training could benefit from more episodes\n"
            report += "  - Consider increasing episode budget\n"
        
        if convergence['achievements_trend']['trend'] == 'improving':
            report += "\n✓ **Achievement unlocks improving**\n"
            report += "  - Agent is discovering new strategies\n"
            report += "  - Continue training to maximize coverage\n"
        else:
            report += "\n⚠ **Achievement unlocks plateaued**\n"
            report += "  - May need curriculum learning or reward shaping adjustment\n"
        
        if convergence['helper_calls_trend']['trend'] == 'declining':
            report += "\n✓ **Helper dependency decreasing**\n"
            report += "  - Agent learning to act independently\n"
            report += "  - Threshold decay strategy effective\n"
        else:
            report += "\n⚠ **Helper dependency stable or increasing**\n"
            report += "  - Agent may not be learning from LLM feedback\n"
            report += "  - Consider reviewing reward shaping or LLM prompt quality\n"
        
        report += f"\n\n---\n\n## Files Generated\n\n- Summary: `{config_name}_summary_report.md`\n"
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"[ReportGenerator] Generated summary report: {output_file}")
        return report
    
    def generate_comparison_report(self, configs: Dict[str, Dict], 
                                  output_file: Optional[str] = None) -> str:
        """Generate comparison report across multiple configurations."""
        
        if output_file is None:
            output_file = f"{self.output_dir}/comparison_report.md"
        
        report = """# F10 Comparison Report: HeRoN vs Baselines

## Configuration Overview

| Configuration | Episodes | Description |
|---------------|----------|-------------|
"""
        
        config_descriptions = {
            'heron_crafter': 'HeRoN (DQN + Helper + Threshold Decay)',
            'baseline_crafter_dqn': 'Baseline: Pure DQN (No LLM)',
            'baseline_crafter_helper': 'Baseline: Helper Only (Always-On, No DQN)'
        }
        
        for config_name, data in configs.items():
            episodes = data['summary_statistics']['total_episodes']
            desc = config_descriptions.get(config_name, config_name)
            report += f"| {config_name} | {episodes} | {desc} |\n"
        
        report += "\n---\n\n## Performance Comparison\n\n"
        
        # Create comparison table
        report += "| Metric | "
        for config_name in configs.keys():
            report += f"{config_name} | "
        report += "\n"
        
        report += "|--------|"
        for _ in configs:
            report += "----------|"
        report += "\n"
        
        # Shaped reward
        report += "| Avg Shaped Reward | "
        for config_name, data in configs.items():
            mean = data['summary_statistics']['shaped_reward']['mean']
            report += f"{mean:.3f} | "
        report += "\n"
        
        # Total achievements
        report += "| Total Achievements | "
        for config_name, data in configs.items():
            total = data['summary_statistics']['achievements_unlocked']['total']
            report += f"{total} | "
        report += "\n"
        
        # Unique achievements
        report += "| Unique Achievements | "
        for config_name, data in configs.items():
            unique = data['achievement_statistics']['unique_achievements_unlocked']
            total_possible = data['achievement_statistics']['total_possible_achievements']
            report += f"{unique}/{total_possible} | "
        report += "\n"
        
        # Avg moves per episode
        report += "| Avg Moves/Episode | "
        for config_name, data in configs.items():
            moves = data['summary_statistics']['moves']['mean']
            report += f"{moves:.1f} | "
        report += "\n"
        
        # Total helper calls
        report += "| Total Helper Calls | "
        for config_name, data in configs.items():
            helper = data['summary_statistics']['helper_calls']['total']
            report += f"{helper} | "
        report += "\n"
        
        # Hallucination rate
        report += "| Hallucination Rate | "
        for config_name, data in configs.items():
            rate = data['summary_statistics']['hallucinations']['mean_rate']
            report += f"{rate:.1%} | "
        report += "\n"
        
        # Efficiency metrics
        report += "\n### Efficiency Comparison\n\n"
        
        report += "| Metric | "
        for config_name in configs.keys():
            report += f"{config_name} | "
        report += "\n"
        
        report += "|--------|"
        for _ in configs:
            report += "----------|"
        report += "\n"
        
        # Reward per move
        report += "| Reward per Move | "
        for config_name, data in configs.items():
            rpm = data['efficiency_statistics']['reward_per_move']['mean']
            report += f"{rpm:.4f} | "
        report += "\n"
        
        # Achievements per move
        report += "| Achievements per Move | "
        for config_name, data in configs.items():
            apm = data['efficiency_statistics']['achievements_per_move']['mean']
            report += f"{apm:.4f} | "
        report += "\n"
        
        report += "\n---\n\n## Analysis\n\n"
        
        # Find best configurations
        best_reward_config = max(configs.items(), 
                                key=lambda x: x[1]['summary_statistics']['shaped_reward']['mean'])
        best_achievements_config = max(configs.items(),
                                      key=lambda x: x[1]['summary_statistics']['achievements_unlocked']['total'])
        best_efficiency_config = max(configs.items(),
                                    key=lambda x: x[1]['efficiency_statistics']['reward_per_move']['mean'])
        
        report += f"### Best Performance by Metric\n\n"
        report += f"- **Highest Avg Reward**: {best_reward_config[0]} ({best_reward_config[1]['summary_statistics']['shaped_reward']['mean']:.3f})\n"
        report += f"- **Most Achievements**: {best_achievements_config[0]} ({best_achievements_config[1]['summary_statistics']['achievements_unlocked']['total']})\n"
        report += f"- **Best Reward/Move**: {best_efficiency_config[0]} ({best_efficiency_config[1]['efficiency_statistics']['reward_per_move']['mean']:.4f})\n"
        
        report += f"\n### Key Insights\n\n"
        
        # LLM impact analysis
        if 'heron_crafter' in configs and 'baseline_crafter_dqn' in configs:
            heron_reward = configs['heron_crafter']['summary_statistics']['shaped_reward']['mean']
            dqn_reward = configs['baseline_crafter_dqn']['summary_statistics']['shaped_reward']['mean']
            improvement = ((heron_reward - dqn_reward) / dqn_reward) * 100 if dqn_reward != 0 else 0
            
            report += f"**HeRoN vs Pure DQN**: HeRoN achieves {heron_reward:.3f} vs DQN {dqn_reward:.3f} "
            report += f"({improvement:+.1f}% {'improvement' if improvement > 0 else 'regression'})\n"
        
        # Helper analysis
        if 'baseline_crafter_helper' in configs:
            helper_calls = configs['baseline_crafter_helper']['summary_statistics']['helper_calls']['total']
            helper_episodes = configs['baseline_crafter_helper']['summary_statistics']['total_episodes']
            report += f"\n**Helper-Only Baseline**: {helper_calls} calls across {helper_episodes} episodes "
            report += f"({helper_calls/max(1, helper_episodes):.1f} calls/episode)\n"
        
        report += "\n---\n\n## Recommendations\n\n"
        
        report += "1. **Best Overall Strategy**: "
        best_config = max(configs.items(), 
                         key=lambda x: (x[1]['summary_statistics']['shaped_reward']['mean'],
                                       x[1]['summary_statistics']['achievements_unlocked']['total']))
        report += f"{best_config[0]}\n"
        
        report += "\n2. **LLM Integration**: "
        if 'heron_crafter' in configs:
            heron_helper = configs['heron_crafter']['summary_statistics']['helper_calls']['mean']
            if heron_helper > 0:
                report += f"Helper provides value (avg {heron_helper:.1f} calls/episode with threshold decay)\n"
            else:
                report += "Threshold decay effective - minimal LLM usage in late training\n"
        
        report += "\n3. **Scalability**: Consider adjusting episode budget based on convergence patterns\n"
        
        report += "\n4. **Next Steps**:\n"
        report += "   - Implement Reviewer fine-tuning (F06) for better LLM feedback quality\n"
        report += "   - Analyze per-achievement unlock patterns for bottleneck identification\n"
        report += "   - Run multiple seeds for statistical significance testing\n"
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"[ReportGenerator] Generated comparison report: {output_file}")
        return report
    
    def generate_markdown_summary_tables(self, configs: Dict[str, Dict],
                                        output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate exportable summary tables in markdown."""
        
        rows = []
        
        for config_name, data in configs.items():
            summary = data['summary_statistics']
            achievements = data['achievement_statistics']
            efficiency = data['efficiency_statistics']
            
            rows.append({
                'Configuration': config_name,
                'Episodes': summary['total_episodes'],
                'Avg Reward': f"{summary['shaped_reward']['mean']:.3f}",
                'Reward Std': f"{summary['shaped_reward']['std']:.3f}",
                'Total Achievements': summary['achievements_unlocked']['total'],
                'Unique Achievements': f"{achievements['unique_achievements_unlocked']}/{achievements['total_possible_achievements']}",
                'Avg Moves': f"{summary['moves']['mean']:.1f}",
                'Total Helper Calls': summary['helper_calls']['total'],
                'Hallucination Rate': f"{summary['hallucinations']['mean_rate']:.1%}",
                'Reward/Move': f"{efficiency['reward_per_move']['mean']:.4f}",
                'Achievements/Move': f"{efficiency['achievements_per_move']['mean']:.4f}"
            })
        
        df = pd.DataFrame(rows)
        
        if output_file is None:
            output_file = f"{self.output_dir}/summary_comparison_table.csv"
        
        df.to_csv(output_file, index=False)
        print(f"[ReportGenerator] Generated summary table: {output_file}")
        
        return df


def generate_full_evaluation_report(eval_configs: Dict[str, str], 
                                   output_dir: str = "./evaluation_reports"):
    """
    Convenience function: generate all reports from configuration JSON files.
    
    Args:
        eval_configs: Dict mapping config_name -> path/to/json
        output_dir: Output directory for all reports
    
    Example:
        configs = {
            'heron_crafter': 'heron_crafter_evaluation.json',
            'baseline_crafter_dqn': 'baseline_crafter_dqn_evaluation.json',
            'baseline_crafter_helper': 'baseline_crafter_helper_evaluation.json'
        }
        generate_full_evaluation_report(configs)
    """
    
    generator = ReportGenerator(output_dir=output_dir)
    all_configs = {}
    
    # Load all configurations
    for config_name, json_path in eval_configs.items():
        try:
            with open(json_path, 'r') as f:
                all_configs[config_name] = json.load(f)
            print(f"[ReportGenerator] Loaded: {config_name}")
        except FileNotFoundError:
            print(f"[ReportGenerator] WARNING: File not found: {json_path}")
    
    # Generate individual reports
    for config_name, data in all_configs.items():
        # Create minimal EvaluationSystem object for reporting
        # (In real usage, you'd pass the actual EvaluationSystem object)
        generator.configurations[config_name] = data
    
    # Generate comparison report
    if len(all_configs) > 1:
        generator.generate_comparison_report(all_configs)
        generator.generate_markdown_summary_tables(all_configs)
    
    print(f"\n[ReportGenerator] All reports generated in {output_dir}/")
