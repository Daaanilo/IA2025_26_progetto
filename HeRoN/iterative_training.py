"""
F09: Iterative Dataset Refinement Cycle
Loop: Dataset Generation → Reviewer Fine-Tuning → HeRoN Training → Evaluation → Repeat
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class IterativeTrainingCycle:
    """
    Manages iterative refinement of HeRoN system through dataset-reviewer-training loops.
    
    Workflow:
    1. Generate training dataset using current Reviewer
    2. Fine-tune Reviewer on new dataset
    3. Train HeRoN with updated Reviewer
    4. Evaluate performance
    5. Repeat for N iterations
    """
    
    def __init__(self, num_iterations=3, episodes_per_iteration=100, output_dir="iterative_results"):
        self.num_iterations = num_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths
        self.dataset_script = os.path.join("dataset Reviewer", "crafter_dataset_generation.py")
        self.finetuning_script = os.path.join("reviewer", "reviewer_fine_tuning.py")
        self.training_script = os.path.join("HeRoN", "run_iterative_training.py")
        
    def run_iteration(self, iteration):
        """Execute one complete iteration of the refinement cycle."""
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{self.num_iterations}")
        print(f"{'='*80}\n")
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        iteration_results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        # ===== STAGE 1: Dataset Generation =====
        print(f"\n[Stage 1/{4}] Generating training dataset...")
        dataset_status = self._generate_dataset(iteration, iteration_dir)
        iteration_results['stages']['dataset_generation'] = dataset_status
        
        if not dataset_status['success']:
            print(f"[ERROR] Dataset generation failed. Skipping iteration {iteration}")
            return iteration_results
        
        # ===== STAGE 2: Reviewer Fine-Tuning =====
        print(f"\n[Stage 2/{4}] Fine-tuning Reviewer model...")
        finetuning_status = self._finetune_reviewer(iteration, iteration_dir)
        iteration_results['stages']['reviewer_finetuning'] = finetuning_status
        
        if not finetuning_status['success']:
            print(f"[ERROR] Reviewer fine-tuning failed. Skipping iteration {iteration}")
            return iteration_results
        
        # ===== STAGE 3: HeRoN Training =====
        print(f"\n[Stage 3/{4}] Training HeRoN with updated Reviewer...")
        training_status = self._train_heron(iteration, iteration_dir)
        iteration_results['stages']['heron_training'] = training_status
        
        if not training_status['success']:
            print(f"[ERROR] HeRoN training failed. Skipping iteration {iteration}")
            return iteration_results
        
        # ===== STAGE 4: Evaluation =====
        print(f"\n[Stage 4/{4}] Evaluating performance...")
        evaluation_status = self._evaluate_performance(iteration, iteration_dir)
        iteration_results['stages']['evaluation'] = evaluation_status
        
        # Save iteration results
        results_path = os.path.join(iteration_dir, "iteration_results.json")
        with open(results_path, 'w') as f:
            json.dump(iteration_results, f, indent=2)
        
        print(f"\n[Iteration {iteration + 1}] Complete!")
        print(f"Results saved to: {results_path}")
        
        return iteration_results
    
    def _generate_dataset(self, iteration, output_dir):
        """Generate training dataset using crafter_dataset_generation.py"""
        
        dataset_output = os.path.join(output_dir, f"dataset_iter{iteration}.csv")
        
        try:
            # Run dataset generation script
            cmd = [
                sys.executable,
                self.dataset_script,
                "--episodes", str(50),  # Generate 50 episodes
                "--output", dataset_output
            ]
            
            print(f"[Dataset] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # Check if dataset file exists
                if os.path.exists(dataset_output):
                    import pandas as pd
                    df = pd.read_csv(dataset_output)
                    
                    return {
                        'success': True,
                        'dataset_path': dataset_output,
                        'num_samples': len(df),
                        'avg_quality_score': df['quality_score'].mean() if 'quality_score' in df.columns else None
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Dataset file not created'
                    }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Dataset generation timeout (>1 hour)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _finetune_reviewer(self, iteration, output_dir):
        """Fine-tune Reviewer model on new dataset"""
        
        dataset_path = os.path.join(output_dir, f"dataset_iter{iteration}.csv")
        model_output = os.path.join(output_dir, f"reviewer_model_iter{iteration}")
        
        try:
            # Run fine-tuning script
            cmd = [
                sys.executable,
                self.finetuning_script,
                "--dataset", dataset_path,
                "--output", model_output,
                "--epochs", "3",
                "--batch_size", "8",
                "--learning_rate", "5e-5"
            ]
            
            print(f"[Fine-tuning] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'model_path': model_output,
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Fine-tuning timeout (>2 hours)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_heron(self, iteration, output_dir):
        """Train HeRoN with updated Reviewer model"""
        
        reviewer_model_path = os.path.join(output_dir, f"reviewer_model_iter{iteration}")
        training_output = os.path.join(output_dir, "training_results")
        
        try:
            # Run HeRoN training script
            cmd = [
                sys.executable,
                self.training_script,
                "--episodes", str(self.episodes_per_iteration),
                "--reviewer_model", reviewer_model_path,
                "--output_dir", training_output,
                "--checkpoint_interval", "10"
            ]
            
            print(f"[Training] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            
            if result.returncode == 0:
                # Load training metrics
                metrics_path = os.path.join(training_output, "heron_crafter_extended_metrics.csv")
                if os.path.exists(metrics_path):
                    import pandas as pd
                    df = pd.read_csv(metrics_path)
                    
                    return {
                        'success': True,
                        'output_dir': training_output,
                        'avg_shaped_reward': df['shaped_reward'].mean(),
                        'avg_native_reward': df['native_reward'].mean(),
                        'total_achievements': df['achievements_unlocked'].sum(),
                        'avg_achievements_per_episode': df['achievements_unlocked'].mean()
                    }
                else:
                    return {
                        'success': True,
                        'output_dir': training_output,
                        'warning': 'Metrics file not found'
                    }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Training timeout (>4 hours)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_performance(self, iteration, output_dir):
        """Evaluate HeRoN performance and compare with previous iterations"""
        
        training_output = os.path.join(output_dir, "training_results")
        metrics_path = os.path.join(training_output, "heron_crafter_extended_metrics.csv")
        
        if not os.path.exists(metrics_path):
            return {
                'success': False,
                'error': 'Metrics file not found'
            }
        
        try:
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv(metrics_path)
            
            evaluation = {
                'success': True,
                'metrics': {
                    'avg_shaped_reward': float(df['shaped_reward'].mean()),
                    'std_shaped_reward': float(df['shaped_reward'].std()),
                    'avg_native_reward': float(df['native_reward'].mean()),
                    'total_achievements': int(df['achievements_unlocked'].sum()),
                    'avg_achievements_per_episode': float(df['achievements_unlocked'].mean()),
                    'total_helper_calls': int(df['helper_calls'].sum()),
                    'avg_hallucination_rate': float(df['hallucination_rate'].mean()),
                    'final_10_episodes_avg_reward': float(df['shaped_reward'].tail(10).mean())
                }
            }
            
            # Compare with previous iteration
            if iteration > 0:
                prev_iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration - 1}")
                prev_results_path = os.path.join(prev_iteration_dir, "iteration_results.json")
                
                if os.path.exists(prev_results_path):
                    with open(prev_results_path, 'r') as f:
                        prev_results = json.load(f)
                    
                    prev_metrics = prev_results['stages']['evaluation']['metrics']
                    
                    evaluation['comparison'] = {
                        'reward_improvement': evaluation['metrics']['avg_shaped_reward'] - prev_metrics['avg_shaped_reward'],
                        'achievement_improvement': evaluation['metrics']['total_achievements'] - prev_metrics['total_achievements'],
                        'hallucination_improvement': prev_metrics['avg_hallucination_rate'] - evaluation['metrics']['avg_hallucination_rate']
                    }
            
            return evaluation
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_full_cycle(self):
        """Execute complete iterative refinement cycle."""
        
        print(f"\n{'='*80}")
        print(f"ITERATIVE TRAINING CYCLE")
        print(f"Iterations: {self.num_iterations}")
        print(f"Episodes per iteration: {self.episodes_per_iteration}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        for iteration in range(self.num_iterations):
            iteration_results = self.run_iteration(iteration)
            self.results.append(iteration_results)
            
            # Save cumulative results
            cumulative_path = os.path.join(self.output_dir, "cumulative_results.json")
            with open(cumulative_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        # Generate final comparison report
        self._generate_comparison_report()
        
        print(f"\n{'='*80}")
        print(f"ITERATIVE TRAINING COMPLETE")
        print(f"All results saved to: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _generate_comparison_report(self):
        """Generate markdown comparison report across all iterations."""
        
        report_path = os.path.join(self.output_dir, "iteration_comparison_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Iterative Training Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Iterations: {self.num_iterations}\n")
            f.write(f"Episodes per Iteration: {self.episodes_per_iteration}\n\n")
            
            # Performance summary table
            f.write("## Performance Summary\n\n")
            f.write("| Iteration | Avg Shaped Reward | Total Achievements | Avg Hallucination Rate |\n")
            f.write("|-----------|-------------------|--------------------|-----------------------|\n")
            
            for result in self.results:
                if result['stages'].get('evaluation', {}).get('success'):
                    metrics = result['stages']['evaluation']['metrics']
                    f.write(f"| {result['iteration'] + 1} | "
                           f"{metrics['avg_shaped_reward']:.2f} | "
                           f"{metrics['total_achievements']} | "
                           f"{metrics['avg_hallucination_rate']:.3f} |\n")
            
            # Improvement analysis
            f.write("\n## Improvement Analysis\n\n")
            
            for i, result in enumerate(self.results):
                if i == 0:
                    continue
                
                if result['stages'].get('evaluation', {}).get('comparison'):
                    comp = result['stages']['evaluation']['comparison']
                    f.write(f"### Iteration {i} → {i + 1}\n\n")
                    f.write(f"- Reward Improvement: {comp['reward_improvement']:+.2f}\n")
                    f.write(f"- Achievement Improvement: {comp['achievement_improvement']:+d}\n")
                    f.write(f"- Hallucination Rate Improvement: {comp['hallucination_improvement']:+.3f}\n\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            if len(self.results) >= 2:
                last_result = self.results[-1]
                if last_result['stages'].get('evaluation', {}).get('comparison'):
                    comp = last_result['stages']['evaluation']['comparison']
                    
                    if comp['reward_improvement'] > 0:
                        f.write("- ✅ Reward is improving - continue iterative refinement\n")
                    else:
                        f.write("- ⚠️ Reward is decreasing - consider adjusting hyperparameters\n")
                    
                    if comp['achievement_improvement'] > 0:
                        f.write("- ✅ Achievement count is improving\n")
                    else:
                        f.write("- ⚠️ Achievement count is stagnant - consider curriculum adjustment\n")
                    
                    if comp['hallucination_improvement'] > 0:
                        f.write("- ✅ Hallucination rate is decreasing - Reviewer is improving\n")
                    else:
                        f.write("- ⚠️ Hallucination rate is increasing - review dataset quality\n")
        
        print(f"\n[Report] Comparison report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="F09: Iterative Training Cycle for HeRoN")
    parser.add_argument("--iterations", type=int, default=3, help="Number of refinement iterations")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per iteration")
    parser.add_argument("--output_dir", type=str, default="iterative_results", help="Output directory")
    
    args = parser.parse_args()
    
    cycle = IterativeTrainingCycle(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        output_dir=args.output_dir
    )
    
    cycle.run_full_cycle()
