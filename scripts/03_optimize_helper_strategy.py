"""
Step 2: Optimize Helper Strategy and Analyze Move Sequences

Obiettivo: Preparare e adattare l'Helper (LLM zero-shot) per generare sequenze
di azioni coerenti. Analizzare il numero ottimale di mosse per chiamata strategica.

Questo script:
1. Testa diverse configurazioni dell'Helper (numero di azioni per chiamata)
2. Analizza la qualità delle strategie generate
3. Determina il numero ottimale di mosse da suggerire
4. Valida la coerenza delle sequenze di azioni

Usage (PowerShell):
  conda activate ia2025
  python scripts/03_optimize_helper_strategy.py --config configs/helper_config.yaml --iterations 100
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.llm import Helper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize Helper strategy configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/helper_config.yaml",
        help="Path to Helper configuration file",
    )
    parser.add_argument(
        "--dqn_config",
        type=str,
        default="configs/dqn_config.yaml",
        help="Path to DQN configuration file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of test iterations per configuration",
    )
    parser.add_argument(
        "--test_configs",
        nargs="+",
        type=int,
        default=[3, 5, 7, 10],
        help="Different action counts to test (e.g., 3 5 7 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/helper_analysis",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def analyze_action_coherence(actions: List[str], env) -> Dict[str, Any]:
    """
    Analyze coherence of suggested action sequence.

    Args:
        actions: List of action names
        env: Crafter environment for validation

    Returns:
        Dictionary with coherence metrics
    """
    metrics = {
        "num_actions": len(actions),
        "unique_actions": len(set(actions)),
        "diversity_ratio": len(set(actions)) / len(actions) if actions else 0,
        "valid_actions": 0,
        "action_types": {
            "movement": 0,
            "crafting": 0,
            "placement": 0,
            "combat": 0,
            "other": 0,
        },
    }

    # Categorize actions
    movement_actions = ["move_left", "move_right", "move_up", "move_down"]
    crafting_actions = [
        "make_wood_pickaxe",
        "make_stone_pickaxe",
        "make_iron_pickaxe",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
    ]
    placement_actions = ["place_stone", "place_table", "place_furnace", "place_plant"]
    combat_actions = ["do"]  # 'do' is used for attacking

    for action in actions:
        # Check if action is valid
        if action in env.action_names:
            metrics["valid_actions"] += 1

        # Categorize
        if action in movement_actions:
            metrics["action_types"]["movement"] += 1
        elif action in crafting_actions:
            metrics["action_types"]["crafting"] += 1
        elif action in placement_actions:
            metrics["action_types"]["placement"] += 1
        elif action in combat_actions:
            metrics["action_types"]["combat"] += 1
        else:
            metrics["action_types"]["other"] += 1

    metrics["validity_ratio"] = (
        metrics["valid_actions"] / len(actions) if actions else 0
    )

    return metrics


def test_helper_configuration(
    helper: Helper, env, num_actions: int, iterations: int
) -> Dict[str, Any]:
    """
    Test Helper with a specific action count configuration.

    Args:
        helper: Helper instance
        env: Crafter environment
        num_actions: Number of actions to request per call
        iterations: Number of test iterations

    Returns:
        Dictionary with test results
    """
    print(f"\nTesting configuration: {num_actions} actions per call")

    results = {
        "num_actions": num_actions,
        "iterations": iterations,
        "successful_calls": 0,
        "failed_calls": 0,
        "coherence_scores": [],
        "validity_ratios": [],
        "diversity_ratios": [],
        "action_type_distributions": [],
        "generation_times": [],
    }

    for i in range(iterations):
        # Reset environment to get a fresh state
        obs, _ = env.reset()
        state_info = env.get_state_description()

        # Generate random recent actions for context
        recent_actions = []
        if i > 0:
            num_recent = min(i, 10)
            recent_actions = [
                env.action_names[np.random.randint(0, len(env.action_names))]
                for _ in range(num_recent)
            ]

        # Call Helper
        try:
            start_time = datetime.now()
            suggested_actions = helper.suggest_actions(
                state_info=state_info,
                recent_actions=recent_actions,
                num_actions=num_actions,
            )
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            results["successful_calls"] += 1
            results["generation_times"].append(generation_time)

            # Analyze coherence
            coherence = analyze_action_coherence(suggested_actions, env)
            results["coherence_scores"].append(coherence)
            results["validity_ratios"].append(coherence["validity_ratio"])
            results["diversity_ratios"].append(coherence["diversity_ratio"])
            results["action_type_distributions"].append(coherence["action_types"])

            if (i + 1) % 10 == 0:
                print(
                    f"  Progress: {i+1}/{iterations} | "
                    f"Valid: {coherence['validity_ratio']:.2%} | "
                    f"Diverse: {coherence['diversity_ratio']:.2%} | "
                    f"Time: {generation_time:.2f}s"
                )

        except Exception as e:
            print(f"  Error at iteration {i+1}: {e}")
            results["failed_calls"] += 1

    # Calculate summary statistics
    if results["validity_ratios"]:
        results["mean_validity"] = np.mean(results["validity_ratios"])
        results["mean_diversity"] = np.mean(results["diversity_ratios"])
        results["mean_generation_time"] = np.mean(results["generation_times"])
        results["std_generation_time"] = np.std(results["generation_times"])

        # Average action type distribution
        action_types = ["movement", "crafting", "placement", "combat", "other"]
        avg_distribution = {}
        for action_type in action_types:
            counts = [d[action_type] for d in results["action_type_distributions"]]
            avg_distribution[action_type] = np.mean(counts)
        results["avg_action_distribution"] = avg_distribution

    return results


def main():
    """Main optimization loop."""
    args = parse_args()

    # Load configurations
    print("Loading configurations...")
    with open(args.config, "r") as f:
        helper_config = yaml.safe_load(f)

    with open(args.dqn_config, "r") as f:
        dqn_config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 2: OPTIMIZE HELPER STRATEGY")
    print("=" * 60)

    # Initialize environment
    print("\nCreating Crafter environment...")
    env = make_crafter_env(dqn_config.get("environment"))

    # Initialize Helper
    print("Initializing Helper LLM...")
    helper = Helper(helper_config)

    # Test different configurations
    print(f"\nTesting {len(args.test_configs)} different configurations:")
    print(f"Action counts to test: {args.test_configs}")
    print(f"Iterations per configuration: {args.iterations}")

    all_results = []

    for num_actions in args.test_configs:
        results = test_helper_configuration(
            helper=helper, env=env, num_actions=num_actions, iterations=args.iterations
        )
        all_results.append(results)

        # Print summary
        print(f"\nConfiguration Summary ({num_actions} actions):")
        print(f"  Success Rate: {results['successful_calls']}/{args.iterations}")
        if results["validity_ratios"]:
            print(f"  Mean Validity: {results['mean_validity']:.2%}")
            print(f"  Mean Diversity: {results['mean_diversity']:.2%}")
            print(
                f"  Mean Generation Time: {results['mean_generation_time']:.2f}s ± {results['std_generation_time']:.2f}s"
            )
            print("  Action Distribution:")
            for action_type, count in results["avg_action_distribution"].items():
                print(f"    {action_type}: {count:.2f} actions")

    # Determine optimal configuration
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    # Find best configuration based on validity and diversity
    valid_results = [r for r in all_results if r.get("mean_validity")]

    if valid_results:
        # Score based on validity (70%) and diversity (30%)
        for r in valid_results:
            r["score"] = 0.7 * r["mean_validity"] + 0.3 * r["mean_diversity"]

        best_config = max(valid_results, key=lambda x: x["score"])

        print(f"\nOptimal Configuration: {best_config['num_actions']} actions per call")
        print(f"  Score: {best_config['score']:.4f}")
        print(f"  Validity: {best_config['mean_validity']:.2%}")
        print(f"  Diversity: {best_config['mean_diversity']:.2%}")
        print(f"  Generation Time: {best_config['mean_generation_time']:.2f}s")

        print("\nRecommendations:")
        print(
            f"1. Set 'num_actions_per_call' to {best_config['num_actions']} in helper_config.yaml"
        )
        print(
            f"2. For two strategic calls per episode, use {best_config['num_actions']} x 2 = {best_config['num_actions']*2} total actions"
        )

        # Calculate optimal query frequency
        avg_episode_length = dqn_config["training"].get("max_steps_per_episode", 10000)
        recommended_freq = avg_episode_length // 2  # Two calls per episode
        print(f"3. Recommended helper_query_freq: {recommended_freq} steps")

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f'helper_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "test_configs": args.test_configs,
                "iterations": args.iterations,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 60)
    print("HELPER OPTIMIZATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
