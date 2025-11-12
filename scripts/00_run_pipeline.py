"""
Headless pipeline validator and lightweight executor.

This helper script validates the linear pipeline defined in `01_pipeline_manager.py` by
checking for expected outputs of each step and (optionally) executing lightweight
variants of the steps to produce test artifacts. It is intentionally conservative: by
default it only performs a dry-run and will not launch heavy training jobs unless
`--execute --force-heavy` are provided.

Usage (PowerShell):
  # Dry-run validation only
  python scripts/00_run_pipeline.py

  # Execute lightweight versions of missing steps (safe, short runs)
  python scripts/00_run_pipeline.py --execute

  # Execute including heavy training steps (USE WITH CAUTION)
  python scripts/00_run_pipeline.py --execute --force-heavy
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    {
        "id": 1,
        "name": "Train NPC Baseline",
        "script": "02_train_npc_baseline.py",
        "outputs": ["models/baseline/npc_baseline_best.pth"],
        "heavy": True,
        "light_args": [
            "--episodes",
            "2",
            "--save_dir",
            "models/baseline_test",
            "--eval_freq",
            "1",
            "--save_freq",
            "1",
        ],
    },
    {
        "id": 2,
        "name": "Optimize Helper Strategy",
        "script": "03_optimize_helper_strategy.py",
        "outputs": ["data/helper_analysis/*.json"],
        "heavy": False,
        "light_args": [
            "--iterations",
            "5",
            "--test_configs",
            "3",
            "5",
            "--output_dir",
            "data/helper_analysis_test",
        ],
    },
    {
        "id": 3,
        "name": "Collect Reviewer Data",
        "script": "04_collect_reviewer_data.py",
        "outputs": ["data/reviewer_dataset.jsonl"],
        "heavy": False,
        "light_args": [
            "--episodes",
            "50",
            "--output",
            "data/reviewer_dataset_test.jsonl",
        ],
    },
    {
        "id": 4,
        "name": "Fine-tune Reviewer",
        "script": "05_finetune_reviewer.py",
        "outputs": ["models/reviewer_finetuned/*"],
        "heavy": True,
        "light_args": [
            "--input",
            "data/reviewer_dataset_test.jsonl",
            "--output_dir",
            "models/reviewer_finetuned_test",
            "--max_examples",
            "20",
            "--ppo_epochs",
            "1",
            "--batch_size",
            "1",
        ],
    },
    {
        "id": 5,
        "name": "Train HeRoN Architecture",
        "script": "06_train_heron.py",
        "outputs": ["models/heron/heron_best.pth"],
        "heavy": True,
        "light_args": [
            "--episodes",
            "5",
            "--save_dir",
            "models/heron_test",
            "--eval_freq",
            "1",
            "--save_freq",
            "1",
        ],
    },
    {
        "id": 6,
        "name": "Evaluate and Compare",
        "script": "07_evaluate_and_compare.py",
        "outputs": ["data/evaluation/*.json"],
        "heavy": False,
        "light_args": [
            "--heron_model",
            "models/heron_test/heron_best.pth",
            "--baseline_model",
            "models/baseline_test/npc_baseline_best.pth",
            "--episodes",
            "3",
            "--output_dir",
            "data/evaluation_test",
        ],
    },
]


def outputs_exist(patterns):
    """Return True if at least one of the output patterns exists."""
    for p in patterns:
        matches = list(ROOT.glob(p))
        if any(matches):
            return True
    return False


def run_step(step, execute=False, force_heavy=False):
    print("\n" + "=" * 60)
    print(f"Step {step['id']}: {step['name']}")
    print("-" * 60)

    exists = outputs_exist(step["outputs"])
    print("Expected outputs patterns:")
    for p in step["outputs"]:
        print(f"  - {p}")
    print(f"Outputs found: {'YES' if exists else 'NO'}")

    if not execute:
        return exists

    if step["heavy"] and not force_heavy:
        print("This step is marked as heavy. Skipping execution without --force-heavy.")
        return exists

    # Build command
    cmd = [sys.executable, str(SCRIPTS_DIR / step["script"])]
    cmd.extend(step.get("light_args", []))

    print(f"Running (light) command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, cwd=str(ROOT))
        if res.returncode == 0:
            print("Execution finished (exit 0)")
        else:
            print(f"Execution returned exit code {res.returncode}")
    except Exception as e:
        print(f"Error running step: {e}")

    return outputs_exist(step["outputs"])


def main():
    p = argparse.ArgumentParser(
        description="Validate and optionally execute pipeline steps (lightweight)"
    )
    p.add_argument(
        "--execute", action="store_true", help="Run light versions of missing steps"
    )
    p.add_argument(
        "--force-heavy",
        action="store_true",
        help="Allow execution of heavy training steps",
    )
    args = p.parse_args()

    print("\nPIPELINE VALIDATION (linear: 01->02->03->04->05->06)")
    print("Project root:", ROOT)

    all_ok = True
    for step in STEPS:
        ok = run_step(step, execute=args.execute, force_heavy=args.force_heavy)
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Pipeline validation: all expected outputs found (or created)")
    else:
        print("⚠️  Pipeline validation: some outputs are missing")
        print(
            'Check the steps that reported "Outputs found: NO" and run this script with --execute (and --force-heavy if required) to attempt light execution.'
        )


if __name__ == "__main__":
    main()
