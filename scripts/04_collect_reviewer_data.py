"""
Step 3: Collect Reviewer Training Data

Obiettivo: Creare un dataset specifico per l'ambiente Crafter che contenga:
- Stati dell'ambiente (scenari di gioco)
- Azioni ideali per quello stato
- Istruzioni/feedback per il Reviewer

Questo script genera dati sintetici per l'addestramento RL del Reviewer,
similmente a dataset_generation.py del codice originale HeRoN.

Usage (PowerShell):
  conda activate ia2025
  python scripts/04_collect_reviewer_data.py --episodes 5000 --output data/reviewer_dataset.csv
"""

import argparse
import os
import random
import json


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser("Generate Crafter Reviewer training dataset")
    p.add_argument(
        "--episodes", type=int, default=5000, help="Number of scenarios to generate"
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/reviewer_dataset.jsonl",
        help="Output JSONL file (one JSON record per line)",
    )
    return p.parse_args()


def ensure_dir_for_file(path: str):
    """Create directory for file if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def generate_crafter_scenario() -> dict:
    """
    Generate a random Crafter game scenario.

    Returns:
        Dictionary with game state information
    """
    # Random health/food/drink (1-9)
    health = random.randint(1, 9)
    food = random.randint(1, 9)
    drink = random.randint(1, 9)
    energy = random.randint(1, 9)

    # Random inventory
    possible_items = ["wood", "stone", "coal", "iron", "diamond", "sapling"]
    inventory = {}
    for item in possible_items:
        if random.random() < 0.3:  # 30% chance to have item
            inventory[item] = random.randint(1, 10)

    # Random nearby objects
    possible_objects = [
        "tree",
        "stone",
        "water",
        "cow",
        "zombie",
        "skeleton",
        "furnace",
        "table",
    ]
    nearby_objects = []
    for obj in possible_objects:
        if random.random() < 0.4:  # 40% chance
            nearby_objects.append(obj)

    # Random achievements progress
    achievements = {
        "collect_wood": random.randint(0, 5),
        "collect_stone": random.randint(0, 5),
        "eat_plant": random.randint(0, 3),
        "drink": random.randint(0, 3),
        "make_wood_pickaxe": random.randint(0, 1),
        "place_stone": random.randint(0, 3),
    }

    return {
        "health": health,
        "food": food,
        "drink": drink,
        "energy": energy,
        "inventory": inventory,
        "nearby_objects": nearby_objects,
        "achievements": achievements,
    }


def generate_ideal_action(scenario: dict) -> str:
    """
    Generate ideal action for the scenario.

    Returns:
        Ideal action string
    """
    # Simple rule-based ideal action generation
    health, food, drink = scenario["health"], scenario["food"], scenario["drink"]
    nearby = scenario["nearby_objects"]
    inventory = scenario["inventory"]

    # Priority: survival
    if health <= 3:
        if "cow" in nearby and food > 0:
            return "eat_cow"
        elif food > 0:
            return "eat_plant"
        return "noop"

    if drink <= 3 and "water" in nearby:
        return "drink"

    if food <= 3 and "cow" in nearby:
        return "eat_cow"
    elif food <= 3 and "plant" in nearby:
        return "eat_plant"

    # Gather resources
    if "tree" in nearby and inventory.get("wood", 0) < 5:
        return "do"  # gather wood

    if "stone" in nearby and inventory.get("stone", 0) < 5:
        return "do"  # gather stone

    # Craft tools
    if inventory.get("wood", 0) >= 2 and not scenario["achievements"].get(
        "make_wood_pickaxe", 0
    ):
        return "make_wood_pickaxe"

    # Build
    if inventory.get("stone", 0) >= 1:
        return "place_stone"

    # Default movement
    return "move_right"


def generate_instructions(scenario: dict, response: str) -> str:
    """
    Generate instructions/feedback for the Reviewer.

    Args:
        scenario: Game scenario
        response: Suggested action

    Returns:
        Instructions string
    """
    ideal = generate_ideal_action(scenario)

    if response == ideal:
        return "Correct action for the current state. Good job!"
    else:
        return f"The suggested action '{response}' is not optimal. The better action would be '{ideal}' because it addresses the current priorities in the game state."


def generate_dataset(n: int) -> list:
    """
    Generate dataset of n scenarios.

    Args:
        n: Number of scenarios to generate

    Returns:
        List of dataset records
    """
    dataset = []

    for i in range(n):
        scenario = generate_crafter_scenario()

        # Format scenario as text
        scenario_text = f"""
Current State:
- Health: {scenario['health']}
- Food: {scenario['food']} 
- Drink: {scenario['drink']}
- Energy: {scenario['energy']}
- Inventory: {scenario['inventory']}
- Nearby objects: {scenario['nearby_objects']}
- Achievements: {scenario['achievements']}

What is the next best action to take?
"""

        response = generate_ideal_action(scenario)
        instructions = generate_instructions(scenario, response)

        record = {
            "prompt": scenario_text.strip(),
            "response": response,
            "instructions": instructions,
        }

        dataset.append(record)

    return dataset


def save_dataset_to_jsonl(dataset: list, filename: str):
    """
    Save dataset to JSONL file (one JSON object per line).

    Args:
        dataset: List of records
        filename: Output filename
    """
    if not dataset:
        print("No data to save")
        return

    with open(filename, "w", encoding="utf-8") as fh:
        for rec in dataset:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {filename} with {len(dataset)} records")


def main():
    """Main data collection function."""
    args = parse_args()

    print("Generating Crafter Reviewer dataset...")
    print(f"Target scenarios: {args.episodes}")

    # Generate dataset
    dataset = generate_dataset(args.episodes)

    # Convert to training schema expected by Reviewer fine-tuning
    # Each record will have: state (dict), suggested_actions (list), review_feedback (dict)
    records = []
    for rec in dataset:
        # rec currently has keys: 'prompt', 'response', 'instructions'
        state = {"prompt": rec["prompt"]}
        suggested = [rec["response"]] if rec.get("response") else []
        feedback = {
            "rating": 10,
            "comment": rec.get("instructions", ""),
            "improved_actions": suggested,
        }

        records.append(
            {
                "state": state,
                "suggested_actions": suggested,
                "review_feedback": feedback,
            }
        )

    # Save to JSONL
    ensure_dir_for_file(args.output)
    save_dataset_to_jsonl(records, args.output)

    print("\nDataset generation completed!")
    print(
        f"Next step: Run script 05 to train the Reviewer with RL using: python scripts/05_finetune_reviewer.py --input {args.output} --output_dir models/reviewer_finetuned --max_examples 1000"
    )


if __name__ == "__main__":
    main()
