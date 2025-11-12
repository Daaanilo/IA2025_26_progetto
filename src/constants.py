"""
Project-wide constants (achievement list and related helpers)
"""

from typing import List, Dict

# Canonical list of 22 Crafter achievements (snake_case keys)
ACHIEVEMENTS: List[str] = [
    "collect_wood",
    "collect_stone",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "collect_sapling",
    "collect_drink",
    "eat_plant",
    "eat_cow",
    "defeat_zombie",
    "defeat_skeleton",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "wake_up",
]

NUM_ACHIEVEMENTS: int = len(ACHIEVEMENTS)

# Optional mapping to human-readable titles
ACHIEVEMENTS_HUMAN: Dict[str, str] = {
    k: k.replace("_", " ").title() for k in ACHIEVEMENTS
}
