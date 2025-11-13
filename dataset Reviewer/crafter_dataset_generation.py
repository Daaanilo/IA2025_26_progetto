"""
F05: Dataset Generation per Reviewer (Crafter Environment)
==========================================================

Episodic data collector for generating training data for Reviewer fine-tuning.
Captures: game states → Helper suggestions → outcome evaluation → strategic feedback

Components:
1. EpisodicDataCollector: Manages episode simulation and data capture
2. OutcomeEvaluator: Analyzes outcome quality (achievements, resources, efficiency)
3. FeedbackGenerator: Hand-crafted rule-based strategic feedback
4. CrafterDatasetGenerator: Orchestrates dataset generation and CSV export

Configuration:
- Episodes: 50-100 (configurable)
- Helper calls per episode: every 5 steps
- Episode length: 500 steps
- Target dataset size: 2000-5000 samples
- Success ratio: 80% achievement-unlocking, 20% exploratory
"""

import json
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# Import game classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from classes.crafter_environment import CrafterEnv
    from classes.crafter_helper import CrafterHelper
except ImportError as e:
    print(f"Warning: Could not import classes: {e}")
    print("Using mock implementations for testing")


@dataclass
class EpisodeData:
    """Container for a single Helper call data point."""
    episode_id: int
    step: int
    state_description: str
    helper_response: str
    action_sequence: Optional[List[int]]
    
    # Outcome metrics
    achievements_before: set
    achievements_after: set
    achievements_unlocked: set
    health_before: int
    health_after: int
    resources_before: Dict[str, int]
    resources_after: Dict[str, int]
    
    # Derived metrics
    quality_score: float  # 0-1 based on progress
    feedback: str


class OutcomeEvaluator:
    """Evaluates outcome quality and generates strategic feedback."""
    
    # Achievement progression tiers (for hierarchical feedback)
    ACHIEVEMENT_TIERS = {
        'tier_1_collect': ['collect_wood', 'collect_stone', 'collect_iron', 'collect_coal', 'collect_diamond'],
        'tier_2_resources': ['collect_food', 'collect_drink', 'collect_fence'],
        'tier_3_build': ['place_stone', 'place_table', 'place_furnace', 'place_plant'],
        'tier_4_craft': [
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
        ],
        'tier_5_interact': ['eat_plant', 'eat_cow', 'defeat_zombie', 'defeat_skeleton']
    }
    
    # Resource importance ranking
    RESOURCE_IMPORTANCE = {
        'health': 10,
        'wood': 7,
        'stone': 6,
        'coal': 5,
        'iron': 8,
        'diamond': 9,
        'food': 4,
        'drink': 3,
        'energy': 2
    }
    
    def __init__(self):
        self.evaluation_count = 0
        self.high_quality_count = 0
    
    def evaluate(self, episode_data: EpisodeData) -> Tuple[float, str]:
        """
        Evaluate outcome quality and generate strategic feedback.
        
        Returns:
            (quality_score: 0-1, feedback: str)
        """
        self.evaluation_count += 1
        quality_score = 0.0
        feedback_parts = []
        
        # === CRITERION 1: Achievement Unlocks (PRIMARY METRIC) ===
        if episode_data.achievements_unlocked:
            num_achievements = len(episode_data.achievements_unlocked)
            quality_score += min(0.5, num_achievements * 0.15)  # Max 0.5 for achievements
            
            achieved_names = list(episode_data.achievements_unlocked)
            feedback_parts.append(
                f"Unlocked {num_achievements} achievement(s): {', '.join(achieved_names)}. "
                f"Your sequence was strategic and effective."
            )
        else:
            feedback_parts.append("No achievements unlocked. Prioritize resource collection or tier advancement.")
        
        # === CRITERION 2: Resource Efficiency ===
        resource_efficiency = self._evaluate_resource_efficiency(
            episode_data.resources_before,
            episode_data.resources_after,
            episode_data.achievements_unlocked
        )
        quality_score += resource_efficiency[0]
        if resource_efficiency[1]:
            feedback_parts.append(resource_efficiency[1])
        
        # === CRITERION 3: Health Management ===
        if episode_data.health_after > episode_data.health_before:
            quality_score += 0.15
            feedback_parts.append(
                f"Good health management: increased from {episode_data.health_before} to {episode_data.health_after}."
            )
        elif episode_data.health_after == episode_data.health_before:
            # Neutral if no change
            pass
        else:
            health_loss = episode_data.health_before - episode_data.health_after
            if health_loss > 5:
                feedback_parts.append(
                    f"WARNING: Lost {health_loss} health. Consider defensive actions or resource gathering instead."
                )
                quality_score -= 0.1
        
        # === CRITERION 4: Achievement Tier Progression ===
        tier_advancement = self._evaluate_tier_advancement(
            episode_data.achievements_before,
            episode_data.achievements_after
        )
        if tier_advancement > 0:
            quality_score += tier_advancement
            feedback_parts.append(
                "Your actions advanced your progress through the achievement tiers. "
                "Keep building on this progression."
            )
        
        # === CRITERION 5: Coherence & Efficiency ===
        if episode_data.action_sequence:
            seq_length = len(episode_data.action_sequence)
            if 3 <= seq_length <= 5:
                quality_score += 0.1
                feedback_parts.append("Sequence length is optimal (3-5 actions).")
            elif seq_length < 3:
                feedback_parts.append("Sequence too short. Generate 3-5 coherent actions per request.")
                quality_score -= 0.05
            else:
                feedback_parts.append("Sequence exceeds optimal length. Keep sequences to 3-5 actions.")
        
        # === Normalize Quality Score ===
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Combine feedback
        final_feedback = " ".join(feedback_parts)
        
        # Track high-quality samples
        if quality_score >= 0.6:
            self.high_quality_count += 1
        
        return quality_score, final_feedback
    
    def _evaluate_resource_efficiency(
        self, 
        resources_before: Dict[str, int],
        resources_after: Dict[str, int],
        achievements_unlocked: set
    ) -> Tuple[float, Optional[str]]:
        """Evaluate resource management efficiency."""
        score = 0.0
        feedback = None
        
        # Check key resources
        critical_resources = ['wood', 'stone', 'iron']
        gained_critical = 0
        
        for resource in critical_resources:
            before = resources_before.get(resource, 0)
            after = resources_after.get(resource, 0)
            
            if after > before:
                gained_critical += 1
        
        if gained_critical >= 2:
            score += 0.15
            feedback = "Efficient resource gathering—collected multiple critical resources."
        elif gained_critical == 1:
            score += 0.08
            feedback = "Collected key resources, but could expand gathering efficiency."
        elif not achievements_unlocked:
            # If no achievements and no resources, efficiency is poor
            score -= 0.1
            feedback = "Resource efficiency is low. Prioritize gathering wood or stone."
        
        return score, feedback
    
    def _evaluate_tier_advancement(self, achievements_before: set, achievements_after: set) -> float:
        """Evaluate progression through achievement tiers."""
        score = 0.0
        
        # Check each tier
        for tier_name, achievements in self.ACHIEVEMENT_TIERS.items():
            tier_before = sum(1 for a in achievements if a in achievements_before)
            tier_after = sum(1 for a in achievements if a in achievements_after)
            
            if tier_after > tier_before:
                # Advancement in this tier
                tier_weight = {
                    'tier_1_collect': 0.15,
                    'tier_2_resources': 0.10,
                    'tier_3_build': 0.12,
                    'tier_4_craft': 0.18,
                    'tier_5_interact': 0.12
                }
                score += tier_weight.get(tier_name, 0.05)
        
        return min(0.25, score)  # Cap at 0.25
    
    def get_statistics(self) -> Dict:
        """Return evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'high_quality_samples': self.high_quality_count,
            'high_quality_ratio': (
                self.high_quality_count / max(1, self.evaluation_count)
            )
        }


class FeedbackGenerator:
    """Hand-crafted rule-based feedback generation."""
    
    def __init__(self):
        self.evaluator = OutcomeEvaluator()
    
    def generate_feedback(self, episode_data: EpisodeData) -> str:
        """
        Generate strategic feedback using hand-crafted rules.
        
        Args:
            episode_data: Complete episode data
        
        Returns:
            str: Strategic feedback for Reviewer fine-tuning
        """
        quality_score, base_feedback = self.evaluator.evaluate(episode_data)
        episode_data.quality_score = quality_score
        
        # Add meta-commentary based on quality tier
        if quality_score >= 0.75:
            tier_comment = "EXCELLENT strategy. This sequence exemplifies optimal decision-making."
        elif quality_score >= 0.6:
            tier_comment = "GOOD approach. You're making solid progress toward achievements."
        elif quality_score >= 0.4:
            tier_comment = "FAIR attempt. Consider focusing more on resource gathering before crafting."
        elif quality_score >= 0.2:
            tier_comment = "NEEDS IMPROVEMENT. Prioritize essential resources (wood, stone) and health management."
        else:
            tier_comment = "POOR outcome. Start with basic resource gathering. Avoid dangerous actions early."
        
        final_feedback = f"{base_feedback} {tier_comment}"
        
        return final_feedback.strip()


class EpisodicDataCollector:
    """Collects episode data during Crafter simulation."""
    
    # Resource keys for tracking
    RESOURCE_KEYS = ['wood', 'stone', 'iron', 'coal', 'diamond', 'food', 'drink', 'energy', 'health']
    
    def __init__(self, env: 'CrafterEnv', helper: 'CrafterHelper'):
        """
        Initialize collector.
        
        Args:
            env: CrafterEnv instance
            helper: CrafterHelper instance
        """
        self.env = env
        self.helper = helper
        self.feedback_gen = FeedbackGenerator()
        
        self.episode_data_list: List[EpisodeData] = []
        self.current_episode_id = 0
        self.current_step = 0
    
    def run_episode(self, episode_id: int, helper_call_interval: int = 5) -> List[EpisodeData]:
        """
        Run a complete episode and collect data.
        
        Args:
            episode_id: Episode identifier
            helper_call_interval: How often to call Helper (every N steps)
        
        Returns:
            List of EpisodeData collected during episode
        """
        self.current_episode_id = episode_id
        self.current_step = 0
        episode_data = []
        
        state = self.env.reset()
        previous_info = None
        
        print(f"[Episode {episode_id}] Starting...")
        
        while self.current_step < 500:  # 500 steps per episode
            # === Call Helper every helper_call_interval steps ===
            if self.current_step % helper_call_interval == 0:
                try:
                    # Get state description for Helper
                    obs, reward, done, info = self.env.step(16)  # noop to get info
                    if done:
                        state = self.env.reset()
                        previous_info = None
                        continue
                    
                    # Generate action sequence from Helper
                    try:
                        action_sequence, llm_response = self.helper.generate_action_sequence(
                            state, info, previous_info
                        )
                    except Exception as e:
                        print(f"  [Step {self.current_step}] Helper error (using random): {e}")
                        action_sequence = None
                        llm_response = f"[Error: {str(e)}]"
                    
                    # Capture state BEFORE sequence execution
                    state_description = self.helper.describe_crafter_state(state, info, previous_info)
                    achievements_before = set(
                        k for k, v in info.get('achievements', {}).items() if v >= 1
                    )
                    resources_before = self._extract_resources(info)
                    health_before = resources_before.get('health', 10)
                    
                    # === Execute sequence ===
                    steps_executed = 0
                    for action in (action_sequence or []):
                        next_state, reward, done, next_info = self.env.step(action)
                        steps_executed += 1
                        self.current_step += 1
                        
                        if done:
                            break
                    
                    # Fallback to random action if sequence failed
                    if not action_sequence:
                        action = np.random.randint(0, self.env.action_size)
                        next_state, reward, done, next_info = self.env.step(action)
                        steps_executed = 1
                        self.current_step += 1
                    
                    # Capture state AFTER sequence execution
                    achievements_after = set(
                        k for k, v in next_info.get('achievements', {}).items() if v >= 1
                    )
                    resources_after = self._extract_resources(next_info)
                    health_after = resources_after.get('health', 10)
                    
                    # Create data point
                    achievements_unlocked = achievements_after - achievements_before
                    
                    episode_data_point = EpisodeData(
                        episode_id=episode_id,
                        step=self.current_step - steps_executed,
                        state_description=state_description,
                        helper_response=llm_response,
                        action_sequence=action_sequence,
                        
                        achievements_before=achievements_before,
                        achievements_after=achievements_after,
                        achievements_unlocked=achievements_unlocked,
                        health_before=health_before,
                        health_after=health_after,
                        resources_before=resources_before,
                        resources_after=resources_after,
                        
                        quality_score=0.0,
                        feedback=""
                    )
                    
                    # Generate feedback
                    episode_data_point.feedback = self.feedback_gen.generate_feedback(episode_data_point)
                    
                    episode_data.append(episode_data_point)
                    
                    previous_info = next_info
                    state = next_state
                    
                    # Check for episode end
                    if done:
                        break
                
                except Exception as e:
                    print(f"  [Episode {episode_id}, Step {self.current_step}] Error collecting data: {e}")
                    continue
            
            else:
                # Regular step (no Helper call)
                action = np.random.randint(0, self.env.action_size)
                next_state, reward, done, next_info = self.env.step(action)
                self.current_step += 1
                
                previous_info = next_info
                state = next_state
                
                if done:
                    break
        
        print(f"[Episode {episode_id}] Completed with {len(episode_data)} data points")
        return episode_data
    
    def _extract_resources(self, info: Dict) -> Dict[str, int]:
        """Extract resource dict from info."""
        inventory = info.get('inventory', {})
        resources = {}
        for key in self.RESOURCE_KEYS:
            resources[key] = inventory.get(key, 0)
        return resources


class CrafterDatasetGenerator:
    """Orchestrates dataset generation and export."""
    
    def __init__(self, num_episodes: int = 50, output_filename: str = 'game_scenarios_dataset_crafter.jsonl'):
        """
        Initialize generator.
        
        Args:
            num_episodes: Number of episodes to simulate
            output_filename: Output CSV filename
        """
        self.num_episodes = num_episodes
        self.output_filename = output_filename
        self.env = None
        self.helper = None
        self.collector = None
        self.all_data: List[EpisodeData] = []
    
    def initialize(self, use_helper_lm_studio: bool = True):
        """
        Initialize environment and helper.
        
        Args:
            use_helper_lm_studio: If True, use real LLM; if False, use synthetic responses
        """
        print("[Initialization] Creating CrafterEnv...")
        self.env = CrafterEnv(reward=True, length=10000, seed=None)
        
        if use_helper_lm_studio:
            print("[Initialization] Creating CrafterHelper (LM Studio)...")
            try:
                self.helper = CrafterHelper(
                    server_host="http://127.0.0.1:1234",
                    model_name="llama-3.2-3b-instruct"
                )
                print("  ✓ Helper ready")
            except Exception as e:
                print(f"  ✗ LM Studio not available: {e}")
                print("  Using synthetic responses instead")
                self.helper = SyntheticHelper()
        else:
            print("[Initialization] Using SyntheticHelper...")
            self.helper = SyntheticHelper()
        
        self.collector = EpisodicDataCollector(self.env, self.helper)
        print("[Initialization] Ready to start data collection\n")
    
    def generate(self) -> List[EpisodeData]:
        """
        Generate dataset by running episodes.
        
        Returns:
            List of all collected EpisodeData
        """
        print(f"{'='*70}")
        print(f"CRAFTER DATASET GENERATION")
        print(f"{'='*70}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Target samples: {self.num_episodes * 100} (100 Helper calls per episode)")
        print(f"Output: {self.output_filename}\n")
        
        start_time = datetime.now()
        
        for episode_id in range(self.num_episodes):
            episode_data = self.collector.run_episode(
                episode_id=episode_id,
                helper_call_interval=5
            )
            self.all_data.extend(episode_data)
        
        elapsed = datetime.now() - start_time
        
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Episodes completed: {self.num_episodes}")
        print(f"Total samples collected: {len(self.all_data)}")
        print(f"Average quality score: {np.mean([d.quality_score for d in self.all_data]):.3f}")
        print(f"High-quality ratio: {self.collector.feedback_gen.evaluator.get_statistics()['high_quality_ratio']:.2%}")
        print(f"Time elapsed: {elapsed}")
        
        return self.all_data
    
    def export_to_jsonl(self) -> str:
        """
        Export collected data to JSONL (one JSON object per line).

        Returns:
            str: Path to exported JSONL
        """
        print(f"\nExporting to JSONL: {self.output_filename}")

        # Prepare data for JSONL
        jsonl_count = 0
        if self.all_data:
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                for data_point in self.all_data:
                    record = {
                        'episode_id': data_point.episode_id,
                        'step': data_point.step,
                        'prompt': data_point.state_description,
                        'response': data_point.helper_response,
                        'instructions': data_point.feedback,
                        'quality_score': float(f"{data_point.quality_score:.3f}"),
                        'achievements_unlocked': ', '.join(data_point.achievements_unlocked) if data_point.achievements_unlocked else 'none',
                        'action_sequence': ', '.join([str(a) for a in (data_point.action_sequence or [])]),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_count += 1

            print(f"✓ Exported {jsonl_count} samples to {self.output_filename}")
        else:
            print("✗ No data to export")

        return self.output_filename
    
    def cleanup(self):
        """Close environment."""
        if self.env:
            self.env.close()


class SyntheticHelper:
    """Rule-based synthetic helper for generating diverse, state-aware action sequences without LLM."""
    
    ACTION_NAMES = {
        0: 'move_up', 1: 'move_down', 2: 'move_left', 3: 'move_right',
        4: 'do', 5: 'sleep',
        6: 'place_stone', 7: 'place_table', 8: 'place_furnace', 9: 'place_plant',
        10: 'make_wood_pickaxe', 11: 'make_stone_pickaxe', 12: 'make_iron_pickaxe',
        13: 'make_wood_sword', 14: 'make_stone_sword', 15: 'make_iron_sword',
        16: 'noop'
    }
    
    ACTION_ID_MAP = {v: k for k, v in ACTION_NAMES.items()}
    
    # Achievement progression tiers (same as OutcomeEvaluator)
    ACHIEVEMENT_TIERS = {
        'tier_1_collect': ['collect_wood', 'collect_stone', 'collect_iron', 'collect_coal', 'collect_diamond'],
        'tier_2_resources': ['collect_food', 'collect_drink', 'collect_fence'],
        'tier_3_build': ['place_stone', 'place_table', 'place_furnace', 'place_plant'],
        'tier_4_craft': [
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
        ],
        'tier_5_interact': ['eat_plant', 'eat_cow', 'defeat_zombie', 'defeat_skeleton']
    }
    
    def __init__(self):
        """Initialize synthetic helper with statistics tracking."""
        self.sequence_count = 0
        self.hallucination_count = 0
    
    def describe_crafter_state(self, state, info, previous_info=None):
        """
        Generate state-aware description from game state.
        
        Args:
            state: 41-dim numpy array from CrafterEnv
            info: info dict from env.step()
            previous_info: optional previous info for change detection
        
        Returns:
            str: Human-readable game state description
        """
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        player_pos = info.get('player_pos', [32, 32])
        discount = info.get('discount', 1.0)
        
        # Inventory summary (only non-zero items)
        inventory_items = []
        for key in ['health', 'food', 'drink', 'energy', 'wood', 'stone', 'iron', 
                    'coal', 'diamond', 'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'potion']:
            val = inventory.get(key, 0)
            if val > 0:
                inventory_items.append(f"{key}: {val}")
        
        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"
        
        # Achievements unlocked
        unlocked = [k for k, v in achievements.items() if v >= 1]
        achievements_str = ", ".join(unlocked) if unlocked else "none yet"
        
        # Determine current goal
        current_goal = self._determine_current_goal(inventory, achievements, unlocked)
        
        # Health status
        alive = "alive" if discount > 0 else "dead"
        health = inventory.get('health', 10)
        
        description = (
            f"Current Status:\n"
            f"- Position: ({player_pos[0]}, {player_pos[1]})\n"
            f"- Health: {health} ({alive})\n"
            f"- Inventory: {inventory_str}\n"
            f"- Achievements Unlocked: {achievements_str}\n"
            f"- Next Priority: {current_goal}\n"
        )
        
        return description
    
    def _determine_current_goal(self, inventory, achievements, unlocked):
        """Determine next logical goal based on achievement progression."""
        if 'collect_wood' not in unlocked:
            return "Collect wood (primary resource)"
        elif 'collect_stone' not in unlocked:
            return "Collect stone (build structures)"
        elif 'collect_iron' not in unlocked:
            return "Collect iron (craft better tools)"
        elif 'place_stone' not in unlocked:
            return "Place stone (build structures)"
        elif 'make_wood_pickaxe' not in unlocked:
            return "Craft wood pickaxe (mine faster)"
        elif 'make_stone_pickaxe' not in unlocked:
            return "Craft stone pickaxe (mine stone)"
        elif 'collect_coal' not in unlocked:
            return "Collect coal (fuel for smelting)"
        elif 'make_iron_pickaxe' not in unlocked:
            return "Craft iron pickaxe (mine iron)"
        elif 'collect_diamond' not in unlocked:
            return "Collect diamond (highest tier resource)"
        else:
            return "Explore and unlock remaining achievements"
    
    def generate_action_sequence(self, state, info, previous_info=None):
        """
        Generate state-aware action sequence using rule-based heuristics.
        
        Args:
            state: 41-dim numpy array from CrafterEnv
            info: info dict from env.step()
            previous_info: optional previous info
        
        Returns:
            tuple: (action_sequence: List[int], response: str)
        """
        self.sequence_count += 1
        
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        unlocked = [k for k, v in achievements.items() if v >= 1]
        health = inventory.get('health', 10)
        
        # === DETERMINE SEQUENCE TYPE ===
        sequence_type = self._select_sequence_type(inventory, achievements, unlocked, health)
        
        # === GENERATE ACTION SEQUENCE ===
        if sequence_type == 'resource_collection':
            actions = self._generate_collection_sequence(inventory)
        elif sequence_type == 'building':
            actions = self._generate_building_sequence(inventory)
        elif sequence_type == 'crafting':
            actions = self._generate_crafting_sequence(inventory, unlocked)
        elif sequence_type == 'survival':
            actions = self._generate_survival_sequence(inventory, health)
        else:  # exploration/fallback
            actions = self._generate_exploration_sequence()
        
        # Convert actions to response format
        response = self._format_response(actions)
        
        return actions, response
    
    def _select_sequence_type(self, inventory, achievements, unlocked, health):
        """
        Select sequence type based on game state.
        
        Returns:
            str: Sequence type ('resource_collection', 'building', 'crafting', 'survival', 'exploration')
        """
        # Priority 1: Survival (low health)
        if health < 5:
            return 'survival'
        
        # Priority 2: Building (have resources, missing build achievements)
        stone_count = inventory.get('stone', 0)
        if stone_count >= 1:
            build_missing = any(
                ach not in unlocked for ach in self.ACHIEVEMENT_TIERS['tier_3_build']
            )
            if build_missing:
                return 'building'
        
        # Priority 3: Crafting (have resources for tools)
        wood_count = inventory.get('wood', 0)
        if wood_count >= 1:
            craft_missing = any(
                ach not in unlocked for ach in self.ACHIEVEMENT_TIERS['tier_4_craft']
            )
            if craft_missing:
                return 'crafting'
        
        # Priority 4: Resource Collection (low resources)
        if wood_count < 3 or inventory.get('stone', 0) < 2:
            return 'resource_collection'
        
        # Fallback: Exploration
        return 'exploration'
    
    def _generate_collection_sequence(self, inventory):
        """Generate 3-5 action sequence for resource gathering."""
        wood = inventory.get('wood', 0)
        stone = inventory.get('stone', 0)
        
        if wood < 3:
            # Move toward trees, gather wood
            return [0, 0, 4, 1, 16]  # up, up, do, down, noop
        else:
            # Move toward stone, gather stone
            return [2, 2, 4, 3, 16]  # left, left, do, right, noop
    
    def _generate_building_sequence(self, inventory):
        """Generate sequence for placing structures."""
        # Move, place stone, move away, noop
        return [0, 6, 1, 1, 16]  # up, place_stone, down, down, noop
    
    def _generate_crafting_sequence(self, inventory, unlocked):
        """Generate sequence for crafting tools."""
        wood = inventory.get('wood', 0)
        
        if 'make_wood_pickaxe' not in unlocked and wood >= 1:
            # Craft wood pickaxe
            return [10, 2, 4, 3, 16]  # make_wood_pickaxe, left, do, right, noop
        elif 'make_stone_pickaxe' not in unlocked:
            # Craft stone pickaxe
            return [11, 0, 4, 1, 16]  # make_stone_pickaxe, up, do, down, noop
        else:
            # Craft iron pickaxe
            return [12, 3, 4, 2, 16]  # make_iron_pickaxe, right, do, left, noop
    
    def _generate_survival_sequence(self, inventory, health):
        """Generate sequence for survival (healing, eating, sleeping)."""
        food = inventory.get('food', 0)
        
        if food > 0:
            # Eat food to heal
            return [4, 4, 5, 16, 16]  # do, do (eat), sleep, noop, noop
        else:
            # Sleep to recover
            return [5, 5, 0, 1, 16]  # sleep, sleep, up, down, noop
    
    def _generate_exploration_sequence(self):
        """Generate random exploration sequence as fallback."""
        # Varied movement patterns
        patterns = [
            [0, 0, 4, 1, 16],      # up, up, do, down, noop
            [3, 3, 4, 2, 16],      # right, right, do, left, noop
            [1, 1, 4, 0, 16],      # down, down, do, up, noop
            [2, 2, 4, 3, 16],      # left, left, do, right, noop
            [0, 3, 4, 1, 2],       # up, right, do, down, left
        ]
        selected_pattern = patterns[np.random.randint(0, len(patterns))]
        return selected_pattern
    
    def _format_response(self, actions):
        """Convert action IDs to response string format."""
        action_names = [self.ACTION_NAMES[action] for action in actions]
        formatted = ", ".join([f"[{name}]" for name in action_names])
        
        # Add reasoning comment
        reasoning = self._generate_reasoning(actions)
        response = f"{formatted}\nReasoning: {reasoning}"
        
        return response
    
    def _generate_reasoning(self, actions):
        """Generate simple reasoning explanation for sequence."""
        action_names = [self.ACTION_NAMES[action] for action in actions]
        
        if 'do' in action_names:
            if 'place_stone' in action_names:
                return "Move to position, place stone, and move away to secure placement."
            elif any('make_' in name for name in action_names):
                return "Craft tools to improve mining efficiency and progress through tiers."
            else:
                return "Move to resource location, gather item, and return to safety."
        elif 'sleep' in action_names:
            return "Sleep to recover health and prepare for next activity."
        else:
            return "Explore and move around to find resources and opportunities."
    
    def should_replan(self, state, info, previous_info, action_sequence):
        """Check if sequence should be interrupted."""
        # For synthetic helper, never replan
        return False
    
    def get_statistics(self):
        """Return generation statistics."""
        return {
            'sequences_generated': self.sequence_count,
            'hallucinations': self.hallucination_count,
            'hallucination_rate': (
                self.hallucination_count / max(1, self.sequence_count)
            )
        }


def main():
    """Main entry point for dataset generation."""
    
    # Configuration
    NUM_EPISODES = 1  # Quick test: 1 episode
    OUTPUT_FILENAME = 'game_scenarios_dataset_crafter_test.jsonl'
    USE_LM_STUDIO = False # Set to False to use synthetic responses
    
    # Generate dataset
    generator = CrafterDatasetGenerator(
        num_episodes=NUM_EPISODES,
        output_filename=OUTPUT_FILENAME
    )
    
    try:
        generator.initialize(use_helper_lm_studio=USE_LM_STUDIO)
        dataset = generator.generate()
        output_path = generator.export_to_jsonl()
        
        print(f"\n✓ SUCCESS: Dataset generated at '{output_path}'")
        print(f"  Ready for F06 (Reviewer Fine-tuning)")
        
    except KeyboardInterrupt:
        print("\n✗ Generation interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()