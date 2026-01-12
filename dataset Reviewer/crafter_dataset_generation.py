import json
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from classes.crafter_environment import CrafterEnv
    from classes.crafter_helper import CrafterHelper
except ImportError as e:
    print(f"Warning: Could not import classes: {e}")
    print("Using mock implementations for testing")

@dataclass
class EpisodeData:
    episode_id: int
    step: int
    state_description: str
    helper_response: str
    action_sequence: Optional[List[int]]

    achievements_before: set
    achievements_after: set
    achievements_unlocked: set
    health_before: int
    health_after: int
    resources_before: Dict[str, int]
    resources_after: Dict[str, int]

    reviewer_feedback: str = ""
    refined_sequence: Optional[List[int]] = None
    improvement_score: float = 0.0

    quality_score: float = 0.0
    feedback: str = ""

class OutcomeEvaluator:

    ACHIEVEMENT_TIERS = {
        'tier_1_collect': ['collect_wood', 'collect_stone', 'collect_iron', 'collect_coal', 'collect_diamond', 'collect_sapling'],
        'tier_2_survival': ['collect_drink', 'eat_plant', 'eat_cow', 'wake_up'],
        'tier_3_build': ['place_stone', 'place_table', 'place_furnace', 'place_plant'],
        'tier_4_craft': [
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
        ],
        'tier_5_combat': ['defeat_zombie', 'defeat_skeleton']
    }

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
        self.evaluation_count += 1
        quality_score = 0.0
        feedback_parts = []

        if episode_data.achievements_unlocked:
            num_achievements = len(episode_data.achievements_unlocked)
            quality_score += min(0.5, num_achievements * 0.15)

            achieved_names = list(episode_data.achievements_unlocked)
            feedback_parts.append(
                f"Unlocked {num_achievements} achievement(s): {', '.join(achieved_names)}. "
                f"Your sequence was strategic and effective."
            )
        else:
            feedback_parts.append("No achievements unlocked. Prioritize resource collection or tier advancement.")

        resource_efficiency = self._evaluate_resource_efficiency(
            episode_data.resources_before,
            episode_data.resources_after,
            episode_data.achievements_unlocked
        )
        quality_score += resource_efficiency[0]
        if resource_efficiency[1]:
            feedback_parts.append(resource_efficiency[1])

        if episode_data.health_after > episode_data.health_before:
            quality_score += 0.15
            feedback_parts.append(
                f"Good health management: increased from {episode_data.health_before} to {episode_data.health_after}."
            )
        elif episode_data.health_after == episode_data.health_before:
            pass
        else:
            health_loss = episode_data.health_before - episode_data.health_after
            if health_loss > 5:
                feedback_parts.append(
                    f"WARNING: Lost {health_loss} health. Consider defensive actions or resource gathering instead."
                )
                quality_score -= 0.1
                feedback_parts.append(
                    f"WARNING: Lost {health_loss} health. Consider defensive actions or resource gathering instead."
                )
                quality_score -= 0.1

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

        if episode_data.action_sequence:
            seq_length = len(episode_data.action_sequence)
            if 3 <= seq_length <= 5:
                quality_score += 0.1
                feedback_parts.append("Sequence length is optimal (3-5 actions).")

                unique_actions = len(set(episode_data.action_sequence))
                if unique_actions >= seq_length * 0.6:
                    quality_score += 0.05
                    feedback_parts.append("Good action diversity in sequence.")
                elif unique_actions <= 2:
                    quality_score -= 0.05
                    feedback_parts.append("Too repetitive. Vary your actions more.")
            elif seq_length < 3:
                feedback_parts.append("Sequence too short. Generate 3-5 coherent actions per request.")
                quality_score -= 0.05
            else:
                feedback_parts.append("Sequence exceeds optimal length. Keep sequences to 3-5 actions.")

        quality_score = max(0.0, min(1.0, quality_score))

        final_feedback = " ".join(feedback_parts)

        if quality_score >= 0.6:
            self.high_quality_count += 1

        return quality_score, final_feedback

    def _evaluate_resource_efficiency(
        self,
        resources_before: Dict[str, int],
        resources_after: Dict[str, int],
        achievements_unlocked: set
    ) -> Tuple[float, Optional[str]]:
        score = 0.0
        feedback = None

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
            score -= 0.1
            feedback = "Resource efficiency is low. Prioritize gathering wood or stone."

        return score, feedback

    def _evaluate_tier_advancement(self, achievements_before: set, achievements_after: set) -> float:
        score = 0.0

        for tier_name, achievements in self.ACHIEVEMENT_TIERS.items():
            tier_before = sum(1 for a in achievements if a in achievements_before)
            tier_after = sum(1 for a in achievements if a in achievements_after)

            if tier_after > tier_before:
                tier_weight = {
                    'tier_1_collect': 0.15,
                    'tier_2_survival': 0.10,
                    'tier_3_build': 0.12,
                    'tier_4_craft': 0.18,
                    'tier_5_combat': 0.12
                }
                score += tier_weight.get(tier_name, 0.05)

        return min(0.25, score)

    def get_statistics(self) -> Dict:
        return {
            'total_evaluations': self.evaluation_count,
            'high_quality_samples': self.high_quality_count,
            'high_quality_ratio': (
                self.high_quality_count / max(1, self.evaluation_count)
            )
        }

class FeedbackGenerator:

    def __init__(self):
        self.evaluator = OutcomeEvaluator()
        self.reviewer_feedback_count = 0

    def simulate_reviewer_feedback(self, episode_data: EpisodeData) -> str:
        self.reviewer_feedback_count += 1
        feedback_parts = []

        inventory = episode_data.resources_after
        health = episode_data.health_after
        achievements = episode_data.achievements_unlocked
        sequence = episode_data.action_sequence or []

        if not sequence or len(sequence) < 3:
            feedback_parts.append("CRITICAL: Sequence too short. Generate 3-5 strategic actions.")

        if health <= 5:
            has_sleep = any(action == 5 for action in sequence)
            if not has_sleep:
                feedback_parts.append("WARNING: Health critical but no [sleep] action. Prioritize survival!")

        has_do = any(action == 4 for action in sequence)
        if not achievements and not has_do:
            feedback_parts.append("SUGGESTION: No [do] action detected. Include resource gathering to progress.")

        unique_actions = len(set(sequence))
        if unique_actions <= 2:
            feedback_parts.append("IMPROVEMENT: Sequence too repetitive. Add diverse actions for better exploration.")

        if achievements:
            feedback_parts.append(f"POSITIVE: Good job unlocking {len(achievements)} achievement(s). Build on this momentum.")
        else:
            wood = inventory.get('wood', 0)
            stone = inventory.get('stone', 0)

            if wood == 0:
                feedback_parts.append("RECOMMENDATION: Prioritize collecting wood first. Use: [move], [do] near trees.")
            elif wood > 0 and stone == 0:
                feedback_parts.append("RECOMMENDATION: You have wood. Now collect stone for progression.")
            elif wood > 0 and not any(action == 7 for action in sequence):
                feedback_parts.append("RECOMMENDATION: With wood, consider [place_table] to unlock crafting.")

        if not feedback_parts:
            return "Sequence looks reasonable. Continue with current strategy."

        return " ".join(feedback_parts)

    def generate_feedback(self, episode_data: EpisodeData, simulate_reviewer: bool = True) -> str:
        quality_score, base_feedback = self.evaluator.evaluate(episode_data)
        episode_data.quality_score = quality_score

        if simulate_reviewer:
            reviewer_feedback = self.simulate_reviewer_feedback(episode_data)
            episode_data.reviewer_feedback = reviewer_feedback

            if "CRITICAL" not in reviewer_feedback and "WARNING" not in reviewer_feedback:
                episode_data.improvement_score = 0.15
            elif "SUGGESTION" in reviewer_feedback or "IMPROVEMENT" in reviewer_feedback:
                episode_data.improvement_score = 0.10
            else:
                episode_data.improvement_score = 0.05

        if quality_score >= 0.75:
            tier_comment = "EXCELLENT strategy. This sequence exemplifies optimal decision-making."
        elif quality_score >= 0.6:
            tier_comment = "GOOD approach. You're making solid progress toward achievements."
        elif quality_score >= 0.4:
            tier_comment = "FAIR attempt. Consider focusing more on resource gathering before crafting."
            if not episode_data.achievements_unlocked:
                tier_comment += " Try: collect wood → place table → make pickaxe."
        elif quality_score >= 0.2:
            tier_comment = "NEEDS IMPROVEMENT. Prioritize essential resources (wood, stone) and health management."
            tier_comment += " Start with: [move], [do] near trees, [move] back."
        else:
            tier_comment = "POOR outcome. Start with basic resource gathering. Avoid dangerous actions early."
            tier_comment += " Example: [move_right], [do], [move_left], [noop] for safe wood collection."

        final_feedback = f"{base_feedback} {tier_comment}"

        return final_feedback.strip()

class EpisodicDataCollector:

    RESOURCE_KEYS = ['wood', 'stone', 'iron', 'coal', 'diamond', 'food', 'drink', 'energy', 'health']

    def __init__(self, env: 'CrafterEnv', helper: 'CrafterHelper'):
        self.env = env
        self.helper = helper
        self.feedback_gen = FeedbackGenerator()

        self.episode_data_list: List[EpisodeData] = []
        self.current_episode_id = 0
        self.current_step = 0

    def run_episode(self, episode_id: int, helper_call_interval: int = 5) -> List[EpisodeData]:
        self.current_episode_id = episode_id
        self.current_step = 0
        episode_data = []

        state = self.env.reset()
        previous_info = None

        print(f"[Episode {episode_id}] Starting...")

        while self.current_step < 500:
            if self.current_step % helper_call_interval == 0:
                try:
                    obs, reward, done, info = self.env.step(16)
                    if done:
                        state = self.env.reset()
                        previous_info = None
                        continue

                    try:
                        action_sequence, llm_response = self.helper.generate_action_sequence(
                            state, info, previous_info
                        )
                    except Exception as e:
                        print(f"  [Step {self.current_step}] Helper error (using random): {e}")
                        action_sequence = None
                        llm_response = f"[Error: {str(e)}]"

                    state_description = self.helper.describe_crafter_state(state, info, previous_info)
                    achievements_before = set(
                        k for k, v in info.get('achievements', {}).items() if v >= 1
                    )
                    resources_before = self._extract_resources(info)
                    health_before = resources_before.get('health', 10)

                    steps_executed = 0
                    for action in (action_sequence or []):
                        next_state, reward, done, next_info = self.env.step(action)
                        steps_executed += 1
                        self.current_step += 1

                        if done:
                            break

                    if not action_sequence:
                        action = np.random.randint(0, self.env.action_size)
                        next_state, reward, done, next_info = self.env.step(action)
                        steps_executed = 1
                        self.current_step += 1

                    achievements_after = set(
                        k for k, v in next_info.get('achievements', {}).items() if v >= 1
                    )
                    resources_after = self._extract_resources(next_info)
                    health_after = resources_after.get('health', 10)

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

                    episode_data_point.feedback = self.feedback_gen.generate_feedback(episode_data_point)

                    episode_data.append(episode_data_point)

                    previous_info = next_info
                    state = next_state

                    if done:
                        break

                except Exception as e:
                    print(f"  [Episode {episode_id}, Step {self.current_step}] Error collecting data: {e}")
                    continue

            else:
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
        inventory = info.get('inventory', {})
        resources = {}
        for key in self.RESOURCE_KEYS:
            resources[key] = inventory.get(key, 0)
        return resources

class CrafterDatasetGenerator:

    def __init__(self, num_episodes: int = 50, output_filename: str = 'game_scenarios_dataset_crafter.jsonl'):
        self.num_episodes = num_episodes
        self.output_filename = output_filename
        self.env = None
        self.helper = None
        self.collector = None
        self.all_data: List[EpisodeData] = []

    def initialize(self, use_helper_lm_studio: bool = True):
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

        quality_scores = [d.quality_score for d in self.all_data]
        print(f"\nQuality Score Distribution:")
        print(f"  Average: {np.mean(quality_scores):.3f}")
        print(f"  Std Dev: {np.std(quality_scores):.3f}")
        print(f"  Min: {np.min(quality_scores):.3f}, Max: {np.max(quality_scores):.3f}")
        print(f"  Median: {np.median(quality_scores):.3f}")

        excellent = sum(1 for s in quality_scores if s >= 0.75)
        good = sum(1 for s in quality_scores if 0.6 <= s < 0.75)
        fair = sum(1 for s in quality_scores if 0.4 <= s < 0.6)
        poor = sum(1 for s in quality_scores if s < 0.4)

        print(f"\nQuality Tiers:")
        print(f"  Excellent (≥0.75): {excellent} ({excellent/len(quality_scores)*100:.1f}%)")
        print(f"  Good (0.6-0.75): {good} ({good/len(quality_scores)*100:.1f}%)")
        print(f"  Fair (0.4-0.6): {fair} ({fair/len(quality_scores)*100:.1f}%)")
        print(f"  Poor (<0.4): {poor} ({poor/len(quality_scores)*100:.1f}%)")

        achievement_unlocks = [d for d in self.all_data if d.achievements_unlocked]
        print(f"\nAchievement Unlocks:")
        print(f"  Samples with achievements: {len(achievement_unlocks)} ({len(achievement_unlocks)/len(self.all_data)*100:.1f}%)")

        avg_improvement = np.mean([d.improvement_score for d in self.all_data])
        samples_with_feedback = sum(1 for d in self.all_data if d.reviewer_feedback)
        print(f"\nReviewer Simulation:")
        print(f"  Samples with Reviewer feedback: {samples_with_feedback} ({samples_with_feedback/len(self.all_data)*100:.1f}%)")
        print(f"  Average improvement score: {avg_improvement:.3f}")

        all_achievements = []
        for d in achievement_unlocks:
            all_achievements.extend(list(d.achievements_unlocked))

        if all_achievements:
            from collections import Counter
            achievement_counts = Counter(all_achievements)
            print(f"  Top 5 achievements unlocked:")
            for ach, count in achievement_counts.most_common(5):
                print(f"    - {ach}: {count} times")

        print(f"\nTime elapsed: {elapsed}")

        return self.all_data

    def export_to_jsonl(self) -> str:
        print(f"\nExporting to JSONL: {self.output_filename}")

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
                        'reviewer_feedback': data_point.reviewer_feedback,
                        'improvement_score': float(f"{data_point.improvement_score:.3f}"),
                        'refined_sequence': ', '.join([str(a) for a in (data_point.refined_sequence or [])]) if data_point.refined_sequence else 'none',
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_count += 1

            print(f"✓ Exported {jsonl_count} samples to {self.output_filename}")
        else:
            print("✗ No data to export")

        return self.output_filename

    def cleanup(self):
        if self.env:
            self.env.close()

class SyntheticHelper:
    ACTION_NAMES = {
        0: 'noop',
        1: 'move_left', 2: 'move_right', 3: 'move_up', 4: 'move_down',
        5: 'do', 6: 'sleep',
        7: 'place_stone', 8: 'place_table', 9: 'place_furnace', 10: 'place_plant',
        11: 'make_wood_pickaxe', 12: 'make_stone_pickaxe', 13: 'make_iron_pickaxe',
        14: 'make_wood_sword', 15: 'make_stone_sword', 16: 'make_iron_sword'
    }

    ACTION_ID_MAP = {v: k for k, v in ACTION_NAMES.items()}

    ACHIEVEMENT_TIERS = {
        'tier_1_collect': ['collect_wood', 'collect_stone', 'collect_iron', 'collect_coal', 'collect_diamond', 'collect_sapling'],
        'tier_2_survival': ['collect_drink', 'eat_plant', 'eat_cow', 'wake_up'],
        'tier_3_build': ['place_stone', 'place_table', 'place_furnace', 'place_plant'],
        'tier_4_craft': [
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
        ],
        'tier_5_combat': ['defeat_zombie', 'defeat_skeleton']
    }

    def __init__(self):
        self.sequence_count = 0
        self.hallucination_count = 0

    def describe_crafter_state(self, state, info, previous_info=None):
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        player_pos = info.get('player_pos', [32, 32])
        discount = info.get('discount', 1.0)

        inventory_items = []
        for key in ['health', 'food', 'drink', 'energy', 'sapling',
                    'wood', 'stone', 'coal', 'iron', 'diamond',
                    'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                    'wood_sword', 'stone_sword', 'iron_sword']:
            val = inventory.get(key, 0)
            if val > 0:
                inventory_items.append(f"{key}: {val}")

        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"

        unlocked = [k for k, v in achievements.items() if v >= 1]
        achievements_str = ", ".join(unlocked) if unlocked else "none yet"

        current_goal = self._determine_current_goal(inventory, achievements, unlocked)

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
        self.sequence_count += 1

        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        unlocked = [k for k, v in achievements.items() if v >= 1]
        health = inventory.get('health', 10)

        sequence_type = self._select_sequence_type(inventory, achievements, unlocked, health)

        if sequence_type == 'resource_collection':
            actions = self._generate_collection_sequence(inventory)
        elif sequence_type == 'building':
            actions = self._generate_building_sequence(inventory)
        elif sequence_type == 'crafting':
            actions = self._generate_crafting_sequence(inventory, unlocked)
        elif sequence_type == 'survival':
            actions = self._generate_survival_sequence(inventory, health)
        else:
            actions = self._generate_exploration_sequence()

        response = self._format_response(actions)

        return actions, response

    def _select_sequence_type(self, inventory, achievements, unlocked, health):
        if health <= 3:
            return 'survival'

        wood_count = inventory.get('wood', 0)
        stone_count = inventory.get('stone', 0)
        iron_count = inventory.get('iron', 0)

        if wood_count == 0:
            return 'resource_collection'

        if wood_count >= 1 and 'place_table' not in unlocked:
            return 'building'

        if 'place_table' in unlocked and wood_count >= 1:
            if 'make_wood_pickaxe' not in unlocked:
                return 'crafting'

        if 'make_wood_pickaxe' in unlocked and stone_count == 0:
            return 'resource_collection'

        if stone_count >= 1:
            build_missing = any(
                ach not in unlocked for ach in self.ACHIEVEMENT_TIERS['tier_3_build']
            )
            if build_missing:
                return 'building'

        if stone_count >= 1 and 'make_stone_pickaxe' not in unlocked:
            return 'crafting'

        if health < 7:
            return 'survival'

        if wood_count < 5 or stone_count < 3 or iron_count < 2:
            return 'resource_collection'

        return 'exploration'

    def _generate_collection_sequence(self, inventory):
        wood = inventory.get('wood', 0)
        stone = inventory.get('stone', 0)
        coal = inventory.get('coal', 0)
        iron = inventory.get('iron', 0)

        if wood < 3:
            patterns = [
                [0, 0, 4, 1, 16],
                [3, 4, 4, 2, 16],
                [1, 3, 4, 0, 2],
                [2, 2, 4, 4, 3],
            ]
            return patterns[np.random.randint(0, len(patterns))]
        elif stone < 3:
            patterns = [
                [2, 2, 4, 3, 16],
                [0, 4, 4, 1, 16],
                [3, 3, 4, 2, 2],
            ]
            return patterns[np.random.randint(0, len(patterns))]
        elif coal < 2:
            patterns = [
                [1, 1, 4, 4, 0],
                [2, 4, 4, 3, 16],
            ]
            return patterns[np.random.randint(0, len(patterns))]
        elif iron < 2:
            patterns = [
                [0, 0, 4, 1, 1],
                [3, 4, 4, 4, 2],
            ]
            return patterns[np.random.randint(0, len(patterns))]
        else:
            return [0, 3, 4, 1, 2]

    def _generate_building_sequence(self, inventory):
        stone = inventory.get('stone', 0)
        wood = inventory.get('wood', 0)

        if wood >= 1:
            patterns = [
                [7, 0, 1, 16, 16],
                [0, 7, 1, 2, 16],
                [3, 7, 2, 16, 16],
            ]
            return patterns[np.random.randint(0, len(patterns))]

        if stone >= 1:
            patterns = [
                [0, 6, 1, 1, 16],
                [2, 6, 3, 16, 16],
                [3, 6, 6, 2, 16],
            ]
            return patterns[np.random.randint(0, len(patterns))]

        if stone >= 2:
            return [0, 8, 1, 16, 16]

        return [9, 0, 1, 16, 16]

    def _generate_crafting_sequence(self, inventory, unlocked):
        wood = inventory.get('wood', 0)
        stone = inventory.get('stone', 0)
        iron = inventory.get('iron', 0)

        if 'make_wood_pickaxe' not in unlocked and wood >= 1:
            patterns = [
                [10, 2, 4, 3, 16],
                [10, 0, 1, 16, 16],
                [0, 10, 1, 4, 16],
            ]
            return patterns[np.random.randint(0, len(patterns))]

        elif 'make_stone_pickaxe' not in unlocked and stone >= 1:
            patterns = [
                [11, 0, 4, 1, 16],
                [11, 3, 2, 16, 16],
            ]
            return patterns[np.random.randint(0, len(patterns))]

        elif 'make_wood_sword' not in unlocked and wood >= 2:
            return [13, 0, 1, 16, 16]

        elif 'make_iron_pickaxe' not in unlocked and iron >= 1:
            return [12, 3, 4, 2, 16]

        elif 'make_stone_sword' not in unlocked and stone >= 2:
            return [14, 2, 3, 16, 16]

        elif 'make_iron_sword' not in unlocked and iron >= 2:
            return [15, 0, 1, 4, 16]

        else:
            return [10, 0, 1, 16, 16]

    def _generate_survival_sequence(self, inventory, health):
        food = inventory.get('food', 0)
        drink = inventory.get('drink', 0)
        energy = inventory.get('energy', 10)

        if health <= 3:
            if food > 0:
                return [4, 5, 5, 5, 16]
            else:
                return [5, 5, 5, 16, 16]

        elif health <= 6:
            if food > 0:
                return [4, 4, 5, 16, 16]
            elif drink > 0:
                return [4, 5, 0, 1, 16]
            else:
                return [5, 5, 0, 1, 16]

        else:
            if food > 0 and drink > 0:
                return [4, 4, 16, 16, 16]
            elif food > 0:
                return [4, 0, 1, 16, 16]
            elif energy < 5:
                return [5, 0, 1, 16, 16]
            else:
                return [0, 4, 4, 1, 16]

    def _generate_exploration_sequence(self):
        patterns = [
            [0, 0, 4, 1, 16],
            [3, 3, 4, 2, 16],
            [1, 1, 4, 0, 16],
            [2, 2, 4, 3, 16],

            [0, 3, 4, 1, 2],
            [3, 1, 4, 0, 2],

            [4, 4, 4, 16, 16],
            [0, 4, 3, 4, 2],

            [0, 3, 3, 1, 2],
            [1, 2, 2, 0, 3],

            [0, 4, 5, 1, 16],
            [3, 4, 4, 5, 2],
        ]
        selected_pattern = patterns[np.random.randint(0, len(patterns))]
        return selected_pattern

    def _format_response(self, actions):
        action_names = [self.ACTION_NAMES[action] for action in actions]
        formatted = ", ".join([f"[{name}]" for name in action_names])

        reasoning = self._generate_reasoning(actions)
        response = f"{formatted}\nReasoning: {reasoning}"

        return response

    def _generate_reasoning(self, actions):
        action_names = [self.ACTION_NAMES[action] for action in actions]

        has_do = 'do' in action_names
        has_sleep = 'sleep' in action_names
        has_place = any('place_' in name for name in action_names)
        has_make = any('make_' in name for name in action_names)
        move_count = sum(1 for name in action_names if 'move_' in name)
        do_count = action_names.count('do')

        if has_make:
            tool_type = next((name for name in action_names if 'make_' in name), 'tool')
            return f"Craft {tool_type} to unlock new gathering capabilities and progress through achievement tiers."

        elif has_place:
            structure = next((name for name in action_names if 'place_' in name), 'structure')
            return f"Place {structure} strategically to unlock crafting or building achievements."

        elif do_count >= 2:
            return f"Intensive resource gathering: execute {do_count} collection actions in quick succession for efficiency."

        elif has_do and move_count >= 2:
            return "Navigate to resource location, collect materials, and safely return to base position."

        elif has_sleep and do_count >= 1:
            return "Gather resources then rest immediately to maintain health and energy for continued exploration."

        elif has_sleep:
            sleep_count = action_names.count('sleep')
            if sleep_count >= 2:
                return f"Emergency health recovery: sleep {sleep_count} times to maximize healing and avoid death."
            else:
                return "Rest to recover health/energy and prepare for the next strategic action."

        elif move_count >= 3:
            return "Wide-range exploration to discover new biomes, resources, and achievement opportunities."

        else:
            return "Balanced exploration and resource gathering to progress toward next achievement milestone."

    def should_replan(self, state, info, previous_info, action_sequence):
        return False

    def get_statistics(self):
        return {
            'sequences_generated': self.sequence_count,
            'hallucinations': self.hallucination_count,
            'hallucination_rate': (
                self.hallucination_count / max(1, self.sequence_count)
            )
        }

def main():

    NUM_EPISODES = 150
    OUTPUT_FILENAME = 'game_scenarios_dataset_crafter.jsonl'
    USE_LM_STUDIO = False

    generator = CrafterDatasetGenerator(
        num_episodes=NUM_EPISODES,
        output_filename=OUTPUT_FILENAME
    )

    try:
        generator.initialize(use_helper_lm_studio=USE_LM_STUDIO)
        dataset = generator.generate()
        output_path = generator.export_to_jsonl()

        print(f"\n✓ SUCCESS: Dataset generated at '{output_path}'")

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