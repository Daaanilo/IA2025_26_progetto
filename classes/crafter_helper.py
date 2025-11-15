"""
F04: Crafter Helper - Zero-shot LLM for generating action sequences
This module handles:
1. State description for Crafter environment
2. Prompt engineering for action sequence generation
3. LLM integration with LM Studio
4. Action sequence parsing and validation
5. State change detection for re-planning
"""

import re
import numpy as np
import lmstudio as lms
import time


class CrafterHelper:
    """Zero-shot LLM helper for generating 3-5 action sequences in Crafter."""
    
    # Action mapping (OFFICIAL CRAFTER ORDER)
    ACTION_NAMES = {
        0: 'noop',
        1: 'move_left', 2: 'move_right', 3: 'move_up', 4: 'move_down',
        5: 'do', 6: 'sleep',
        7: 'place_stone', 8: 'place_table', 9: 'place_furnace', 10: 'place_plant',
        11: 'make_wood_pickaxe', 12: 'make_stone_pickaxe', 13: 'make_iron_pickaxe',
        14: 'make_wood_sword', 15: 'make_stone_sword', 16: 'make_iron_sword'
    }
    
    ACTION_ID_MAP = {v: k for k, v in ACTION_NAMES.items()}
    
    # Re-planning thresholds
    REPLAN_THRESHOLD_HP = 0.3  # Re-plan if health drops below 30% (was 20%)
    REPLAN_THRESHOLD_HP_CRITICAL = 5  # Re-plan immediately if health <= 5
    REPLAN_THRESHOLD_ACHIEVEMENT = True  # Re-plan on achievement unlock
    REPLAN_THRESHOLD_INVENTORY_CHANGE = True  # Re-plan on significant inventory change
    
    def __init__(self, server_host="http://127.0.0.1:1234", model_name="llama-3.2-3b-instruct",
                 min_sequence_length=3, max_sequence_length=5, default_sequence_length=4):
        """
        Initialize Crafter Helper with LM Studio connection.
        
        Args:
            server_host: LM Studio API host
            model_name: LLM model name (default: llama-3.2-3b-instruct)
            min_sequence_length: Minimum actions per sequence (default: 3)
            max_sequence_length: Maximum actions per sequence (default: 5)
            default_sequence_length: Target sequence length for prompts (default: 4)
        """
        self.server_host = server_host
        self.model_name = model_name
        
        # Sequence generation parameters (configurable for F07 analysis)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.default_sequence_length = default_sequence_length
        
        # Validate configuration
        assert 1 <= min_sequence_length <= max_sequence_length <= 15, \
            f"Invalid sequence bounds: min={min_sequence_length}, max={max_sequence_length}"
        assert min_sequence_length <= default_sequence_length <= max_sequence_length, \
            f"Default {default_sequence_length} must be within [{min_sequence_length}, {max_sequence_length}]"
        
        # Statistics
        self.sequence_count = 0
        self.hallucination_count = 0
        # LLM safety limits (CRITICAL: prevent context overflow)
        self.max_messages_history = 3  # Keep only last 3 exchanges (was 16 - caused 8900+ token overflow)
        self.llm_timeout_seconds = 40   # fail fast instead of hanging
        # Conversation history - MUST be cleared between episodes
        self._message_history = []
    
    def describe_crafter_state(self, state, info, previous_info=None):
        """
        Convert 43-dim Crafter state vector + info dict to human-readable description.
        
        Args:
            state: 43-dim numpy array from CrafterEnv (16 inventory + 2 pos + 3 status + 22 achievements)
            info: info dict from env.step() containing inventory, achievements, player_pos
            previous_info: optional previous info dict for change detection
        
        Returns:
            str: Human-readable game state description
        """
        # Extract components from state vector
        # State structure: [inventory(16), pos(2), status(3), achievements(22)]
        
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        player_pos = info.get('player_pos', [32, 32])
        discount = info.get('discount', 1.0)
        
        # Inventory summary (include only non-zero items)
        inventory_items = []
        for key in ['health', 'food', 'drink', 'energy', 'sapling',
                    'wood', 'stone', 'coal', 'iron', 'diamond',
                    'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                    'wood_sword', 'stone_sword', 'iron_sword']:
            val = inventory.get(key, 0)
            if val > 0:
                inventory_items.append(f"{key}: {val}")
        
        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"
        
        # Achievements unlocked
        unlocked = [k for k, v in achievements.items() if v >= 1]
        achievements_str = ", ".join(unlocked) if unlocked else "none yet"
        
        # Determine current priority/goal based on achievements and inventory
        current_goal = self._determine_current_goal(inventory, achievements, unlocked)
        
        # Status
        alive = "alive" if discount > 0 else "dead"
        
        # Available actions (all 17 are always valid in Crafter)
        available_actions = ", ".join([
            "[move_up], [move_down], [move_left], [move_right], [do], [sleep]",
            "[place_stone], [place_table], [place_furnace], [place_plant]",
            "[make_wood_pickaxe], [make_stone_pickaxe], [make_iron_pickaxe]",
            "[make_wood_sword], [make_stone_sword], [make_iron_sword], [noop]"
        ])
        
        description = (
            f"Current Status:\n"
            f"- Position: ({player_pos[0]}, {player_pos[1]})\n"
            f"- Health: {alive}\n"
            f"- Inventory: {inventory_str}\n"
            f"- Achievements Unlocked: {achievements_str}\n"
            f"- Next Priority: {current_goal}\n\n"
            f"Available Actions:\n{available_actions}"
        )
        
        return description
    
    def _determine_current_goal(self, inventory, achievements, unlocked):
        """Determine the next logical goal based on current state."""
        # PRIORITY 1: Survival - check health/food/drink first
        health = inventory.get('health', 10)
        food = inventory.get('food', 10)
        drink = inventory.get('drink', 10)
        
        if health <= 3:
            return "URGENT: Restore health (eat/drink/sleep) - SURVIVAL CRITICAL!"
        elif health <= 5:
            return "Low health - prioritize eating, drinking, or sleeping"
        elif food <= 2:
            return "Low food - find and eat plants or hunt animals"
        elif drink <= 2:
            return "Low hydration - collect and drink water"
        
        # PRIORITY 2: Achievement progression
        if 'collect_wood' not in unlocked:
            return "Collect wood (primary resource)"
        elif 'collect_stone' not in unlocked:
            return "Collect stone (build structures)"
        elif 'place_table' not in unlocked:
            return "Place table (craft tools)"
        elif 'make_wood_pickaxe' not in unlocked:
            return "Craft wood pickaxe (mine stone/coal)"
        elif 'collect_coal' not in unlocked:
            return "Collect coal (fuel for smelting)"
        elif 'collect_iron' not in unlocked:
            return "Collect iron (advanced tools)"
        elif 'place_stone' not in unlocked:
            return "Place stone (build structures)"
        elif 'make_stone_pickaxe' not in unlocked:
            return "Craft stone pickaxe (mine iron)"
        elif 'place_furnace' not in unlocked:
            return "Place furnace (smelt iron)"
        elif 'make_iron_pickaxe' not in unlocked:
            return "Craft iron pickaxe (mine diamond)"
        elif 'collect_diamond' not in unlocked:
            return "Collect diamond (highest tier resource)"
        elif 'collect_sapling' not in unlocked:
            return "Collect sapling (plant trees)"
        elif 'place_plant' not in unlocked:
            return "Place plant (grow resources)"
        elif 'eat_plant' not in unlocked:
            return "Eat plant (restore food)"
        elif 'collect_drink' not in unlocked:
            return "Collect drink (restore hydration)"
        elif 'wake_up' not in unlocked:
            return "Sleep and wake up (restore energy)"
        else:
            return "Explore and unlock remaining achievements (combat, crafting)"
    
    def generate_action_sequence(self, state, info, previous_info=None, override_prompt=None):
        """
        Generate 3-5 action sequence using LLM.
        
        Args:
            state: 41-dim numpy array from CrafterEnv
            info: info dict from env.step()
            previous_info: optional previous info for change context
            override_prompt: optional custom prompt (for Reviewer refinement workflow)
        
        Returns:
            tuple: (action_sequence: List[int], llm_response: str)
            If LLM fails, returns (None, error_message)
        """
        game_description = self.describe_crafter_state(state, info, previous_info)

        # Craft prompt for action sequence (or use override for Reviewer refinement)
        if override_prompt:
            prompt = override_prompt
        else:
            prompt = self._build_sequence_prompt(game_description)

        # CRITICAL: Aggressive context management to prevent 4096 token overflow
        # Keep only last N prompts (sliding window)
        self._message_history.append(prompt)
        if len(self._message_history) > self.max_messages_history:
            self._message_history = self._message_history[-self.max_messages_history:]
            print(f"[Helper] Context trimmed to {len(self._message_history)} messages")

        try:
            start_time = time.time()
            with lms.Client() as client:
                model = client.llm.model(self.model_name)
                # lmstudio-python non espone max_tokens direttamente su respond();
                # ci affidiamo quindi a timeout lato client per evitare blocchi.
                llm_response = model.respond("\n\n".join(self._message_history))
                elapsed = time.time() - start_time
                if elapsed > self.llm_timeout_seconds:
                    raise TimeoutError(f"LLM response exceeded {self.llm_timeout_seconds}s ({elapsed:.1f}s)")

                llm_response = str(llm_response)

                # Clean thinking tags
                llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()

                print(f"[Helper] LLM Response:\n{llm_response}\n")

                # Parse action sequence
                action_sequence = self.parse_action_sequence(llm_response)

                if action_sequence:
                    self.sequence_count += 1
                    return action_sequence, llm_response
                else:
                    self.hallucination_count += 1
                    print("[Helper] Failed to parse valid action sequence - hallucination detected")
                    return None, llm_response

        except TimeoutError as e:
            print(f"[Helper] LLM Timeout: {e}")
            return None, str(e)
        except Exception as e:
            print(f"[Helper] LLM Error: {e}")
            return None, str(e)
    
    def _build_sequence_prompt(self, game_description):
        """Build LLM prompt requesting action sequence."""
        # List of ONLY the 17 official valid actions
        official_actions = [
            "move_up", "move_down", "move_left", "move_right",
            "do", "sleep",
            "place_stone", "place_table", "place_furnace", "place_plant",
            "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
            "make_wood_sword", "make_stone_sword", "make_iron_sword",
            "noop"
        ]
        
        actions_list = ", ".join(official_actions)
        
        prompt = (
            "You are a strategic AI for Crafter. PRIMARY GOALS:\n"
            "1. STAY ALIVE - avoid death at all costs\n"
            "2. MAXIMIZE achievements unlocked\n"
            "3. Be efficient and strategic\n\n"
            
            "=== ALLOWED ACTIONS ONLY ===\n"
            f"{actions_list}\n\n"
            
            "COMMON MISTAKES:\n"
            "❌ collect_wood, gather, mine → use [do]\n"
            "❌ place_rock → use [place_stone]\n"
            "❌ wait, rest → use [noop] or [sleep]\n\n"
            
            f"{game_description}\n\n"
            
            "=== SURVIVAL FIRST ===\n"
            "• Health ≤ 5? URGENT - use [sleep] to recover\n"
            "• Food/drink low? Prioritize gathering\n"
            "• Avoid enemies until armed\n\n"
            
            "=== ACHIEVEMENT PATH ===\n"
            "Fast progression:\n"
            "1. Wood: [do] near trees\n"
            "2. Table: [place_table]\n"
            "3. Pickaxe: [make_wood_pickaxe]\n"
            "4. Stone: [do] near rocks\n"
            "5. Continue: coal → iron → advanced\n\n"
            
            f"TASK: Generate {self.min_sequence_length}-{self.max_sequence_length} actions (target: {self.default_sequence_length}) for:\n"
            "• Survival (monitor health/food/drink)\n"
            "• Next Priority achievement\n"
            "• Strategic movement\n\n"
            
            f"FORMAT: [action1], [action2], ... [{self.default_sequence_length} actions total]\n"
            "Brief reasoning (max 25 words).\n\n"
            
            "EXAMPLES:\n"
            "• [move_right], [do], [move_left], [noop] - Get wood safely\n"
            "• [place_table], [make_wood_pickaxe], [sleep] - Craft then rest\n"
            "• [sleep], [sleep], [noop] - URGENT health recovery\n\n"
            
            "Generate (prioritize survival + achievements):\n"
        )
        return prompt
    
    def parse_action_sequence(self, llm_response, max_length=None):
        """
        Parse bracketed actions from LLM response.
        
        Args:
            llm_response: String response from LLM
            max_length: Maximum sequence length (default: use self.max_sequence_length)
        
        Returns:
            List[int]: Action IDs, or None if parsing fails
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        # Extract all bracketed text
        matches = re.findall(r'\[(.*?)\]', llm_response)
        
        if not matches:
            print("[Parser] No bracketed actions found")
            return None
        
        action_sequence = []
        for match in matches[:max_length]:
            action_str = match.strip().lower()
            action_id = self.ACTION_ID_MAP.get(action_str)
            
            if action_id is not None:
                action_sequence.append(action_id)
            else:
                # Try fuzzy matching for common typos
                action_id = self._fuzzy_match_action(action_str)
                if action_id is not None:
                    action_sequence.append(action_id)
                else:
                    print(f"[Parser] Unknown action: '{action_str}' - skipping")
        
        # Validate sequence length
        if len(action_sequence) < self.min_sequence_length:
            print(f"[Parser] Sequence too short ({len(action_sequence)} < {self.min_sequence_length})")
            return None
        
        return action_sequence
    
    def _fuzzy_match_action(self, action_str):
        """Handle common typos or variations in action names."""
        typo_map = {
            'move up': 'move_up',
            'move down': 'move_down',
            'move left': 'move_left',
            'move right': 'move_right',
            'place_rock': 'place_stone',  # Common mistake
            'place wood': 'place_stone',
            'pickaxe': 'make_wood_pickaxe',  # Abbreviated
            'sword': 'make_wood_sword',  # Abbreviated
            'wood pickaxe': 'make_wood_pickaxe',
            'stone pickaxe': 'make_stone_pickaxe',
            'iron pickaxe': 'make_iron_pickaxe',
            'wood sword': 'make_wood_sword',
            'stone sword': 'make_stone_sword',
            'iron sword': 'make_iron_sword',
        }
        
        normalized = typo_map.get(action_str)
        if normalized:
            return self.ACTION_ID_MAP.get(normalized)
        
        return None
    
    def should_replan(self, state, info, previous_info, action_sequence):
        """
        Determine if the current action sequence should be abandoned and re-planned.
        
        Strategy B: Use DQN for missing actions when LLM sequence is interrupted.
        
        Args:
            state: current state
            info: current info dict
            previous_info: previous info dict
            action_sequence: remaining actions in sequence
        
        Returns:
            bool: True if should re-plan and get new sequence
        """
        if previous_info is None:
            return False
        
        # Check for achievement unlocks
        if self.REPLAN_THRESHOLD_ACHIEVEMENT:
            prev_achievements = set(
                k for k, v in previous_info.get('achievements', {}).items() if v >= 1
            )
            curr_achievements = set(
                k for k, v in info.get('achievements', {}).items() if v >= 1
            )
            if curr_achievements > prev_achievements:
                print(f"[Replan] Achievement unlocked: {curr_achievements - prev_achievements}")
                return True
        
        # Check for critical health (if applicable in Crafter)
        curr_health = info.get('inventory', {}).get('health', 10)
        prev_health = previous_info.get('inventory', {}).get('health', 10)
        
        # Immediate re-plan if health is critically low
        if curr_health <= self.REPLAN_THRESHOLD_HP_CRITICAL:
            print(f"[Replan] CRITICAL HEALTH: {curr_health} - immediate survival action needed!")
            return True
        
        # Re-plan if health drops below threshold
        if curr_health < self.REPLAN_THRESHOLD_HP * 20:  # Assuming max health ~20
            print(f"[Replan] Low health: {curr_health}")
            return True
        
        # Re-plan if health is decreasing rapidly
        if curr_health < prev_health - 2:
            print(f"[Replan] Rapid health loss: {prev_health} → {curr_health}")
            return True
        
        # Check for significant inventory changes
        if self.REPLAN_THRESHOLD_INVENTORY_CHANGE:
            # Check if resources changed unexpectedly (death, unexpected use)
            for resource in ['wood', 'stone', 'iron', 'coal', 'diamond']:
                prev_count = previous_info.get('inventory', {}).get(resource, 0)
                curr_count = info.get('inventory', {}).get(resource, 0)
                # If resource decreased unexpectedly (not by plan), re-plan
                if prev_count > 0 and curr_count == 0:
                    print(f"[Replan] Resource depleted: {resource}")
                    return True
        
        return False
    
    def get_statistics(self):
        """Return helper statistics."""
        return {
            'sequences_generated': self.sequence_count,
            'hallucinations': self.hallucination_count,
            'hallucination_rate': self.hallucination_count / max(1, self.sequence_count)
        }
    
    def reset_conversation(self):
        """Clear conversation history - call at episode start to prevent context overflow."""
        self._message_history = []
        print("[Helper] Conversation history cleared for new episode")


# Strategy B Fallback: Use DQN for remaining sequence actions after LLM interruption
class SequenceExecutor:
    """Executes action sequences with fallback to DQN."""
    
    def __init__(self, agent, env):
        """
        Initialize executor.
        
        Args:
            agent: DQNAgent instance
            env: CrafterEnv instance
        """
        self.agent = agent
        self.env = env
        self.current_sequence = []
        self.current_sequence_index = 0
    
    def execute_action(self, state, info, previous_info=None):
        """
        Execute next action from sequence, or get new sequence, or use DQN fallback.
        
        Args:
            state: current state (43-dim array)
            info: current info dict
            previous_info: previous info dict
        
        Returns:
            int: Next action to execute
        """
        # If no current sequence, get new one from LLM
        if not self.current_sequence:
            return self._get_next_action_from_llm(state, info, previous_info)
        
        # If current sequence has actions remaining
        if self.current_sequence_index < len(self.current_sequence):
            action = self.current_sequence[self.current_sequence_index]
            self.current_sequence_index += 1
            return action
        
        # Sequence exhausted, get new one
        self.current_sequence = []
        self.current_sequence_index = 0
        return self._get_next_action_from_llm(state, info, previous_info)
    
    def _get_next_action_from_llm(self, state, info, previous_info):
        """Request new action sequence from Helper."""
        # This will be implemented in the training loop
        # For now, fallback to DQN
        return self.agent.act(state, self.env)
    
    def interrupt_sequence(self, action_id=None):
        """
        Interrupt current sequence (Strategy B: Use DQN for remaining actions).
        If no action_id provided, will use DQN for next action.
        
        Args:
            action_id: optional action ID to execute instead
        """
        self.current_sequence = []
        self.current_sequence_index = 0
        return action_id
