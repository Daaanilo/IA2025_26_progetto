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


class CrafterHelper:
    """Zero-shot LLM helper for generating 3-5 action sequences in Crafter."""
    
    # Action mapping
    ACTION_NAMES = {
        0: 'move_up', 1: 'move_down', 2: 'move_left', 3: 'move_right',
        4: 'do', 5: 'sleep',
        6: 'place_stone', 7: 'place_table', 8: 'place_furnace', 9: 'place_plant',
        10: 'make_wood_pickaxe', 11: 'make_stone_pickaxe', 12: 'make_iron_pickaxe',
        13: 'make_wood_sword', 14: 'make_stone_sword', 15: 'make_iron_sword',
        16: 'noop'
    }
    
    ACTION_ID_MAP = {v: k for k, v in ACTION_NAMES.items()}
    
    # Sequence generation parameters
    SEQUENCE_LENGTH = 5  # Generate sequences of 3-5 actions (use 5 as default)
    MIN_SEQUENCE_LENGTH = 3
    MAX_SEQUENCE_LENGTH = 5
    
    # Re-planning thresholds
    REPLAN_THRESHOLD_HP = 0.2  # Re-plan if health drops below 20%
    REPLAN_THRESHOLD_ACHIEVEMENT = True  # Re-plan on achievement unlock
    REPLAN_THRESHOLD_INVENTORY_CHANGE = True  # Re-plan on significant inventory change
    
    def __init__(self, server_host="http://127.0.0.1:1234", model_name="llama-3.2-3b-instruct"):
        """
        Initialize Crafter Helper with LM Studio connection.
        
        Args:
            server_host: LM Studio API host
            model_name: LLM model name (default: llama-3.2-3b-instruct)
        """
        self.server_host = server_host
        self.model_name = model_name
        
        # Statistics
        self.sequence_count = 0
        self.hallucination_count = 0
    
    def describe_crafter_state(self, state, info, previous_info=None):
        """
        Convert 41-dim Crafter state vector + info dict to human-readable description.
        
        Args:
            state: 41-dim numpy array from CrafterEnv
            info: info dict from env.step() containing inventory, achievements, player_pos
            previous_info: optional previous info dict for change detection
        
        Returns:
            str: Human-readable game state description
        """
        # Extract components from state vector
        # State structure: [inventory(13), pos(2), status(3), achievements(22), fence(1)]
        
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        player_pos = info.get('player_pos', [32, 32])
        discount = info.get('discount', 1.0)
        
        # Inventory summary (include only non-zero items)
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
        # Simple heuristic: suggest next missing achievement tier
        
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
        Generate 3-5 action sequence using LLM.
        
        Args:
            state: 41-dim numpy array from CrafterEnv
            info: info dict from env.step()
            previous_info: optional previous info for change context
        
        Returns:
            tuple: (action_sequence: List[int], llm_response: str)
            If LLM fails, returns (None, error_message)
        """
        game_description = self.describe_crafter_state(state, info, previous_info)
        
        # Craft prompt for action sequence
        prompt = self._build_sequence_prompt(game_description)
        
        try:
            with lms.Client() as client:
                model = client.llm.model(self.model_name)
                llm_response = model.respond(prompt)
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
                    
        except Exception as e:
            print(f"[Helper] LLM Error: {e}")
            return None, str(e)
    
    def _build_sequence_prompt(self, game_description):
        """Build LLM prompt requesting action sequence."""
        prompt = (
            "You are a strategic game assistant for a survival crafting game (Crafter).\n\n"
            
            "Your task is to generate a sequence of 3-5 coherent actions to progress toward "
            "unlocking achievements and survival.\n\n"
            
            f"{game_description}\n\n"
            
            "Generate a sequence of 3-5 actions:\n"
            "1. Each action MUST be in square brackets, e.g., [move_right], [do], [make_wood_pickaxe]\n"
            "2. Separate actions with commas\n"
            "3. Actions should be COHERENT and STRATEGIC (e.g., move to wood → do → move away)\n"
            "4. Explain your reasoning (max 100 words)\n\n"
            
            "Example:\n"
            "[move_right], [move_right], [do], [move_left], [noop]\n"
            "Reasoning: Move right twice to reach the tree, collect wood with [do], "
            "then move left once and wait.\n\n"
            
            "Now generate your sequence:\n"
        )
        return prompt
    
    def parse_action_sequence(self, llm_response, max_length=5):
        """
        Parse bracketed actions from LLM response.
        
        Args:
            llm_response: String response from LLM
            max_length: Maximum sequence length (default 5)
        
        Returns:
            List[int]: Action IDs, or None if parsing fails
        """
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
        if len(action_sequence) < self.MIN_SEQUENCE_LENGTH:
            print(f"[Parser] Sequence too short ({len(action_sequence)} < {self.MIN_SEQUENCE_LENGTH})")
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
        if curr_health < self.REPLAN_THRESHOLD_HP * 20:  # Assuming max health ~20
            print(f"[Replan] Health critical: {curr_health}")
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
            state: current state (41-dim array)
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
