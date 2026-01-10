"""
Crafter Helper - Un LLM Zero-shot che suggerisce sequenze di azioni.
"""

import re
import numpy as np
import lmstudio as lms
import time
from transformers import AutoTokenizer


class CrafterHelper:
    """Helper che usa un LLM per generare sequenze di 3-5 azioni in Crafter."""
    
    # Mappa azioni (ordine ufficiale di Crafter)
    ACTION_NAMES = {
        0: 'noop',
        1: 'move_left', 2: 'move_right', 3: 'move_up', 4: 'move_down',
        5: 'do', 6: 'sleep',
        7: 'place_stone', 8: 'place_table', 9: 'place_furnace', 10: 'place_plant',
        11: 'make_wood_pickaxe', 12: 'make_stone_pickaxe', 13: 'make_iron_pickaxe',
        14: 'make_wood_sword', 15: 'make_stone_sword', 16: 'make_iron_sword'
    }
    
    ACTION_ID_MAP = {v: k for k, v in ACTION_NAMES.items()}
    
    # Re-planning
    REPLAN_THRESHOLD_HP = 0.3  # Se HP < 30%
    REPLAN_THRESHOLD_HP_CRITICAL = 5  # Se HP <= 5 
    REPLAN_THRESHOLD_ACHIEVEMENT = True  
    REPLAN_THRESHOLD_INVENTORY_CHANGE = True  
    
    def __init__(self, server_host="http://127.0.0.1:1234", model_name="qwen/qwen3-4b-2507",
                 min_sequence_length=3, max_sequence_length=5, default_sequence_length=4):
        self.server_host = server_host
        self.model_name = model_name
        
        
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.default_sequence_length = default_sequence_length
        
        
        assert 1 <= min_sequence_length <= max_sequence_length <= 15, \
            f"Invalid sequence bounds: min={min_sequence_length}, max={max_sequence_length}"
        assert min_sequence_length <= default_sequence_length <= max_sequence_length, \
            f"Default {default_sequence_length} must be within [{min_sequence_length}, {max_sequence_length}]"
        
       
        self.sequence_count = 0
        self.hallucination_count = 0
        
       
        self.max_messages_history = 12  
        self.llm_timeout_seconds = 60
      
        self._message_history = []
        
       
        try:
   
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            self.max_context_tokens = 8192  
            self.safe_context_threshold = 6500  
            self.token_counting_enabled = True
            print(f"[Helper] Token counting attivato - max: {self.max_context_tokens}, threshold: {self.safe_context_threshold}")
        except Exception as e:
            print(f"[Helper WARNING] Tokenizer non caricato: {e}")
            print("[Helper] Uso il conteggio messaggi classico")
            self.tokenizer = None
            self.token_counting_enabled = False
        
        
        self._episode_achievements = []
        self._recent_reviewer_feedback = None
        self._episode_step_count = 0
        self._episode_reward = 0.0
        
        
        self._recent_sequences = []
        self._max_sequence_history = 5
    
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
        """Capisce qual è la prossima mossa logica."""
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
        game_description = self.describe_crafter_state(state, info, previous_info)

    
        if override_prompt:
            prompt = override_prompt
        else:
            prompt = self._build_sequence_prompt(game_description)

      
        if self.token_counting_enabled:
            
            current_context = "\n\n".join(self._message_history)
            current_tokens = self._count_tokens(current_context)
            new_prompt_tokens = self._count_tokens(prompt)
            total_tokens = current_tokens + new_prompt_tokens
            
            
            if total_tokens > self.safe_context_threshold:
                print(f"[Helper] Token overflow: {total_tokens}/{self.max_context_tokens}")
                print(f"[Helper] Resetto contesto con riassunto episodio...")
            
                episode_summary = self._generate_episode_summary(game_description)
                self._message_history = [episode_summary]
                
                summary_tokens = self._count_tokens(episode_summary)
                print(f"[Helper] Reset completato: {summary_tokens} token (risparmiati {current_tokens - summary_tokens})")
            else:
                self._message_history.append(prompt)
        else:
            self._message_history.append(prompt)
            if len(self._message_history) > self.max_messages_history:
                self._message_history = self._message_history[-self.max_messages_history:]
                print(f"[Helper] Contesto tagliato a {len(self._message_history)} messaggi")

        try:
            start_time = time.time()
            with lms.Client() as client:
                model = client.llm.model(self.model_name)
                llm_response = model.respond("\n\n".join(self._message_history))
                elapsed = time.time() - start_time
                if elapsed > self.llm_timeout_seconds:
                    raise TimeoutError(f"LLM response exceeded {self.llm_timeout_seconds}s ({elapsed:.1f}s)")

                llm_response = str(llm_response)

                llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
                
                if len(llm_response) > 500:
                    print(f"[Helper] WARNING: Risposta troppo lunga ({len(llm_response)} chars) - taglio a 500")
                    llm_response = llm_response[:500]

                print(f"[Helper] LLM Response:\n{llm_response}\n")

                action_sequence = self.parse_action_sequence(llm_response)

                if action_sequence:
                    self.sequence_count += 1
                    
                    seq_str = str(action_sequence)
                    self._recent_sequences.append(seq_str)
                    if len(self._recent_sequences) > self._max_sequence_history:
                        self._recent_sequences.pop(0)
        
                    if len(self._recent_sequences) >= 3:
                        if self._recent_sequences[-1] == self._recent_sequences[-2] == self._recent_sequences[-3]:
                            print(f"[Helper] WARNING: Loop rilevato - stesse azioni per 3 volte!")
                    
                    return action_sequence, llm_response
                else:
                    self.hallucination_count += 1
                    print("[Helper] Parsing fallito - allucinazione rilevata")
                    return None, llm_response

        except TimeoutError as e:
            print(f"[Helper] LLM Timeout: {e}")
            return None, str(e)
        except Exception as e:
            print(f"[Helper] LLM Error: {e}")
            return None, str(e)
    
    def _build_sequence_prompt(self, game_description):
        """Costruisce il prompt per l'LLM."""
        official_actions = [
            "move_up", "move_down", "move_left", "move_right",
            "do", "sleep",
            "place_stone", "place_table", "place_furnace", "place_plant",
            "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
            "make_wood_sword", "make_stone_sword", "make_iron_sword",
            "noop"
        ]
        
        actions_list = ", ".join(official_actions)
        
        variety_reminder = ""
        if len(self._message_history) > 3:
            if len(self._recent_sequences) >= 3 and self._recent_sequences[-1] == self._recent_sequences[-2]:
                variety_reminder = "\nCRITICAL: Last sequences were IDENTICAL! Try COMPLETELY DIFFERENT actions!\n" \
                                 "Example alternatives: [do], [place_table], [make_wood_pickaxe], [sleep]\n"
            else:
                variety_reminder = "\nIMPORTANT: Try DIFFERENT actions if previous sequences didn't unlock achievements!\n"
        
        prompt = (
            "You are a Crafter AI. GOALS: 1) Survive 2) Unlock achievements 3) Be efficient.\n\n"
            f"{variety_reminder}"
            
            f"VALID ACTIONS: {actions_list}\n\n"
            
            "MISTAKES TO AVOID:\n"
            "collect_wood/gather/mine → use [do]\n"
            "place_rock → use [place_stone]\n"
            "NEVER use placeholders: 'action1', 'action2', etc. → use REAL actions\n\n"
            
            f"CURRENT STATE:\n{game_description}\n\n"
            
            "SURVIVAL:\n"
            "• Health ≤5? Use [sleep]\n"
            "• Food/drink low? Gather resources\n\n"
            
            "ACHIEVEMENT CHAIN:\n"
            "Wood→Table→Pickaxe→Stone→Coal→Iron→Diamond\n\n"
            
            f"TASK: Generate EXACTLY ONE sequence of {self.default_sequence_length} actions.\n\n"
            
            "FORMAT (MANDATORY - NO PLACEHOLDERS!):\n"
            "You MUST replace 'action1/action2/etc' with REAL action names from the list above!\n"
            f"[REAL_ACTION_1], [REAL_ACTION_2], [REAL_ACTION_3], [REAL_ACTION_4]\n"
            "One-line reason (max 15 words).\n\n"
            
            "STOP! Do NOT write 'action1', 'action2', etc. Use ONLY these 17 real actions:\n"
            f"{actions_list}\n\n"
            
            "EXAMPLES:\n"
            "GOOD:\n"
            "[move_right], [do], [move_left], [noop]\n"
            "Collect wood safely.\n\n"
            
            "GOOD:\n"
            "[place_table], [make_wood_pickaxe], [sleep], [noop]\n"
            "Craft pickaxe then rest.\n\n"
            
            "BAD (NEVER DO THIS):\n"
            "[action1], [action2], [action3]  ← WRONG! These are placeholders!\n"
            "[do_something], [gather_wood]  ← WRONG! Not in the 17 valid actions!\n"
            "(move_left), (do)  ← WRONG! Use square brackets [ ], not parentheses!\n"
            "Use ONLY the 17 valid actions listed above!\n\n"
            
            "YOUR TURN - Write your sequence using REAL action names from the list:\n"
        )
        return prompt
    
    def parse_action_sequence(self, llm_response, max_length=None):
        if max_length is None:
            max_length = self.max_sequence_length
        
        lines = llm_response.split('\n')
        first_action_line = None
        for line in lines:
            if '[' in line and ']' in line:
                first_action_line = line
                break
        
        if not first_action_line:
            print("[Parser] Nessuna azione tra parentesi trovata")
            return None
        
        matches = re.findall(r'\[(.*?)\]', first_action_line)
        
        if not matches:
            print("[Parser] Parentesi vuote o malformate")
            return None
        
        action_sequence = []
        for match in matches[:max_length]:
            action_str = match.strip().lower()
            action_id = self.ACTION_ID_MAP.get(action_str)
            
            if action_id is not None:
                action_sequence.append(action_id)
            else:
                action_id = self._fuzzy_match_action(action_str)
                if action_id is not None:
                    action_sequence.append(action_id)
                else:
                    print(f"[Parser] Azione sconosciuta: '{action_str}' - salto")
        
        if len(action_sequence) < self.min_sequence_length:
            print(f"[Parser] Sequenza troppo corta ({len(action_sequence)} < {self.min_sequence_length})")
            return None
        
        return action_sequence
    
    def _fuzzy_match_action(self, action_str):
        """Corregge errori di battitura comuni."""
        typo_map = {
            'move up': 'move_up',
            'move down': 'move_down',
            'move left': 'move_left',
            'move right': 'move_right',
            'place_rock': 'place_stone',
            'place wood': 'place_stone',
            'pickaxe': 'make_wood_pickaxe',
            'sword': 'make_wood_sword',
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
        if previous_info is None:
            return False
        
        if self.REPLAN_THRESHOLD_ACHIEVEMENT:
            prev_achievements = set(
                k for k, v in previous_info.get('achievements', {}).items() if v >= 1
            )
            curr_achievements = set(
                k for k, v in info.get('achievements', {}).items() if v >= 1
            )
            if curr_achievements > prev_achievements:
                print(f"[Replan] Achievement sbloccato: {curr_achievements - prev_achievements}")
                return True
        
        curr_health = info.get('inventory', {}).get('health', 10)
        prev_health = previous_info.get('inventory', {}).get('health', 10)
        
        if curr_health <= self.REPLAN_THRESHOLD_HP_CRITICAL:
            print(f"[Replan] SALUTE CRITICA: {curr_health} - serve agire subito!")
            return True
        
        if curr_health < self.REPLAN_THRESHOLD_HP * 20:
            print(f"[Replan] Salute bassa: {curr_health}")
            return True
        
        if curr_health < prev_health - 2:
            print(f"[Replan] Danno subito rapido: {prev_health} → {curr_health}")
            return True
        
        if self.REPLAN_THRESHOLD_INVENTORY_CHANGE:
            for resource in ['wood', 'stone', 'iron', 'coal', 'diamond']:
                prev_count = previous_info.get('inventory', {}).get(resource, 0)
                curr_count = info.get('inventory', {}).get(resource, 0)
                
                if prev_count > 0 and curr_count == 0:
                    print(f"[Replan] Risorsa esaurita: {resource}")
                    return True
        
        return False
    
    def get_statistics(self):
        return {
            'sequences_generated': self.sequence_count,
            'hallucinations': self.hallucination_count,
            'hallucination_rate': self.hallucination_count / max(1, self.sequence_count)
        }
    
    def reset_conversation(self):
        """Pulisce la memoria per iniziare un nuovo episodio."""
        self._message_history = []
        self._episode_achievements = []
        self._recent_reviewer_feedback = None
        self._episode_step_count = 0
        self._episode_reward = 0.0
        self._recent_sequences = []
    
    def update_episode_progress(self, achievements=None, step_count=0, reward=0.0, reviewer_feedback=None):
        """
        Tiene traccia dei progressi per fare riassunti intelligenti.
        """
        if achievements:
            self._episode_achievements = achievements
        self._episode_step_count = step_count
        self._episode_reward = reward
        if reviewer_feedback:
            self._recent_reviewer_feedback = reviewer_feedback
    
    def _count_tokens(self, text):
        """Conta i token usando il tokenizer di Qwen."""
        if not self.token_counting_enabled or self.tokenizer is None:
            return 0
        
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"[Helper] Errore conteggio token: {e}")
            return 0
    
    def _generate_episode_summary(self, current_state_description):
        """
        Crea un riassunto compatto dell'episodio quando stiamo finendo la memoria.
        Sostituisce tutta la chat history.
        """
        achievement_summary = ", ".join(self._episode_achievements[-10:]) if self._episode_achievements else "none"
        
        summary_parts = [
            "[CONTEXT RESET - Episode Summary]",
            f"Step {self._episode_step_count} | Reward: {self._episode_reward:.1f} | Achievements: {len(self._episode_achievements)}",
            f"Unlocked: {achievement_summary}",
        ]
        
        if self._recent_reviewer_feedback:
            feedback_short = self._recent_reviewer_feedback[:100] + "..." if len(self._recent_reviewer_feedback) > 100 else self._recent_reviewer_feedback
            summary_parts.append(f"Reviewer: {feedback_short}")
        
        summary_parts.extend([
            "",
            current_state_description,
            "",
            "CRITICAL REMINDER:",
            "- Generate EXACTLY ONE sequence of 3-5 actions",
            "- Use ONLY bracketed actions: [move_right], [do], [move_left], [noop]",
            "- NO explanations, NO strategy text, NO markdown",
            "- Format: [action1], [action2], [action3], [action4]",
            "- One-line reason (max 10 words)",
            "",
            "Generate your action sequence NOW:"
        ])
        
        return "\n".join(summary_parts)


class SequenceExecutor:
    """Esegue le sequenze, ma se finiscono usa la DQN."""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.current_sequence = []
        self.current_sequence_index = 0
    
    def execute_action(self, state, info, previous_info=None):
        """
        Esegue la prossima azione della sequenza.
        Se finita, chiede all'LLM (o usa DQN se non implementato qui).
        """
        if not self.current_sequence:
            return self._get_next_action_from_llm(state, info, previous_info)
        
        if self.current_sequence_index < len(self.current_sequence):
            action = self.current_sequence[self.current_sequence_index]
            self.current_sequence_index += 1
            return action
        
        self.current_sequence = []
        self.current_sequence_index = 0
        return self._get_next_action_from_llm(state, info, previous_info)
    
    def _get_next_action_from_llm(self, state, info, previous_info):
        """Chiede nuova sequenza (per ora fallback su DQN)."""
        return self.agent.act(state, self.env)
    
    def interrupt_sequence(self, action_id=None):
        """
        Interrompe la sequenza corrente (es. pericolo).
        """
        self.current_sequence = []
        self.current_sequence_index = 0
        return action_id
