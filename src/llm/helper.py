"""
Helper LLM - Generates sequences of actions for the NPC
"""
import json
import os
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class Helper:
    """
    Helper LLM that suggests sequences of coherent actions.
    Uses OpenAI GPT models for zero-shot action generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Helper LLM.
        
        Args:
            config: Configuration dictionary from helper_config.yaml
        """
        self.config = config
        helper_config = config.get('helper', {})
        
        # API configuration
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key
        
        # Model configuration
        self.model = helper_config.get('model', 'gpt-4-turbo-preview')
        self.temperature = helper_config.get('temperature', 0.7)
        self.max_tokens = helper_config.get('max_tokens', 500)
        self.top_p = helper_config.get('top_p', 0.9)
        
        # Action generation configuration
        self.num_actions_per_call = helper_config.get('num_actions_per_call', 5)
        self.context_window = helper_config.get('context_window', 10)
        
        # Prompts
        self.system_prompt = helper_config.get('system_prompt', '').format(
            num_actions=self.num_actions_per_call
        )
        self.user_prompt_template = helper_config.get('user_prompt_template', '')
        
        # API configuration
        api_config = config.get('api', {})
        self.max_retries = api_config.get('max_retries', 3)
        self.timeout = api_config.get('timeout', 30)
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def suggest_actions(self, state_info: Dict[str, Any], 
                       recent_actions: List[str],
                       num_actions: Optional[int] = None) -> List[str]:
        """
        Suggest a sequence of actions based on the current state.
        
        Args:
            state_info: Current game state information
            recent_actions: List of recently taken actions
            num_actions: Number of actions to suggest (uses default if None)
            
        Returns:
            List of suggested action names
        """
        if num_actions is None:
            num_actions = self.num_actions_per_call
        
        # Format the user prompt
        user_prompt = self._format_user_prompt(state_info, recent_actions, num_actions)
        
        # Call OpenAI API
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    timeout=self.timeout
                )
                
                # Parse response
                content = response.choices[0].message.content
                actions = self._parse_actions(content)
                
                self.total_calls += 1
                self.successful_calls += 1
                
                return actions[:num_actions]
            
            except Exception as e:
                print(f"Helper API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    self.total_calls += 1
                    self.failed_calls += 1
                    # Return default conservative actions
                    return self._get_fallback_actions(num_actions)
        
        return self._get_fallback_actions(num_actions)
    
    def _format_user_prompt(self, state_info: Dict[str, Any], 
                           recent_actions: List[str],
                           num_actions: int) -> str:
        """Format the user prompt with current state information."""
        # Extract state information (with defaults)
        health = state_info.get('health', 'unknown')
        food = state_info.get('food', 'unknown')
        drink = state_info.get('drink', 'unknown')
        energy = state_info.get('energy', 'unknown')
        inventory = state_info.get('inventory', {})
        nearby_objects = state_info.get('nearby_objects', [])
        achievements = state_info.get('achievements', {})
        
        # Format achievements (only unlocked ones)
        unlocked_achievements = [k for k, v in achievements.items() if v > 0]
        
        # Format recent actions (last N actions)
        recent = recent_actions[-self.context_window:] if recent_actions else ['none']
        
        # Create prompt
        prompt = self.user_prompt_template.format(
            health=health,
            food=food,
            drink=drink,
            energy=energy,
            inventory=json.dumps(inventory) if inventory else 'empty',
            nearby_objects=', '.join(nearby_objects) if nearby_objects else 'none',
            achievements=', '.join(unlocked_achievements) if unlocked_achievements else 'none',
            recent_actions=', '.join(recent) if recent else 'none',
            num_actions=num_actions
        )
        
        return prompt
    
    def _parse_actions(self, content: str) -> List[str]:
        """Parse actions from LLM response."""
        try:
            # Try to parse as JSON
            actions = json.loads(content)
            if isinstance(actions, list):
                return actions
        except json.JSONDecodeError:
            pass
        
        # Try to extract list-like structure
        import re
        
        # Look for ["action1", "action2", ...]
        match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if match:
            items = match.group(1).split(',')
            actions = [item.strip().strip('"').strip("'") for item in items]
            return actions
        
        # Look for line-by-line actions
        lines = content.strip().split('\n')
        actions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering and quotes
                action = re.sub(r'^\d+[\.\)]\s*', '', line)
                action = action.strip('"').strip("'")
                if action:
                    actions.append(action)
        
        return actions if actions else self._get_fallback_actions(1)
    
    def _get_fallback_actions(self, num_actions: int) -> List[str]:
        """Return safe fallback actions when API fails."""
        # Conservative fallback: mostly exploration
        fallback_pool = ['move_right', 'move_left', 'move_up', 'move_down', 'do']
        import random
        return [random.choice(fallback_pool) for _ in range(num_actions)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Helper usage statistics."""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': f"{success_rate:.2f}%"
        }
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt (useful for adaptive learning)."""
        self.system_prompt = new_prompt
