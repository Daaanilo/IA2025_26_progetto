import numpy as np
import gymnasium as gym
from gymnasium import spaces
import crafter


class CrafterEnv:
    """Wrapper per Crafter che estrae stato 43D (16 inventario + 2 posizione + 3 status + 22 achievement)."""
    def __init__(self, 
                 area=(64, 64), 
                 view=(9, 9), 
                 size=(64, 64),
                 reward=True, 
                 length=10000, 
                 seed=None):
        self.env = crafter.Env(area=area, view=view, size=size, reward=reward, length=length, seed=seed)
        self._env = self.env
        
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        self.action_names = [
            'noop',
            'move_left',
            'move_right',
            'move_up',
            'move_down',
            'do',
            'sleep',
            'place_stone',
            'place_table',
            'place_furnace',
            'place_plant',
            'make_wood_pickaxe',
            'make_stone_pickaxe',
            'make_iron_pickaxe',
            'make_wood_sword',
            'make_stone_sword',
            'make_iron_sword'
        ]
        
        self.state_size = 43
        self.action_size = 17
        self._last_info = None
        
    def reset(self):
        """Resetta l'ambiente e restituisce lo stato iniziale."""
        obs = self.env.reset()
        self._last_info = self._get_dummy_info()
        state = self._extract_state()
        self.state_size = len(state)
        return state, self._last_info
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_info = info
        state = self._extract_state()
        return state, reward, done, info
    
    def _get_dummy_info(self):
        obs, reward, done, info = self.env.step(0)
        if done:
            self.env.reset()
        return info
    
    def _extract_state(self):
        """Estrae vettore di stato 43D da osservazioni RGB e info."""
        info = self._last_info
        if info is None:
            return np.zeros(43, dtype=np.float32)
        
        # Composizione stato 43D:
        # [0-15]  = 16 elementi inventario (health, food, drink, energy, risorse, tools)
        # [16-17] = 2 coordinate posizione normalizzate
        # [18-20] = 3 status (discount/alive, sleeping, daylight)
        # [21-42] = 22 achievement binari
        state_parts = []
        
        # 16 elementi inventario (health, food, drink, energy, risorse, tools)
        inventory = info.get('inventory', {})
        inventory_keys = [
            'health', 'food', 'drink', 'energy', 'sapling',
            'wood', 'stone', 'coal', 'iron', 'diamond',
            'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
            'wood_sword', 'stone_sword', 'iron_sword'
        ]

        for key in inventory_keys:
            state_parts.append(float(inventory.get(key, 0)))

        # Posizione normalizzata 0-1
        player_pos = info.get('player_pos', np.array([32, 32]))
        state_parts.append(float(player_pos[0]) / 64.0)
        state_parts.append(float(player_pos[1]) / 64.0)
        
        # 3 status: discount (alive/dead), sleeping, daylight
        discount = info.get('discount', 1.0)
        state_parts.append(float(discount))
        
        sleeping = float(self._env._player.sleeping) if hasattr(self._env, '_player') else 0.0
        state_parts.append(sleeping)
        
        daylight = float(self._env._world.daylight) if hasattr(self._env, '_world') else 0.5
        state_parts.append(daylight)
        
        # 22 achievement binari (0 o 1)
        achievements = info.get('achievements', {})
        achievement_keys = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood',
            'defeat_skeleton', 'defeat_zombie',
            'eat_cow', 'eat_plant',
            'make_iron_pickaxe', 'make_iron_sword',
            'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword',
            'place_furnace', 'place_plant', 'place_stone', 'place_table',
            'wake_up'
        ]
        
        for key in achievement_keys:
            achievement_value = achievements.get(key, 0)
            state_parts.append(1.0 if achievement_value >= 1 else 0.0)
        
        state = np.array(state_parts, dtype=np.float32)
        return state
    
    def get_state_size(self):
        return self.state_size
    
    def get_action_size(self):
        return self.action_size
    
    def get_valid_actions(self):
        return list(range(self.action_size))
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()


class CrafterEnvRecorded(CrafterEnv):
    
    def __init__(self, 
                 area=(64, 64), 
                 view=(9, 9), 
                 size=(64, 64),
                 reward=True, 
                 length=10000, 
                 seed=None,
                 record_dir=None):
        env_name = 'CrafterReward-v1' if reward else 'CrafterNoReward-v1'
        base_env = crafter.Env(area=area, view=view, size=size, reward=reward, length=length, seed=seed)
        
        if record_dir:
            self.env = crafter.Recorder(
                base_env,
                record_dir,
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        else:
            self.env = base_env
            
        self._env = self.env

        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        self.action_names = [
            'noop',
            'move_left', 'move_right', 'move_up', 'move_down',
            'do', 'sleep',
            'place_stone', 'place_table', 'place_furnace', 'place_plant',
            'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
            'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
        ]
        
        self.state_size = 43
        self.action_size = 17
        self._last_info = None