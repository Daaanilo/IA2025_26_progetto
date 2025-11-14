import numpy as np
import gymnasium as gym
from gymnasium import spaces
import crafter


class CrafterEnv:
    """
    Wrapper per l'environment Crafter che estrae feature semantiche da info dict
    e le converte in vettore numerico compatibile con DQN.
    
    Feature vector (43 dims) - Official Crafter Specification:
    - Inventario (16 dims): health, food, drink, energy, sapling, wood, stone, coal, iron, diamond,
                            wood_pickaxe, stone_pickaxe, iron_pickaxe,
                            wood_sword, stone_sword, iron_sword
    - Posizione giocatore (2 dims): x, y
    - Status (3 dims): discount (alive/dead), sleeping, daylight
    - Achievements (22 dims): one-hot encoding per ogni achievement
    
    Totale: 16 + 2 + 3 + 22 = 43 dims
    """
    
    def __init__(self, 
                 area=(64, 64), 
                 view=(9, 9), 
                 size=(64, 64),
                 reward=True, 
                 length=10000, 
                 seed=None):
        """
        Inizializza l'environment Crafter con wrapper semantico.
        
        Args:
            area: dimensioni del mondo Crafter (64x64 default)
            view: dimensioni della vista locale (9x9 default)
            size: dimensioni render (64x64 default)
            reward: se True usa rewards, altrimenti no
            length: lunghezza episodio in steps (10000 default)
            seed: seed per RNG
        """
        # Usa crafter.Env() direttamente invece di gym.make()
        self.env = crafter.Env(area=area, view=view, size=size, reward=reward, length=length, seed=seed)
        
        # Store per accesso a funzionalità interne (per feature extraction avanzata)
        self._env = self.env
        
        # Crafter ha 17 azioni discrete
        self.action_space = spaces.Discrete(17)
        # Observation space è immagine RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        # Action names mapping (OFFICIAL CRAFTER ORDER)
        self.action_names = [
            'noop',              # 0
            'move_left',         # 1
            'move_right',        # 2
            'move_up',           # 3
            'move_down',         # 4
            'do',                # 5
            'sleep',             # 6
            'place_stone',       # 7
            'place_table',       # 8
            'place_furnace',     # 9
            'place_plant',       # 10
            'make_wood_pickaxe', # 11
            'make_stone_pickaxe',# 12
            'make_iron_pickaxe', # 13
            'make_wood_sword',   # 14
            'make_stone_sword',  # 15
            'make_iron_sword'    # 16
        ]
        
        self.state_size = 43  # 16 inventory + 2 position + 3 status + 22 achievements
        self.action_size = 17
        self._last_info = None
        
    def reset(self):
        """
        Reset environment e ritorna stato iniziale come vettore numerico + info dict.
        """
        obs = self.env.reset()
        # Dummy step per ottenere info dict
        self._last_info = self._get_dummy_info()
        state = self._extract_state()
        self.state_size = len(state)
        return state, self._last_info
    
    def step(self, action):
        """
        Esegui azione e ritorna (state, reward, done, info).
        """
        obs, reward, done, info = self.env.step(action)
        self._last_info = info
        state = self._extract_state()
        return state, reward, done, info
    
    def _get_dummy_info(self):
        """Ritorna un info dict dummy con tutti i campi necessari."""
        obs, reward, done, info = self.env.step(0)  # noop action (now at index 0)
        if done:
            self.env.reset()
        return info
    
    def _extract_state(self):
        """
        Estrai feature semantiche da info dict e stato interno.
        Ritorna vettore numpy (43 dims).
        """
        info = self._last_info
        if info is None:
            return np.zeros(43, dtype=np.float32)
        
        state_parts = []
        
        # === Inventario (16 dims) - Official Crafter Items ===
        inventory = info.get('inventory', {})
        inventory_keys = [
            'health', 'food', 'drink', 'energy', 'sapling',
            'wood', 'stone', 'coal', 'iron', 'diamond',
            'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
            'wood_sword', 'stone_sword', 'iron_sword'
        ]
        
        # Alcuni items potrebbero non essere in inventory, usa default 0
        for key in inventory_keys:
            state_parts.append(float(inventory.get(key, 0)))
        
        # === Posizione giocatore (2 dims) ===
        player_pos = info.get('player_pos', np.array([32, 32]))
        # Normalizza a [0, 1]
        state_parts.append(float(player_pos[0]) / 64.0)
        state_parts.append(float(player_pos[1]) / 64.0)
        
        # === Status del giocatore (3 dims) ===
        # Discount: 1.0 se vivo, 0.0 se morto
        discount = info.get('discount', 1.0)
        state_parts.append(float(discount))
        
        # Sleeping status
        sleeping = float(self._env._player.sleeping) if hasattr(self._env, '_player') else 0.0
        state_parts.append(sleeping)
        
        # Daylight (0-1)
        daylight = float(self._env._world.daylight) if hasattr(self._env, '_world') else 0.5
        state_parts.append(daylight)
        
        # === Achievements (22 dims) - Official Crafter Achievements ===
        # One-hot: 1.0 se achievement sbloccato (count >= 1), 0.0 altrimenti
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
        """Ritorna dimensione dello stato."""
        return self.state_size
    
    def get_action_size(self):
        """Ritorna numero di azioni possibili."""
        return self.action_size
    
    def get_valid_actions(self):
        """
        In Crafter, tutte le azioni sono sempre valide.
        Ritorna lista [0, 1, ..., 16].
        """
        return list(range(self.action_size))
    
    def render(self, mode='human'):
        """Renderizza l'environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Chiudi l'environment."""
        self.env.close()


class CrafterEnvRecorded(CrafterEnv):
    """
    Estensione di CrafterEnv con Recorder per salvare video/statistiche.
    """
    
    def __init__(self, 
                 area=(64, 64), 
                 view=(9, 9), 
                 size=(64, 64),
                 reward=True, 
                 length=10000, 
                 seed=None,
                 record_dir=None):
        """
        Inizializza CrafterEnv con Recorder opzionale.
        
        Args:
            record_dir: directory per salvare stats/video (es. './logdir/experiment_0')
        """
        # Inizializza base environment
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
        
        # Inizializza resto come CrafterEnv
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