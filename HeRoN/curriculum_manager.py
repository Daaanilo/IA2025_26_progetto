"""
F09: Curriculum Learning Manager for HeRoN
Implements progressive difficulty, staged achievement targets, and adaptive reward shaping
"""

import numpy as np


class CurriculumManager:
    """
    Manages curriculum learning stages for Crafter environment training.
    
    Stages:
    1. Early (episodes 0-33%): Focus on basic resource collection (wood, stone)
    2. Mid (episodes 33%-66%): Focus on crafting and structure placement
    3. Late (episodes 66%-100%): Focus on advanced resources and combat
    """
    
    ACHIEVEMENT_TIERS = {
        'early': [
            'collect_wood', 'collect_stone', 'collect_drink', 
            'place_stone', 'eat_plant'
        ],
        'mid': [
            'place_table', 'place_plant', 'place_furnace',
            'make_wood_pickaxe', 'make_wood_sword', 'make_stone_pickaxe'
        ],
        'late': [
            'collect_iron', 'collect_coal', 'collect_diamond',
            'make_iron_pickaxe', 'make_iron_sword', 
            'defeat_zombie', 'defeat_skeleton', 'eat_cow',
            'wake_up'
        ]
    }
    
    def __init__(self, total_episodes):
        self.total_episodes = total_episodes
        self.current_stage = 'early'
        self.stage_thresholds = {
            'early': int(total_episodes * 0.33),
            'mid': int(total_episodes * 0.66)
        }
    
    def get_stage(self, episode):
        """Determine current curriculum stage based on episode number."""
        if episode < self.stage_thresholds['early']:
            return 'early'
        elif episode < self.stage_thresholds['mid']:
            return 'mid'
        else:
            return 'late'
    
    def get_target_achievements(self, episode):
        """Return priority achievement targets for current stage."""
        stage = self.get_stage(episode)
        return self.ACHIEVEMENT_TIERS[stage]
    
    def get_episode_length(self, episode):
        """Progressive episode length: 500 → 2000 steps over training."""
        min_length = 500
        max_length = 2000
        
        # Linear interpolation
        progress = episode / max(1, self.total_episodes - 1)
        length = int(min_length + (max_length - min_length) * progress)
        
        return min(length, max_length)
    
    def should_increase_difficulty(self, episode, achievements_unlocked):
        """
        Check if agent is ready for next stage based on performance.
        
        Criteria: Agent should unlock at least 50% of current stage's achievements
        before progressing to next stage.
        """
        stage = self.get_stage(episode)
        target_achievements = set(self.ACHIEVEMENT_TIERS[stage])
        
        # Check how many target achievements have been unlocked
        unlocked_count = len(achievements_unlocked & target_achievements)
        required_count = len(target_achievements) * 0.5
        
        return unlocked_count >= required_count
    
    def get_reward_shaping_weights(self, episode, achievement_rate):
        """
        Adaptive reward shaping weights based on episode and performance.
        
        Args:
            episode: Current episode number
            achievement_rate: Recent achievement unlock rate (achievements/episode)
        
        Returns:
            dict: Reward shaping weight multipliers
        """
        stage = self.get_stage(episode)
        
        # Base weights
        weights = {
            'resource_collection': 0.1,
            'health_management': 0.05,
            'tier_progression': 0.05,
            'tool_usage': 0.02
        }
        
        # Stage-specific adjustments
        if stage == 'early':
            # Emphasize resource collection in early stage
            weights['resource_collection'] *= 1.5
        elif stage == 'mid':
            # Emphasize tool usage and tier progression in mid stage
            weights['tool_usage'] *= 2.0
            weights['tier_progression'] *= 1.5
        else:  # late
            # Emphasize tier progression and health management in late stage
            weights['tier_progression'] *= 2.0
            weights['health_management'] *= 1.5
        
        # Adaptive adjustment based on achievement rate
        if achievement_rate < 0.1:
            # Low achievement rate: increase exploration bonuses
            weights['resource_collection'] *= 1.3
            weights['tool_usage'] *= 1.3
        elif achievement_rate > 0.5:
            # High achievement rate: reduce bonuses (agent is doing well)
            for key in weights:
                weights[key] *= 0.8
        
        return weights


class HyperparameterScheduler:
    """
    Manages dynamic hyperparameter adjustment during training.
    
    Schedules:
    - Learning rate decay
    - Epsilon decay (exploration)
    - Threshold decay (LLM involvement)
    """
    
    def __init__(self, initial_lr=0.001, initial_epsilon=1.0, initial_threshold=1.0):
        self.initial_lr = initial_lr
        self.initial_epsilon = initial_epsilon
        self.initial_threshold = initial_threshold
        
        # Tracking
        self.lr_history = []
        self.epsilon_history = []
        self.threshold_history = []
    
    def get_learning_rate(self, episode, total_episodes, strategy='step_decay'):
        """
        Calculate learning rate for current episode.
        
        Strategies:
            - 'constant': No decay
            - 'step_decay': Decay by 0.5 every 100 episodes
            - 'exponential': Exponential decay to 10% of initial
            - 'cosine': Cosine annealing
        """
        if strategy == 'constant':
            lr = self.initial_lr
        
        elif strategy == 'step_decay':
            decay_factor = 0.9
            decay_interval = 100
            decay_count = episode // decay_interval
            lr = self.initial_lr * (decay_factor ** decay_count)
        
        elif strategy == 'exponential':
            # Decay to 10% over total episodes
            decay_rate = -np.log(0.1) / total_episodes
            lr = self.initial_lr * np.exp(-decay_rate * episode)
        
        elif strategy == 'cosine':
            # Cosine annealing
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * episode / total_episodes))
        
        else:
            lr = self.initial_lr
        
        # Minimum learning rate floor
        lr = max(lr, 1e-6)
        
        self.lr_history.append(lr)
        return lr
    
    def get_epsilon(self, episode, total_episodes, strategy='linear_decay'):
        """
        Calculate epsilon (exploration rate) for current episode.
        
        Strategies:
            - 'linear_decay': Linear decay from 1.0 to 0.01
            - 'exponential_decay': Exponential decay
            - 'staged': High exploration early, low exploration late
        """
        if strategy == 'linear_decay':
            # Linear decay from initial_epsilon to 0.01
            epsilon = self.initial_epsilon - (self.initial_epsilon - 0.01) * (episode / total_episodes)
        
        elif strategy == 'exponential_decay':
            # Exponential decay
            decay_rate = 0.995
            epsilon = max(0.01, self.initial_epsilon * (decay_rate ** episode))
        
        elif strategy == 'staged':
            # High exploration first 30%, moderate next 40%, low final 30%
            progress = episode / total_episodes
            if progress < 0.3:
                epsilon = self.initial_epsilon
            elif progress < 0.7:
                epsilon = 0.5
            else:
                epsilon = 0.1
        
        else:
            epsilon = self.initial_epsilon
        
        epsilon = max(0.01, min(1.0, epsilon))
        self.epsilon_history.append(epsilon)
        return epsilon
    
    def get_threshold(self, episode, threshold_episodes, strategy='linear_decay'):
        """
        Calculate threshold (LLM involvement probability) for current episode.
        
        Strategies:
            - 'linear_decay': Linear decay from 1.0 to 0.0
            - 'staged': Full LLM early, gradual reduction mid, DQN-only late
        """
        if episode >= threshold_episodes:
            threshold = 0.0
        elif strategy == 'linear_decay':
            threshold = self.initial_threshold * (1 - episode / threshold_episodes)
        elif strategy == 'staged':
            progress = episode / threshold_episodes
            if progress < 0.2:
                threshold = 1.0
            elif progress < 0.5:
                threshold = 0.7
            elif progress < 0.8:
                threshold = 0.3
            else:
                threshold = 0.1
        else:
            threshold = self.initial_threshold
        
        threshold = max(0.0, min(1.0, threshold))
        self.threshold_history.append(threshold)
        return threshold
    
    def should_adjust_hyperparameters(self, episode, performance_history, window=10):
        """
        Determine if hyperparameters should be adjusted based on performance plateau.
        
        Returns True if performance has plateaued (no improvement in last window episodes).
        """
        if len(performance_history) < window + 1:
            return False
        
        recent_performance = performance_history[-window:]
        previous_performance = performance_history[-(window + 1):-1]
        
        recent_mean = np.mean(recent_performance)
        previous_mean = np.mean(previous_performance)
        
        # Performance plateau: less than 5% improvement
        improvement = (recent_mean - previous_mean) / max(abs(previous_mean), 1e-6)
        
        return abs(improvement) < 0.05


class EarlyStoppingManager:
    """
    Manages early stopping criteria for training convergence.
    """
    
    def __init__(self, patience=100, min_episodes=200, convergence_threshold=0.05):
        self.patience = patience
        self.min_episodes = min_episodes
        self.convergence_threshold = convergence_threshold
        
        self.best_achievement_count = 0
        self.episodes_without_improvement = 0
        self.convergence_history = []
    
    def should_stop(self, episode, achievements_history):
        """
        Determine if training should stop early.
        
        Criteria:
        1. Minimum episodes reached
        2. No improvement for 'patience' episodes
        3. Achievement rate has converged (low variance)
        """
        if episode < self.min_episodes:
            return False
        
        # Check for improvement
        recent_achievements = sum(achievements_history[-10:]) if len(achievements_history) >= 10 else 0
        
        if recent_achievements > self.best_achievement_count:
            self.best_achievement_count = recent_achievements
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Check patience
        if self.episodes_without_improvement >= self.patience:
            print(f"\n[Early Stopping] No improvement for {self.patience} episodes")
            return True
        
        # Check convergence (achievement rate variance)
        if len(achievements_history) >= 50:
            recent_variance = np.var(achievements_history[-50:])
            self.convergence_history.append(recent_variance)
            
            if recent_variance < self.convergence_threshold:
                print(f"\n[Early Stopping] Achievement rate converged (variance: {recent_variance:.4f})")
                return True
        
        return False
    
    def get_status(self):
        """Return current early stopping status."""
        return {
            'best_achievement_count': self.best_achievement_count,
            'episodes_without_improvement': self.episodes_without_improvement,
            'patience_remaining': self.patience - self.episodes_without_improvement
        }
