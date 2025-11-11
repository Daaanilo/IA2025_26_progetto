"""
Deep Q-Network (DQN) Agent for Crafter Environment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
import random


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Crafter environment.
    Uses a CNN to process visual observations and outputs Q-values for actions.
    """
    
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int, config: Dict[str, Any]):
        """
        Initialize DQN network.
        
        Args:
            observation_shape: Shape of observation (C, H, W)
            num_actions: Number of available actions
            config: Network configuration
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.config = config
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        conv_out_size = self._get_conv_output_size(observation_shape)
        
        # Fully connected layers
        hidden_layers = config.get('hidden_layers', [256, 256])
        self.fc1 = nn.Linear(conv_out_size, hidden_layers[0])
        
        # Dueling DQN architecture (optional)
        self.dueling = config.get('dueling', True)
        
        if self.dueling:
            # Value stream
            self.value_fc = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.value = nn.Linear(hidden_layers[1], 1)
            
            # Advantage stream
            self.advantage_fc = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.advantage = nn.Linear(hidden_layers[1], num_actions)
        else:
            self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.output = nn.Linear(hidden_layers[1], num_actions)
    
    def _get_conv_output_size(self, shape: Tuple[int, ...]) -> int:
        """Calculate the output size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input observation tensor (batch_size, C, H, W)
            
        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        # Normalize pixel values
        x = x.float() / 255.0
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        if self.dueling:
            # Dueling architecture
            value = F.relu(self.value_fc(x))
            value = self.value(value)
            
            advantage = F.relu(self.advantage_fc(x))
            advantage = self.advantage(advantage)
            
            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            x = F.relu(self.fc2(x))
            q_values = self.output(x)
        
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer for DQN."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for playing Crafter.
    """
    
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int, 
                 config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize DQN agent.
        
        Args:
            observation_shape: Shape of observations
            num_actions: Number of available actions
            config: Agent configuration
            device: Device to use (cuda/cpu)
        """
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Networks
        network_config = config.get('network', {})
        self.policy_net = DQNNetwork(observation_shape, num_actions, network_config).to(self.device)
        self.target_net = DQNNetwork(observation_shape, num_actions, network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training parameters
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon = training_config.get('epsilon_start', 1.0)
        self.epsilon_end = training_config.get('epsilon_end', 0.01)
        self.epsilon_decay = training_config.get('epsilon_decay', 0.995)
        self.batch_size = training_config.get('batch_size', 64)
        self.target_update_freq = training_config.get('target_update_freq', 1000)
        self.learning_starts = training_config.get('learning_starts', 10000)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        buffer_size = training_config.get('buffer_size', 100000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training state
        self.total_steps = 0
        self.training_steps = 0
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            epsilon: Exploration rate (uses agent's epsilon if None)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size or self.total_steps < self.learning_starts:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def step(self):
        """Increment step counter and decay epsilon."""
        self.total_steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
