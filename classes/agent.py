from collections import deque
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """PyTorch neural network for DQN (GPU-compatible)."""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size, load_model_path=None, learning_rate=0.001, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = epsilon  # exploration rate (F09: now parameterized, default 1.0)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate  # F09: now parameterized for dynamic adjustment
        
        # GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_model_path:
            self.load(load_model_path)
        else:
            self.model = self._build_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        return model
    
    def update_learning_rate(self, new_lr):
        """F09: Update learning rate dynamically during training."""
        self.learning_rate = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def load(self, path_prefix):
        # F09: Convert to relative paths for cross-platform compatibility
        model_path = f"{path_prefix}.pth"  # PyTorch format
        if os.path.exists(model_path):
            self.model = self._build_model()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        memory_path = f"{path_prefix}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
        epsilon_path = f"{path_prefix}_epsilon.txt"
        if os.path.exists(epsilon_path):
            with open(epsilon_path, 'r') as f:
                self.epsilon = float(f.read().strip())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        valid_actions = env.get_valid_actions()
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # PyTorch prediction on GPU
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor).cpu().numpy()[0]
        
        if len(valid_actions) < self.action_size:
            masked_q_values = np.full_like(act_values, -np.inf)  
            masked_q_values[valid_actions] = act_values[valid_actions] 
            return np.argmax(masked_q_values)

        return np.argmax(act_values)

    def replay(self, batch_size, env):
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # PyTorch prediction for next_state on GPU
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor).cpu().numpy()[0]
                
                valid_actions = env.get_valid_actions()
                if len(valid_actions) < self.action_size:
                    masked_q_values = np.full_like(next_q_values, -np.inf)
                    masked_q_values[valid_actions] = next_q_values[valid_actions]
                    target = reward + self.gamma * np.amax(masked_q_values)
                else:
                    target = reward + self.gamma * np.amax(next_q_values)
            
            # Get current Q-values
            state_tensor = torch.FloatTensor(state).to(self.device)
            self.model.eval()
            with torch.no_grad():
                target_f = self.model(state_tensor).cpu().numpy()
            
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])
        
        # Batch training on GPU
        self.model.train()
        states_batch = torch.FloatTensor(np.array(states)).to(self.device)
        targets_batch = torch.FloatTensor(np.array(targets)).to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(states_batch)
        loss = self.criterion(predictions, targets_batch)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path_prefix):
        # F09: Convert to relative paths for cross-platform compatibility
        # Create directory if needed
        save_dir = os.path.dirname(path_prefix)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        model_path = f"{path_prefix}.pth"  # PyTorch format
        torch.save(self.model.state_dict(), model_path)
        with open(f"{path_prefix}_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)
        with open(f"{path_prefix}_epsilon.txt", 'w') as f:
            f.write(str(self.epsilon))

'''
    def replay(self, batch_size, env):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                valid_actions = env.get_valid_actions()
                q_values_next = self.model.predict(next_state, verbose=0)[0]
                filtered_q = [q_values_next[i] for i in valid_actions]
                target = reward + self.gamma * max(filtered_q) if filtered_q else reward

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
'''


