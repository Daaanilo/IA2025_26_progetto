from collections import deque, namedtuple
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))


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
    def __init__(self, state_size, action_size, load_model_path=None, learning_rate=0.0001, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        
        # Memory più grande, ma occhio alla GPU da 6GB
        self.memory = deque(maxlen=5000)
        self.priorities = deque(maxlen=5000)  # Prioritized Replay
        
        # Parametri per Crafter
        self.gamma = 0.99  # Discount alto per rewards sparse
        self.epsilon = epsilon
        self.epsilon_min = 0.05  # Più esplorazione alla fine
        self.epsilon_decay_episodes = 300  # Scende lineare in 300 episodi
        self.learning_rate = learning_rate
        
        # Target network separata
        self.target_update_freq = 100  # Ogni quanto aggiorno la target
        self.train_step_counter = 0
        
        # Parametri Prioritized Replay
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001
        self.priority_epsilon = 1e-6  # Per evitare priorità zero
        
        # Usa GPU se c'è
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_model_path:
            self.load(load_model_path)
        else:
            # Due reti per Double DQN
            self.model = self._build_model()
            self.target_model = self._build_model()
            self._update_target_model()  # Sincronizza subito
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        return model
    
    def _update_target_model(self):
        """Copia i pesi sulla target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def update_learning_rate(self, new_lr):
        """Aggiorna il learning rate al volo."""
        self.learning_rate = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def load(self, path_prefix):
        model_path = f"{path_prefix}.pth"
        if os.path.exists(model_path):
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self._update_target_model()  
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        memory_path = f"{path_prefix}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
        
        
        priorities_path = f"{path_prefix}_priorities.pkl"
        if os.path.exists(priorities_path):
            with open(priorities_path, 'rb') as f:
                self.priorities = pickle.load(f)
        
        epsilon_path = f"{path_prefix}_epsilon.txt"
        if os.path.exists(epsilon_path):
            with open(epsilon_path, 'r') as f:
                self.epsilon = float(f.read().strip())

    def remember(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def act(self, state, env):
        valid_actions = env.get_valid_actions()
        if np.random.rand() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            self.model.eval()
            with torch.no_grad():
                act_values = self.model(state_tensor).cpu().numpy()[0]
            
            if len(valid_actions) < self.action_size:
                masked_q_values = np.full_like(act_values, -np.inf)
                masked_q_values[valid_actions] = act_values[valid_actions]
                action = np.argmax(masked_q_values)
            else:
                action = np.argmax(act_values)

        return action

    def decay_epsilon_linear(self, episode, total_episodes=None):
        if total_episodes is None:
            total_episodes = self.epsilon_decay_episodes

        if episode >= total_episodes:
            self.epsilon = self.epsilon_min
        else:
            decay_progress = episode / total_episodes
            self.epsilon = max(self.epsilon_min, 1.0 - decay_progress * (1.0 - self.epsilon_min))

    def replay(self, batch_size, env):
        """Training con Prioritized Replay e Double DQN."""
        if len(self.memory) < batch_size:
            return

        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.priority_alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        
        weights = (len(self.memory) * probabilities[indices]) ** (-self.priority_beta)
        weights /= weights.max() 
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        # Prepara i batch
        states = []
        targets = []
        td_errors = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.memory[idx]
            
            target = reward
            if not done:
                # Double DQN: Policy sceglie, Target valuta
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                self.model.eval()
                self.target_model.eval()
                
                with torch.no_grad():
                    # La Policy sceglie l'azione migliore
                    next_q_policy = self.model(next_state_tensor).cpu().numpy()[0]
                    valid_actions = env.get_valid_actions()
                    
                    if len(valid_actions) < self.action_size:
                        masked_q = np.full_like(next_q_policy, -np.inf)
                        masked_q[valid_actions] = next_q_policy[valid_actions]
                        best_action = np.argmax(masked_q)
                    else:
                        best_action = np.argmax(next_q_policy)
                    
                    # La Target valuta quell'azione
                    next_q_target = self.target_model(next_state_tensor).cpu().numpy()[0]
                    target = reward + self.gamma * next_q_target[best_action]
            
            # Q-values attuali
            state_tensor = torch.FloatTensor(state).to(self.device)
            self.model.eval()
            with torch.no_grad():
                current_q = self.model(state_tensor).cpu().numpy()[0]
            
            # Errore TD per aggiornare le priorità
            td_error = abs(target - current_q[action])
            td_errors.append(td_error)
            
            target_f = current_q.copy()
            target_f[action] = target
            states.append(state[0])
            targets.append(target_f)
        
        # Training su GPU con i pesi
        self.model.train()
        states_batch = torch.FloatTensor(np.array(states)).to(self.device)
        targets_batch = torch.FloatTensor(np.array(targets)).to(self.device)
        weights_batch = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(states_batch)
        
        # Loss pesata
        loss = (weights_batch * ((predictions - targets_batch) ** 2).mean(dim=1)).mean()
        loss.backward()
        
        # Clipping per stabilità
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Aggiorna priorità
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = td_error + self.priority_epsilon
        
        # Aggiorna la target network 
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_model()
        
        # Epsilon decay viene fatto in act()

    def save(self, path_prefix):
        save_dir = os.path.dirname(path_prefix)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        model_path = f"{path_prefix}.pth"
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.target_model.state_dict(), f"{path_prefix}_target.pth")
        
        with open(f"{path_prefix}_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)
        

        with open(f"{path_prefix}_priorities.pkl", 'wb') as f:
            pickle.dump(self.priorities, f)
        
        with open(f"{path_prefix}_epsilon.txt", 'w') as f:
            f.write(str(self.epsilon))



