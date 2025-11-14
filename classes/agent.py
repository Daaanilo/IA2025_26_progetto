from collections import deque, namedtuple
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Named tuple for prioritized replay
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
    def __init__(self, state_size, action_size, load_model_path=None, learning_rate=0.0003, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        
        # OTTIMIZZAZIONE: Memory più grande per esperienze diversificate
        self.memory = deque(maxlen=10000)  # Era 2000
        self.priorities = deque(maxlen=10000)  # Per Prioritized Replay
        
        # OTTIMIZZAZIONE: Iperparametri ottimizzati per Crafter
        self.gamma = 0.99  # Era 0.95 - discount maggiore per rewards sparse
        self.epsilon = epsilon
        self.epsilon_min = 0.05  # Era 0.01 - più esplorazione
        self.epsilon_decay = 0.998  # Era 0.995 - decay più lento
        self.learning_rate = learning_rate  # Ridotto da 0.001 a 0.0003
        
        # DOUBLE DQN: Target network separata
        self.target_update_freq = 100  # Update target network ogni N steps
        self.train_step_counter = 0
        
        # PRIORITIZED REPLAY: Parametri
        self.priority_alpha = 0.6  # Quanto prioritizzare esperienze importanti
        self.priority_beta = 0.4  # Importance sampling correction
        self.priority_beta_increment = 0.001  # Incremento graduale
        self.priority_epsilon = 1e-6  # Evita priorità zero
        
        # GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if load_model_path:
            self.load(load_model_path)
        else:
            # DOUBLE DQN: Due reti separate
            self.model = self._build_model()
            self.target_model = self._build_model()
            self._update_target_model()  # Sincronizza inizialmente
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        return model
    
    def _update_target_model(self):
        """DOUBLE DQN: Copia i pesi dalla policy network alla target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
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
            self.target_model = self._build_model()  # DOUBLE DQN
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self._update_target_model()  # Sincronizza target
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        memory_path = f"{path_prefix}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
        
        # Carica anche le priorità se esistono
        priorities_path = f"{path_prefix}_priorities.pkl"
        if os.path.exists(priorities_path):
            with open(priorities_path, 'rb') as f:
                self.priorities = pickle.load(f)
        
        epsilon_path = f"{path_prefix}_epsilon.txt"
        if os.path.exists(epsilon_path):
            with open(epsilon_path, 'r') as f:
                self.epsilon = float(f.read().strip())

    def remember(self, state, action, reward, next_state, done):
        """PRIORITIZED REPLAY: Aggiungi esperienza con priorità iniziale massima."""
        # Nuove esperienze hanno priorità massima per essere campionate subito
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

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
        """OTTIMIZZATO: Prioritized Replay + Double DQN + Gradient Clipping."""
        if len(self.memory) < batch_size:
            return
        
        # PRIORITIZED REPLAY: Campiona in base alle priorità
        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.priority_alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        
        # Importance sampling weights per bias correction
        weights = (len(self.memory) * probabilities[indices]) ** (-self.priority_beta)
        weights /= weights.max()  # Normalizza
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        # Prepare batch tensors
        states = []
        targets = []
        td_errors = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.memory[idx]
            
            target = reward
            if not done:
                # DOUBLE DQN: Policy network seleziona azione, target network valuta
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                self.model.eval()
                self.target_model.eval()
                
                with torch.no_grad():
                    # Policy network sceglie best action
                    next_q_policy = self.model(next_state_tensor).cpu().numpy()[0]
                    valid_actions = env.get_valid_actions()
                    
                    if len(valid_actions) < self.action_size:
                        masked_q = np.full_like(next_q_policy, -np.inf)
                        masked_q[valid_actions] = next_q_policy[valid_actions]
                        best_action = np.argmax(masked_q)
                    else:
                        best_action = np.argmax(next_q_policy)
                    
                    # Target network valuta quella action
                    next_q_target = self.target_model(next_state_tensor).cpu().numpy()[0]
                    target = reward + self.gamma * next_q_target[best_action]
            
            # Get current Q-values
            state_tensor = torch.FloatTensor(state).to(self.device)
            self.model.eval()
            with torch.no_grad():
                current_q = self.model(state_tensor).cpu().numpy()[0]
            
            # TD error per aggiornare priorità
            td_error = abs(target - current_q[action])
            td_errors.append(td_error)
            
            target_f = current_q.copy()
            target_f[action] = target
            states.append(state[0])
            targets.append(target_f)
        
        # Batch training on GPU con importance sampling weights
        self.model.train()
        states_batch = torch.FloatTensor(np.array(states)).to(self.device)
        targets_batch = torch.FloatTensor(np.array(targets)).to(self.device)
        weights_batch = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(states_batch)
        
        # Loss pesato per importance sampling
        loss = (weights_batch * ((predictions - targets_batch) ** 2).mean(dim=1)).mean()
        loss.backward()
        
        # GRADIENT CLIPPING: Stabilizza training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # PRIORITIZED REPLAY: Aggiorna priorità con TD errors
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = td_error + self.priority_epsilon
        
        # DOUBLE DQN: Aggiorna target network periodicamente
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_model()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path_prefix):
        # F09: Convert to relative paths for cross-platform compatibility
        # Create directory if needed
        save_dir = os.path.dirname(path_prefix)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Salva entrambe le reti (DOUBLE DQN)
        model_path = f"{path_prefix}.pth"  # PyTorch format
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.target_model.state_dict(), f"{path_prefix}_target.pth")
        
        with open(f"{path_prefix}_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)
        
        # Salva anche le priorità
        with open(f"{path_prefix}_priorities.pkl", 'wb') as f:
            pickle.dump(self.priorities, f)
        
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


