import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    """Réseau de neurones pour DQN"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Agent DQN (Deep Q-Network) pour RL.
    
    Implémente:
    - Experience replay buffer
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(self, state_dim, action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=32,
                 device='cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Réseaux
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Statistiques
        self.training_losses = []
        self.episode_rewards = []
    
    def choose_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        
        Args:
            state: vecteur d'état
            training: si True, utilise epsilon-greedy; sinon greedy pur
        """
        if training and random.random() < self.epsilon:
            # Action aléatoire
            return random.randint(0, self.action_dim - 1)
        else:
            # Action greedy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Ajoute transition au replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Entraîne le réseau sur un batch du replay buffer.
        
        Returns:
            loss: MSE loss du batch
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir en tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calcul Q-target
        with torch.no_grad():
            q_next = self.target_network(next_states)
            q_max_next = q_next.max(dim=1)[0]
            q_target = rewards + self.gamma * q_max_next * (1 - dones)
        
        # Calcul Q-predicted
        q_predicted = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Perte
        loss = self.loss_fn(q_predicted, q_target)
        
        # Rétropropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Copie poids du Q-network au target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Décroissance epsilon-greedy"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.training_losses,
            'rewards': self.episode_rewards
        }, filepath)
    
    def load(self, filepath):
        """Charge le modèle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint['losses']
        self.episode_rewards = checkpoint['rewards']
    
    def get_statistics(self):
        """Retourne stats d'entraînement"""
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }
