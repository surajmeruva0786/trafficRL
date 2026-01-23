"""
DQN Agent for Traffic Signal Control.

This module implements the DQN agent with epsilon-greedy exploration,
experience replay, and target network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from .network import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Implements:
    - Epsilon-greedy action selection
    - Experience replay
    - Target network for stable training
    - Huber loss for robust learning
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any],
        device: str = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            config: Configuration dictionary with hyperparameters
            device: Device to use ('cuda' or 'cpu')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Set device with explicit GPU detection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
            else:
                self.device = torch.device("cpu")
                print("WARNING: CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create main and target networks
        hidden_layers = config.get('hidden_layers', [128, 128])
        self.policy_net = DQN(state_size, action_size, hidden_layers).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        buffer_size = config.get('buffer_size', 50000)
        self.memory = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.min_buffer_size = config.get('min_buffer_size', 1000)
        
        # Epsilon-greedy parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 50000)
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        
        # Target network update
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.steps_done = 0
        
        # Training metrics
        self.loss_history = []
    
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use agent's current epsilon)
        
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor.unsqueeze(0))
                return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value (0 if not enough samples)
        """
        # Check if enough samples in buffer
        if not self.memory.is_ready(self.min_buffer_size):
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Store loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save(self, filepath: str) -> None:
        """
        Save agent's networks and training state.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent's networks and training state.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }
