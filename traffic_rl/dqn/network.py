"""
Deep Q-Network (DQN) neural network architecture.

This module defines the neural network used for Q-value approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """
    Deep Q-Network with fully connected layers.
    
    Architecture:
        Input -> FC1 -> ReLU -> FC2 -> ReLU -> ... -> Output
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = None):
        """
        Initialize DQN network.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_layers: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values for each action, shape (batch_size, action_size)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get greedy action for a single state.
        
        Args:
            state: State tensor of shape (state_size,)
        
        Returns:
            Action with highest Q-value
        """
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    # Test network
    state_size = 9
    action_size = 2
    batch_size = 32
    
    model = DQN(state_size, action_size, hidden_layers=[128, 128])
    print(model)
    
    # Test forward pass
    dummy_state = torch.randn(batch_size, state_size)
    q_values = model(dummy_state)
    print(f"\nInput shape: {dummy_state.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values sample: {q_values[0]}")
