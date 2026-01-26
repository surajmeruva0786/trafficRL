"""
Deep Q-Network (DQN) neural network architecture.

This module defines both standard DQN and Multi-Head DQN architectures.
The Multi-Head DQN uses a shared encoder with multiple specialized Q-heads
for different traffic regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


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


class SharedEncoder(nn.Module):
    """
    Shared feature encoder for Multi-Head DQN.
    
    Extracts common features from state that are shared across all Q-heads.
    """
    
    def __init__(self, state_size: int, hidden_layers: List[int] = None, dropout: float = 0.1):
        """
        Initialize shared encoder.
        
        Args:
            state_size: Dimension of state space
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability for regularization
        """
        super(SharedEncoder, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 256]
        
        self.state_size = state_size
        self.output_size = hidden_layers[-1]
        
        # Build encoder layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state into feature representation.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
        
        Returns:
            Encoded features of shape (batch_size, output_size)
        """
        return self.encoder(state)


class QHead(nn.Module):
    """
    Individual Q-value head specialized for a specific traffic regime.
    """
    
    def __init__(self, input_size: int, action_size: int, hidden_layers: List[int] = None):
        """
        Initialize Q-head.
        
        Args:
            input_size: Size of input features (from encoder)
            action_size: Number of actions
            hidden_layers: List of hidden layer sizes
        """
        super(QHead, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128]
        
        self.input_size = input_size
        self.action_size = action_size
        
        # Build head layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, action_size))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values from encoded features.
        
        Args:
            features: Encoded features of shape (batch_size, input_size)
        
        Returns:
            Q-values of shape (batch_size, action_size)
        """
        return self.head(features)


class MultiHeadDQN(nn.Module):
    """
    Multi-Head Deep Q-Network with regime-specific heads.
    
    Architecture:
        State → Shared Encoder → Encoded Features
                                      ↓
                        ┌─────────────┼─────────────┐
                        ↓             ↓             ↓
                    Q-Head 0      Q-Head 1      Q-Head 2
                    (Low)         (Medium)      (High)
        
        State → Regime Classifier → Regime Probabilities
        
        Final Q-values = Gating(Q-heads, Regime Probabilities)
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_heads: int = 3,
        encoder_layers: List[int] = None,
        head_layers: List[int] = None,
        classifier_layers: List[int] = None,
        gating_type: str = "soft",
        encoder_dropout: float = 0.1,
        classifier_dropout: float = 0.2
    ):
        """
        Initialize Multi-Head DQN.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            num_heads: Number of Q-heads (one per regime)
            encoder_layers: Hidden layer sizes for shared encoder
            head_layers: Hidden layer sizes for each Q-head
            classifier_layers: Hidden layer sizes for regime classifier
            gating_type: "hard" (select one head) or "soft" (weighted mixture)
            encoder_dropout: Dropout for encoder
            classifier_dropout: Dropout for classifier
        """
        super(MultiHeadDQN, self).__init__()
        
        if encoder_layers is None:
            encoder_layers = [256, 256]
        if head_layers is None:
            head_layers = [128]
        if classifier_layers is None:
            classifier_layers = [128, 64]
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_heads = num_heads
        self.gating_type = gating_type
        
        # Shared encoder
        self.encoder = SharedEncoder(state_size, encoder_layers, encoder_dropout)
        encoder_output_size = encoder_layers[-1]
        
        # Multiple Q-heads (one per regime)
        self.heads = nn.ModuleList([
            QHead(encoder_output_size, action_size, head_layers)
            for _ in range(num_heads)
        ])
        
        # Regime classifier (operates on raw state)
        classifier_net = []
        input_size = state_size
        for hidden_size in classifier_layers:
            classifier_net.append(nn.Linear(input_size, hidden_size))
            classifier_net.append(nn.ReLU())
            classifier_net.append(nn.Dropout(classifier_dropout))
            input_size = hidden_size
        classifier_net.append(nn.Linear(input_size, num_heads))
        
        self.classifier = nn.Sequential(*classifier_net)
    
    def forward(
        self,
        state: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through multi-head network.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
            return_all: If True, return all head outputs and regime probs
        
        Returns:
            If return_all=False:
                q_values: Final Q-values of shape (batch_size, action_size)
            If return_all=True:
                (q_values, all_head_outputs, regime_probs)
        """
        # Encode state
        features = self.encoder(state)
        
        # Get Q-values from all heads
        head_outputs = torch.stack([head(features) for head in self.heads], dim=1)
        # Shape: (batch_size, num_heads, action_size)
        
        # Get regime probabilities
        regime_logits = self.classifier(state)
        regime_probs = F.softmax(regime_logits, dim=1)
        # Shape: (batch_size, num_heads)
        
        # Apply gating mechanism
        if self.gating_type == "hard":
            # Hard selection: choose head with highest probability
            selected_heads = regime_probs.argmax(dim=1)  # (batch_size,)
            q_values = head_outputs[torch.arange(head_outputs.size(0)), selected_heads]
            # Shape: (batch_size, action_size)
        else:
            # Soft gating: weighted mixture of all heads
            regime_probs_expanded = regime_probs.unsqueeze(2)  # (batch_size, num_heads, 1)
            q_values = (head_outputs * regime_probs_expanded).sum(dim=1)
            # Shape: (batch_size, action_size)
        
        if return_all:
            return q_values, head_outputs, regime_probs
        else:
            return q_values
    
    def get_regime_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get regime probabilities for given state.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
        
        Returns:
            Regime probabilities of shape (batch_size, num_heads)
        """
        with torch.no_grad():
            regime_logits = self.classifier(state)
            return F.softmax(regime_logits, dim=1)
    
    def get_head_q_values(self, state: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Get Q-values from a specific head.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
            head_idx: Index of the head to use
        
        Returns:
            Q-values from specified head
        """
        with torch.no_grad():
            features = self.encoder(state)
            return self.heads[head_idx](features)
    
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
    print("="*60)
    print("Testing Standard DQN")
    print("="*60)
    
    state_size = 9
    action_size = 2
    batch_size = 32
    
    # Test standard DQN
    model = DQN(state_size, action_size, hidden_layers=[128, 128])
    print(f"\nStandard DQN architecture:\n{model}")
    
    dummy_state = torch.randn(batch_size, state_size)
    q_values = model(dummy_state)
    print(f"\nInput shape: {dummy_state.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values sample: {q_values[0]}")
    
    print("\n" + "="*60)
    print("Testing Multi-Head DQN")
    print("="*60)
    
    # Test Multi-Head DQN with soft gating
    multihead_model = MultiHeadDQN(
        state_size=state_size,
        action_size=action_size,
        num_heads=3,
        encoder_layers=[256, 256],
        head_layers=[128],
        classifier_layers=[128, 64],
        gating_type="soft"
    )
    print(f"\nMulti-Head DQN architecture (soft gating):")
    print(f"  Encoder: {multihead_model.encoder}")
    print(f"  Number of heads: {multihead_model.num_heads}")
    print(f"  Classifier: {multihead_model.classifier}")
    
    # Test forward pass
    q_values_soft, all_heads, regime_probs = multihead_model(dummy_state, return_all=True)
    print(f"\nSoft Gating Results:")
    print(f"  Input shape: {dummy_state.shape}")
    print(f"  Final Q-values shape: {q_values_soft.shape}")
    print(f"  All heads output shape: {all_heads.shape}")
    print(f"  Regime probabilities shape: {regime_probs.shape}")
    print(f"  Sample regime probs: {regime_probs[0]} (sum={regime_probs[0].sum():.3f})")
    print(f"  Sample Q-values: {q_values_soft[0]}")
    
    # Test hard gating
    multihead_model_hard = MultiHeadDQN(
        state_size=state_size,
        action_size=action_size,
        num_heads=3,
        gating_type="hard"
    )
    q_values_hard = multihead_model_hard(dummy_state)
    print(f"\nHard Gating Results:")
    print(f"  Final Q-values shape: {q_values_hard.shape}")
    print(f"  Sample Q-values: {q_values_hard[0]}")
    
    # Test individual head access
    head_0_q = multihead_model.get_head_q_values(dummy_state, head_idx=0)
    head_1_q = multihead_model.get_head_q_values(dummy_state, head_idx=1)
    head_2_q = multihead_model.get_head_q_values(dummy_state, head_idx=2)
    print(f"\nIndividual Head Q-values (first sample):")
    print(f"  Head 0 (Low): {head_0_q[0]}")
    print(f"  Head 1 (Med): {head_1_q[0]}")
    print(f"  Head 2 (High): {head_2_q[0]}")
    
    # Test single state action selection
    single_state = torch.randn(state_size)
    action = multihead_model.get_action(single_state)
    print(f"\nSingle state action selection: {action}")
    
    print("\n✓ All network tests passed!")
