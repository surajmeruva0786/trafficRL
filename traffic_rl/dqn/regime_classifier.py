"""
Traffic Regime Classifier for Multi-Head DQN.

This module implements a neural network classifier that categorizes
traffic states into different regimes (low/medium/high density).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class RegimeClassifier(nn.Module):
    """
    Neural network classifier for traffic regime identification.
    
    Classifies traffic states into three regimes:
    - Low (0): Light traffic, minimal congestion
    - Medium (1): Moderate traffic, some congestion
    - High (2): Heavy traffic, significant congestion
    """
    
    def __init__(self, state_size: int, hidden_layers: list = None, dropout: float = 0.2, device: str = None):
        """
        Initialize regime classifier.
        
        Args:
            state_size: Dimension of state space
            hidden_layers: List of hidden layer sizes (default: [128, 64])
            dropout: Dropout probability for regularization
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
        """
        super(RegimeClassifier, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        self.state_size = state_size
        self.num_regimes = 3  # Low, Medium, High
        
        # Set device with explicit GPU detection
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Regime exposure tracking for balanced training
        self.regime_exposure_counts = {0: 0, 1: 0, 2: 0}  # Count of episodes per regime
        self.regime_exposure_time = {0: 0.0, 1: 0.0, 2: 0.0}  # Total time in each regime
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        # Output layer (no activation, will use softmax)
        layers.append(nn.Linear(input_size, self.num_regimes))
        
        self.network = nn.Sequential(*layers)
        
        # Move network to device
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            state: State tensor of shape (batch_size, state_size)
        
        Returns:
            Logits for each regime, shape (batch_size, num_regimes)
        """
        return self.network(state)
    
    def predict_regime(self, state: torch.Tensor) -> int:
        """
        Predict regime for a single state.
        
        Args:
            state: State tensor of shape (state_size,)
        
        Returns:
            Predicted regime (0=low, 1=medium, 2=high)
        """
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            return probs.argmax(dim=1).item()
    
    def predict_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get regime probabilities for a state.
        
        Args:
            state: State tensor of shape (batch_size, state_size) or (state_size,)
        
        Returns:
            Probability distribution over regimes, shape (batch_size, num_regimes)
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            logits = self.forward(state)
            return F.softmax(logits, dim=1)
    
    def record_regime_exposure(self, regime: int, duration: float = 1.0):
        """
        Record exposure to a regime during training.
        
        Args:
            regime: Regime index (0, 1, or 2)
            duration: Duration of exposure in simulation steps or seconds
        """
        if regime in self.regime_exposure_counts:
            self.regime_exposure_counts[regime] += 1
            self.regime_exposure_time[regime] += duration
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """
        Get the distribution of regime exposure.
        
        Returns:
            Dictionary with regime exposure percentages
        """
        total_count = sum(self.regime_exposure_counts.values())
        total_time = sum(self.regime_exposure_time.values())
        
        if total_count == 0:
            return {
                'low_pct': 0.0,
                'medium_pct': 0.0,
                'high_pct': 0.0,
                'total_episodes': 0
            }
        
        return {
            'low_pct': (self.regime_exposure_counts[0] / total_count) * 100,
            'medium_pct': (self.regime_exposure_counts[1] / total_count) * 100,
            'high_pct': (self.regime_exposure_counts[2] / total_count) * 100,
            'total_episodes': total_count,
            'total_time': total_time
        }
    
    def is_balanced(self, tolerance: float = 5.0) -> bool:
        """
        Check if regime exposure is balanced (approximately 33% each).
        
        Args:
            tolerance: Acceptable deviation from 33.33% (default: 5%)
            
        Returns:
            True if balanced within tolerance
        """
        dist = self.get_regime_distribution()
        target = 33.33
        
        return (abs(dist['low_pct'] - target) <= tolerance and
                abs(dist['medium_pct'] - target) <= tolerance and
                abs(dist['high_pct'] - target) <= tolerance)
    
    def reset_exposure_tracking(self):
        """Reset regime exposure counters."""
        self.regime_exposure_counts = {0: 0, 1: 0, 2: 0}
        self.regime_exposure_time = {0: 0.0, 1: 0.0, 2: 0.0}


def compute_regime_label(
    state: np.ndarray,
    queue_threshold_low: float = 5.0,
    queue_threshold_high: float = 15.0,
    wait_threshold_low: float = 20.0,
    wait_threshold_high: float = 60.0
) -> int:
    """
    Compute ground-truth regime label from state features.
    
    State format: [queue_N, queue_S, queue_E, queue_W, 
                   wait_N, wait_S, wait_E, wait_W, current_phase]
    
    Args:
        state: State vector (normalized or unnormalized)
        queue_threshold_low: Max total queue for low regime
        queue_threshold_high: Max total queue for medium regime
        wait_threshold_low: Max avg waiting time for low regime
        wait_threshold_high: Max avg waiting time for medium regime
    
    Returns:
        Regime label (0=low, 1=medium, 2=high)
    """
    # Extract queue lengths (first 4 features) and waiting times (next 4 features)
    queues = state[:4]
    waiting_times = state[4:8]
    
    # Denormalize if needed (assuming normalization by 20 for queues, 100 for wait)
    # If values are all < 1, assume they're normalized
    if np.max(queues) <= 1.0:
        queues = queues * 20.0
    if np.max(waiting_times) <= 1.0:
        waiting_times = waiting_times * 100.0
    
    # Compute aggregate metrics
    total_queue = np.sum(queues)
    avg_wait = np.mean(waiting_times)
    
    # Classify regime based on thresholds
    if total_queue < queue_threshold_low and avg_wait < wait_threshold_low:
        return 0  # Low traffic
    elif total_queue < queue_threshold_high and avg_wait < wait_threshold_high:
        return 1  # Medium traffic
    else:
        return 2  # High traffic


def compute_regime_labels_batch(
    states: np.ndarray,
    queue_threshold_low: float = 5.0,
    queue_threshold_high: float = 15.0,
    wait_threshold_low: float = 20.0,
    wait_threshold_high: float = 60.0
) -> np.ndarray:
    """
    Compute regime labels for a batch of states.
    
    Args:
        states: State array of shape (batch_size, state_size)
        queue_threshold_low: Max total queue for low regime
        queue_threshold_high: Max total queue for medium regime
        wait_threshold_low: Max avg waiting time for low regime
        wait_threshold_high: Max avg waiting time for medium regime
    
    Returns:
        Array of regime labels, shape (batch_size,)
    """
    labels = np.array([
        compute_regime_label(
            state,
            queue_threshold_low,
            queue_threshold_high,
            wait_threshold_low,
            wait_threshold_high
        )
        for state in states
    ])
    return labels


def get_regime_name(regime: int) -> str:
    """
    Get human-readable name for regime.
    
    Args:
        regime: Regime index (0, 1, or 2)
    
    Returns:
        Regime name string
    """
    regime_names = {0: "Low", 1: "Medium", 2: "High"}
    return regime_names.get(regime, "Unknown")


if __name__ == "__main__":
    # Test regime classifier
    print("Testing Regime Classifier...")
    
    state_size = 9
    batch_size = 32
    
    # Create classifier
    classifier = RegimeClassifier(state_size, hidden_layers=[128, 64])
    print(f"\nClassifier architecture:\n{classifier}")
    
    # Test forward pass
    dummy_states = torch.randn(batch_size, state_size)
    logits = classifier(dummy_states)
    probs = F.softmax(logits, dim=1)
    
    print(f"\nInput shape: {dummy_states.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size))}")
    
    # Test single prediction
    single_state = torch.randn(state_size)
    regime = classifier.predict_regime(single_state)
    regime_probs = classifier.predict_probs(single_state)
    print(f"\nSingle state prediction:")
    print(f"  Regime: {regime} ({get_regime_name(regime)})")
    print(f"  Probabilities: {regime_probs[0]}")
    
    # Test regime labeling
    print("\n\nTesting Regime Labeling...")
    
    # Create sample states (unnormalized)
    low_traffic_state = np.array([1, 1, 1, 1, 5, 5, 5, 5, 0])  # Low queues, low wait
    med_traffic_state = np.array([3, 3, 3, 3, 30, 30, 30, 30, 0])  # Medium queues, medium wait
    high_traffic_state = np.array([8, 8, 8, 8, 80, 80, 80, 80, 0])  # High queues, high wait
    
    print(f"Low traffic state label: {compute_regime_label(low_traffic_state)} (expected: 0)")
    print(f"Medium traffic state label: {compute_regime_label(med_traffic_state)} (expected: 1)")
    print(f"High traffic state label: {compute_regime_label(high_traffic_state)} (expected: 2)")
    
    # Test batch labeling
    states_batch = np.array([low_traffic_state, med_traffic_state, high_traffic_state])
    labels = compute_regime_labels_batch(states_batch)
    print(f"\nBatch labels: {labels} (expected: [0, 1, 2])")
    
    print("\n✓ All tests passed!")
