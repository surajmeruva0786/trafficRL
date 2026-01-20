"""
Experience Replay Buffer for DQN.

This module implements a circular buffer to store and sample
experience transitions for training.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Stores transitions (state, action, reward, next_state, done)
    and provides random sampling for training.
    """
    
    def __init__(self, capacity: int, seed: int = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of samples required
        
        Returns:
            True if buffer size >= min_size
        """
        return len(self.buffer) >= min_size
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000, seed=42)
    
    # Add some dummy experiences
    for i in range(100):
        state = np.random.randn(9)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(9)
        done = i % 50 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready for training (min 50): {buffer.is_ready(50)}")
    
    # Sample a batch
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"\nBatch shapes:")
    print(f"States: {states.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Rewards: {rewards.shape}")
    print(f"Next states: {next_states.shape}")
    print(f"Dones: {dones.shape}")
