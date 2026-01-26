"""
Multi-Head DQN Agent for Traffic Signal Control.

This module implements a DQN agent with multiple specialized Q-heads
for different traffic regimes, with integrated regime classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List
from .network import MultiHeadDQN
from .regime_classifier import compute_regime_label, compute_regime_labels_batch
from .replay_buffer import ReplayBuffer


class MultiHeadDQNAgent:
    """
    Multi-Head DQN agent with regime-aware learning.
    
    Features:
    - Multiple Q-heads specialized for different traffic regimes
    - Integrated regime classifier
    - Both hard selection and soft gating mechanisms
    - Joint training of encoder, heads, and classifier
    - Head specialization tracking
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any],
        device: str = None
    ):
        """
        Initialize Multi-Head DQN agent.
        
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
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.device = torch.device("cpu")
                print("⚠ WARNING: CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
        
        print(f"✓ Using device: {self.device}")
        
        # Multi-head configuration
        multihead_config = config.get('multihead_dqn', {})
        self.num_heads = multihead_config.get('num_heads', 3)
        self.gating_type = multihead_config.get('gating_type', 'soft')
        self.classifier_loss_weight = multihead_config.get('classifier_loss_weight', 0.1)
        
        # Regime thresholds
        regime_thresholds = multihead_config.get('regime_thresholds', {})
        self.queue_threshold_low = regime_thresholds.get('low_queue_max', 5.0)
        self.queue_threshold_high = regime_thresholds.get('med_queue_max', 15.0)
        self.wait_threshold_low = regime_thresholds.get('low_wait_max', 20.0)
        self.wait_threshold_high = regime_thresholds.get('med_wait_max', 60.0)
        
        # Create main and target networks
        encoder_layers = multihead_config.get('encoder_layers', [256, 256])
        head_layers = multihead_config.get('head_layers', [128])
        classifier_layers = multihead_config.get('classifier_layers', [128, 64])
        
        self.policy_net = MultiHeadDQN(
            state_size=state_size,
            action_size=action_size,
            num_heads=self.num_heads,
            encoder_layers=encoder_layers,
            head_layers=head_layers,
            classifier_layers=classifier_layers,
            gating_type=self.gating_type
        ).to(self.device)
        
        self.target_net = MultiHeadDQN(
            state_size=state_size,
            action_size=action_size,
            num_heads=self.num_heads,
            encoder_layers=encoder_layers,
            head_layers=head_layers,
            classifier_layers=classifier_layers,
            gating_type=self.gating_type
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        print(f"✓ Multi-Head DQN initialized:")
        print(f"  Heads: {self.num_heads}, Gating: {self.gating_type}")
        print(f"  Encoder: {encoder_layers}, Head: {head_layers}")
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )
        self.q_criterion = nn.SmoothL1Loss()  # Huber loss for Q-values
        self.classifier_criterion = nn.CrossEntropyLoss()  # For regime classification
        
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
        self.q_loss_history = []
        self.classifier_loss_history = []
        
        # Regime tracking
        self.regime_history = []
        self.predicted_regime_history = []
        self.head_usage_count = [0] * self.num_heads
        
        # Head specialization tracking
        self.track_specialization = multihead_config.get('track_head_specialization', True)
        self.specialization_history = []
    
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy policy with multi-head network.
        
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
        
        # Track regime for current state
        regime = compute_regime_label(
            state,
            self.queue_threshold_low,
            self.queue_threshold_high,
            self.wait_threshold_low,
            self.wait_threshold_high
        )
        self.regime_history.append(regime)
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        Perform one training step with joint Q-learning and classifier training.
        
        Returns:
            Tuple of (total_loss, q_loss, classifier_loss)
        """
        # Check if enough samples in buffer
        if not self.memory.is_ready(self.min_buffer_size):
            return 0.0, 0.0, 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute ground-truth regime labels for classifier training
        regime_labels = compute_regime_labels_batch(
            states.cpu().numpy(),
            self.queue_threshold_low,
            self.queue_threshold_high,
            self.wait_threshold_low,
            self.wait_threshold_high
        )
        regime_labels = torch.LongTensor(regime_labels).to(self.device)
        
        # Forward pass through policy network
        current_q_values, all_heads, regime_probs = self.policy_net(states, return_all=True)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Q-learning loss
        q_loss = self.q_criterion(current_q_values, target_q_values)
        
        # Classifier loss (regime classification)
        regime_logits = self.policy_net.classifier(states)
        classifier_loss = self.classifier_criterion(regime_logits, regime_labels)
        
        # Combined loss
        total_loss = q_loss + self.classifier_loss_weight * classifier_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
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
        
        # Store losses
        total_loss_value = total_loss.item()
        q_loss_value = q_loss.item()
        classifier_loss_value = classifier_loss.item()
        
        self.loss_history.append(total_loss_value)
        self.q_loss_history.append(q_loss_value)
        self.classifier_loss_history.append(classifier_loss_value)
        
        # Track predicted regimes
        predicted_regimes = regime_probs.argmax(dim=1).cpu().numpy()
        self.predicted_regime_history.extend(predicted_regimes.tolist())
        
        # Track head usage (for hard gating)
        if self.gating_type == "hard":
            for regime in predicted_regimes:
                self.head_usage_count[regime] += 1
        
        return total_loss_value, q_loss_value, classifier_loss_value
    
    def compute_head_specialization(self, sample_states: np.ndarray = None) -> float:
        """
        Compute head specialization metric (Q-value divergence between heads).
        
        Args:
            sample_states: States to evaluate (if None, sample from buffer)
        
        Returns:
            Specialization score (higher = more specialized)
        """
        if sample_states is None:
            if len(self.memory) < 100:
                return 0.0
            sample_states, _, _, _, _ = self.memory.sample(min(100, len(self.memory)))
        
        states = torch.FloatTensor(sample_states).to(self.device)
        
        with torch.no_grad():
            features = self.policy_net.encoder(states)
            
            # Get Q-values from each head
            head_q_values = []
            for head in self.policy_net.heads:
                q_vals = head(features)
                head_q_values.append(q_vals.cpu().numpy())
            
            # Compute variance across heads for each state-action pair
            head_q_values = np.array(head_q_values)  # (num_heads, batch_size, action_size)
            variance = np.var(head_q_values, axis=0)  # (batch_size, action_size)
            specialization = np.mean(variance)
        
        return float(specialization)
    
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
            'config': self.config,
            'regime_history': self.regime_history,
            'predicted_regime_history': self.predicted_regime_history,
            'head_usage_count': self.head_usage_count,
            'specialization_history': self.specialization_history
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Multi-Head model saved to {filepath}")
    
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
        
        # Load regime tracking data if available
        if 'regime_history' in checkpoint:
            self.regime_history = checkpoint['regime_history']
        if 'predicted_regime_history' in checkpoint:
            self.predicted_regime_history = checkpoint['predicted_regime_history']
        if 'head_usage_count' in checkpoint:
            self.head_usage_count = checkpoint['head_usage_count']
        if 'specialization_history' in checkpoint:
            self.specialization_history = checkpoint['specialization_history']
        
        print(f"✓ Multi-Head model loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics including regime and head metrics.
        
        Returns:
            Dictionary of statistics
        """
        # Compute classifier accuracy if we have predictions
        classifier_accuracy = 0.0
        if len(self.regime_history) > 0 and len(self.predicted_regime_history) > 0:
            min_len = min(len(self.regime_history), len(self.predicted_regime_history))
            true_regimes = np.array(self.regime_history[-min_len:])
            pred_regimes = np.array(self.predicted_regime_history[-min_len:])
            classifier_accuracy = np.mean(true_regimes == pred_regimes)
        
        # Regime distribution
        regime_distribution = [0, 0, 0]
        if len(self.regime_history) > 0:
            for regime in self.regime_history[-1000:]:  # Last 1000 steps
                regime_distribution[regime] += 1
            total = sum(regime_distribution)
            regime_distribution = [count / total for count in regime_distribution]
        
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'buffer_size': len(self.memory),
            'avg_total_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            'avg_q_loss': np.mean(self.q_loss_history[-100:]) if self.q_loss_history else 0.0,
            'avg_classifier_loss': np.mean(self.classifier_loss_history[-100:]) if self.classifier_loss_history else 0.0,
            'classifier_accuracy': classifier_accuracy,
            'regime_distribution': regime_distribution,
            'head_usage_count': self.head_usage_count,
            'current_specialization': self.specialization_history[-1] if self.specialization_history else 0.0
        }
    
    def get_regime_info(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Get regime classification info for a given state.
        
        Args:
            state: State to classify
        
        Returns:
            Dictionary with regime probabilities and predicted regime
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            regime_probs = self.policy_net.get_regime_probs(state_tensor.unsqueeze(0))
            regime_probs = regime_probs[0].cpu().numpy()
            predicted_regime = int(np.argmax(regime_probs))
        
        true_regime = compute_regime_label(
            state,
            self.queue_threshold_low,
            self.queue_threshold_high,
            self.wait_threshold_low,
            self.wait_threshold_high
        )
        
        regime_names = ["Low", "Medium", "High"]
        
        return {
            'true_regime': true_regime,
            'true_regime_name': regime_names[true_regime],
            'predicted_regime': predicted_regime,
            'predicted_regime_name': regime_names[predicted_regime],
            'regime_probabilities': regime_probs.tolist(),
            'is_correct': true_regime == predicted_regime
        }
