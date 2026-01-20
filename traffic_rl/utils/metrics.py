"""
Metrics computation for traffic simulation.

This module provides functions to compute and track various
traffic performance metrics.
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class MetricsTracker:
    """
    Track metrics across multiple episodes.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
    
    def add(self, metrics: Dict[str, float]) -> None:
        """
        Add metrics for one episode.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        for key, values in self.metrics.items():
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return summary
    
    def get_last_n(self, n: int) -> Dict[str, List[float]]:
        """
        Get last n values for all metrics.
        
        Args:
            n: Number of recent values to return
        
        Returns:
            Dictionary of metric name -> list of last n values
        """
        return {key: values[-n:] for key, values in self.metrics.items()}
    
    def get_all(self) -> Dict[str, List[float]]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary of metric name -> list of all values
        """
        return dict(self.metrics)
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()


def compute_metrics(
    waiting_times: List[float],
    queue_lengths: List[int],
    num_completed: int,
    episode_duration: float
) -> Dict[str, float]:
    """
    Compute traffic performance metrics.
    
    Args:
        waiting_times: List of waiting times for all vehicles
        queue_lengths: List of queue lengths at each step
        num_completed: Number of vehicles that completed their journey
        episode_duration: Total episode duration in seconds
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Average waiting time
    if waiting_times:
        metrics['avg_waiting_time'] = np.mean(waiting_times)
        metrics['max_waiting_time'] = np.max(waiting_times)
        metrics['std_waiting_time'] = np.std(waiting_times)
    else:
        metrics['avg_waiting_time'] = 0.0
        metrics['max_waiting_time'] = 0.0
        metrics['std_waiting_time'] = 0.0
    
    # Average queue length
    if queue_lengths:
        metrics['avg_queue_length'] = np.mean(queue_lengths)
        metrics['max_queue_length'] = np.max(queue_lengths)
        metrics['std_queue_length'] = np.std(queue_lengths)
    else:
        metrics['avg_queue_length'] = 0.0
        metrics['max_queue_length'] = 0.0
        metrics['std_queue_length'] = 0.0
    
    # Throughput (vehicles per hour)
    metrics['num_completed'] = num_completed
    metrics['throughput'] = (num_completed / episode_duration) * 3600 if episode_duration > 0 else 0.0
    
    return metrics


def compare_metrics(
    baseline_metrics: Dict[str, float],
    dqn_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare DQN metrics against baseline.
    
    Args:
        baseline_metrics: Metrics from baseline controller
        dqn_metrics: Metrics from DQN agent
    
    Returns:
        Dictionary of improvement percentages
    """
    improvements = {}
    
    # Metrics where lower is better
    lower_is_better = ['avg_waiting_time', 'max_waiting_time', 'avg_queue_length', 'max_queue_length']
    
    # Metrics where higher is better
    higher_is_better = ['throughput', 'num_completed']
    
    for metric in lower_is_better:
        if metric in baseline_metrics and metric in dqn_metrics:
            baseline_val = baseline_metrics[metric]
            dqn_val = dqn_metrics[metric]
            if baseline_val > 0:
                # Negative improvement means DQN is better (lower value)
                improvement = ((dqn_val - baseline_val) / baseline_val) * 100
                improvements[f'{metric}_improvement'] = -improvement  # Flip sign
    
    for metric in higher_is_better:
        if metric in baseline_metrics and metric in dqn_metrics:
            baseline_val = baseline_metrics[metric]
            dqn_val = dqn_metrics[metric]
            if baseline_val > 0:
                # Positive improvement means DQN is better (higher value)
                improvement = ((dqn_val - baseline_val) / baseline_val) * 100
                improvements[f'{metric}_improvement'] = improvement
    
    return improvements


if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker()
    
    # Add some dummy metrics
    for i in range(10):
        metrics = {
            'avg_waiting_time': np.random.uniform(20, 50),
            'avg_queue_length': np.random.uniform(5, 15),
            'throughput': np.random.uniform(500, 800)
        }
        tracker.add(metrics)
    
    # Get summary
    summary = tracker.get_summary()
    print("Metrics Summary:")
    for metric, stats in summary.items():
        print(f"{metric}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}")
