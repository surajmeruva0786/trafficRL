"""
Plotting utilities for visualization.

This module provides functions to create plots for training curves,
metric comparisons, and evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(
    rewards: List[float],
    losses: List[float],
    save_path: str = None
) -> None:
    """
    Plot training reward and loss curves.
    
    Args:
        rewards: List of episode rewards
        losses: List of training losses
        save_path: Path to save plot (if None, display only)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.6, label='Episode Reward')
    
    # Add moving average
    if len(rewards) >= 10:
        window = min(10, len(rewards))
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        ax1.plot(episodes, moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        steps = range(1, len(losses) + 1)
        ax2.plot(steps, losses, alpha=0.4)
        
        # Add moving average
        if len(losses) >= 100:
            window = 100
            moving_avg = pd.Series(losses).rolling(window=window).mean()
            ax2.plot(steps, moving_avg, linewidth=2, color='red', label=f'{window}-Step Moving Avg')
            ax2.legend()
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Over Time')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    baseline_metrics: Dict[str, float],
    dqn_metrics: Dict[str, float],
    save_path: str = None
) -> None:
    """
    Plot comparison between baseline and DQN metrics.
    
    Args:
        baseline_metrics: Metrics from baseline controller
        dqn_metrics: Metrics from DQN agent
        save_path: Path to save plot (if None, display only)
    """
    # Select metrics to compare
    metrics_to_plot = [
        'avg_waiting_time',
        'avg_queue_length',
        'throughput'
    ]
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in baseline_metrics and m in dqn_metrics]
    
    if not available_metrics:
        print("No common metrics to plot")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        baseline_val = baseline_metrics[metric]
        dqn_val = dqn_metrics[metric]
        
        # Create bar chart
        x = ['Fixed-Time\nBaseline', 'DQN Agent']
        y = [baseline_val, dqn_val]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Calculate improvement
        if baseline_val > 0:
            improvement = ((dqn_val - baseline_val) / baseline_val) * 100
            
            # For waiting time and queue length, lower is better
            if 'waiting' in metric or 'queue' in metric:
                improvement = -improvement
            
            color = 'green' if improvement > 0 else 'red'
            ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=11, fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format title
        title = metric.replace('_', ' ').title()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('DQN vs Fixed-Time Baseline Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_metrics(
    metrics_history: Dict[str, List[float]],
    save_path: str = None
) -> None:
    """
    Plot metrics over episodes.
    
    Args:
        metrics_history: Dictionary of metric name -> list of values
        save_path: Path to save plot
    """
    n_metrics = len(metrics_history)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics_history.items()):
        episodes = range(1, len(values) + 1)
        ax.plot(episodes, values, marker='o', markersize=4, alpha=0.6)
        
        # Add moving average
        if len(values) >= 5:
            window = min(5, len(values))
            moving_avg = pd.Series(values).rolling(window=window).mean()
            ax.plot(episodes, moving_avg, linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
            ax.legend()
        
        title = metric_name.replace('_', ' ').title()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_all_plots(
    training_data: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    dqn_metrics: Dict[str, float],
    output_dir: str
) -> None:
    """
    Save all plots to output directory.
    
    Args:
        training_data: Dictionary with 'rewards' and 'losses' lists
        baseline_metrics: Baseline controller metrics
        dqn_metrics: DQN agent metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Training curves
    plot_training_curves(
        training_data.get('rewards', []),
        training_data.get('losses', []),
        save_path=os.path.join(output_dir, 'training_curves.png')
    )
    
    # Comparison plot
    plot_comparison(
        baseline_metrics,
        dqn_metrics,
        save_path=os.path.join(output_dir, 'dqn_vs_baseline.png')
    )
    
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    # Test plotting functions
    import numpy as np
    
    # Generate dummy data
    rewards = -1000 + np.cumsum(np.random.randn(100) * 10)
    losses = 10 * np.exp(-np.arange(1000) / 200) + np.random.randn(1000) * 0.5
    
    baseline_metrics = {
        'avg_waiting_time': 45.2,
        'avg_queue_length': 12.5,
        'throughput': 650.0
    }
    
    dqn_metrics = {
        'avg_waiting_time': 32.1,
        'avg_queue_length': 8.3,
        'throughput': 780.0
    }
    
    # Test plots
    plot_training_curves(rewards, losses)
    plot_comparison(baseline_metrics, dqn_metrics)
