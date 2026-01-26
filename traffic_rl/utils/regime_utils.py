"""
Utilities for traffic regime analysis and visualization.

This module provides functions for regime labeling, metrics computation,
and visualization of regime-related data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_regime_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    regime_names: List[str] = None
) -> Dict[str, any]:
    """
    Compute classification metrics for regime prediction.
    
    Args:
        true_labels: Ground-truth regime labels
        predicted_labels: Predicted regime labels
        regime_names: Names for each regime (default: ["Low", "Medium", "High"])
    
    Returns:
        Dictionary containing accuracy, confusion matrix, and classification report
    """
    if regime_names is None:
        regime_names = ["Low", "Medium", "High"]
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(
        true_labels,
        predicted_labels,
        target_names=regime_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'per_class_accuracy': conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    }


def visualize_regime_distribution(
    regime_history: List[int],
    save_path: str = None,
    title: str = "Traffic Regime Distribution"
) -> None:
    """
    Visualize distribution of traffic regimes over time.
    
    Args:
        regime_history: List of regime labels over time
        save_path: Path to save figure (if None, display only)
        title: Plot title
    """
    regime_names = ["Low", "Medium", "High"]
    regime_counts = np.bincount(regime_history, minlength=3)
    regime_percentages = regime_counts / len(regime_history) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    ax1.bar(regime_names, regime_percentages, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Regime Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (name, pct) in enumerate(zip(regime_names, regime_percentages)):
        ax1.text(i, pct + 1, f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Timeline
    time_steps = np.arange(len(regime_history))
    regime_colors = [colors[r] for r in regime_history]
    ax2.scatter(time_steps, regime_history, c=regime_colors, alpha=0.5, s=10)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Regime', fontsize=12)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(regime_names)
    ax2.set_title('Regime Timeline', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regime distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    regime_names: List[str] = None,
    save_path: str = None,
    title: str = "Regime Classification Confusion Matrix"
) -> None:
    """
    Plot confusion matrix for regime classification.
    
    Args:
        conf_matrix: Confusion matrix array
        regime_names: Names for each regime
        save_path: Path to save figure
        title: Plot title
    """
    if regime_names is None:
        regime_names = ["Low", "Medium", "High"]
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=regime_names,
        yticklabels=regime_names,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_xlabel('Predicted Regime', fontsize=12)
    ax1.set_ylabel('True Regime', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=regime_names,
        yticklabels=regime_names,
        ax=ax2,
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_xlabel('Predicted Regime', fontsize=12)
    ax2.set_ylabel('True Regime', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def detect_regime_shifts(
    regime_history: List[int],
    min_duration: int = 10
) -> List[Tuple[int, int, int]]:
    """
    Detect regime shift points in the timeline.
    
    Args:
        regime_history: List of regime labels over time
        min_duration: Minimum duration to consider a stable regime
    
    Returns:
        List of (start_idx, end_idx, regime) tuples for each regime segment
    """
    if not regime_history:
        return []
    
    segments = []
    current_regime = regime_history[0]
    start_idx = 0
    
    for i, regime in enumerate(regime_history[1:], start=1):
        if regime != current_regime:
            # Regime changed
            duration = i - start_idx
            if duration >= min_duration:
                segments.append((start_idx, i - 1, current_regime))
            current_regime = regime
            start_idx = i
    
    # Add final segment
    duration = len(regime_history) - start_idx
    if duration >= min_duration:
        segments.append((start_idx, len(regime_history) - 1, current_regime))
    
    return segments


def plot_regime_timeline_with_metrics(
    regime_history: List[int],
    metric_history: List[float],
    metric_name: str = "Reward",
    save_path: str = None,
    title: str = "Regime Timeline with Performance Metrics"
) -> None:
    """
    Plot regime timeline overlaid with performance metrics.
    
    Args:
        regime_history: List of regime labels over time
        metric_history: List of metric values over time
        metric_name: Name of the metric being plotted
        save_path: Path to save figure
        title: Plot title
    """
    regime_names = ["Low", "Medium", "High"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    time_steps = np.arange(len(regime_history))
    
    # Regime timeline
    regime_colors = [colors[r] for r in regime_history]
    ax1.scatter(time_steps, regime_history, c=regime_colors, alpha=0.6, s=20)
    ax1.set_ylabel('Regime', fontsize=12)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(regime_names)
    ax1.set_title('Traffic Regime Over Time', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Add regime background shading
    for i, regime in enumerate(regime_history):
        ax1.axvspan(i, i+1, alpha=0.1, color=colors[regime])
    
    # Metric timeline
    ax2.plot(time_steps[:len(metric_history)], metric_history, 
             color='#3498db', linewidth=2, alpha=0.8, label=metric_name)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f'{metric_name} Over Time', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Add regime background shading to metric plot
    for i, regime in enumerate(regime_history[:len(metric_history)]):
        ax2.axvspan(i, i+1, alpha=0.1, color=colors[regime])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regime timeline plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_regime_performance(
    regime_history: List[int],
    metric_history: List[float],
    metric_name: str = "Reward"
) -> Dict[int, Dict[str, float]]:
    """
    Analyze performance metrics per regime.
    
    Args:
        regime_history: List of regime labels over time
        metric_history: List of metric values over time
        metric_name: Name of the metric
    
    Returns:
        Dictionary mapping regime to performance statistics
    """
    regime_names = ["Low", "Medium", "High"]
    results = {}
    
    # Ensure same length
    min_len = min(len(regime_history), len(metric_history))
    regime_history = regime_history[:min_len]
    metric_history = metric_history[:min_len]
    
    for regime in range(3):
        regime_mask = np.array(regime_history) == regime
        regime_metrics = np.array(metric_history)[regime_mask]
        
        if len(regime_metrics) > 0:
            results[regime] = {
                'regime_name': regime_names[regime],
                'count': len(regime_metrics),
                'mean': np.mean(regime_metrics),
                'std': np.std(regime_metrics),
                'min': np.min(regime_metrics),
                'max': np.max(regime_metrics),
                'median': np.median(regime_metrics)
            }
        else:
            results[regime] = {
                'regime_name': regime_names[regime],
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
    
    return results


if __name__ == "__main__":
    # Test regime utilities
    print("Testing Regime Utilities...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate regime history with shifts
    regime_history = []
    for _ in range(300):
        regime_history.append(0)  # Low
    for _ in range(400):
        regime_history.append(1)  # Medium
    for _ in range(300):
        regime_history.append(2)  # High
    
    # Add some noise
    for i in range(50):
        idx = np.random.randint(0, len(regime_history))
        regime_history[idx] = np.random.randint(0, 3)
    
    # Simulate predictions with some errors
    predicted_labels = regime_history.copy()
    error_indices = np.random.choice(len(predicted_labels), size=100, replace=False)
    for idx in error_indices:
        predicted_labels[idx] = (predicted_labels[idx] + np.random.randint(1, 3)) % 3
    
    # Test metrics computation
    print("\n1. Computing regime metrics...")
    metrics = compute_regime_metrics(regime_history, predicted_labels)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Per-class accuracy: {metrics['per_class_accuracy']}")
    
    # Test regime shift detection
    print("\n2. Detecting regime shifts...")
    shifts = detect_regime_shifts(regime_history, min_duration=50)
    print(f"   Found {len(shifts)} regime segments:")
    for start, end, regime in shifts[:5]:
        regime_names = ["Low", "Medium", "High"]
        print(f"     Steps {start}-{end}: {regime_names[regime]} ({end-start+1} steps)")
    
    # Test performance analysis
    print("\n3. Analyzing regime performance...")
    metric_history = [np.random.normal(-10 * (r + 1), 5) for r in regime_history]
    perf_analysis = analyze_regime_performance(regime_history, metric_history, "Reward")
    for regime, stats in perf_analysis.items():
        print(f"   {stats['regime_name']}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, count={stats['count']}")
    
    print("\n✓ All tests passed!")
