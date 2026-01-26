"""
Visualization utilities for Multi-Head DQN analysis.

This module provides specialized visualization functions for analyzing
multi-head DQN behavior, head specialization, and Q-value patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import List, Dict, Tuple
from pathlib import Path


def plot_q_value_heatmap(
    states: np.ndarray,
    multihead_model,
    regime_labels: np.ndarray,
    device: str = "cpu",
    save_path: str = None,
    title: str = "Q-Value Specialization by Head and Regime"
):
    """
    Plot Q-value heatmaps showing how each head responds to different regimes.
    
    Args:
        states: Array of states (n_samples, state_size)
        multihead_model: Trained MultiHeadDQN model
        regime_labels: True regime labels for states
        device: Device to use for computation
        save_path: Path to save figure
        title: Plot title
    """
    regime_names = ["Low", "Medium", "High"]
    action_names = ["NS Green", "EW Green"]
    num_heads = multihead_model.num_heads
    
    # Move model to device
    multihead_model = multihead_model.to(device)
    multihead_model.eval()
    
    # Organize states by regime
    regime_states = {i: states[regime_labels == i] for i in range(3)}
    
    # Compute average Q-values per head per regime
    fig, axes = plt.subplots(1, num_heads, figsize=(6*num_heads, 5))
    if num_heads == 1:
        axes = [axes]
    
    for head_idx in range(num_heads):
        q_values_by_regime = []
        
        for regime in range(3):
            if len(regime_states[regime]) > 0:
                # Sample up to 100 states per regime
                sample_states = regime_states[regime][:min(100, len(regime_states[regime]))]
                states_tensor = torch.FloatTensor(sample_states).to(device)
                
                with torch.no_grad():
                    q_vals = multihead_model.get_head_q_values(states_tensor, head_idx)
                    avg_q = q_vals.mean(dim=0).cpu().numpy()
            else:
                avg_q = np.zeros(2)
            
            q_values_by_regime.append(avg_q)
        
        # Create heatmap
        q_matrix = np.array(q_values_by_regime)  # Shape: (3 regimes, 2 actions)
        
        sns.heatmap(
            q_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            xticklabels=action_names,
            yticklabels=regime_names,
            ax=axes[head_idx],
            cbar_kws={'label': 'Q-Value'}
        )
        
        axes[head_idx].set_title(f'Head {head_idx} ({regime_names[head_idx]} Traffic)',
                                fontweight='bold', fontsize=12)
        axes[head_idx].set_xlabel('Action', fontsize=11)
        if head_idx == 0:
            axes[head_idx].set_ylabel('Traffic Regime', fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Q-value heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_head_selection_distribution(
    regime_history: List[int],
    predicted_regime_history: List[int],
    save_path: str = None,
    title: str = "Head Selection vs True Regime"
):
    """
    Plot distribution of head selections compared to true regimes.
    
    Args:
        regime_history: True regime labels
        predicted_regime_history: Predicted regime labels (head selections)
        save_path: Path to save figure
        title: Plot title
    """
    regime_names = ["Low", "Medium", "High"]
    
    # Create contingency table
    contingency = np.zeros((3, 3))
    for true_regime, pred_regime in zip(regime_history, predicted_regime_history):
        contingency[true_regime, pred_regime] += 1
    
    # Normalize by row (true regime)
    contingency_norm = contingency / contingency.sum(axis=1, keepdims=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(
        contingency,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=[f"Head {i}\n({regime_names[i]})" for i in range(3)],
        yticklabels=[f"True: {name}" for name in regime_names],
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_xlabel('Selected Head', fontsize=12)
    ax1.set_ylabel('True Regime', fontsize=12)
    ax1.set_title('Head Selection Counts', fontsize=14, fontweight='bold')
    
    # Normalized
    sns.heatmap(
        contingency_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[f"Head {i}\n({regime_names[i]})" for i in range(3)],
        yticklabels=[f"True: {name}" for name in regime_names],
        ax=ax2,
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_xlabel('Selected Head', fontsize=12)
    ax2.set_ylabel('True Regime', fontsize=12)
    ax2.set_title('Head Selection Proportions', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved head selection distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_q_value_divergence(
    states: np.ndarray,
    multihead_model,
    regime_labels: np.ndarray,
    device: str = "cpu",
    save_path: str = None,
    title: str = "Q-Value Divergence Between Heads"
):
    """
    Plot Q-value divergence between heads to show specialization.
    
    Args:
        states: Array of states
        multihead_model: Trained MultiHeadDQN model
        regime_labels: True regime labels
        device: Device for computation
        save_path: Path to save figure
        title: Plot title
    """
    regime_names = ["Low", "Medium", "High"]
    multihead_model = multihead_model.to(device)
    multihead_model.eval()
    
    # Organize states by regime
    regime_states = {i: states[regime_labels == i] for i in range(3)}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for regime in range(3):
        if len(regime_states[regime]) == 0:
            continue
        
        # Sample states
        sample_states = regime_states[regime][:min(200, len(regime_states[regime]))]
        states_tensor = torch.FloatTensor(sample_states).to(device)
        
        # Get Q-values from all heads
        with torch.no_grad():
            head_q_values = []
            for head_idx in range(multihead_model.num_heads):
                q_vals = multihead_model.get_head_q_values(states_tensor, head_idx)
                head_q_values.append(q_vals.cpu().numpy())
        
        # Plot Q-value distributions for each head
        ax = axes[regime]
        
        for head_idx, q_vals in enumerate(head_q_values):
            # Flatten Q-values (both actions)
            q_flat = q_vals.flatten()
            
            ax.hist(q_flat, bins=30, alpha=0.5, label=f'Head {head_idx}',
                   edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Q-Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{regime_names[regime]} Traffic Regime', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Q-value divergence plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_multihead_analysis_report(
    model_path: str,
    eval_data_path: str,
    output_dir: str,
    device: str = "cpu"
):
    """
    Create comprehensive analysis report for multi-head DQN.
    
    Args:
        model_path: Path to trained model
        eval_data_path: Path to evaluation data (.npz)
        output_dir: Directory to save analysis plots
        device: Device for computation
    """
    from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
    import yaml
    
    print("Creating Multi-Head DQN Analysis Report...")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    # Note: This requires config - in practice, load from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load evaluation data
    print("Loading evaluation data...")
    data = np.load(eval_data_path, allow_pickle=True)
    
    # Extract data
    states = data.get('states', None)
    regime_history = data.get('regime_history', [])
    predicted_regime_history = data.get('predicted_regime_history', [])
    
    if states is not None and len(regime_history) > 0:
        regime_labels = np.array(regime_history)
        
        # Note: Model reconstruction would require config
        # For now, provide template for when model is available
        
        print("\nGenerating visualizations...")
        
        # 1. Head selection distribution
        if len(predicted_regime_history) > 0:
            plot_head_selection_distribution(
                regime_history,
                predicted_regime_history,
                save_path=str(output_dir / "head_selection_distribution.png")
            )
        
        print("\n✓ Analysis report generated")
        print(f"  Output directory: {output_dir}")
    else:
        print("⚠ Insufficient data for analysis")
    
    print("=" * 70)


if __name__ == "__main__":
    print("Multi-Head DQN Visualization Utilities")
    print("=" * 70)
    print("\nThis module provides visualization functions for:")
    print("  • Q-value heatmaps by head and regime")
    print("  • Head selection distribution analysis")
    print("  • Q-value divergence between heads")
    print("  • Comprehensive analysis reports")
    print("\nImport this module in your evaluation scripts to use these functions.")
    print("=" * 70)
