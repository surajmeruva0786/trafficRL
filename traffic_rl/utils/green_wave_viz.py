#!/usr/bin/env python3
"""
Green Wave Visualization
Creates time-space diagrams and green wave band visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def plot_time_space_diagram(
    signal_data: Dict,
    arterial_route: List[str],
    start_time: float = 0.0,
    end_time: float = 300.0,
    output_path: Path = None
):
    """
    Create time-space diagram for an arterial route.
    
    Args:
        signal_data: Dictionary with 'green_periods' and 'red_periods'
        arterial_route: List of intersection IDs along the arterial
        start_time: Start time for visualization
        end_time: End time for visualization
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot settings
    num_intersections = len(arterial_route)
    y_positions = list(range(num_intersections))
    
    # Plot green and red periods for each intersection
    for idx, intersection_id in enumerate(arterial_route):
        y_pos = y_positions[idx]
        
        # Plot green periods
        if intersection_id in signal_data['green_periods']:
            for start, end in signal_data['green_periods'][intersection_id]:
                if start >= start_time and end <= end_time:
                    width = end - start
                    rect = patches.Rectangle(
                        (start, y_pos - 0.3), width, 0.6,
                        linewidth=1, edgecolor='darkgreen',
                        facecolor='green', alpha=0.7
                    )
                    ax.add_patch(rect)
        
        # Plot red periods
        if intersection_id in signal_data['red_periods']:
            for start, end in signal_data['red_periods'][intersection_id]:
                if start >= start_time and end <= end_time:
                    width = end - start
                    rect = patches.Rectangle(
                        (start, y_pos - 0.3), width, 0.6,
                        linewidth=1, edgecolor='darkred',
                        facecolor='red', alpha=0.5
                    )
                    ax.add_patch(rect)
    
    # Formatting
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(-0.5, num_intersections - 0.5)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Intersection', fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(arterial_route)
    ax.set_title('Time-Space Diagram: Signal Coordination', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    green_patch = patches.Patch(color='green', alpha=0.7, label='Green Phase')
    red_patch = patches.Patch(color='red', alpha=0.5, label='Red Phase')
    ax.legend(handles=[green_patch, red_patch], loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-space diagram to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_green_wave_bands(
    coordination_data: Dict[str, float],
    output_path: Path = None
):
    """
    Visualize green wave coordination scores for arterial routes.
    
    Args:
        coordination_data: Dictionary mapping route_id to coordination score
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    routes = list(coordination_data.keys())
    scores = list(coordination_data.values())
    
    # Create bar chart
    colors = ['green' if s >= 0.7 else 'orange' if s >= 0.4 else 'red' for s in scores]
    bars = ax.bar(routes, scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Arterial Route', fontsize=12)
    ax.set_ylabel('Coordination Score', fontsize=12)
    ax.set_title('Green Wave Coordination Quality', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥0.7)')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.4)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved coordination scores to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_coordination_heatmap(
    coordination_scores: Dict[str, float],
    grid_shape: Tuple[int, int] = (3, 3),
    output_path: Path = None
):
    """
    Create heatmap of coordination quality across the network.
    
    Args:
        coordination_scores: Dictionary mapping route_id to score
        grid_shape: Tuple of (rows, cols) for the grid
        output_path: Path to save the figure
    """
    rows, cols = grid_shape
    
    # Create matrices for horizontal and vertical routes
    h_scores = np.zeros(rows)
    v_scores = np.zeros(cols)
    
    for route_id, score in coordination_scores.items():
        if route_id.startswith('H'):
            idx = int(route_id[1:])
            if idx < rows:
                h_scores[idx] = score
        elif route_id.startswith('V'):
            idx = int(route_id[1:])
            if idx < cols:
                v_scores[idx] = score
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Horizontal routes heatmap
    im1 = ax1.imshow(h_scores.reshape(-1, 1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks(range(rows))
    ax1.set_yticklabels([f'H{i}' for i in range(rows)])
    ax1.set_xticks([])
    ax1.set_title('Horizontal Arterials', fontsize=12, fontweight='bold')
    
    # Add score labels
    for i in range(rows):
        ax1.text(0, i, f'{h_scores[i]:.2f}', ha='center', va='center',
                color='white' if h_scores[i] < 0.5 else 'black', fontweight='bold')
    
    # Vertical routes heatmap
    im2 = ax2.imshow(v_scores.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(cols))
    ax2.set_xticklabels([f'V{i}' for i in range(cols)])
    ax2.set_yticks([])
    ax2.set_title('Vertical Arterials', fontsize=12, fontweight='bold')
    
    # Add score labels
    for i in range(cols):
        ax2.text(i, 0, f'{v_scores[i]:.2f}', ha='center', va='center',
                color='white' if v_scores[i] < 0.5 else 'black', fontweight='bold')
    
    # Add colorbar
    fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', pad=0.1,
                label='Coordination Score')
    
    fig.suptitle('Arterial Coordination Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved coordination heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Testing Green Wave Visualization...")
    
    # Create sample data
    arterial_route = ['I_0_0', 'I_0_1', 'I_0_2']
    
    # Simulate coordinated signals
    signal_data = {
        'green_periods': {},
        'red_periods': {}
    }
    
    for i, intersection_id in enumerate(arterial_route):
        green_periods = []
        red_periods = []
        
        for cycle in range(5):
            green_start = cycle * 60 + i * 10  # 10 second offset
            green_end = green_start + 30
            red_start = green_end
            red_end = (cycle + 1) * 60 + i * 10
            
            green_periods.append((green_start, green_end))
            red_periods.append((red_start, red_end))
        
        signal_data['green_periods'][intersection_id] = green_periods
        signal_data['red_periods'][intersection_id] = red_periods
    
    # Test time-space diagram
    print("\n1. Creating time-space diagram...")
    plot_time_space_diagram(signal_data, arterial_route, 0, 300)
    
    # Test coordination scores
    print("\n2. Creating coordination scores plot...")
    coordination_scores = {
        'H0': 0.85,
        'H1': 0.72,
        'H2': 0.45,
        'V0': 0.68,
        'V1': 0.91,
        'V2': 0.55
    }
    plot_green_wave_bands(coordination_scores)
    
    # Test heatmap
    print("\n3. Creating coordination heatmap...")
    plot_coordination_heatmap(coordination_scores, (3, 3))
    
    print("\n✓ All visualizations created successfully!")
