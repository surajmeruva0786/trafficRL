"""
Evaluation script for Multi-Head DQN agent.

This script evaluates a trained multi-head DQN agent and compares
it with baselines (single-head DQN, fixed-time control).
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.dqn.agent import DQNAgent
from traffic_rl.env.sumo_env import SUMOTrafficEnv
from traffic_rl.utils.regime_utils import (
    compute_regime_metrics,
    plot_confusion_matrix,
    analyze_regime_performance,
    plot_regime_timeline_with_metrics
)


def evaluate_agent(
    agent,
    env: SUMOTrafficEnv,
    num_episodes: int = 10,
    agent_name: str = "Agent",
    track_regimes: bool = False
) -> Dict:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of evaluation episodes
        agent_name: Name for logging
        track_regimes: Whether to track regime information
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {agent_name}...")
    
    episode_rewards = []
    episode_waiting_times = []
    episode_queue_lengths = []
    episode_throughputs = []
    episode_phase_changes = []
    
    regime_history = []
    predicted_regime_history = []
    reward_history = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Select action (greedy for evaluation)
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, epsilon=0.0)
            else:
                # For baseline agents
                action = agent.get_action(state)
            
            # Track regime if multi-head agent
            if track_regimes and hasattr(agent, 'get_regime_info'):
                regime_info = agent.get_regime_info(state)
                regime_history.append(regime_info['true_regime'])
                predicted_regime_history.append(regime_info['predicted_regime'])
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            reward_history.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # Get episode metrics
        metrics = env.get_metrics()
        
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(metrics['avg_waiting_time'])
        episode_queue_lengths.append(metrics['avg_queue_length'])
        episode_throughputs.append(metrics['throughput'])
        episode_phase_changes.append(metrics['num_phase_changes'])
        
        print(f"  Episode {episode}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Wait={metrics['avg_waiting_time']:.2f}s, "
              f"Throughput={metrics['throughput']:.2f} veh/h")
    
    results = {
        'agent_name': agent_name,
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_queue_lengths': episode_queue_lengths,
        'episode_throughputs': episode_throughputs,
        'episode_phase_changes': episode_phase_changes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times),
        'mean_throughput': np.mean(episode_throughputs),
        'std_throughput': np.std(episode_throughputs),
        'mean_phase_changes': np.mean(episode_phase_changes),
        'regime_history': regime_history,
        'predicted_regime_history': predicted_regime_history,
        'reward_history': reward_history
    }
    
    print(f"\n{agent_name} Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}s")
    print(f"  Mean Throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f} veh/h")
    
    return results


def compare_agents(config_path: str, multihead_model_path: str, baseline_model_path: str = None):
    """
    Compare multi-head DQN with baselines.
    
    Args:
        config_path: Path to configuration file
        multihead_model_path: Path to trained multi-head model
        baseline_model_path: Path to trained single-head DQN (optional)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"MULTI-HEAD DQN EVALUATION")
    print(f"{'='*70}")
    print(f"Device: {device.upper()}")
    print(f"Multi-Head Model: {multihead_model_path}")
    if baseline_model_path:
        print(f"Baseline Model: {baseline_model_path}")
    print(f"{'='*70}")
    
    # Create results directory
    results_dir = Path(config['logging']['results_dir']) / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = SUMOTrafficEnv(
        net_file=config['sumo']['network_file'],
        route_file=config['sumo']['route_file'],
        config_file=config['sumo']['config_file'],
        use_gui=True,  # No GUI for evaluation
        step_length=config['sumo']['step_length'],
        yellow_time=config['sumo']['yellow_time'],
        min_green_time=config['sumo']['min_green_time'],
        max_steps=config['traffic']['episode_duration'],
        reward_weights=config['reward']
    )
    
    num_eval_episodes = config['evaluation']['num_episodes']
    all_results = []
    
    # Evaluate Multi-Head DQN
    print("\n" + "="*70)
    print("1. MULTI-HEAD DQN")
    print("="*70)
    
    multihead_agent = MultiHeadDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn'] | config,
        device=device
    )
    multihead_agent.load(multihead_model_path)
    
    multihead_results = evaluate_agent(
        multihead_agent,
        env,
        num_episodes=num_eval_episodes,
        agent_name="Multi-Head DQN",
        track_regimes=True
    )
    all_results.append(multihead_results)
    
    # Evaluate Single-Head DQN baseline (if provided)
    if baseline_model_path and os.path.exists(baseline_model_path):
        print("\n" + "="*70)
        print("2. SINGLE-HEAD DQN BASELINE")
        print("="*70)
        
        baseline_agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            config=config['dqn'],
            device=device
        )
        baseline_agent.load(baseline_model_path)
        
        baseline_results = evaluate_agent(
            baseline_agent,
            env,
            num_episodes=num_eval_episodes,
            agent_name="Single-Head DQN"
        )
        all_results.append(baseline_results)
    
    env.close()
    
    # Generate comparison plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    # Plot 1: Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['mean_reward', 'mean_waiting_time', 'mean_throughput', 'mean_phase_changes']
    titles = ['Average Reward', 'Average Waiting Time (s)', 'Average Throughput (veh/h)', 'Average Phase Changes']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        agent_names = [r['agent_name'] for r in all_results]
        means = [r[metric] for r in all_results]
        stds = [r.get(f"std_{metric.replace('mean_', '')}", 0) for r in all_results]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(agent_names, means, yerr=stds, capsize=5, 
                      color=colors[:len(agent_names)], alpha=0.7, edgecolor='black')
        
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance comparison plot")
    plt.close()
    
    # Plot 2: Regime classification analysis (for multi-head)
    if multihead_results['regime_history']:
        true_regimes = np.array(multihead_results['regime_history'])
        pred_regimes = np.array(multihead_results['predicted_regime_history'])
        
        metrics = compute_regime_metrics(true_regimes, pred_regimes)
        
        print(f"\nRegime Classification Metrics:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Per-class Accuracy: {metrics['per_class_accuracy']}")
        
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=str(results_dir / "regime_confusion_matrix.png")
        )
        print(f"✓ Saved regime confusion matrix")
        
        # Regime performance analysis
        regime_perf = analyze_regime_performance(
            multihead_results['regime_history'],
            multihead_results['reward_history'],
            "Reward"
        )
        
        print(f"\nPerformance by Regime:")
        for regime, stats in regime_perf.items():
            print(f"  {stats['regime_name']}: "
                  f"mean={stats['mean']:.2f}, "
                  f"std={stats['std']:.2f}, "
                  f"count={stats['count']}")
    
    # Plot 3: Box plots for detailed comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = [
        ('episode_rewards', 'Episode Rewards'),
        ('episode_waiting_times', 'Waiting Time (s)'),
        ('episode_throughputs', 'Throughput (veh/h)')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        data = [r[metric] for r in all_results]
        labels = [r['agent_name'] for r in all_results]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "detailed_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed comparison plot")
    plt.close()
    
    # Save evaluation results
    np.savez(
        results_dir / "evaluation_results.npz",
        **{f"{r['agent_name'].replace(' ', '_').lower()}": r for r in all_results}
    )
    print(f"✓ Saved evaluation data")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Multi-Head DQN agent")
    parser.add_argument(
        "--config",
        type=str,
        default="traffic_rl/config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--multihead-model",
        type=str,
        default="models/multihead_dqn_best.pth",
        help="Path to trained multi-head model"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="Path to trained single-head DQN baseline model"
    )
    
    args = parser.parse_args()
    
    compare_agents(
        config_path=args.config,
        multihead_model_path=args.multihead_model,
        baseline_model_path=args.baseline_model
    )
