"""
Comprehensive comparison script for Multi-Head DQN vs Fixed-Time Baseline.

This script runs both the trained Multi-Head DQN and the fixed-time baseline,
then generates detailed comparison reports and visualizations.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.sumo_env import SUMOTrafficEnv
from traffic_rl.env.route_generator import generate_routes, get_distribution_by_name


class FixedTimeController:
    """Fixed-time traffic light controller."""
    
    def __init__(self, ns_green_time: int = 30, ew_green_time: int = 30):
        self.ns_green_time = ns_green_time
        self.ew_green_time = ew_green_time
        self.cycle_length = ns_green_time + ew_green_time
        self.current_time = 0
    
    def select_action(self, state: np.ndarray) -> int:
        time_in_cycle = self.current_time % self.cycle_length
        if time_in_cycle < self.ns_green_time:
            action = 0  # NS green
        else:
            action = 1  # EW green
        self.current_time += 1
        return action
    
    def reset(self):
        self.current_time = 0


def evaluate_controller(controller, env, num_episodes, controller_name, is_dqn=False):
    """Evaluate a controller over multiple episodes."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating {controller_name}")
    print(f"{'='*70}")
    
    episode_rewards = []
    episode_waiting_times = []
    episode_queue_lengths = []
    episode_throughputs = []
    episode_completed = []
    episode_phase_changes = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        if not is_dqn:
            controller.reset()
        
        episode_reward = 0
        done = False
        step = 0
        max_steps = 3600  # 1 hour simulation
        
        while not done and step < max_steps:
            # Select action
            if is_dqn:
                action = controller.select_action(state, epsilon=0.0)  # Greedy
            else:
                action = controller.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
        
        # Get metrics
        metrics = env.get_metrics()
        
        episode_rewards.append(episode_reward)
        episode_waiting_times.append(metrics['avg_waiting_time'])
        episode_queue_lengths.append(metrics['avg_queue_length'])
        episode_throughputs.append(metrics['throughput'])
        episode_completed.append(metrics['num_completed'])
        episode_phase_changes.append(metrics['num_phase_changes'])
        
        print(f"  Episode {episode:2d}/{num_episodes} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Wait: {metrics['avg_waiting_time']:6.2f}s | "
              f"Throughput: {metrics['throughput']:6.1f} veh/h | "
              f"Completed: {metrics['num_completed']:4d}")
    
    results = {
        'controller_name': controller_name,
        'episode_rewards': episode_rewards,
        'episode_waiting_times': episode_waiting_times,
        'episode_queue_lengths': episode_queue_lengths,
        'episode_throughputs': episode_throughputs,
        'episode_completed': episode_completed,
        'episode_phase_changes': episode_phase_changes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times),
        'mean_throughput': np.mean(episode_throughputs),
        'std_throughput': np.std(episode_throughputs),
        'mean_queue_length': np.mean(episode_queue_lengths),
        'std_queue_length': np.std(episode_queue_lengths),
        'mean_completed': np.mean(episode_completed),
        'mean_phase_changes': np.mean(episode_phase_changes)
    }
    
    print(f"\n{controller_name} Summary:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}s")
    print(f"  Mean Throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f} veh/h")
    print(f"  Mean Completed: {results['mean_completed']:.1f} vehicles")
    
    return results


def generate_comparison_report(multihead_results, baseline_results, output_dir):
    """Generate comprehensive comparison report and visualizations."""
    
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance Comparison Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Head DQN vs Fixed-Time Baseline Comparison', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('mean_reward', 'Average Reward', 'Reward'),
        ('mean_waiting_time', 'Average Waiting Time', 'Time (s)'),
        ('mean_throughput', 'Average Throughput', 'Vehicles/hour'),
        ('mean_queue_length', 'Average Queue Length', 'Vehicles'),
        ('mean_completed', 'Vehicles Completed', 'Count'),
        ('mean_phase_changes', 'Phase Changes', 'Count')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        names = [multihead_results['controller_name'], baseline_results['controller_name']]
        values = [multihead_results[metric], baseline_results[metric]]
        stds = [multihead_results.get(f"std_{metric.replace('mean_', '')}", 0),
                baseline_results.get(f"std_{metric.replace('mean_', '')}", 0)]
        
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(names, values, yerr=stds, capsize=5, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Rotate x labels if needed
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance comparison chart")
    plt.close()
    
    # 2. Box Plots for Detailed Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribution Comparison: Multi-Head DQN vs Fixed-Time Baseline',
                 fontsize=14, fontweight='bold')
    
    box_metrics = [
        ('episode_rewards', 'Episode Rewards'),
        ('episode_waiting_times', 'Waiting Time (s)'),
        ('episode_throughputs', 'Throughput (veh/h)')
    ]
    
    for idx, (metric, title) in enumerate(box_metrics):
        ax = axes[idx]
        
        data = [multihead_results[metric], baseline_results[metric]]
        labels = ['Multi-Head DQN', 'Fixed-Time']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution comparison chart")
    plt.close()
    
    # 3. Episode-by-Episode Comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Episode-by-Episode Performance Comparison',
                 fontsize=14, fontweight='bold')
    
    episode_metrics = [
        ('episode_rewards', 'Reward'),
        ('episode_waiting_times', 'Waiting Time (s)'),
        ('episode_throughputs', 'Throughput (veh/h)')
    ]
    
    episodes = range(1, len(multihead_results['episode_rewards']) + 1)
    
    for idx, (metric, ylabel) in enumerate(episode_metrics):
        ax = axes[idx]
        
        ax.plot(episodes, multihead_results[metric], 'o-', 
               label='Multi-Head DQN', color='#3498db', linewidth=2, markersize=6)
        ax.plot(episodes, baseline_results[metric], 's-',
               label='Fixed-Time', color='#e74c3c', linewidth=2, markersize=6)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} per Episode', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episode_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved episode-by-episode comparison chart")
    plt.close()
    
    # 4. Generate Text Report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE COMPARISON REPORT\n")
        f.write("Multi-Head DQN vs Fixed-Time Baseline\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Multi-Head DQN Results
        f.write("Multi-Head DQN:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Average Reward:        {multihead_results['mean_reward']:8.2f} ± {multihead_results['std_reward']:.2f}\n")
        f.write(f"  Average Waiting Time:  {multihead_results['mean_waiting_time']:8.2f} ± {multihead_results['std_waiting_time']:.2f} s\n")
        f.write(f"  Average Throughput:    {multihead_results['mean_throughput']:8.2f} ± {multihead_results['std_throughput']:.2f} veh/h\n")
        f.write(f"  Average Queue Length:  {multihead_results['mean_queue_length']:8.2f} ± {multihead_results['std_queue_length']:.2f} vehicles\n")
        f.write(f"  Vehicles Completed:    {multihead_results['mean_completed']:8.1f}\n")
        f.write(f"  Phase Changes:         {multihead_results['mean_phase_changes']:8.1f}\n\n")
        
        # Fixed-Time Baseline Results
        f.write("Fixed-Time Baseline:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Average Reward:        {baseline_results['mean_reward']:8.2f} ± {baseline_results['std_reward']:.2f}\n")
        f.write(f"  Average Waiting Time:  {baseline_results['mean_waiting_time']:8.2f} ± {baseline_results['std_waiting_time']:.2f} s\n")
        f.write(f"  Average Throughput:    {baseline_results['mean_throughput']:8.2f} ± {baseline_results['std_throughput']:.2f} veh/h\n")
        f.write(f"  Average Queue Length:  {baseline_results['mean_queue_length']:8.2f} ± {baseline_results['std_queue_length']:.2f} vehicles\n")
        f.write(f"  Vehicles Completed:    {baseline_results['mean_completed']:8.1f}\n")
        f.write(f"  Phase Changes:         {baseline_results['mean_phase_changes']:8.1f}\n\n")
        
        # Improvement Analysis
        f.write("="*70 + "\n")
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        reward_improvement = ((multihead_results['mean_reward'] - baseline_results['mean_reward']) / 
                            abs(baseline_results['mean_reward']) * 100)
        wait_improvement = ((baseline_results['mean_waiting_time'] - multihead_results['mean_waiting_time']) / 
                          baseline_results['mean_waiting_time'] * 100)
        throughput_improvement = ((multihead_results['mean_throughput'] - baseline_results['mean_throughput']) / 
                                baseline_results['mean_throughput'] * 100)
        
        f.write(f"Reward Improvement:           {reward_improvement:+.2f}%\n")
        f.write(f"Waiting Time Reduction:       {wait_improvement:+.2f}%\n")
        f.write(f"Throughput Improvement:       {throughput_improvement:+.2f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        if reward_improvement > 0:
            f.write("✓ Multi-Head DQN achieved higher rewards than the baseline\n")
        if wait_improvement > 0:
            f.write("✓ Multi-Head DQN reduced average waiting times\n")
        if throughput_improvement > 0:
            f.write("✓ Multi-Head DQN improved traffic throughput\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n\n")
        
        if reward_improvement > 10:
            f.write("The Multi-Head DQN agent significantly outperforms the fixed-time baseline,\n")
            f.write("demonstrating the effectiveness of adaptive traffic signal control using\n")
            f.write("deep reinforcement learning with regime-specific policy heads.\n")
        elif reward_improvement > 0:
            f.write("The Multi-Head DQN agent shows improvement over the fixed-time baseline,\n")
            f.write("indicating potential for adaptive traffic signal control.\n")
        else:
            f.write("The Multi-Head DQN agent performance is comparable to the fixed-time baseline.\n")
            f.write("Further training or hyperparameter tuning may be beneficial.\n")
    
    print(f"✓ Saved text comparison report")
    
    # 5. Save CSV Data
    df_multihead = pd.DataFrame({
        'Episode': range(1, len(multihead_results['episode_rewards']) + 1),
        'Controller': 'Multi-Head DQN',
        'Reward': multihead_results['episode_rewards'],
        'Waiting_Time': multihead_results['episode_waiting_times'],
        'Throughput': multihead_results['episode_throughputs'],
        'Queue_Length': multihead_results['episode_queue_lengths'],
        'Completed': multihead_results['episode_completed']
    })
    
    df_baseline = pd.DataFrame({
        'Episode': range(1, len(baseline_results['episode_rewards']) + 1),
        'Controller': 'Fixed-Time',
        'Reward': baseline_results['episode_rewards'],
        'Waiting_Time': baseline_results['episode_waiting_times'],
        'Throughput': baseline_results['episode_throughputs'],
        'Queue_Length': baseline_results['episode_queue_lengths'],
        'Completed': baseline_results['episode_completed']
    })
    
    df_combined = pd.concat([df_multihead, df_baseline], ignore_index=True)
    df_combined.to_csv(output_dir / 'comparison_data.csv', index=False)
    print(f"✓ Saved comparison data CSV")
    
    print(f"\n{'='*70}")
    print(f"All reports saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main comparison function."""
    
    # Load configuration
    config_path = "traffic_rl/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes = config['evaluation']['num_episodes']
    
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    print(f"Device: {device.upper()}")
    print(f"Episodes: {num_episodes}")
    print("="*70)
    
    # Initialize environment (no GUI for automated evaluation)
    env = SUMOTrafficEnv(
        net_file=config['sumo']['network_file'],
        route_file=config['sumo']['route_file'],
        config_file=config['sumo']['config_file'],
        use_gui=False,  # No GUI for automated comparison
        step_length=config['sumo']['step_length'],
        yellow_time=config['sumo']['yellow_time'],
        min_green_time=config['sumo']['min_green_time'],
        max_steps=config['traffic']['episode_duration'],
        reward_weights=config['reward']
    )
    
    # 1. Evaluate Multi-Head DQN
    multihead_agent = MultiHeadDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn'] | config,
        device=device
    )
    multihead_agent.load("models/multihead_dqn_final.pth")
    
    multihead_results = evaluate_controller(
        multihead_agent, env, num_episodes, 
        "Multi-Head DQN", is_dqn=True
    )
    
    # 2. Evaluate Fixed-Time Baseline
    baseline_controller = FixedTimeController(
        ns_green_time=config['baseline']['ns_green_time'],
        ew_green_time=config['baseline']['ew_green_time']
    )
    
    baseline_results = evaluate_controller(
        baseline_controller, env, num_episodes,
        "Fixed-Time Baseline", is_dqn=False
    )
    
    # Close environment
    env.close()
    
    # 3. Generate Comparison Report
    output_dir = Path(config['logging']['results_dir']) / 'comparison'
    generate_comparison_report(multihead_results, baseline_results, output_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
