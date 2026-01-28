"""
Script to evaluate multiple model checkpoints and generate performance plots.
Evaluates models from models_old_network_backup on the current fixed network.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.sumo_env import SUMOTrafficEnv
import yaml

def evaluate_models():
    # Configuration
    models_dir = Path("e:/github_projects/trafficRL/models_old_network_backup")
    config_path = "e:/github_projects/trafficRL/traffic_rl/config/config.yaml"
    output_dir = Path("e:/github_projects/trafficRL/presentation_materials")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup environment
    # Use shorter episodes for faster evaluation
    config['traffic']['episode_duration'] = 1000 
    config['sumo']['gui'] = False
    
    env = SUMOTrafficEnv(
        net_file=config['sumo']['network_file'],
        route_file=config['sumo']['route_file'],
        config_file=config['sumo']['config_file'],
        use_gui=config['sumo']['gui'],
        step_length=config['sumo']['step_length'],
        yellow_time=config['sumo']['yellow_time'],
        min_green_time=config['sumo']['min_green_time'],
        max_steps=config['traffic']['episode_duration'],
        reward_weights=config['reward']
    )
    
    # Checkpoints to evaluate
    checkpoints = [f"multihead_dqn_ep{i}.pth" for i in range(10, 101, 10)]
    
    results = []
    
    print(f"Evaluating {len(checkpoints)} checkpoints...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize agent
    agent = MultiHeadDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn'] | config,
        device=device
    )
    
    for ckpt in checkpoints:
        ckpt_path = models_dir / ckpt
        if not ckpt_path.exists():
            print(f"Skipping {ckpt} (not found)")
            continue
            
        print(f"Evaluating {ckpt}...")
        
        # Load model
        agent.load(str(ckpt_path))
        
        # Run one episode per model
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Greedy action
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            if done:
                break
        
        metrics = env.get_metrics()
        results.append({
            'episode': int(ckpt.replace('multihead_dqn_ep', '').replace('.pth', '')),
            'reward': episode_reward,
            'avg_waiting_time': metrics['avg_waiting_time'],
            'throughput': metrics['throughput'],
            'queue_length': metrics['avg_queue_length']
        })
        print(f"  Reward: {episode_reward:.2f}, Wait: {metrics['avg_waiting_time']:.2f}s")
    
    env.close()
    
    # Generate plots
    df = pd.DataFrame(results)
    df = df.sort_values('episode')
    
    # Set style
    plt.style.use('default')  # Use default instead of seaborn for reliability
    
    # Plot 1: Reward & Waiting Time
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(df['episode'], df['reward'], color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Waiting Time (s)', color=color)
    ax2.plot(df['episode'], df['avg_waiting_time'], color=color, marker='s', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Agent Performance: Reward vs Waiting Time')
    plt.tight_layout()
    plt.savefig(output_dir / "performance_reward_wait.png", dpi=300)
    
    # Plot 2: Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['throughput'], color='green', marker='^', linewidth=2)
    plt.xlabel('Training Episode')
    plt.ylabel('Throughput (veh/h)')
    plt.title('Agent Performance: Throughput')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "performance_throughput.png", dpi=300)
    
    print(f"\nAnalysis complete. Plots saved to {output_dir}")
    return results

if __name__ == "__main__":
    evaluate_models()
