"""
Training script for Multi-Head DQN agent.

This script trains a multi-head DQN agent with regime classification
for traffic signal control.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.sumo_env import SUMOTrafficEnv
from traffic_rl.utils.regime_utils import (
    compute_regime_metrics,
    visualize_regime_distribution,
    plot_confusion_matrix,
    analyze_regime_performance
)


def train_multihead_dqn(config_path: str, num_episodes: int = None, use_gpu: bool = True):
    """
    Train multi-head DQN agent.
    
    Args:
        config_path: Path to configuration file
        num_episodes: Number of episodes to train (overrides config)
        use_gpu: Whether to use GPU for training
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override episodes if specified
    if num_episodes is not None:
        config['dqn']['max_episodes'] = num_episodes
    
    # Force GPU/CPU usage based on parameter
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if use_gpu and not torch.cuda.is_available():
        print("⚠ WARNING: GPU requested but CUDA not available, using CPU")
    
    # Create directories
    log_dir = Path(config['logging']['log_dir'])
    results_dir = Path(config['logging']['results_dir'])
    models_dir = Path("models")
    
    for directory in [log_dir, results_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"multihead_dqn_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-HEAD DQN TRAINING")
    print("=" * 70)
    print(f"Configuration: {config_path}")
    print(f"Device: {device.upper()}")
    print(f"Episodes: {config['dqn']['max_episodes']}")
    print(f"Gating type: {config['multihead_dqn']['gating_type']}")
    print(f"Results directory: {run_dir}")
    print("=" * 70)
    
    # Initialize environment
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
    
    # Initialize multi-head agent
    agent = MultiHeadDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn'] | config,  # Merge configs
        device=device
    )
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_q_losses = []
    episode_classifier_losses = []
    episode_waiting_times = []
    episode_queue_lengths = []
    episode_throughputs = []
    episode_classifier_accuracies = []
    episode_specializations = []
    
    max_episodes = config['dqn']['max_episodes']
    save_frequency = config['dqn']['save_frequency']
    spec_log_freq = config['multihead_dqn']['specialization_log_frequency']
    
    best_reward = float('-inf')
    
    print("\nStarting training...")
    print("-" * 70)
    
    try:
        for episode in range(1, max_episodes + 1):
            state = env.reset()
            episode_reward = 0
            episode_loss_sum = 0
            episode_q_loss_sum = 0
            episode_classifier_loss_sum = 0
            steps = 0
            
            while True:
                # Select action
                action = agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                total_loss, q_loss, classifier_loss = agent.train_step()
                
                episode_reward += reward
                episode_loss_sum += total_loss
                episode_q_loss_sum += q_loss
                episode_classifier_loss_sum += classifier_loss
                steps += 1
                
                state = next_state
                
                if done:
                    break
            
            # Get episode metrics
            metrics = env.get_metrics()
            stats = agent.get_stats()
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss_sum / max(steps, 1))
            episode_q_losses.append(episode_q_loss_sum / max(steps, 1))
            episode_classifier_losses.append(episode_classifier_loss_sum / max(steps, 1))
            episode_waiting_times.append(metrics['avg_waiting_time'])
            episode_queue_lengths.append(metrics['avg_queue_length'])
            episode_throughputs.append(metrics['throughput'])
            episode_classifier_accuracies.append(stats['classifier_accuracy'])
            
            # Compute head specialization periodically
            if episode % spec_log_freq == 0:
                specialization = agent.compute_head_specialization()
                agent.specialization_history.append(specialization)
                episode_specializations.append((episode, specialization))
            
            # Print progress
            if episode % config['logging']['log_frequency'] == 0:
                print(f"Episode {episode}/{max_episodes}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Avg Waiting Time: {metrics['avg_waiting_time']:.2f}s")
                print(f"  Throughput: {metrics['throughput']:.2f} veh/h")
                print(f"  Epsilon: {stats['epsilon']:.3f}")
                print(f"  Q Loss: {episode_q_losses[-1]:.4f}")
                print(f"  Classifier Loss: {episode_classifier_losses[-1]:.4f}")
                print(f"  Classifier Accuracy: {stats['classifier_accuracy']:.3f}")
                print(f"  Regime Dist: Low={stats['regime_distribution'][0]:.2f}, "
                      f"Med={stats['regime_distribution'][1]:.2f}, "
                      f"High={stats['regime_distribution'][2]:.2f}")
                if episode_specializations:
                    print(f"  Head Specialization: {episode_specializations[-1][1]:.4f}")
                print("-" * 70)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(str(models_dir / "multihead_dqn_best.pth"))
            
            # Save checkpoint periodically
            if episode % save_frequency == 0:
                agent.save(str(models_dir / f"multihead_dqn_ep{episode}.pth"))
        
        # Save final model
        agent.save(str(models_dir / "multihead_dqn_final.pth"))
        
    finally:
        env.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final classifier accuracy: {episode_classifier_accuracies[-1]:.3f}")
    print(f"Models saved to: {models_dir}")
    
    # Generate plots
    print("\nGenerating training plots...")
    
    # Plot 1: Rewards and losses
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
                    label='Moving Avg (10)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Losses
    axes[0, 1].plot(episode_q_losses, alpha=0.6, label='Q Loss')
    axes[0, 1].plot(episode_classifier_losses, alpha=0.6, label='Classifier Loss')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Waiting time
    axes[1, 0].plot(episode_waiting_times, alpha=0.6)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Waiting Time (s)')
    axes[1, 0].set_title('Average Waiting Time')
    axes[1, 0].grid(alpha=0.3)
    
    # Classifier accuracy
    axes[1, 1].plot(episode_classifier_accuracies, alpha=0.6, color='green')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Regime Classifier Accuracy')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(run_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved training metrics plot")
    plt.close()
    
    # Plot 2: Head specialization over time
    if episode_specializations:
        episodes_spec, specializations = zip(*episode_specializations)
        plt.figure(figsize=(10, 6))
        plt.plot(episodes_spec, specializations, marker='o', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Specialization Score')
        plt.title('Head Specialization Over Training')
        plt.grid(alpha=0.3)
        plt.savefig(run_dir / "head_specialization.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved head specialization plot")
        plt.close()
    
    # Plot 3: Regime distribution
    if len(agent.regime_history) > 0:
        visualize_regime_distribution(
            agent.regime_history,
            save_path=str(run_dir / "regime_distribution.png"),
            title="Traffic Regime Distribution During Training"
        )
        print(f"✓ Saved regime distribution plot")
    
    # Plot 4: Regime classification confusion matrix
    if len(agent.regime_history) > 0 and len(agent.predicted_regime_history) > 0:
        min_len = min(len(agent.regime_history), len(agent.predicted_regime_history))
        true_regimes = np.array(agent.regime_history[-min_len:])
        pred_regimes = np.array(agent.predicted_regime_history[-min_len:])
        
        metrics = compute_regime_metrics(true_regimes, pred_regimes)
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=str(run_dir / "regime_confusion_matrix.png"),
            title="Regime Classification Confusion Matrix"
        )
        print(f"✓ Saved confusion matrix plot")
    
    # Save training data
    np.savez(
        run_dir / "training_data.npz",
        episode_rewards=episode_rewards,
        episode_losses=episode_losses,
        episode_q_losses=episode_q_losses,
        episode_classifier_losses=episode_classifier_losses,
        episode_waiting_times=episode_waiting_times,
        episode_queue_lengths=episode_queue_lengths,
        episode_throughputs=episode_throughputs,
        episode_classifier_accuracies=episode_classifier_accuracies,
        regime_history=agent.regime_history,
        predicted_regime_history=agent.predicted_regime_history,
        head_usage_count=agent.head_usage_count
    )
    print(f"✓ Saved training data")
    
    print(f"\nAll results saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multi-Head DQN agent")
    parser.add_argument(
        "--config",
        type=str,
        default="traffic_rl/config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to train (overrides config)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage (use CPU only)"
    )
    
    args = parser.parse_args()
    
    train_multihead_dqn(
        config_path=args.config,
        num_episodes=args.episodes,
        use_gpu=not args.no_gpu
    )
