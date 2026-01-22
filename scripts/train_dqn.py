"""
Training script for DQN agent on traffic signal control.

This script trains a DQN agent to control traffic lights at a 4-way
intersection in SUMO, with logging and checkpointing.
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.env.sumo_env import SUMOTrafficEnv
from traffic_rl.env.route_generator import generate_routes, get_distribution_by_name
from traffic_rl.dqn.agent import DQNAgent
from traffic_rl.utils.metrics import MetricsTracker
from traffic_rl.utils.plotting import plot_training_curves
from traffic_rl.utils.logging_utils import setup_logger, CSVLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_dqn(config: dict, args: argparse.Namespace):
    """
    Train DQN agent.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Setup logging
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    logger = setup_logger(
        'train_dqn',
        os.path.join(config['logging']['log_dir'], 'training.log')
    )
    
    logger.info("=" * 80)
    logger.info("Starting DQN Training for Traffic Signal Control")
    logger.info("=" * 80)
    
    # Generate routes
    logger.info("Generating traffic routes...")
    distribution = get_distribution_by_name(config['traffic']['distribution'])
    route_file = os.path.abspath(config['sumo']['route_file'])
    
    generate_routes(
        output_file=route_file,
        num_vehicles=config['traffic']['num_vehicles_per_episode'],
        episode_duration=config['traffic']['episode_duration'],
        distribution=distribution,
        seed=args.seed
    )
    
    # Initialize environment
    logger.info("Initializing SUMO environment...")
    env = SUMOTrafficEnv(
        net_file=os.path.abspath(config['sumo']['network_file']),
        route_file=route_file,
        config_file=os.path.abspath(config['sumo']['config_file']),
        use_gui=config['sumo']['gui'] or args.gui,
        step_length=config['sumo']['step_length'],
        yellow_time=config['sumo']['yellow_time'],
        min_green_time=config['sumo']['min_green_time'],
        max_steps=config['dqn']['max_steps_per_episode'],
        reward_weights=config['reward']
    )
    
    # Perform initial reset to determine state size
    logger.info("Performing initial reset to determine state dimensions...")
    env.reset(seed=args.seed)
    
    # Initialize agent
    logger.info("Initializing DQN agent...")
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn']
    )
    
    logger.info(f"State size: {env.state_size}")
    logger.info(f"Action size: {env.action_size}")
    logger.info(f"Device: {agent.device}")
    
    # Setup CSV logging
    csv_logger = CSVLogger(
        filepath=os.path.join(config['logging']['log_dir'], 'training_log.csv'),
        fieldnames=[
            'episode', 'total_reward', 'avg_waiting_time', 'avg_queue_length',
            'throughput', 'num_phase_changes', 'epsilon', 'avg_loss'
        ]
    )
    
    # Training metrics
    metrics_tracker = MetricsTracker()
    all_rewards = []
    all_losses = []
    best_reward = -float('inf')
    
    # Training loop
    logger.info(f"Starting training for {config['dqn']['max_episodes']} episodes...")
    
    for episode in range(1, config['dqn']['max_episodes'] + 1):
        # Reset environment
        state = env.reset(seed=args.seed + episode if args.seed else None)
        episode_reward = 0
        done = False
        
        # Episode loop
        pbar = tqdm(total=config['dqn']['max_steps_per_episode'], 
                   desc=f"Episode {episode}/{config['dqn']['max_episodes']}",
                   leave=False)
        
        step = 0
        while not done and step < config['dqn']['max_steps_per_episode']:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss > 0:
                all_losses.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            pbar.update(1)
        
        pbar.close()
        
        # Get episode metrics
        env_metrics = env.get_metrics()
        agent_stats = agent.get_stats()
        
        # Track metrics
        all_rewards.append(episode_reward)
        metrics_tracker.add(env_metrics)
        
        # Log episode
        log_data = {
            'episode': episode,
            'total_reward': episode_reward,
            'avg_waiting_time': env_metrics['avg_waiting_time'],
            'avg_queue_length': env_metrics['avg_queue_length'],
            'throughput': env_metrics['throughput'],
            'num_phase_changes': env_metrics['num_phase_changes'],
            'epsilon': agent_stats['epsilon'],
            'avg_loss': agent_stats['avg_loss']
        }
        csv_logger.log(log_data)
        
        if episode % config['logging']['log_frequency'] == 0:
            logger.info(
                f"Episode {episode:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Wait: {env_metrics['avg_waiting_time']:6.2f}s | "
                f"Queue: {env_metrics['avg_queue_length']:5.2f} | "
                f"Throughput: {env_metrics['throughput']:6.1f} veh/h | "
                f"eps: {agent_stats['epsilon']:.3f} | "
                f"Loss: {agent_stats['avg_loss']:.4f}"
            )
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs(os.path.dirname(config['dqn']['model_save_path']), exist_ok=True)
            agent.save(config['dqn']['model_save_path'])
            logger.info(f"New best model saved! Reward: {best_reward:.2f}")
        
        # Save checkpoint periodically
        if episode % config['dqn']['save_frequency'] == 0:
            checkpoint_path = config['dqn']['model_save_path'].replace('.pth', f'_ep{episode}.pth')
            agent.save(checkpoint_path)
    
    # Close environment
    env.close()
    
    # Save final plots
    logger.info("Generating training plots...")
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    plot_training_curves(
        all_rewards,
        all_losses,
        save_path=os.path.join(config['logging']['results_dir'], 'training_curves.png')
    )
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Final epsilon: {agent.get_stats()['epsilon']:.3f}")
    logger.info(f"Total training steps: {agent.steps_done}")
    logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train DQN agent for traffic signal control')
    parser.add_argument('--config', type=str, default='traffic_rl/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train agent
    train_dqn(config, args)


if __name__ == "__main__":
    main()
