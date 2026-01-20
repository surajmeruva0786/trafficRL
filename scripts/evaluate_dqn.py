"""
Evaluation script for trained DQN agent.

This script evaluates a trained DQN agent on traffic signal control
and generates comparison plots.
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
from traffic_rl.utils.logging_utils import setup_logger, CSVLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_dqn(config: dict, args: argparse.Namespace):
    """
    Evaluate trained DQN agent.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Setup logging
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    logger = setup_logger(
        'evaluate_dqn',
        os.path.join(config['logging']['log_dir'], 'evaluation.log')
    )
    
    logger.info("=" * 80)
    logger.info("Evaluating Trained DQN Agent")
    logger.info("=" * 80)
    
    # Generate routes
    logger.info("Generating traffic routes...")
    distribution = get_distribution_by_name(config['traffic']['distribution'])
    route_file = os.path.abspath(config['sumo']['route_file'])
    
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
    
    # Initialize agent
    logger.info("Initializing DQN agent...")
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn']
    )
    
    # Load trained model
    model_path = args.model_path or config['dqn']['model_save_path']
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}")
    agent.load(model_path)
    
    # Setup CSV logging
    csv_logger = CSVLogger(
        filepath=os.path.join(config['logging']['log_dir'], 'evaluation_log.csv'),
        fieldnames=[
            'episode', 'total_reward', 'avg_waiting_time', 'avg_queue_length',
            'throughput', 'num_completed', 'num_phase_changes'
        ]
    )
    
    # Evaluation metrics
    metrics_tracker = MetricsTracker()
    num_eval_episodes = config['evaluation']['num_episodes']
    
    logger.info(f"Running {num_eval_episodes} evaluation episodes...")
    
    for episode in range(1, num_eval_episodes + 1):
        # Generate new routes for each episode
        generate_routes(
            output_file=route_file,
            num_vehicles=config['traffic']['num_vehicles_per_episode'],
            episode_duration=config['traffic']['episode_duration'],
            distribution=distribution,
            seed=args.seed + episode if args.seed else None
        )
        
        # Reset environment
        state = env.reset(seed=args.seed + episode if args.seed else None)
        episode_reward = 0
        done = False
        
        # Episode loop (greedy policy)
        pbar = tqdm(total=config['dqn']['max_steps_per_episode'],
                   desc=f"Eval Episode {episode}/{num_eval_episodes}",
                   leave=False)
        
        step = 0
        while not done and step < config['dqn']['max_steps_per_episode']:
            # Select action (greedy)
            action = agent.select_action(state, epsilon=0.0)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            pbar.update(1)
        
        pbar.close()
        
        # Get episode metrics
        env_metrics = env.get_metrics()
        
        # Track metrics
        metrics_tracker.add(env_metrics)
        
        # Log episode
        log_data = {
            'episode': episode,
            'total_reward': episode_reward,
            'avg_waiting_time': env_metrics['avg_waiting_time'],
            'avg_queue_length': env_metrics['avg_queue_length'],
            'throughput': env_metrics['throughput'],
            'num_completed': env_metrics['num_completed'],
            'num_phase_changes': env_metrics['num_phase_changes']
        }
        csv_logger.log(log_data)
        
        logger.info(
            f"Episode {episode:2d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Wait: {env_metrics['avg_waiting_time']:6.2f}s | "
            f"Queue: {env_metrics['avg_queue_length']:5.2f} | "
            f"Throughput: {env_metrics['throughput']:6.1f} veh/h | "
            f"Completed: {env_metrics['num_completed']:4d}"
        )
    
    # Close environment
    env.close()
    
    # Print summary statistics
    summary = metrics_tracker.get_summary()
    
    logger.info("=" * 80)
    logger.info("Evaluation Summary:")
    logger.info("-" * 80)
    for metric, stats in summary.items():
        logger.info(f"{metric}:")
        logger.info(f"  Mean: {stats['mean']:.2f}")
        logger.info(f"  Std:  {stats['std']:.2f}")
        logger.info(f"  Min:  {stats['min']:.2f}")
        logger.info(f"  Max:  {stats['max']:.2f}")
    logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--config', type=str, default='traffic_rl/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (default: from config)')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate agent
    evaluate_dqn(config, args)


if __name__ == "__main__":
    main()
