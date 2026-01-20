"""
Fixed-time baseline controller for comparison.

This script runs a simple fixed-time traffic light controller
and collects metrics for comparison with the DQN agent.
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
from traffic_rl.utils.metrics import MetricsTracker
from traffic_rl.utils.logging_utils import setup_logger, CSVLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FixedTimeController:
    """
    Fixed-time traffic light controller.
    
    Alternates between NS green and EW green with fixed durations.
    """
    
    def __init__(self, ns_green_time: int, ew_green_time: int):
        """
        Initialize fixed-time controller.
        
        Args:
            ns_green_time: Duration of NS green phase in seconds
            ew_green_time: Duration of EW green phase in seconds
        """
        self.ns_green_time = ns_green_time
        self.ew_green_time = ew_green_time
        self.cycle_length = ns_green_time + ew_green_time
        self.current_time = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on fixed schedule.
        
        Args:
            state: Current state (not used)
        
        Returns:
            Action (0: NS green, 1: EW green)
        """
        time_in_cycle = self.current_time % self.cycle_length
        
        if time_in_cycle < self.ns_green_time:
            action = 0  # NS green
        else:
            action = 1  # EW green
        
        self.current_time += 1
        return action
    
    def reset(self):
        """Reset controller for new episode."""
        self.current_time = 0


def run_baseline(config: dict, args: argparse.Namespace):
    """
    Run fixed-time baseline controller.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Setup logging
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    logger = setup_logger(
        'baseline',
        os.path.join(config['logging']['log_dir'], 'baseline.log')
    )
    
    logger.info("=" * 80)
    logger.info("Running Fixed-Time Baseline Controller")
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
    
    # Initialize fixed-time controller
    controller = FixedTimeController(
        ns_green_time=config['baseline']['ns_green_time'],
        ew_green_time=config['baseline']['ew_green_time']
    )
    
    logger.info(f"NS green time: {config['baseline']['ns_green_time']}s")
    logger.info(f"EW green time: {config['baseline']['ew_green_time']}s")
    logger.info(f"Cycle length: {controller.cycle_length}s")
    
    # Setup CSV logging
    csv_logger = CSVLogger(
        filepath=os.path.join(config['logging']['results_dir'], 'fixed_time_metrics.csv'),
        fieldnames=[
            'episode', 'total_reward', 'avg_waiting_time', 'avg_queue_length',
            'throughput', 'num_completed', 'num_phase_changes'
        ]
    )
    
    # Evaluation metrics
    metrics_tracker = MetricsTracker()
    num_episodes = config['evaluation']['num_episodes']
    
    logger.info(f"Running {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        # Generate new routes for each episode
        generate_routes(
            output_file=route_file,
            num_vehicles=config['traffic']['num_vehicles_per_episode'],
            episode_duration=config['traffic']['episode_duration'],
            distribution=distribution,
            seed=args.seed + episode if args.seed else None
        )
        
        # Reset environment and controller
        state = env.reset(seed=args.seed + episode if args.seed else None)
        controller.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        pbar = tqdm(total=config['dqn']['max_steps_per_episode'],
                   desc=f"Baseline Episode {episode}/{num_episodes}",
                   leave=False)
        
        step = 0
        while not done and step < config['dqn']['max_steps_per_episode']:
            # Select action from fixed-time controller
            action = controller.select_action(state)
            
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
    logger.info("Baseline Summary:")
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
    parser = argparse.ArgumentParser(description='Run fixed-time baseline controller')
    parser.add_argument('--config', type=str, default='traffic_rl/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run baseline
    run_baseline(config, args)


if __name__ == "__main__":
    main()
