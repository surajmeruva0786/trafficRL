"""
Simple script to test a trained Multi-Head DQN model with SUMO GUI visualization.

This script loads a trained model and runs it in SUMO with GUI enabled
so you can visually see how the agent controls the traffic lights.
"""

import os
import sys
import yaml
import torch
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.sumo_env import SUMOTrafficEnv


def test_model_with_gui(model_path: str, num_episodes: int = 3):
    """
    Test a trained model with SUMO GUI visualization.
    
    Args:
        model_path: Path to the trained model file
        num_episodes: Number of episodes to run
    """
    # Load configuration
    config_path = "traffic_rl/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("MULTI-HEAD DQN MODEL TESTING WITH GUI")
    print("="*70)
    print(f"Device: {device.upper()}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("="*70)
    print("\nStarting SUMO GUI...")
    print("NOTE: The SUMO window will open. You can:")
    print("  - Click 'Play' to start/resume simulation")
    print("  - Adjust speed with the slider")
    print("  - Watch the traffic lights change based on the AI agent")
    print("="*70)
    
    # Create environment with GUI enabled
    env = SUMOTrafficEnv(
        net_file=config['sumo']['network_file'],
        route_file=config['sumo']['route_file'],
        config_file=config['sumo']['config_file'],
        use_gui=True,  # Enable GUI
        step_length=config['sumo']['step_length'],
        yellow_time=config['sumo']['yellow_time'],
        min_green_time=config['sumo']['min_green_time'],
        max_steps=config['traffic']['episode_duration'],
        reward_weights=config['reward']
    )
    
    # Load the trained agent
    print("\nLoading trained agent...")
    agent = MultiHeadDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config['dqn'] | config,
        device=device
    )
    agent.load(model_path)
    print(f"✓ Model loaded successfully")
    
    # Run episodes
    for episode in range(1, num_episodes + 1):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode}/{num_episodes}")
        print(f"{'='*70}")
        
        state = env.reset()
        episode_reward = 0
        step_count = 0
        
        print("Running simulation... (watch the SUMO GUI window)")
        
        while True:
            # Select action (greedy - no exploration)
            action = agent.select_action(state, epsilon=0.0)
            
            # Get regime info for display
            if hasattr(agent, 'get_regime_info'):
                regime_info = agent.get_regime_info(state)
                regime_name = ['Low', 'Medium', 'High'][regime_info['predicted_regime']]
            else:
                regime_name = "N/A"
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward={episode_reward:.2f}, "
                      f"Regime={regime_name}, Action={'NS' if action == 0 else 'EW'}")
            
            state = next_state
            
            if done:
                break
        
        # Get final metrics
        metrics = env.get_metrics()
        
        print(f"\nEpisode {episode} Results:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Steps: {step_count}")
        print(f"  Avg Waiting Time: {metrics['avg_waiting_time']:.2f}s")
        print(f"  Avg Queue Length: {metrics['avg_queue_length']:.2f}")
        print(f"  Throughput: {metrics['throughput']:.2f} vehicles/hour")
        print(f"  Phase Changes: {metrics['num_phase_changes']}")
        
        # Small delay between episodes
        if episode < num_episodes:
            print("\nStarting next episode in 3 seconds...")
            time.sleep(3)
    
    # Clean up
    env.close()
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Multi-Head DQN model with SUMO GUI visualization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/multihead_dqn_ep100.pth",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(models_dir, f)}")
        sys.exit(1)
    
    test_model_with_gui(args.model, args.episodes)
