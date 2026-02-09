#!/usr/bin/env python3
"""
Multi-Agent Grid Network Training Script
Trains multiple DQN agents on a grid network with balanced regime distribution.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from datetime import datetime
import json
import traci  # Import TraCI for regime sampling

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork
from traffic_rl.coordination.multi_agent_coordination import MultiAgentCoordination, CoordinationMode
from traffic_rl.coordination.green_wave_analyzer import GreenWaveAnalyzer
from traffic_rl.dqn.regime_classifier import RegimeClassifier
from traffic_rl.dqn.agent import DQNAgent
from traffic_rl.env.grid_sumo_env import GridSUMOEnv
from traffic_rl.utils.logging_utils import setup_logger


def create_grid_config(rows=3, cols=3, spacing=500.0):
    """Create configuration for grid network."""
    return {
        'rows': rows,
        'cols': cols,
        'spacing': spacing,
        'network_file': f'traffic_rl/sumo/grid_network.net.xml',
        'route_file': f'traffic_rl/sumo/grid_routes.rou.xml'
    }


def train_multihead_grid(
    episodes=100,
    rows=3,
    cols=3,
    coordination_mode='independent',
    balanced_regimes=True,
    save_interval=10,
    output_dir='results/grid_training'
):
    """
    Train multi-agent system on grid network.
    
    Args:
        episodes: Number of training episodes
        rows: Number of grid rows
        cols: Number of grid columns
        coordination_mode: 'independent' or 'coordinated'
        balanced_regimes: Whether to enforce balanced regime distribution
        save_interval: Save checkpoints every N episodes
        output_dir: Directory for saving results
    """
    # ===== GPU/CUDA SETUP =====
    print("=" * 60)
    print("GPU/CUDA Setup")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN auto-tuner enabled")
    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: CUDA not available!")
        print("⚠️  Training will run on CPU (this will be VERY slow)")
        print("⚠️  Please install CUDA-enabled PyTorch for GPU acceleration")
    
    print(f"✓ Using device: {device}")
    print("=" * 60)
    print()
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('grid_training', output_path / 'training.log')
    logger.info(f"Starting grid network training: {rows}x{cols} grid")
    logger.info(f"Coordination mode: {coordination_mode}")
    logger.info(f"Balanced regimes: {balanced_regimes}")
    logger.info(f"Device: {device}")
    
    # Create network
    network = MultiIntersectionNetwork(rows=rows, cols=cols, spacing=500.0)
    logger.info(f"Created network with {len(network.intersections)} intersections")
    
    # Setup paths
    net_file = 'traffic_rl/sumo/grid_network.net.xml'
    route_file = 'traffic_rl/sumo/grid_routes.rou.xml'
    
    # Verify files exist
    if not os.path.exists(net_file):
        raise FileNotFoundError(
            f"Network file not found: {net_file}\n"
            f"Please run: python traffic_rl/sumo/generate_grid_network.py"
        )
    if not os.path.exists(route_file):
        raise FileNotFoundError(
            f"Route file not found: {route_file}\n"
            f"Please run: python traffic_rl/sumo/generate_grid_routes.py"
        )
    
    # Initialize Grid SUMO Environment
    logger.info("Initializing Grid SUMO environment...")
    env = GridSUMOEnv(
        network=network,
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        step_length=1.0,
        yellow_time=3,
        min_green_time=10,
        max_steps=3600
    )
    
    # Get intersection IDs
    intersection_ids = network.get_all_intersections()
    num_agents = len(intersection_ids)
    logger.info(f"Number of agents: {num_agents}")
    
    # Initialize DQN agents for each intersection
    logger.info("Initializing DQN agents...")
    agents = {}
    agent_config = {
        'hidden_layers': [128, 128],
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 50000,
        'min_buffer_size': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 50000,
        'target_update_frequency': 1000
    }
    
    for i_id in intersection_ids:
        agents[i_id] = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            config=agent_config,
            device=str(device)
        )
    
    logger.info(f"[OK] Created {num_agents} DQN agents on {device}")
    
    # Setup coordination
    network_topology = {
        i_id: intersection.neighbors
        for i_id, intersection in network.intersections.items()
    }
    
    coordinator = MultiAgentCoordination(network_topology)
    coordinator.set_arterial_routes(network.get_arterial_routes())
    
    mode = CoordinationMode.COORDINATED if coordination_mode == 'coordinated' else CoordinationMode.INDEPENDENT
    coordinator.set_coordination_mode(mode)
    
    # Setup green wave analyzer
    analyzer = GreenWaveAnalyzer()
    analyzer.set_arterial_routes(network.get_arterial_routes())
    
    # Setup regime classifier
    regime_classifier = RegimeClassifier(env.state_size, device=str(device))
    
    # Training metrics
    training_stats = {
        'episodes': [],
        'total_rewards': [],
        'avg_waiting_times': [],
        'throughputs': [],
        'coordination_scores': [],
        'regime_distribution': []
    }
    
    logger.info(f"Starting training for {episodes} episodes...")
    logger.info("=" * 60)
    
    # Training loop
    for episode in range(1, episodes + 1):
        logger.info(f"\nEpisode {episode}/{episodes}")
        logger.info("=" * 60)
        
        # Reset environment
        states = env.reset(seed=42 + episode)
        
        episode_rewards = {i_id: 0.0 for i_id in intersection_ids}
        done = False
        step = 0
        
        # Track regime metrics during episode (not at end when traffic is 0)
        regime_samples = []
        
        # Episode loop
        while not done and step < 3600:
            # Select actions for all agents
            actions = {}
            for i_id in intersection_ids:
                state = states[i_id]
                action = agents[i_id].select_action(state)
                actions[i_id] = action
            
            # Execute actions
            next_states, rewards, done, info = env.step(actions)
            
            # Store transitions and train
            for i_id in intersection_ids:
                agents[i_id].store_transition(
                    states[i_id],
                    actions[i_id],
                    rewards[i_id],
                    next_states[i_id],
                    done
                )
                
                # Train agent
                loss = agents[i_id].train_step()
                
                # Accumulate rewards
                episode_rewards[i_id] += rewards[i_id]
            
            # Record signal changes for green wave analysis
            for i_id in intersection_ids:
                current_phase = info['current_phases'][i_id]
                if current_phase == 0:  # NS green
                    analyzer.record_signal_change(i_id, 0, step, True)
                elif current_phase == 2:  # EW green
                    analyzer.record_signal_change(i_id, 0, step, False)
            
            # Collect regime metrics on EVERY step
            total_queue = 0
            total_waiting = 0
            num_lanes = 0
            
            # Debug on first step only
            if step == 1:
                logger.info(f"  [DEBUG] Step {step}: Starting regime sampling for {len(intersection_ids)} intersections")
            
            for i_id in intersection_ids:
                try:
                    # Get lanes controlled by this traffic light
                    controlled_lanes = traci.trafficlight.getControlledLanes(i_id)
                    # Remove duplicates (same lane can appear multiple times)
                    unique_lanes = list(set(controlled_lanes))
                    
                    if step == 1:
                        logger.info(f"  [DEBUG] Intersection {i_id}: {len(unique_lanes)} unique lanes")
                    
                    for lane in unique_lanes:
                        total_queue += traci.lane.getLastStepHaltingNumber(lane)
                        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                        if vehicle_ids:
                            total_waiting += sum([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
                        num_lanes += 1
                except Exception as e:
                    if step == 1:
                        logger.info(f"  [DEBUG] Error for {i_id}: {e}")
                    pass
            
            if step == 1:
                logger.info(f"  [DEBUG] Step {step}: num_lanes={num_lanes}, total_queue={total_queue}, total_waiting={total_waiting}")
            
            if num_lanes > 0:
                avg_queue = total_queue / num_lanes
                avg_wait = total_waiting / num_lanes
                regime_samples.append((avg_queue, avg_wait))
            elif step == 1:
                logger.info(f"  [DEBUG] Step {step}: num_lanes is 0, NOT appending sample!")
            
            states = next_states
            step += 1
        
        # Debug: Log episode length
        logger.info(f"  Episode completed in {step} steps, collected {len(regime_samples)} regime samples")
        
        # Get episode metrics
        env_metrics = env.get_metrics()
        
        # Calculate total reward
        total_reward = sum(episode_rewards.values())
        
        # Calculate coordination scores
        coordination_scores = {}
        for route_id, route in network.get_arterial_routes().items():
            score = analyzer.calculate_coordination_score(route)
            coordination_scores[route_id] = score
        
        avg_coordination = np.mean(list(coordination_scores.values())) if coordination_scores else 0.0
        
        # Record regime using sampled metrics from during the episode
        if regime_samples:
            # Calculate average queue and wait time from samples taken during episode
            avg_queue_per_lane = np.mean([s[0] for s in regime_samples])
            avg_waiting_time = np.mean([s[1] for s in regime_samples])
        else:
            # Fallback if no samples (shouldn't happen)
            avg_queue_per_lane = 0.0
            avg_waiting_time = 0.0
        
        # Debug: Log actual metrics for regime classification
        logger.info(f"  Regime Metrics: Avg Queue/Lane={avg_queue_per_lane:.2f}, Avg Wait={avg_waiting_time:.2f}s (from {len(regime_samples)} samples)")
        
        # Classify regime based on actual metrics
        # Adjusted thresholds for actual traffic levels (avg ~0.40 queue/lane, ~5s wait)
        # Low: < 0.25 vehicles/lane OR < 3s wait (very light traffic)
        # Medium: 0.25-0.7 vehicles/lane OR 3-10s wait (moderate traffic)
        # High: > 0.7 vehicles/lane OR > 10s wait (heavy traffic)
        if avg_queue_per_lane < 0.25 and avg_waiting_time < 3:
            regime = 0  # Low
        elif avg_queue_per_lane < 0.7 and avg_waiting_time < 10:
            regime = 1  # Medium
        else:
            regime = 2  # High
        
        regime_classifier.record_regime_exposure(regime, duration=step)
        
        # Log episode results
        logger.info(f"Episode {episode} Results:")
        logger.info(f"  Total Reward: {total_reward:.2f}")
        logger.info(f"  Throughput: {env_metrics['throughput']:.2f} vehicles/hour")
        logger.info(f"  Total Arrived: {env_metrics['total_arrived']}")
        logger.info(f"  Avg Coordination Score: {avg_coordination:.3f}")
        logger.info(f"  Total Phase Changes: {env_metrics['total_phase_changes']}")
        
        # Track regime distribution
        regime_dist = regime_classifier.get_regime_distribution()
        logger.info(f"  Regime Distribution: Low={regime_dist['low_pct']:.1f}%, "
                   f"Med={regime_dist['medium_pct']:.1f}%, High={regime_dist['high_pct']:.1f}%")
        
        # Store stats
        training_stats['episodes'].append(episode)
        training_stats['total_rewards'].append(total_reward)
        training_stats['avg_waiting_times'].append(0.0)  # Placeholder
        training_stats['throughputs'].append(env_metrics['throughput'])
        training_stats['coordination_scores'].append(avg_coordination)
        training_stats['regime_distribution'].append(regime_dist)
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = output_path / f'checkpoint_ep{episode}.pth'
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            
            # Save all agents
            checkpoint = {
                'episode': episode,
                'agents': {i_id: {
                    'policy_net': agents[i_id].policy_net.state_dict(),
                    'target_net': agents[i_id].target_net.state_dict(),
                    'optimizer': agents[i_id].optimizer.state_dict(),
                    'epsilon': agents[i_id].epsilon,
                    'steps_done': agents[i_id].steps_done
                } for i_id in intersection_ids},
                'regime_classifier': regime_classifier.state_dict(),
                'training_stats': training_stats,
                'network_config': {'rows': rows, 'cols': cols}
            }
            torch.save(checkpoint, checkpoint_path)
    
    # Close environment
    env.close()
    
    # Final regime distribution check
    final_dist = regime_classifier.get_regime_distribution()
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Final Regime Distribution:")
    logger.info(f"  Low: {final_dist['low_pct']:.2f}%")
    logger.info(f"  Medium: {final_dist['medium_pct']:.2f}%")
    logger.info(f"  High: {final_dist['high_pct']:.2f}%")
    logger.info(f"  Balanced: {regime_classifier.is_balanced()}")
    
    # Save final results
    results_file = output_path / 'training_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_serializable = {
            'episodes': training_stats['episodes'],
            'total_rewards': [float(r) for r in training_stats['total_rewards']],
            'throughputs': [float(t) for t in training_stats['throughputs']],
            'coordination_scores': [float(c) for c in training_stats['coordination_scores']],
            'regime_distribution': training_stats['regime_distribution']
        }
        json.dump(stats_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return training_stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-agent grid network")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes (default: 100)")
    parser.add_argument("--rows", type=int, default=3,
                       help="Number of grid rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3,
                       help="Number of grid columns (default: 3)")
    parser.add_argument("--coordination-mode", type=str, default='independent',
                       choices=['independent', 'coordinated'],
                       help="Coordination mode (default: independent)")
    parser.add_argument("--no-balanced", action="store_true",
                       help="Disable balanced regime distribution")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N episodes (default: 10)")
    parser.add_argument("--output-dir", type=str, default='results/grid_training',
                       help="Output directory (default: results/grid_training)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode (shorter training)")
    
    args = parser.parse_args()
    
    # Adjust for test mode
    if args.test_mode:
        args.episodes = min(args.episodes, 10)
        print(f"Test mode: Running only {args.episodes} episodes")
    
    # Run training
    train_multihead_grid(
        episodes=args.episodes,
        rows=args.rows,
        cols=args.cols,
        coordination_mode=args.coordination_mode,
        balanced_regimes=not args.no_balanced,
        save_interval=args.save_interval,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
