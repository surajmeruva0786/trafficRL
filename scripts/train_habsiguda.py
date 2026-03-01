#!/usr/bin/env python3
"""
Training Script for Habsiguda-Nacharam Corridor.

Fine-tunes pre-trained multihead DQN agents on the real-world
Habsiguda-Nacharam traffic corridor network.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.habsiguda_nacharam import HabsigudaNacharamNetwork
from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.dqn.regime_classifier import RegimeClassifier
from traffic_rl.env.habsiguda_env import HabsigudaSUMOEnv
from traffic_rl.utils.logging_utils import setup_logger
import traci


def train_habsiguda(
    episodes=100,
    pretrained_model_path='models/multihead_dqn_best.pth',
    save_interval=10,
    output_dir='results/habsiguda_training',
    use_gui=False,
    test_mode=False
):
    """
    Train multi-agent system on Habsiguda-Nacharam corridor with transfer learning.

    Args:
        episodes: Number of training episodes
        pretrained_model_path: Path to pre-trained multihead DQN checkpoint
        save_interval: Save checkpoints every N episodes
        output_dir: Directory for saving results
        use_gui: Whether to use SUMO GUI
        test_mode: If True, run shorter episodes
    """
    # ===== GPU/CUDA SETUP =====
    print("=" * 60)
    print("Habsiguda-Nacharam Training - Transfer Learning")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("[WARNING] Using CPU")

    print(f"[OK] Using device: {device}")
    print("=" * 60)
    print()

    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_dir = Path('models/habsiguda')
    model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('habsiguda_training', output_path / 'training.log')
    logger.info(f"Starting Habsiguda-Nacharam training")
    logger.info(f"Pretrained model: {pretrained_model_path}")
    logger.info(f"Device: {device}")

    # Create network
    network = HabsigudaNacharamNetwork()
    junction_ids = network.get_all_junctions()
    num_agents = len(junction_ids)
    logger.info(f"Network: {network}")
    logger.info(f"Number of agents: {num_agents}")

    for jid in junction_ids:
        j = network.get_junction(jid)
        logger.info(f"  {jid}: {j.name} ({j.junction_type})")

    # Setup paths
    net_file = 'traffic_rl/sumo/habsiguda_network.net.xml'
    route_file = 'traffic_rl/sumo/habsiguda_routes.rou.xml'

    # Verify files exist
    if not os.path.exists(net_file):
        raise FileNotFoundError(
            f"Network file not found: {net_file}\n"
            f"Please run: python traffic_rl/sumo/generate_habsiguda_network.py"
        )
    if not os.path.exists(route_file):
        raise FileNotFoundError(
            f"Route file not found: {route_file}\n"
            f"Please run: python traffic_rl/sumo/generate_habsiguda_routes.py"
        )

    # Initialize environment
    max_steps = 1800 if test_mode else 3600
    env = HabsigudaSUMOEnv(
        network=network,
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        step_length=1.0,
        yellow_time=3,
        min_green_time=15,        # Balanced green time
        max_steps=max_steps,
        reward_weights={
            'waiting_time_weight': -0.4,
            'queue_length_weight': -0.2,
            'phase_change_penalty': -0.3,   # Moderate penalty
            'throughput_weight': 0.5,       # Reward completed vehicles
        },
    )

    # Agent configuration for fine-tuning (lower LR, reduced exploration)
    agent_config = {
        'num_heads': 3,
        'encoder_layers': [256, 256],
        'head_layers': [128],
        'classifier_layers': [128, 64],
        'gating_type': 'soft',
        'learning_rate': 0.00005,      # Lower LR for fine-tuning
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 50000,
        'min_buffer_size': 500,        # Can start training sooner
        'epsilon_start': 0.15,          # Low exploration — exploit pretrained knowledge
        'epsilon_end': 0.02,
        'epsilon_decay_steps': 50000,  # Slow decay
        'target_update_frequency': 500,
        'classifier_loss_weight': 0.1,
    }

    # Initialize agents with pre-trained weights
    logger.info("Initializing MultiHead DQN agents with transfer learning...")
    agents = {}

    # Load pre-trained model if available
    pretrained_checkpoint = None
    if os.path.exists(pretrained_model_path):
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
        pretrained_checkpoint = torch.load(
            pretrained_model_path,
            map_location=device,
            weights_only=False
        )
        logger.info("[OK] Pretrained model loaded successfully")
    else:
        logger.warning(f"Pretrained model not found at {pretrained_model_path}")
        logger.warning("Training from scratch...")

    for jid in junction_ids:
        agent = MultiHeadDQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            config=agent_config,
            device=str(device)
        )

        # Transfer pre-trained weights
        if pretrained_checkpoint is not None:
            try:
                agent.policy_net.load_state_dict(
                    pretrained_checkpoint['policy_net_state_dict']
                )
                agent.target_net.load_state_dict(
                    pretrained_checkpoint['target_net_state_dict']
                )
                logger.info(f"  [OK] Loaded pretrained weights for {jid} ({network.get_junction(jid).name})")
            except Exception as e:
                logger.warning(f"  [WARNING] Could not load weights for {jid}: {e}")
                logger.warning(f"  Training {jid} from scratch")

        agents[jid] = agent

    logger.info(f"[OK] Created {num_agents} agents on {device}")

    # Setup regime classifier
    regime_classifier = RegimeClassifier(env.state_size, device=str(device))

    # Training metrics
    training_stats = {
        'episodes': [],
        'total_rewards': [],
        'avg_waiting_times': [],
        'throughputs': [],
        'per_junction_rewards': [],
    }

    logger.info(f"Starting training for {episodes} episodes...")
    logger.info("=" * 60)

    best_reward = float('-inf')
    best_throughput = 0.0

    # Training loop
    for episode in range(1, episodes + 1):
        logger.info(f"\nEpisode {episode}/{episodes}")
        logger.info("-" * 40)

        # Reset environment
        states = env.reset(seed=42 + episode)

        episode_rewards = {jid: 0.0 for jid in junction_ids}
        done = False
        step = 0
        regime_samples = []

        # Episode loop
        while not done and step < max_steps:
            # Select actions
            actions = {}
            for jid in junction_ids:
                state = states[jid]
                action = agents[jid].select_action(state)
                actions[jid] = action

            # Execute
            next_states, rewards, done, info = env.step(actions)

            # Store transitions and train
            for jid in junction_ids:
                agents[jid].store_transition(
                    states[jid], actions[jid], rewards[jid],
                    next_states[jid], done
                )
                agents[jid].train_step()
                episode_rewards[jid] += rewards[jid]

            # Regime sampling
            total_queue = 0
            total_waiting = 0
            num_lanes = 0

            for jid in junction_ids:
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(jid)
                    unique_lanes = list(set(controlled_lanes))
                    for lane in unique_lanes:
                        total_queue += traci.lane.getLastStepHaltingNumber(lane)
                        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                        if vehicle_ids:
                            total_waiting += sum([
                                traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids
                            ])
                        num_lanes += 1
                except:
                    pass

            if num_lanes > 0:
                regime_samples.append((total_queue / num_lanes, total_waiting / num_lanes))

            states = next_states
            step += 1

        # Episode metrics
        env_metrics = env.get_metrics()
        total_reward = sum(episode_rewards.values())
        current_throughput = env_metrics['throughput']

        # Regime classification
        if regime_samples:
            avg_queue = np.mean([s[0] for s in regime_samples])
            avg_wait = np.mean([s[1] for s in regime_samples])
        else:
            avg_queue, avg_wait = 0.0, 0.0

        if avg_queue < 0.25 and avg_wait < 3:
            regime = 0
        elif avg_queue < 0.7 and avg_wait < 10:
            regime = 1
        else:
            regime = 2

        regime_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        regime_classifier.record_regime_exposure(regime, duration=step)

        # Log results
        logger.info(f"  Total Reward: {total_reward:.2f}")
        logger.info(f"  Throughput: {current_throughput:.1f} veh/hr")
        logger.info(f"  Total Arrived: {env_metrics['total_arrived']}")
        logger.info(f"  Total Phase Changes: {env_metrics['total_phase_changes']}")
        logger.info(f"  Traffic Regime: {regime_names[regime]} "
                     f"(queue={avg_queue:.2f}, wait={avg_wait:.1f}s)")

        # Per-junction rewards
        for jid in junction_ids:
            jname = network.get_junction(jid).name
            logger.info(f"    {jid} ({jname}): reward={episode_rewards[jid]:.2f}")

        # Store stats
        training_stats['episodes'].append(episode)
        training_stats['total_rewards'].append(total_reward)
        training_stats['throughputs'].append(current_throughput)
        training_stats['per_junction_rewards'].append(
            {jid: float(r) for jid, r in episode_rewards.items()}
        )

        # Save best model based on THROUGHPUT (not reward)
        if current_throughput > best_throughput:
            best_throughput = current_throughput
            best_reward = total_reward
            for jid in junction_ids:
                agents[jid].save(str(model_dir / f'habsiguda_{jid}_best.pth'))
            logger.info(f"  * New best throughput: {best_throughput:.1f} veh/hr (reward: {best_reward:.2f})")

        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = output_path / f'checkpoint_ep{episode}.pth'
            checkpoint = {
                'episode': episode,
                'agents': {
                    jid: {
                        'policy_net': agents[jid].policy_net.state_dict(),
                        'target_net': agents[jid].target_net.state_dict(),
                    } for jid in junction_ids
                },
                'training_stats': training_stats,
                'best_reward': best_reward,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Saved checkpoint: {checkpoint_path}")

    # Close environment
    env.close()

    # Save final results
    results_file = output_path / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'episodes': training_stats['episodes'],
            'total_rewards': [float(r) for r in training_stats['total_rewards']],
            'throughputs': [float(t) for t in training_stats['throughputs']],
            'best_reward': float(best_reward),
            'network_info': network.get_network_info(),
        }, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"Results saved to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete - Habsiguda-Nacharam Corridor")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Models saved to: {model_dir}")
    print(f"Results saved to: {output_path}")

    return training_stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train multihead DQN on Habsiguda-Nacharam corridor"
    )
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes (default: 100)")
    parser.add_argument("--pretrained-model", type=str,
                        default='models/multihead_dqn_best.pth',
                        help="Path to pretrained model (default: models/multihead_dqn_best.pth)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N episodes (default: 10)")
    parser.add_argument("--output-dir", type=str,
                        default='results/habsiguda_training',
                        help="Output directory (default: results/habsiguda_training)")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode (shorter episodes)")

    args = parser.parse_args()

    if args.test_mode:
        args.episodes = min(args.episodes, 5)
        print(f"Test mode: Running {args.episodes} episodes with shorter duration")

    train_habsiguda(
        episodes=args.episodes,
        pretrained_model_path=args.pretrained_model,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        use_gui=args.gui,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
