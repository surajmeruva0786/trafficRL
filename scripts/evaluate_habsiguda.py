#!/usr/bin/env python3
"""
Evaluation Script for Habsiguda-Nacharam Corridor.

Evaluates trained DQN agents on the Habsiguda-Nacharam corridor
and compares against a fixed-time baseline.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.habsiguda_nacharam import HabsigudaNacharamNetwork
from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.habsiguda_env import HabsigudaSUMOEnv
from traffic_rl.utils.logging_utils import setup_logger
import traci


def run_fixed_time_baseline(network, env, episodes=5, ns_green=30, ew_green=30):
    """Run fixed-time baseline on the Habsiguda-Nacharam corridor."""
    print("\n" + "=" * 60)
    print("Running Fixed-Time Baseline")
    print(f"NS Green: {ns_green}s, EW Green: {ew_green}s")
    print("=" * 60)

    junction_ids = network.get_all_junctions()
    metrics_list = []

    for ep in range(1, episodes + 1):
        states = env.reset(seed=100 + ep)
        done = False
        step = 0
        cycle_length = ns_green + ew_green

        while not done and step < env.max_steps:
            # Determine current phase based on cycle position
            cycle_pos = step % cycle_length
            action = 0 if cycle_pos < ns_green else 1

            # Same action for all junctions
            actions = {jid: action for jid in junction_ids}
            states, _, done, info = env.step(actions)
            step += 1

        ep_metrics = env.get_metrics()
        metrics_list.append(ep_metrics)
        print(f"  Episode {ep}: throughput={ep_metrics['throughput']:.1f} veh/hr, "
              f"arrived={ep_metrics['total_arrived']}")

    env.close()

    avg_metrics = {
        'avg_throughput': np.mean([m['throughput'] for m in metrics_list]),
        'avg_arrived': np.mean([m['total_arrived'] for m in metrics_list]),
        'avg_phase_changes': np.mean([m['total_phase_changes'] for m in metrics_list]),
    }
    return avg_metrics


def run_rl_evaluation(network, env, agents, episodes=5):
    """Run trained RL agents on the Habsiguda-Nacharam corridor."""
    print("\n" + "=" * 60)
    print("Running RL Agent Evaluation")
    print("=" * 60)

    junction_ids = network.get_all_junctions()
    metrics_list = []

    for ep in range(1, episodes + 1):
        states = env.reset(seed=100 + ep)
        done = False
        step = 0
        episode_rewards = {jid: 0.0 for jid in junction_ids}

        while not done and step < env.max_steps:
            actions = {}
            for jid in junction_ids:
                action = agents[jid].select_action(states[jid], epsilon=0.0)
                actions[jid] = action

            next_states, rewards, done, info = env.step(actions)

            for jid in junction_ids:
                episode_rewards[jid] += rewards[jid]

            states = next_states
            step += 1

        ep_metrics = env.get_metrics()
        ep_metrics['total_reward'] = sum(episode_rewards.values())
        metrics_list.append(ep_metrics)

        print(f"  Episode {ep}: throughput={ep_metrics['throughput']:.1f} veh/hr, "
              f"arrived={ep_metrics['total_arrived']}, "
              f"reward={ep_metrics['total_reward']:.2f}")

    env.close()

    avg_metrics = {
        'avg_throughput': np.mean([m['throughput'] for m in metrics_list]),
        'avg_arrived': np.mean([m['total_arrived'] for m in metrics_list]),
        'avg_phase_changes': np.mean([m['total_phase_changes'] for m in metrics_list]),
        'avg_reward': np.mean([m['total_reward'] for m in metrics_list]),
    }
    return avg_metrics


def generate_comparison_plot(rl_metrics, baseline_metrics, output_dir):
    """Generate comparison plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Throughput comparison
    methods = ['Fixed-Time', 'RL (Fine-tuned)']
    throughputs = [baseline_metrics['avg_throughput'], rl_metrics['avg_throughput']]
    colors = ['#e74c3c', '#2ecc71']

    axes[0].bar(methods, throughputs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Throughput (vehicles/hr)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Vehicles per Hour')
    for i, v in enumerate(throughputs):
        axes[0].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')

    # Total arrived
    arrived = [baseline_metrics['avg_arrived'], rl_metrics['avg_arrived']]
    axes[1].bar(methods, arrived, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Vehicles Completed', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Total Vehicles')
    for i, v in enumerate(arrived):
        axes[1].text(i, v + 2, f'{v:.0f}', ha='center', fontweight='bold')

    # Phase changes
    phase_changes = [baseline_metrics['avg_phase_changes'], rl_metrics['avg_phase_changes']]
    axes[2].bar(methods, phase_changes, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title('Phase Changes', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Total Phase Changes')
    for i, v in enumerate(phase_changes):
        axes[2].text(i, v + 2, f'{v:.0f}', ha='center', fontweight='bold')

    fig.suptitle('Habsiguda-Nacharam Corridor: RL vs Fixed-Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'habsiguda_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_path / 'habsiguda_comparison.png'}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained agents on Habsiguda-Nacharam corridor"
    )
    parser.add_argument("--model-dir", type=str, default='models/habsiguda',
                        help="Directory with trained models (default: models/habsiguda)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--output-dir", type=str,
                        default='results/habsiguda_evaluation',
                        help="Output directory (default: results/habsiguda_evaluation)")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the baseline comparison")

    args = parser.parse_args()

    print("=" * 60)
    print("Habsiguda-Nacharam Corridor Evaluation")
    print("=" * 60)

    # Setup
    network = HabsigudaNacharamNetwork()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    junction_ids = network.get_all_junctions()

    net_file = 'traffic_rl/sumo/habsiguda_network.net.xml'
    route_file = 'traffic_rl/sumo/habsiguda_routes.rou.xml'

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ===== BASELINE =====
    env_baseline = HabsigudaSUMOEnv(
        network=network,
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
    )
    baseline_metrics = run_fixed_time_baseline(
        network, env_baseline, episodes=args.episodes
    )

    if args.baseline_only:
        print(f"\nBaseline Results:")
        for k, v in baseline_metrics.items():
            print(f"  {k}: {v:.2f}")
        return

    # ===== RL AGENTS =====
    agent_config = {
        'num_heads': 3,
        'encoder_layers': [256, 256],
        'head_layers': [128],
        'classifier_layers': [128, 64],
        'gating_type': 'soft',
        'learning_rate': 0.00005,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 50000,
        'min_buffer_size': 500,
        'epsilon_start': 0.0,
        'epsilon_end': 0.0,
        'epsilon_decay_steps': 1,
        'target_update_frequency': 1000,
        'classifier_loss_weight': 0.1,
    }

    agents = {}
    model_dir = Path(args.model_dir)

    for jid in junction_ids:
        agent = MultiHeadDQNAgent(
            state_size=9,
            action_size=2,
            config=agent_config,
            device=device,
        )

        model_path = model_dir / f'habsiguda_{jid}_best.pth'
        if model_path.exists():
            agent.load(str(model_path))
            print(f"  ✓ Loaded model for {jid} ({network.get_junction(jid).name})")
        else:
            print(f"  ⚠️  No model found for {jid}, using untrained agent")

        agents[jid] = agent

    env_rl = HabsigudaSUMOEnv(
        network=network,
        net_file=net_file,
        route_file=route_file,
        use_gui=args.gui,
    )
    rl_metrics = run_rl_evaluation(network, env_rl, agents, episodes=args.episodes)

    # ===== COMPARISON =====
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    comparison = {
        'Throughput (veh/hr)': (baseline_metrics['avg_throughput'], rl_metrics['avg_throughput']),
        'Vehicles Completed': (baseline_metrics['avg_arrived'], rl_metrics['avg_arrived']),
        'Phase Changes': (baseline_metrics['avg_phase_changes'], rl_metrics['avg_phase_changes']),
    }

    print(f"{'Metric':<25} {'Fixed-Time':>12} {'RL Agent':>12} {'Improvement':>12}")
    print("-" * 65)

    for metric, (baseline_val, rl_val) in comparison.items():
        if baseline_val > 0:
            improvement = ((rl_val - baseline_val) / baseline_val) * 100
            print(f"{metric:<25} {baseline_val:>12.1f} {rl_val:>12.1f} {improvement:>+11.1f}%")
        else:
            print(f"{metric:<25} {baseline_val:>12.1f} {rl_val:>12.1f} {'N/A':>12}")

    # Generate plots
    generate_comparison_plot(rl_metrics, baseline_metrics, args.output_dir)

    # Save results
    results = {
        'baseline': {k: float(v) for k, v in baseline_metrics.items()},
        'rl': {k: float(v) for k, v in rl_metrics.items()},
        'network_info': network.get_network_info(),
    }

    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
