#!/usr/bin/env python3
"""
Quick evaluation: Apply the original multihead_dqn_best.pth model
directly on the Habsiguda-Nacharam corridor (zero-shot transfer).
Compares RL vs fixed-time baseline.
"""

import os, sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.habsiguda_nacharam import HabsigudaNacharamNetwork
from traffic_rl.dqn.multihead_agent import MultiHeadDQNAgent
from traffic_rl.env.habsiguda_env import HabsigudaSUMOEnv
import traci


def run_episodes(network, env, agents, episodes, mode="rl"):
    """Run evaluation episodes."""
    junction_ids = network.get_all_junctions()
    results = []

    for ep in range(1, episodes + 1):
        states = env.reset(seed=200 + ep)
        done = False
        step = 0
        ep_reward = 0.0

        # Track per-junction waiting times and queues
        total_wait_samples = []
        total_queue_samples = []

        while not done and step < env.max_steps:
            if mode == "rl":
                actions = {}
                for jid in junction_ids:
                    actions[jid] = agents[jid].select_action(states[jid], epsilon=0.0)
            else:
                # Fixed-time: 30s NS, 30s EW
                cycle_pos = step % 60
                action = 0 if cycle_pos < 30 else 1
                actions = {jid: action for jid in junction_ids}

            next_states, rewards, done, info = env.step(actions)

            # Collect traffic metrics
            try:
                vehicle_ids = traci.vehicle.getIDList()
                if vehicle_ids:
                    waits = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                    total_wait_samples.append(np.mean(waits))
                    total_queue_samples.append(sum(1 for s in speeds if s < 0.1))
            except:
                pass

            if mode == "rl":
                ep_reward += sum(rewards.values())
            states = next_states
            step += 1

        metrics = env.get_metrics()
        avg_wait = np.mean(total_wait_samples) if total_wait_samples else 0
        avg_queue = np.mean(total_queue_samples) if total_queue_samples else 0

        results.append({
            'throughput': metrics['throughput'],
            'arrived': metrics['total_arrived'],
            'phase_changes': metrics['total_phase_changes'],
            'avg_waiting_time': avg_wait,
            'avg_queue_length': avg_queue,
            'reward': ep_reward if mode == "rl" else 0,
        })

        print(f"  Ep {ep}: throughput={metrics['throughput']:.1f} veh/hr, "
              f"arrived={metrics['total_arrived']}, "
              f"avg_wait={avg_wait:.1f}s, avg_queue={avg_queue:.1f}")

    env.close()
    return results


def main():
    print("=" * 65)
    print("Habsiguda-Nacharam: Original Best Model Evaluation")
    print("Using: models/multihead_dqn_best.pth (fully trained, 200 eps)")
    print("Traffic: 2000 vehicles/hr (realistic Habsiguda peak conditions)")
    print("=" * 65)

    network = HabsigudaNacharamNetwork()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    junction_ids = network.get_all_junctions()

    net_file = 'traffic_rl/sumo/habsiguda_network.net.xml'
    route_file = 'traffic_rl/sumo/habsiguda_routes.rou.xml'
    model_path = 'models/multihead_dqn_best.pth'

    episodes = 5

    # === FIXED-TIME BASELINE ===
    print("\n--- Fixed-Time Baseline (30s/30s cycle) ---")
    env_b = HabsigudaSUMOEnv(network=network, net_file=net_file,
                              route_file=route_file, max_steps=3600)
    baseline = run_episodes(network, env_b, None, episodes, mode="baseline")

    # === RL AGENT (original best model) ===
    print(f"\n--- RL Agent (multihead_dqn_best.pth) ---")
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    agent_config = {
        'num_heads': 3,
        'encoder_layers': [256, 256],
        'head_layers': [128],
        'classifier_layers': [128, 64],
        'gating_type': 'soft',
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'buffer_size': 1000,
        'min_buffer_size': 100,
        'epsilon_start': 0.0,
        'epsilon_end': 0.0,
        'epsilon_decay_steps': 1,
        'target_update_frequency': 1000,
        'classifier_loss_weight': 0.1,
    }

    agents = {}
    for jid in junction_ids:
        agent = MultiHeadDQNAgent(
            state_size=9, action_size=2,
            config=agent_config, device=device
        )
        try:
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            print(f"  [OK] {jid} ({network.get_junction(jid).name})")
        except Exception as e:
            print(f"  [WARN] {jid}: {e}")
        agents[jid] = agent

    env_r = HabsigudaSUMOEnv(network=network, net_file=net_file,
                              route_file=route_file, max_steps=3600)
    rl_results = run_episodes(network, env_r, agents, episodes, mode="rl")

    # === COMPARISON ===
    print("\n" + "=" * 65)
    print("COMPARISON: Original Best Model vs Fixed-Time Baseline")
    print("=" * 65)

    metrics_names = ['throughput', 'arrived', 'avg_waiting_time', 'avg_queue_length', 'phase_changes']
    labels = ['Throughput (veh/hr)', 'Vehicles Completed', 'Avg Wait Time (s)',
              'Avg Queue Length', 'Phase Changes']

    print(f"\n{'Metric':<25} {'Fixed-Time':>12} {'RL Agent':>12} {'Change':>12}")
    print("-" * 65)

    for name, label in zip(metrics_names, labels):
        b_val = np.mean([r[name] for r in baseline])
        r_val = np.mean([r[name] for r in rl_results])
        if b_val > 0:
            pct = ((r_val - b_val) / b_val) * 100
            print(f"{label:<25} {b_val:>12.1f} {r_val:>12.1f} {pct:>+11.1f}%")
        else:
            print(f"{label:<25} {b_val:>12.1f} {r_val:>12.1f} {'N/A':>12}")

    # Save results
    output_dir = Path('results/habsiguda_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'original_model_evaluation.json', 'w') as f:
        json.dump({
            'model': 'multihead_dqn_best.pth (200 episodes, fully trained)',
            'traffic': '2000 vehicles/hr (realistic Habsiguda peak)',
            'baseline': [{k: float(v) for k, v in r.items()} for r in baseline],
            'rl': [{k: float(v) for k, v in r.items()} for r in rl_results],
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'original_model_evaluation.json'}")


if __name__ == "__main__":
    main()
