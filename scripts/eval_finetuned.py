#!/usr/bin/env python3
"""
Full evaluation of the fine-tuned Habsiguda-Nacharam model (100 episodes)
vs fixed-time baseline. Uses the same environment settings as training.
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
    """Run evaluation episodes and collect detailed metrics."""
    junction_ids = network.get_all_junctions()
    results = []

    for ep in range(1, episodes + 1):
        states = env.reset(seed=300 + ep)
        done = False
        step = 0
        ep_reward = 0.0

        total_wait_samples = []
        total_queue_samples = []
        total_speed_samples = []

        while not done and step < env.max_steps:
            if mode == "rl":
                actions = {}
                for jid in junction_ids:
                    actions[jid] = agents[jid].select_action(states[jid], epsilon=0.0)
            else:
                cycle_pos = step % 60
                action = 0 if cycle_pos < 30 else 1
                actions = {jid: action for jid in junction_ids}

            next_states, rewards, done, info = env.step(actions)

            try:
                vehicle_ids = traci.vehicle.getIDList()
                if vehicle_ids:
                    waits = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                    total_wait_samples.append(np.mean(waits))
                    total_queue_samples.append(sum(1 for s in speeds if s < 0.1))
                    total_speed_samples.append(np.mean(speeds))
            except:
                pass

            if mode == "rl":
                ep_reward += sum(rewards.values())
            states = next_states
            step += 1

        metrics = env.get_metrics()
        avg_wait = np.mean(total_wait_samples) if total_wait_samples else 0
        avg_queue = np.mean(total_queue_samples) if total_queue_samples else 0
        avg_speed = np.mean(total_speed_samples) if total_speed_samples else 0

        results.append({
            'throughput': metrics['throughput'],
            'arrived': metrics['total_arrived'],
            'phase_changes': metrics['total_phase_changes'],
            'avg_waiting_time': avg_wait,
            'avg_queue_length': avg_queue,
            'avg_speed': avg_speed,
            'reward': ep_reward if mode == "rl" else 0,
        })

        print(f"  Ep {ep}: throughput={metrics['throughput']:.1f} veh/hr, "
              f"arrived={metrics['total_arrived']}, "
              f"avg_wait={avg_wait:.1f}s, avg_queue={avg_queue:.1f}, "
              f"avg_speed={avg_speed:.1f} m/s, "
              f"phase_changes={metrics['total_phase_changes']}")

    env.close()
    return results


def main():
    print("=" * 70)
    print("FULL EVALUATION: Fine-tuned Model (100 eps) vs Fixed-Time Baseline")
    print("Traffic: 2000 vehicles/hr (realistic Habsiguda peak conditions)")
    print("Environment: min_green=20s, phase_change_penalty=-1.0")
    print("=" * 70)

    network = HabsigudaNacharamNetwork()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    junction_ids = network.get_all_junctions()

    net_file = 'traffic_rl/sumo/habsiguda_network.net.xml'
    route_file = 'traffic_rl/sumo/habsiguda_routes.rou.xml'
    model_dir = Path('models/habsiguda')

    episodes = 5
    env_kwargs = dict(
        network=network, net_file=net_file, route_file=route_file,
        max_steps=3600, min_green_time=20,
        reward_weights={
            'waiting_time_weight': -0.5,
            'queue_length_weight': -0.3,
            'phase_change_penalty': -1.0,
            'throughput_weight': 1.0,
        },
    )

    # === BASELINE ===
    print("\n--- Fixed-Time Baseline (30s/30s cycle) ---")
    env_b = HabsigudaSUMOEnv(**env_kwargs)
    baseline = run_episodes(network, env_b, None, episodes, mode="baseline")

    # === FINE-TUNED RL AGENT ===
    print(f"\n--- Fine-tuned RL Agent (100 episodes training) ---")

    agent_config = {
        'num_heads': 3, 'encoder_layers': [256, 256], 'head_layers': [128],
        'classifier_layers': [128, 64], 'gating_type': 'soft',
        'learning_rate': 0.00005, 'gamma': 0.99, 'batch_size': 64,
        'buffer_size': 1000, 'min_buffer_size': 100,
        'epsilon_start': 0.0, 'epsilon_end': 0.0, 'epsilon_decay_steps': 1,
        'target_update_frequency': 1000, 'classifier_loss_weight': 0.1,
    }

    agents = {}
    for jid in junction_ids:
        agent = MultiHeadDQNAgent(
            state_size=9, action_size=2, config=agent_config, device=device
        )
        model_path = model_dir / f'habsiguda_{jid}_best.pth'
        if model_path.exists():
            agent.load(str(model_path))
            print(f"  [OK] {jid} ({network.get_junction(jid).name})")
        else:
            print(f"  [WARN] No model for {jid}")
        agents[jid] = agent

    env_r = HabsigudaSUMOEnv(**env_kwargs)
    rl_results = run_episodes(network, env_r, agents, episodes, mode="rl")

    # === COMPARISON TABLE ===
    print("\n" + "=" * 70)
    print("RESULTS: Fine-tuned RL Agent vs Fixed-Time Baseline")
    print("=" * 70)

    metrics_map = [
        ('throughput', 'Throughput (veh/hr)'),
        ('arrived', 'Vehicles Completed'),
        ('avg_waiting_time', 'Avg Waiting Time (s)'),
        ('avg_queue_length', 'Avg Queue Length'),
        ('avg_speed', 'Avg Speed (m/s)'),
        ('phase_changes', 'Phase Changes'),
    ]

    print(f"\n{'Metric':<25} {'Fixed-Time':>12} {'RL Agent':>12} {'Change':>12}")
    print("-" * 65)

    comparison = {}
    for key, label in metrics_map:
        b = np.mean([r[key] for r in baseline])
        r = np.mean([r[key] for r in rl_results])
        if b > 0:
            pct = ((r - b) / b) * 100
            print(f"{label:<25} {b:>12.1f} {r:>12.1f} {pct:>+11.1f}%")
        else:
            print(f"{label:<25} {b:>12.1f} {r:>12.1f} {'N/A':>12}")
        comparison[label] = {'baseline': float(b), 'rl': float(r)}

    # Save
    output_dir = Path('results/habsiguda_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'finetuned_evaluation.json', 'w') as f:
        json.dump({
            'model': 'Fine-tuned multihead DQN (100 episodes on Habsiguda corridor)',
            'traffic': '2000 vehicles/hr (realistic peak)',
            'env_settings': {'min_green_time': 20, 'phase_change_penalty': -1.0},
            'comparison': comparison,
            'baseline_episodes': [{k: float(v) for k, v in r.items()} for r in baseline],
            'rl_episodes': [{k: float(v) for k, v in r.items()} for r in rl_results],
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'finetuned_evaluation.json'}")


if __name__ == "__main__":
    main()
