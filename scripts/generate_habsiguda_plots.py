#!/usr/bin/env python3
"""Generate evaluation and training plots for Habsiguda-Nacharam corridor."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = Path('results/habsiguda_evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 1. COMPARISON BAR CHART: Fine-tuned RL vs Fixed-Time Baseline
# ================================================================
eval_file = OUTPUT_DIR / 'finetuned_evaluation.json'
with open(eval_file) as f:
    data = json.load(f)

baseline_eps = data['baseline_episodes']
rl_eps = data['rl_episodes']

metrics = {
    'Throughput\n(veh/hr)': ('throughput', False),
    'Avg Wait\nTime (s)': ('avg_waiting_time', True),
    'Avg Queue\nLength': ('avg_queue_length', True),
    'Avg Speed\n(m/s)': ('avg_speed', False),
    'Phase\nChanges': ('phase_changes', True),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Habsiguda-Nacharam Corridor: Fine-tuned RL vs Fixed-Time Baseline\n'
             '(2000 vehicles/hr, realistic peak traffic)', fontsize=14, fontweight='bold', y=1.02)

for ax, (label, (key, lower_better)) in zip(axes, metrics.items()):
    b_vals = [ep[key] for ep in baseline_eps]
    r_vals = [ep[key] for ep in rl_eps]
    b_mean = np.mean(b_vals)
    r_mean = np.mean(r_vals)
    b_std = np.std(b_vals)
    r_std = np.std(r_vals)

    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(['Fixed-Time', 'RL Agent'], [b_mean, r_mean],
                  yerr=[b_std, r_std], capsize=5,
                  color=colors, edgecolor='#333', linewidth=0.8, alpha=0.9)

    # Percentage change annotation
    if b_mean > 0:
        pct = ((r_mean - b_mean) / b_mean) * 100
        better = (pct < 0) if lower_better else (pct > 0)
        color = '#27ae60' if better else '#c0392b'
        ax.annotate(f'{pct:+.1f}%', xy=(1, r_mean), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontweight='bold',
                    fontsize=11, color=color)

    # Value labels
    for bar, val in zip(bars, [b_mean, r_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + b_std + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparison_finetuned.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'comparison_finetuned.png'}")
plt.close()


# ================================================================
# 2. TRAINING CURVE (from training log)
# ================================================================
training_log = Path('results/habsiguda_training/training.log')
if training_log.exists():
    episodes = []
    rewards = []
    throughputs = []
    phase_changes = []

    with open(training_log, 'r', errors='replace') as f:
        lines = f.readlines()

    current_ep = None
    for line in lines:
        if 'Episode ' in line and '/' in line:
            try:
                part = line.split('Episode ')[1].split('/')[0].strip()
                current_ep = int(part)
            except:
                pass
        elif 'Total Reward:' in line and current_ep is not None:
            try:
                val = float(line.split('Total Reward:')[1].strip())
                rewards.append(val)
                episodes.append(current_ep)
            except:
                pass
        elif 'Throughput:' in line and current_ep is not None:
            try:
                val = float(line.split('Throughput:')[1].split('veh')[0].strip())
                throughputs.append(val)
            except:
                pass
        elif 'Total Phase Changes:' in line and current_ep is not None:
            try:
                val = int(line.split('Total Phase Changes:')[1].strip())
                phase_changes.append(val)
            except:
                pass

    if episodes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Habsiguda-Nacharam Training Progress (100 Episodes)',
                     fontsize=14, fontweight='bold')

        # Reward curve
        axes[0].plot(episodes, rewards, color='#3498db', linewidth=1.2, alpha=0.6)
        if len(rewards) > 5:
            window = min(10, len(rewards)//3)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(episodes[window-1:], smoothed, color='#e74c3c',
                        linewidth=2.5, label=f'{window}-ep moving avg')
            axes[0].legend()
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Episode Reward', fontweight='bold')
        axes[0].grid(alpha=0.3)

        # Throughput curve
        if throughputs:
            axes[1].plot(episodes[:len(throughputs)], throughputs,
                        color='#2ecc71', linewidth=1.2, alpha=0.6)
            if len(throughputs) > 5:
                window = min(10, len(throughputs)//3)
                smoothed = np.convolve(throughputs, np.ones(window)/window, mode='valid')
                axes[1].plot(episodes[window-1:len(throughputs)], smoothed,
                            color='#e74c3c', linewidth=2.5, label=f'{window}-ep moving avg')
                axes[1].legend()
            # Baseline reference line
            b_throughput = np.mean([ep['throughput'] for ep in baseline_eps])
            axes[1].axhline(y=b_throughput, color='#95a5a6', linestyle='--',
                           linewidth=1.5, label=f'Baseline ({b_throughput:.0f})')
            axes[1].legend()
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Throughput (veh/hr)')
            axes[1].set_title('Throughput per Episode', fontweight='bold')
            axes[1].grid(alpha=0.3)

        # Phase changes curve
        if phase_changes:
            axes[2].plot(episodes[:len(phase_changes)], phase_changes,
                        color='#9b59b6', linewidth=1.2, alpha=0.6)
            if len(phase_changes) > 5:
                window = min(10, len(phase_changes)//3)
                smoothed = np.convolve(phase_changes, np.ones(window)/window, mode='valid')
                axes[2].plot(episodes[window-1:len(phase_changes)], smoothed,
                            color='#e74c3c', linewidth=2.5, label=f'{window}-ep moving avg')
                axes[2].legend()
            axes[2].axhline(y=450, color='#95a5a6', linestyle='--',
                           linewidth=1.5, label='Baseline (450)')
            axes[2].legend()
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Phase Changes')
            axes[2].set_title('Phase Changes per Episode', fontweight='bold')
            axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {OUTPUT_DIR / 'training_curves.png'}")
        plt.close()


# ================================================================
# 3. PER-EPISODE COMPARISON (Baseline vs RL side by side)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Per-Episode Comparison: Fixed-Time vs Fine-tuned RL',
             fontsize=14, fontweight='bold')

ep_nums = list(range(1, len(baseline_eps) + 1))

# Throughput per episode
axes[0].bar([x - 0.2 for x in ep_nums],
            [ep['throughput'] for ep in baseline_eps], width=0.35,
            color='#e74c3c', label='Fixed-Time', alpha=0.85)
axes[0].bar([x + 0.2 for x in ep_nums],
            [ep['throughput'] for ep in rl_eps], width=0.35,
            color='#2ecc71', label='RL Agent', alpha=0.85)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Throughput (veh/hr)')
axes[0].set_title('Throughput', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Wait time per episode
axes[1].bar([x - 0.2 for x in ep_nums],
            [ep['avg_waiting_time'] for ep in baseline_eps], width=0.35,
            color='#e74c3c', label='Fixed-Time', alpha=0.85)
axes[1].bar([x + 0.2 for x in ep_nums],
            [ep['avg_waiting_time'] for ep in rl_eps], width=0.35,
            color='#2ecc71', label='RL Agent', alpha=0.85)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Avg Waiting Time (s)')
axes[1].set_title('Waiting Time', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Queue length per episode
axes[2].bar([x - 0.2 for x in ep_nums],
            [ep['avg_queue_length'] for ep in baseline_eps], width=0.35,
            color='#e74c3c', label='Fixed-Time', alpha=0.85)
axes[2].bar([x + 0.2 for x in ep_nums],
            [ep['avg_queue_length'] for ep in rl_eps], width=0.35,
            color='#2ecc71', label='RL Agent', alpha=0.85)
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Avg Queue Length')
axes[2].set_title('Queue Length', fontweight='bold')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'per_episode_comparison.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'per_episode_comparison.png'}")
plt.close()

print("\nAll plots generated successfully!")
