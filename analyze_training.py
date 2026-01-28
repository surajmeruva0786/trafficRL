"""
Analyze and visualize the multihead DQN training up to episode 100.
Generate presentation-ready graphs and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

# Set style for presentation
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create output directory
output_dir = Path("presentation_materials")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("MULTIHEAD DQN TRAINING ANALYSIS - EPISODE 100")
print("=" * 70)

# Load the most recent training data from results directories
results_dirs = sorted(Path("results").glob("multihead_dqn_202601*"))
if not results_dirs:
    print("❌ No training results found!")
    exit(1)

# Find the directory with training data
training_data_path = None
for results_dir in reversed(results_dirs):
    data_file = results_dir / "training_data.npz"
    if data_file.exists():
        training_data_path = data_file
        print(f"\n✓ Found training data: {results_dir.name}")
        break

if training_data_path is None:
    print("❌ No training_data.npz found in results directories!")
    exit(1)

# Load training data
data = np.load(training_data_path)
episode_rewards = data['episode_rewards']
episode_waiting_times = data['episode_waiting_times']
episode_queue_lengths = data['episode_queue_lengths']
episode_throughputs = data['episode_throughputs']
episode_q_losses = data['episode_q_losses']
episode_classifier_losses = data['episode_classifier_losses']
episode_classifier_accuracies = data['episode_classifier_accuracies']

num_episodes = len(episode_rewards)
print(f"✓ Loaded {num_episodes} episodes of training data")

# Print summary statistics
print("\n" + "=" * 70)
print("TRAINING SUMMARY STATISTICS")
print("=" * 70)
print(f"Total Episodes: {num_episodes}")
print(f"\nRewards:")
print(f"  Mean: {np.mean(episode_rewards):,.2f}")
print(f"  Min: {np.min(episode_rewards):,.2f}")
print(f"  Max: {np.max(episode_rewards):,.2f}")
print(f"\nWaiting Times (seconds):")
print(f"  Mean: {np.mean(episode_waiting_times):.2f}s")
print(f"  Min: {np.min(episode_waiting_times):.2f}s")
print(f"  Max: {np.max(episode_waiting_times):.2f}s")
print(f"\nThroughput (veh/h):")
print(f"  Mean: {np.mean(episode_throughputs):.2f}")
print(f"  Min: {np.min(episode_throughputs):.2f}")
print(f"  Max: {np.max(episode_throughputs):.2f}")
print(f"\nClassifier Accuracy:")
print(f"  Mean: {np.mean(episode_classifier_accuracies):.3f}")
print(f"  Final: {episode_classifier_accuracies[-1]:.3f}")

# ============================================================================
# GRAPH 1: Training Progress Overview (4 subplots)
# ============================================================================
print("\n📊 Generating Graph 1: Training Progress Overview...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Multihead DQN Training Progress (Episodes 1-{})'.format(num_episodes), 
             fontsize=20, fontweight='bold')

# Plot 1: Episode Rewards
ax = axes[0, 0]
episodes = np.arange(1, num_episodes + 1)
ax.plot(episodes, episode_rewards, alpha=0.6, linewidth=1, label='Episode Reward')
if num_episodes >= 10:
    window = min(10, num_episodes // 10)
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Episode Rewards Over Training')
ax.legend()
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='plain', axis='y')

# Plot 2: Average Waiting Time
ax = axes[0, 1]
ax.plot(episodes, episode_waiting_times, alpha=0.6, linewidth=1, label='Avg Waiting Time')
if num_episodes >= 10:
    moving_avg = np.convolve(episode_waiting_times, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Waiting Time (seconds)')
ax.set_title('Average Waiting Time Per Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Throughput
ax = axes[1, 0]
ax.plot(episodes, episode_throughputs, alpha=0.6, linewidth=1, label='Throughput')
if num_episodes >= 10:
    moving_avg = np.convolve(episode_throughputs, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Throughput (vehicles/hour)')
ax.set_title('Traffic Throughput Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Classifier Accuracy
ax = axes[1, 1]
ax.plot(episodes, episode_classifier_accuracies, alpha=0.6, linewidth=1, label='Classifier Accuracy')
if num_episodes >= 10:
    moving_avg = np.convolve(episode_classifier_accuracies, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Accuracy')
ax.set_title('Regime Classifier Accuracy')
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_training_progress_overview.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '1_training_progress_overview.png'}")
plt.close()

# ============================================================================
# GRAPH 2: Loss Analysis
# ============================================================================
print("\n📊 Generating Graph 2: Loss Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Loss Analysis', fontsize=20, fontweight='bold')

# Q-Loss
ax = axes[0]
ax.plot(episodes, episode_q_losses, alpha=0.6, linewidth=1, label='Q-Loss')
if num_episodes >= 10:
    moving_avg = np.convolve(episode_q_losses, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Loss')
ax.set_title('Q-Learning Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Classifier Loss
ax = axes[1]
ax.plot(episodes, episode_classifier_losses, alpha=0.6, linewidth=1, label='Classifier Loss')
if num_episodes >= 10:
    moving_avg = np.convolve(episode_classifier_losses, np.ones(window)/window, mode='valid')
    ax.plot(episodes[window-1:], moving_avg, linewidth=2.5, label=f'{window}-Episode Moving Avg', color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Loss')
ax.set_title('Regime Classifier Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_loss_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '2_loss_analysis.png'}")
plt.close()

# ============================================================================
# GRAPH 3: Performance Metrics Comparison
# ============================================================================
print("\n📊 Generating Graph 3: Performance Metrics Comparison...")

# Compare first 10 vs last 10 episodes
first_10 = slice(0, min(10, num_episodes))
last_10 = slice(max(0, num_episodes - 10), num_episodes)

metrics = {
    'Avg Reward': [np.mean(episode_rewards[first_10]), np.mean(episode_rewards[last_10])],
    'Avg Waiting\nTime (s)': [np.mean(episode_waiting_times[first_10]), np.mean(episode_waiting_times[last_10])],
    'Avg Throughput\n(veh/h)': [np.mean(episode_throughputs[first_10]), np.mean(episode_throughputs[last_10])],
    'Classifier\nAccuracy': [np.mean(episode_classifier_accuracies[first_10]), np.mean(episode_classifier_accuracies[last_10])]
}

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label='First 10 Episodes', alpha=0.8)
bars2 = ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label='Last 10 Episodes', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Value', fontsize=14)
ax.set_title('Performance Comparison: First 10 vs Last 10 Episodes', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys())
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '3_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '3_performance_comparison.png'}")
plt.close()

# ============================================================================
# GRAPH 4: Queue Length Distribution
# ============================================================================
print("\n📊 Generating Graph 4: Queue Length Distribution...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(episode_queue_lengths, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(episode_queue_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_queue_lengths):.2f}')
ax.axvline(np.median(episode_queue_lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(episode_queue_lengths):.2f}')
ax.set_xlabel('Average Queue Length')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Average Queue Lengths Across Episodes', fontsize=18, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '4_queue_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '4_queue_distribution.png'}")
plt.close()

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n📝 Generating Summary Report...")

report = f"""
# Multihead DQN Training Analysis Report
## Episodes 1-{num_episodes}

### Training Overview
- **Total Episodes Completed**: {num_episodes}
- **Model Checkpoint**: multihead_dqn_ep{num_episodes}.pth

### Performance Metrics

#### Rewards
- **Mean Episode Reward**: {np.mean(episode_rewards):,.2f}
- **Best Episode Reward**: {np.max(episode_rewards):,.2f} (Episode {np.argmax(episode_rewards) + 1})
- **Worst Episode Reward**: {np.min(episode_rewards):,.2f} (Episode {np.argmin(episode_rewards) + 1})
- **Improvement (First 10 vs Last 10)**: {((np.mean(episode_rewards[last_10]) - np.mean(episode_rewards[first_10])) / abs(np.mean(episode_rewards[first_10])) * 100):.1f}%

#### Traffic Performance
- **Average Waiting Time**: {np.mean(episode_waiting_times):.2f} seconds
- **Average Queue Length**: {np.mean(episode_queue_lengths):.2f} vehicles
- **Average Throughput**: {np.mean(episode_throughputs):.2f} vehicles/hour
- **Throughput Improvement**: {((np.mean(episode_throughputs[last_10]) - np.mean(episode_throughputs[first_10]))):.2f} veh/h

#### Learning Metrics
- **Final Classifier Accuracy**: {episode_classifier_accuracies[-1]:.1%}
- **Mean Classifier Accuracy**: {np.mean(episode_classifier_accuracies):.1%}
- **Final Q-Loss**: {episode_q_losses[-1]:.4f}
- **Final Classifier Loss**: {episode_classifier_losses[-1]:.4f}

### Key Observations

1. **Reward Trend**: {"Improving" if episode_rewards[-1] > episode_rewards[0] else "Declining" if episode_rewards[-1] < episode_rewards[0] else "Stable"}
2. **Waiting Time**: {"Decreasing (Good)" if episode_waiting_times[-1] < episode_waiting_times[0] else "Increasing (Needs attention)"}
3. **Throughput**: {"Improving" if episode_throughputs[-1] > episode_throughputs[0] else "Declining"}
4. **Classifier Learning**: {"Strong" if episode_classifier_accuracies[-1] > 0.7 else "Moderate" if episode_classifier_accuracies[-1] > 0.5 else "Weak"}

### Generated Visualizations
1. `1_training_progress_overview.png` - Overall training metrics
2. `2_loss_analysis.png` - Q-learning and classifier losses
3. `3_performance_comparison.png` - First vs last episodes comparison
4. `4_queue_distribution.png` - Queue length distribution

---
*Generated automatically from training data*
"""

with open(output_dir / 'analysis_report.md', 'w') as f:
    f.write(report)

print(f"  ✓ Saved: {output_dir / 'analysis_report.md'}")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 All presentation materials saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  - 1_training_progress_overview.png")
print("  - 2_loss_analysis.png")
print("  - 3_performance_comparison.png")
print("  - 4_queue_distribution.png")
print("  - analysis_report.md")
print("\n" + "=" * 70)
