
# Multihead DQN Training Analysis Report
## Episodes 1-5

### Training Overview
- **Total Episodes Completed**: 5
- **Model Checkpoint**: multihead_dqn_ep5.pth

### Performance Metrics

#### Rewards
- **Mean Episode Reward**: -25,752.82
- **Best Episode Reward**: -24,157.06 (Episode 1)
- **Worst Episode Reward**: -27,781.74 (Episode 2)
- **Improvement (First 10 vs Last 10)**: 0.0%

#### Traffic Performance
- **Average Waiting Time**: 645.49 seconds
- **Average Queue Length**: 54.20 vehicles
- **Average Throughput**: 0.40 vehicles/hour
- **Throughput Improvement**: 0.00 veh/h

#### Learning Metrics
- **Final Classifier Accuracy**: 81.4%
- **Mean Classifier Accuracy**: 82.3%
- **Final Q-Loss**: 4.7997
- **Final Classifier Loss**: 0.1632

### Key Observations

1. **Reward Trend**: Declining
2. **Waiting Time**: Increasing (Needs attention)
3. **Throughput**: Declining
4. **Classifier Learning**: Strong

### Generated Visualizations
1. `1_training_progress_overview.png` - Overall training metrics
2. `2_loss_analysis.png` - Q-learning and classifier losses
3. `3_performance_comparison.png` - First vs last episodes comparison
4. `4_queue_distribution.png` - Queue length distribution

---
*Generated automatically from training data*
