"""
Comparison script to visualize DQN vs Baseline performance.

This script loads results from both DQN and baseline runs
and creates comparison plots.
"""

import os
import sys
import yaml
import argparse
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_rl.utils.plotting import plot_comparison
from traffic_rl.utils.metrics import compare_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_metrics_from_csv(filepath: str) -> dict:
    """
    Load metrics from CSV file and compute averages.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Dictionary of average metrics
    """
    df = pd.read_csv(filepath)
    
    metrics = {
        'avg_waiting_time': df['avg_waiting_time'].mean(),
        'avg_queue_length': df['avg_queue_length'].mean(),
        'throughput': df['throughput'].mean(),
        'num_completed': df['num_completed'].mean() if 'num_completed' in df.columns else 0,
        'total_reward': df['total_reward'].mean() if 'total_reward' in df.columns else 0
    }
    
    return metrics


def compare_results(config: dict, args: argparse.Namespace):
    """
    Compare DQN and baseline results.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("=" * 80)
    print("Comparing DQN Agent vs Fixed-Time Baseline")
    print("=" * 80)
    
    # Load results
    baseline_csv = os.path.join(config['logging']['results_dir'], 'fixed_time_metrics.csv')
    dqn_csv = os.path.join(config['logging']['log_dir'], 'evaluation_log.csv')
    
    if not os.path.exists(baseline_csv):
        print(f"Error: Baseline results not found at {baseline_csv}")
        print("Please run: python scripts/run_fixed_time_baseline.py")
        return
    
    if not os.path.exists(dqn_csv):
        print(f"Error: DQN evaluation results not found at {dqn_csv}")
        print("Please run: python scripts/evaluate_dqn.py")
        return
    
    print(f"\nLoading baseline results from: {baseline_csv}")
    baseline_metrics = load_metrics_from_csv(baseline_csv)
    
    print(f"Loading DQN results from: {dqn_csv}")
    dqn_metrics = load_metrics_from_csv(dqn_csv)
    
    # Print metrics
    print("\n" + "-" * 80)
    print("BASELINE (Fixed-Time) METRICS:")
    print("-" * 80)
    for metric, value in baseline_metrics.items():
        print(f"{metric:25s}: {value:10.2f}")
    
    print("\n" + "-" * 80)
    print("DQN AGENT METRICS:")
    print("-" * 80)
    for metric, value in dqn_metrics.items():
        print(f"{metric:25s}: {value:10.2f}")
    
    # Compute improvements
    improvements = compare_metrics(baseline_metrics, dqn_metrics)
    
    print("\n" + "-" * 80)
    print("IMPROVEMENTS (DQN vs Baseline):")
    print("-" * 80)
    for metric, improvement in improvements.items():
        sign = "+" if improvement > 0 else ""
        print(f"{metric:35s}: {sign}{improvement:6.2f}%")
    
    # Create comparison plot
    output_dir = config['logging']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating comparison plot...")
    plot_comparison(
        baseline_metrics,
        dqn_metrics,
        save_path=os.path.join(output_dir, 'dqn_vs_baseline_comparison.png')
    )
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare DQN and baseline results')
    parser.add_argument('--config', type=str, default='traffic_rl/config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Compare results
    compare_results(config, args)


if __name__ == "__main__":
    main()
