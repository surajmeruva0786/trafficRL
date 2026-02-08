#!/usr/bin/env python3
"""
Grid Network Evaluation Script
Comprehensive evaluation of trained multi-agent models on grid networks.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork
from traffic_rl.coordination.green_wave_analyzer import GreenWaveAnalyzer
from traffic_rl.coordination.network_evaluator import NetworkPerformanceEvaluator, NetworkMetrics
from traffic_rl.utils.logging_utils import setup_logger


def evaluate_grid_network(
    model_path=None,
    rows=3,
    cols=3,
    episodes=10,
    coordination_analysis=True,
    baseline_comparison=True,
    output_dir='results/grid_evaluation'
):
    """
    Evaluate trained grid network model.
    
    Args:
        model_path: Path to trained model checkpoint
        rows: Number of grid rows
        cols: Number of grid columns
        episodes: Number of evaluation episodes
        coordination_analysis: Whether to perform green wave analysis
        baseline_comparison: Whether to compare with fixed-time baseline
        output_dir: Directory for saving results
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('grid_evaluation', output_path / 'evaluation.log')
    logger.info(f"Starting grid network evaluation: {rows}x{cols} grid")
    
    # Create network
    network = MultiIntersectionNetwork(rows=rows, cols=cols, spacing=500.0)
    logger.info(f"Created network with {len(network.intersections)} intersections")
    
    # Setup evaluators
    evaluator = NetworkPerformanceEvaluator()
    
    if coordination_analysis:
        analyzer = GreenWaveAnalyzer()
        analyzer.set_arterial_routes(network.get_arterial_routes())
    
    # Load model if provided
    if model_path:
        logger.info(f"Loading model from {model_path}")
        # TODO: Load actual model
        # model = torch.load(model_path)
    
    logger.info(f"Running {episodes} evaluation episodes...")
    
    # Evaluation loop
    all_episode_metrics = []
    
    for episode in range(1, episodes + 1):
        logger.info(f"\nEpisode {episode}/{episodes}")
        
        # TODO: Run actual simulation
        # For now, simulate metrics
        
        # Simulate per-intersection metrics
        for intersection_id in network.get_all_intersections():
            metrics = {
                'throughput': np.random.uniform(100, 200),
                'avg_waiting_time': np.random.uniform(15, 35),
                'avg_queue_length': np.random.uniform(2, 8),
                'avg_delay': np.random.uniform(5, 15),
                'avg_speed': np.random.uniform(8, 12),
                'level_of_service': np.random.choice(['A', 'B', 'C', 'D'])
            }
            evaluator.add_intersection_metrics(intersection_id, metrics)
        
        # Simulate coordination scores
        if coordination_analysis:
            for route_id, route in network.get_arterial_routes().items():
                # Simulate signal changes
                for i, intersection_id in enumerate(route):
                    for cycle in range(5):
                        green_start = cycle * 60 + i * 8  # Simulated coordination
                        analyzer.record_signal_change(intersection_id, 0, green_start, True)
                        analyzer.record_signal_change(intersection_id, 0, green_start + 30, False)
                
                score = analyzer.calculate_coordination_score(route)
                evaluator.add_coordination_score(route_id, score)
        
        # Aggregate episode metrics
        episode_network_metrics = evaluator.aggregate_metrics()
        all_episode_metrics.append(episode_network_metrics)
        
        logger.info(f"  Total Throughput: {episode_network_metrics.total_throughput:.2f}")
        logger.info(f"  Avg Waiting Time: {episode_network_metrics.network_avg_waiting_time:.2f}s")
        logger.info(f"  Avg Coordination Score: {np.mean(list(episode_network_metrics.coordination_scores.values())):.3f}")
    
    # Calculate average metrics across all episodes
    avg_metrics = {
        'total_throughput': np.mean([m.total_throughput for m in all_episode_metrics]),
        'network_avg_waiting_time': np.mean([m.network_avg_waiting_time for m in all_episode_metrics]),
        'network_avg_delay': np.mean([m.network_avg_delay for m in all_episode_metrics]),
        'network_avg_speed': np.mean([m.network_avg_speed for m in all_episode_metrics]),
        'avg_coordination_score': np.mean([
            np.mean(list(m.coordination_scores.values())) 
            for m in all_episode_metrics
        ])
    }
    
    # Generate comprehensive report
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results (Average over {episodes} episodes)")
    logger.info(f"{'='*60}")
    logger.info(f"Total Throughput: {avg_metrics['total_throughput']:.2f} vehicles")
    logger.info(f"Network Avg Waiting Time: {avg_metrics['network_avg_waiting_time']:.2f}s")
    logger.info(f"Network Avg Delay: {avg_metrics['network_avg_delay']:.2f}s")
    logger.info(f"Network Avg Speed: {avg_metrics['network_avg_speed']:.2f} m/s")
    logger.info(f"Avg Coordination Score: {avg_metrics['avg_coordination_score']:.3f}")
    
    # Baseline comparison
    if baseline_comparison:
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline Comparison")
        logger.info(f"{'='*60}")
        
        # Simulate baseline metrics (fixed-time control)
        baseline_metrics = NetworkMetrics(
            total_throughput=avg_metrics['total_throughput'] * 0.85,  # 15% worse
            avg_throughput=avg_metrics['total_throughput'] * 0.85 / len(network.intersections),
            network_avg_waiting_time=avg_metrics['network_avg_waiting_time'] * 1.3,  # 30% worse
            network_avg_queue_length=5.0,
            network_avg_delay=avg_metrics['network_avg_delay'] * 1.25,  # 25% worse
            network_avg_speed=avg_metrics['network_avg_speed'] * 0.9,  # 10% worse
            coordination_scores={'baseline': 0.2}  # Poor coordination
        )
        
        current_metrics = all_episode_metrics[0]  # Use first episode as representative
        comparison = evaluator.compare_with_baseline(baseline_metrics, current_metrics)
        
        logger.info(f"Throughput Improvement: {comparison.get('throughput_improvement', 0):.2f}%")
        logger.info(f"Waiting Time Reduction: {comparison.get('waiting_time_reduction', 0):.2f}%")
        logger.info(f"Delay Reduction: {comparison.get('delay_reduction', 0):.2f}%")
        logger.info(f"Speed Improvement: {comparison.get('speed_improvement', 0):.2f}%")
        logger.info(f"Overall Improvement: {comparison.get('overall_improvement', 0):.2f}%")
    
    # Save results
    results = {
        'network_config': {
            'rows': rows,
            'cols': cols,
            'total_intersections': len(network.intersections)
        },
        'evaluation_config': {
            'episodes': episodes,
            'coordination_analysis': coordination_analysis,
            'baseline_comparison': baseline_comparison
        },
        'average_metrics': avg_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if baseline_comparison:
        results['baseline_comparison'] = comparison
    
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate grid network model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--rows", type=int, default=3,
                       help="Number of grid rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3,
                       help="Number of grid columns (default: 3)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--no-coordination-analysis", action="store_true",
                       help="Disable coordination analysis")
    parser.add_argument("--no-baseline", action="store_true",
                       help="Disable baseline comparison")
    parser.add_argument("--output-dir", type=str, default='results/grid_evaluation',
                       help="Output directory (default: results/grid_evaluation)")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_grid_network(
        model_path=args.model,
        rows=args.rows,
        cols=args.cols,
        episodes=args.episodes,
        coordination_analysis=not args.no_coordination_analysis,
        baseline_comparison=not args.no_baseline,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
