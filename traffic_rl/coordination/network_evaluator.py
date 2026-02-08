#!/usr/bin/env python3
"""
Network Performance Evaluator
Comprehensive evaluation of multi-intersection network performance.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class IntersectionMetrics:
    """Metrics for a single intersection."""
    intersection_id: str
    throughput: float
    avg_waiting_time: float
    avg_queue_length: float
    avg_delay: float
    avg_speed: float
    level_of_service: str


@dataclass
class NetworkMetrics:
    """Aggregated metrics for the entire network."""
    total_throughput: float
    avg_throughput: float
    network_avg_waiting_time: float
    network_avg_queue_length: float
    network_avg_delay: float
    network_avg_speed: float
    coordination_scores: Dict[str, float] = field(default_factory=dict)
    intersection_metrics: Dict[str, IntersectionMetrics] = field(default_factory=dict)


class NetworkPerformanceEvaluator:
    """
    Evaluates performance of multi-intersection traffic networks.
    
    Aggregates metrics across all intersections and calculates network-wide
    performance indicators including coordination quality.
    """
    
    def __init__(self):
        """Initialize the network performance evaluator."""
        self.intersection_data: Dict[str, Dict] = {}
        self.coordination_scores: Dict[str, float] = {}
    
    def add_intersection_metrics(self, intersection_id: str, metrics: Dict):
        """
        Add metrics for a single intersection.
        
        Args:
            intersection_id: ID of the intersection
            metrics: Dictionary with metric values
        """
        self.intersection_data[intersection_id] = metrics
    
    def add_coordination_score(self, route_id: str, score: float):
        """
        Add coordination score for an arterial route.
        
        Args:
            route_id: ID of the arterial route
            score: Coordination score (0-1)
        """
        self.coordination_scores[route_id] = score
    
    def aggregate_metrics(self) -> NetworkMetrics:
        """
        Aggregate metrics across all intersections.
        
        Returns:
            NetworkMetrics object with aggregated data
        """
        if not self.intersection_data:
            raise ValueError("No intersection data available")
        
        # Extract metrics from all intersections
        throughputs = []
        waiting_times = []
        queue_lengths = []
        delays = []
        speeds = []
        intersection_metrics = {}
        
        for intersection_id, data in self.intersection_data.items():
            throughputs.append(data.get('throughput', 0.0))
            waiting_times.append(data.get('avg_waiting_time', 0.0))
            queue_lengths.append(data.get('avg_queue_length', 0.0))
            delays.append(data.get('avg_delay', 0.0))
            speeds.append(data.get('avg_speed', 0.0))
            
            # Create IntersectionMetrics object
            intersection_metrics[intersection_id] = IntersectionMetrics(
                intersection_id=intersection_id,
                throughput=data.get('throughput', 0.0),
                avg_waiting_time=data.get('avg_waiting_time', 0.0),
                avg_queue_length=data.get('avg_queue_length', 0.0),
                avg_delay=data.get('avg_delay', 0.0),
                avg_speed=data.get('avg_speed', 0.0),
                level_of_service=data.get('level_of_service', 'F')
            )
        
        # Calculate aggregates
        network_metrics = NetworkMetrics(
            total_throughput=sum(throughputs),
            avg_throughput=np.mean(throughputs),
            network_avg_waiting_time=np.mean(waiting_times),
            network_avg_queue_length=np.mean(queue_lengths),
            network_avg_delay=np.mean(delays),
            network_avg_speed=np.mean(speeds),
            coordination_scores=self.coordination_scores.copy(),
            intersection_metrics=intersection_metrics
        )
        
        return network_metrics
    
    def compare_with_baseline(self, baseline_metrics: NetworkMetrics,
                             current_metrics: NetworkMetrics) -> Dict:
        """
        Compare current metrics with baseline.
        
        Args:
            baseline_metrics: Baseline network metrics
            current_metrics: Current network metrics
            
        Returns:
            Dictionary with comparison results and improvements
        """
        improvements = {}
        
        # Calculate percentage improvements
        if baseline_metrics.total_throughput > 0:
            improvements['throughput_improvement'] = (
                (current_metrics.total_throughput - baseline_metrics.total_throughput) /
                baseline_metrics.total_throughput * 100
            )
        
        if baseline_metrics.network_avg_waiting_time > 0:
            improvements['waiting_time_reduction'] = (
                (baseline_metrics.network_avg_waiting_time - current_metrics.network_avg_waiting_time) /
                baseline_metrics.network_avg_waiting_time * 100
            )
        
        if baseline_metrics.network_avg_delay > 0:
            improvements['delay_reduction'] = (
                (baseline_metrics.network_avg_delay - current_metrics.network_avg_delay) /
                baseline_metrics.network_avg_delay * 100
            )
        
        if baseline_metrics.network_avg_speed > 0:
            improvements['speed_improvement'] = (
                (current_metrics.network_avg_speed - baseline_metrics.network_avg_speed) /
                baseline_metrics.network_avg_speed * 100
            )
        
        # Overall performance score (weighted combination)
        weights = {
            'throughput': 0.3,
            'waiting_time': 0.3,
            'delay': 0.2,
            'speed': 0.2
        }
        
        overall_score = (
            weights['throughput'] * improvements.get('throughput_improvement', 0) +
            weights['waiting_time'] * improvements.get('waiting_time_reduction', 0) +
            weights['delay'] * improvements.get('delay_reduction', 0) +
            weights['speed'] * improvements.get('speed_improvement', 0)
        )
        
        improvements['overall_improvement'] = overall_score
        
        return improvements
    
    def generate_network_report(self, network_metrics: NetworkMetrics,
                               baseline_metrics: Optional[NetworkMetrics] = None) -> Dict:
        """
        Generate comprehensive network performance report.
        
        Args:
            network_metrics: Current network metrics
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            Dictionary with complete report
        """
        report = {
            'network_summary': {
                'total_intersections': len(network_metrics.intersection_metrics),
                'total_throughput': network_metrics.total_throughput,
                'avg_throughput_per_intersection': network_metrics.avg_throughput,
                'network_avg_waiting_time': network_metrics.network_avg_waiting_time,
                'network_avg_queue_length': network_metrics.network_avg_queue_length,
                'network_avg_delay': network_metrics.network_avg_delay,
                'network_avg_speed': network_metrics.network_avg_speed
            },
            'coordination_analysis': {
                'arterial_routes': len(network_metrics.coordination_scores),
                'avg_coordination_score': (
                    np.mean(list(network_metrics.coordination_scores.values()))
                    if network_metrics.coordination_scores else 0.0
                ),
                'route_scores': network_metrics.coordination_scores
            },
            'intersection_details': {}
        }
        
        # Add per-intersection details
        for intersection_id, metrics in network_metrics.intersection_metrics.items():
            report['intersection_details'][intersection_id] = {
                'throughput': metrics.throughput,
                'avg_waiting_time': metrics.avg_waiting_time,
                'avg_queue_length': metrics.avg_queue_length,
                'avg_delay': metrics.avg_delay,
                'avg_speed': metrics.avg_speed,
                'level_of_service': metrics.level_of_service
            }
        
        # Add comparison if baseline provided
        if baseline_metrics:
            report['comparison'] = self.compare_with_baseline(
                baseline_metrics, network_metrics
            )
        
        return report
    
    def get_best_and_worst_intersections(self, network_metrics: NetworkMetrics,
                                        metric: str = 'throughput') -> Dict:
        """
        Identify best and worst performing intersections.
        
        Args:
            network_metrics: Network metrics
            metric: Metric to use for ranking ('throughput', 'avg_waiting_time', etc.)
            
        Returns:
            Dictionary with best and worst intersections
        """
        if not network_metrics.intersection_metrics:
            return {}
        
        # Get metric values
        metric_values = {
            i_id: getattr(metrics, metric)
            for i_id, metrics in network_metrics.intersection_metrics.items()
        }
        
        # Sort by metric (higher is better for throughput/speed, lower is better for others)
        reverse = metric in ['throughput', 'avg_speed']
        sorted_intersections = sorted(
            metric_values.items(),
            key=lambda x: x[1],
            reverse=reverse
        )
        
        return {
            'best': sorted_intersections[0] if sorted_intersections else None,
            'worst': sorted_intersections[-1] if sorted_intersections else None,
            'ranking': sorted_intersections
        }
    
    def reset(self):
        """Clear all stored data."""
        self.intersection_data.clear()
        self.coordination_scores.clear()


if __name__ == "__main__":
    # Example usage
    evaluator = NetworkPerformanceEvaluator()
    
    # Add metrics for 3x3 grid
    for row in range(3):
        for col in range(3):
            intersection_id = f"I_{row}_{col}"
            metrics = {
                'throughput': np.random.uniform(100, 200),
                'avg_waiting_time': np.random.uniform(10, 30),
                'avg_queue_length': np.random.uniform(2, 8),
                'avg_delay': np.random.uniform(5, 15),
                'avg_speed': np.random.uniform(8, 12),
                'level_of_service': np.random.choice(['A', 'B', 'C', 'D'])
            }
            evaluator.add_intersection_metrics(intersection_id, metrics)
    
    # Add coordination scores
    evaluator.add_coordination_score('H0', 0.75)
    evaluator.add_coordination_score('H1', 0.68)
    evaluator.add_coordination_score('V0', 0.82)
    
    # Generate report
    network_metrics = evaluator.aggregate_metrics()
    report = evaluator.generate_network_report(network_metrics)
    
    print("Network Performance Report:")
    print(f"\nNetwork Summary:")
    for key, value in report['network_summary'].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print(f"\nCoordination Analysis:")
    for key, value in report['coordination_analysis'].items():
        if key != 'route_scores':
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Find best/worst intersections
    best_worst = evaluator.get_best_and_worst_intersections(network_metrics, 'throughput')
    print(f"\nBest throughput: {best_worst['best']}")
    print(f"Worst throughput: {best_worst['worst']}")
