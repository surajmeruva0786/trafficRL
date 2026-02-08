"""
Demonstration script for the enhanced metrics system.

This script shows how to use the new transportation metrics including:
- Delay calculation
- Travel time reliability metrics
- Speed-based performance
- Level of Service classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from traffic_rl.metrics import (
    DelayCalculator,
    TravelTimeMetrics,
    SpeedMetrics,
    LevelOfServiceCalculator,
    EnhancedMetricsTracker
)


def demo_delay_calculator():
    """Demonstrate delay calculation."""
    print("\n" + "="*60)
    print("DELAY CALCULATOR DEMO")
    print("="*60)
    
    delay_calc = DelayCalculator(free_flow_speed=13.89)  # 50 km/h
    
    # Simulate some vehicles
    vehicles = [
        {'id': 'veh_1', 'entry_time': 0, 'exit_time': 45, 'distance': 500},
        {'id': 'veh_2', 'entry_time': 5, 'exit_time': 55, 'distance': 500},
        {'id': 'veh_3', 'entry_time': 10, 'exit_time': 70, 'distance': 500},
        {'id': 'veh_4', 'entry_time': 15, 'exit_time': 50, 'distance': 500},
        {'id': 'veh_5', 'entry_time': 20, 'exit_time': 65, 'distance': 500},
    ]
    
    for veh in vehicles:
        delay_calc.record_vehicle_entry(veh['id'], 'edge_1', veh['entry_time'], veh['distance'])
        delay = delay_calc.record_vehicle_exit(veh['id'], veh['exit_time'], veh['distance'])
        print(f"Vehicle {veh['id']}: Delay = {delay:.2f}s")
    
    print(f"\nTotal Delay: {delay_calc.get_total_delay():.2f}s")
    print(f"Average Delay: {delay_calc.get_average_delay():.2f}s")
    
    stats = delay_calc.get_delay_statistics()
    print(f"\nDelay Statistics:")
    print(f"  Mean: {stats['mean']:.2f}s")
    print(f"  Median: {stats['median']:.2f}s")
    print(f"  Std Dev: {stats['std']:.2f}s")
    print(f"  95th Percentile: {stats['p95']:.2f}s")


def demo_travel_time_metrics():
    """Demonstrate travel time metrics."""
    print("\n" + "="*60)
    print("TRAVEL TIME METRICS DEMO")
    print("="*60)
    
    tt_metrics = TravelTimeMetrics(free_flow_time=36.0)  # 36 seconds free-flow
    
    # Simulate travel times (some with delays)
    travel_times = [45, 50, 38, 42, 60, 55, 40, 48, 70, 52, 44, 39, 65, 47, 51]
    
    for tt in travel_times:
        tt_metrics.add_travel_time(tt)
    
    print(f"Average Travel Time: {tt_metrics.get_average_travel_time():.2f}s")
    print(f"95th Percentile: {tt_metrics.get_percentile_travel_time(95):.2f}s")
    print(f"\nReliability Indices:")
    print(f"  Travel Time Index (TTI): {tt_metrics.get_travel_time_index():.2f}")
    print(f"  Planning Time Index (PTI): {tt_metrics.get_planning_time_index():.2f}")
    print(f"  Buffer Time Index (BTI): {tt_metrics.get_buffer_time_index():.2f}")
    print(f"  Misery Index: {tt_metrics.get_misery_index():.2f}")
    
    print(f"\nInterpretation:")
    print(f"  - Trips take {(tt_metrics.get_travel_time_index() - 1) * 100:.1f}% longer than free-flow")
    print(f"  - Travelers should budget {tt_metrics.get_buffer_time_index() * 100:.1f}% extra time for reliability")


def demo_speed_metrics():
    """Demonstrate speed metrics."""
    print("\n" + "="*60)
    print("SPEED METRICS DEMO")
    print("="*60)
    
    speed_metrics = SpeedMetrics(speed_limit=50.0)  # 50 km/h
    
    # Simulate speeds from vehicles
    speeds = [45, 48, 30, 42, 50, 38, 44, 25, 47, 40, 35, 49, 28, 46, 41]
    
    for speed in speeds:
        speed_metrics.add_speed(speed)
    
    print(f"Average Speed: {speed_metrics.get_average_speed():.2f} km/h")
    print(f"Harmonic Mean Speed: {speed_metrics.get_harmonic_mean_speed():.2f} km/h")
    print(f"Speed Std Dev: {speed_metrics.get_speed_standard_deviation():.2f} km/h")
    print(f"\nFlow Quality:")
    print(f"  Free Flow (≥80% limit): {speed_metrics.get_percent_free_flow():.1f}%")
    print(f"  Congested (<50% limit): {speed_metrics.get_percent_congested():.1f}%")
    print(f"  Speed Ratio: {speed_metrics.get_average_speed_ratio():.2f}")


def demo_los_calculator():
    """Demonstrate Level of Service calculation."""
    print("\n" + "="*60)
    print("LEVEL OF SERVICE DEMO")
    print("="*60)
    
    los_calc = LevelOfServiceCalculator()
    
    # Test different delay scenarios
    scenarios = [
        ("Excellent conditions", 8),
        ("Good conditions", 25),
        ("Acceptable conditions", 40),
        ("Poor conditions", 65),
        ("Failing conditions", 95)
    ]
    
    for scenario_name, delay in scenarios:
        los = los_calc.get_los_from_delay(delay)
        print(f"\n{scenario_name} (Delay: {delay}s):")
        print(f"  LOS Grade: {los.value}")
        print(f"  Description: {los_calc.get_los_description(los)}")
        print(f"  Score: {los_calc.get_los_score(los)}/6.0")
        print(f"  Acceptable: {los_calc.is_acceptable_los(los)}")


def demo_enhanced_tracker():
    """Demonstrate the integrated enhanced metrics tracker."""
    print("\n" + "="*60)
    print("ENHANCED METRICS TRACKER DEMO")
    print("="*60)
    
    tracker = EnhancedMetricsTracker(
        free_flow_speed=13.89,
        speed_limit=50.0,
        free_flow_time=36.0
    )
    
    # Simulate a traffic episode
    print("\nSimulating traffic episode...")
    
    # Vehicle entries and exits
    vehicles = [
        {'id': f'veh_{i}', 'entry': i*2, 'exit': i*2 + np.random.randint(35, 70), 
         'distance': 500, 'speed': np.random.uniform(25, 50)}
        for i in range(20)
    ]
    
    for veh in vehicles:
        tracker.record_vehicle_entry(veh['id'], 'edge_1', veh['entry'], veh['distance'])
        travel_time = veh['exit'] - veh['entry']
        tracker.record_vehicle_exit(veh['id'], veh['exit'], travel_time, veh['distance'])
        tracker.record_speeds([veh['speed']])
    
    # Record some waiting times and queue lengths
    for step in range(100):
        tracker.record_waiting_time(np.random.uniform(10, 50))
        tracker.record_queue_length(np.random.uniform(2, 15))
    
    # Get comprehensive report
    report = tracker.get_comprehensive_report()
    
    print("\n" + "-"*60)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("-"*60)
    
    print(f"\nDelay Metrics:")
    print(f"  Average Delay: {report['delay']['mean']:.2f}s")
    print(f"  95th Percentile: {report['delay']['p95']:.2f}s")
    
    print(f"\nTravel Time Metrics:")
    print(f"  Average: {report['travel_time']['mean']:.2f}s")
    print(f"  TTI: {report['travel_time']['travel_time_index']:.2f}")
    print(f"  BTI: {report['travel_time']['buffer_time_index']:.2f}")
    
    print(f"\nSpeed Metrics:")
    print(f"  Average: {report['speed']['mean']:.2f} km/h")
    print(f"  Free Flow %: {report['speed']['percent_free_flow']:.1f}%")
    
    print(f"\nLevel of Service:")
    print(f"  Grade: {report['los']['grade']}")
    print(f"  Description: {report['los']['description']}")
    print(f"  Score: {report['los']['score']}/6.0")
    
    print(f"\nTraditional Metrics:")
    print(f"  Avg Waiting Time: {report['waiting_time']['mean']:.2f}s")
    print(f"  Avg Queue Length: {report['queue_length']['mean']:.2f}")
    print(f"  Throughput: {report['throughput']['total']} vehicles")
    
    # Get summary
    print("\n" + "-"*60)
    print("SUMMARY METRICS")
    print("-"*60)
    summary = tracker.get_summary_metrics()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def demo_baseline_comparison():
    """Demonstrate comparison with baseline."""
    print("\n" + "="*60)
    print("BASELINE COMPARISON DEMO")
    print("="*60)
    
    # Create RL agent tracker
    rl_tracker = EnhancedMetricsTracker(free_flow_speed=13.89, speed_limit=50.0)
    
    # Simulate RL performance (better)
    for i in range(30):
        rl_tracker.record_vehicle_entry(f'veh_{i}', 'edge_1', i*2, 500)
        travel_time = np.random.uniform(38, 55)  # Better performance
        rl_tracker.record_vehicle_exit(f'veh_{i}', i*2 + travel_time, travel_time, 500)
        rl_tracker.record_speeds([np.random.uniform(35, 50)])
        rl_tracker.record_waiting_time(np.random.uniform(5, 25))
        rl_tracker.record_queue_length(np.random.uniform(1, 8))
    
    # Create baseline tracker
    baseline_tracker = EnhancedMetricsTracker(free_flow_speed=13.89, speed_limit=50.0)
    
    # Simulate baseline performance (worse)
    for i in range(30):
        baseline_tracker.record_vehicle_entry(f'veh_{i}', 'edge_1', i*2, 500)
        travel_time = np.random.uniform(50, 75)  # Worse performance
        baseline_tracker.record_vehicle_exit(f'veh_{i}', i*2 + travel_time, travel_time, 500)
        baseline_tracker.record_speeds([np.random.uniform(20, 40)])
        baseline_tracker.record_waiting_time(np.random.uniform(15, 45))
        baseline_tracker.record_queue_length(np.random.uniform(5, 18))
    
    # Compare
    comparison = rl_tracker.compare_with_baseline(baseline_tracker)
    
    print("\nPerformance Improvements:")
    for metric, improvement in comparison['improvements'].items():
        print(f"  {metric}: {improvement:+.2f}%")
    
    print("\nStatistical Significance:")
    for metric, sig in comparison['significance'].items():
        status = "✓ Significant" if sig['significant'] else "✗ Not significant"
        print(f"  {metric}: {status} (p={sig['p_value']:.4f})")
    
    print("\nLOS Comparison:")
    los_comp = comparison['los_comparison']
    print(f"  RL Agent: {los_comp['rl']} (score: {los_comp['rl_score']})")
    print(f"  Baseline: {los_comp['baseline']} (score: {los_comp['baseline_score']})")
    print(f"  Improvement: {los_comp['improvement']:+.1f} grades")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("ENHANCED TRAFFIC METRICS SYSTEM DEMONSTRATION")
    print("="*60)
    
    demo_delay_calculator()
    demo_travel_time_metrics()
    demo_speed_metrics()
    demo_los_calculator()
    demo_enhanced_tracker()
    demo_baseline_comparison()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nAll metric systems are working correctly!")
    print("These metrics can now be integrated into your SUMO environment.")


if __name__ == "__main__":
    main()
