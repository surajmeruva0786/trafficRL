"""
Enhanced Metrics Tracker Module

Integrates all metric systems into a comprehensive tracking and reporting system.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from scipy import stats

from .delay_calculator import DelayCalculator
from .travel_time_metrics import TravelTimeMetrics
from .speed_metrics import SpeedMetrics
from .los_calculator import LevelOfServiceCalculator, LOSGrade


class EnhancedMetricsTracker:
    """
    Comprehensive metrics tracking system integrating all transportation metrics.
    
    Combines delay, travel time, speed, and LOS metrics with traditional
    RL metrics (waiting time, queue length, throughput) for complete evaluation.
    
    Attributes:
        delay_calc: DelayCalculator instance
        travel_time: TravelTimeMetrics instance
        speed: SpeedMetrics instance
        los_calc: LevelOfServiceCalculator instance
        waiting_times: List of waiting time measurements
        queue_lengths: List of queue length measurements
        throughput: Total number of vehicles that completed trips
        phase_changes: Number of signal phase changes
    """
    
    def __init__(self, free_flow_speed: float = 13.89, speed_limit: float = 50.0,
                 free_flow_time: Optional[float] = None):
        """
        Initialize enhanced metrics tracker.
        
        Args:
            free_flow_speed: Free-flow speed in m/s (default: 13.89 m/s = 50 km/h)
            speed_limit: Speed limit in km/h (default: 50 km/h)
            free_flow_time: Expected free-flow travel time in seconds (optional)
        """
        # New transportation metrics
        self.delay_calc = DelayCalculator(free_flow_speed=free_flow_speed)
        self.travel_time = TravelTimeMetrics(free_flow_time=free_flow_time)
        self.speed = SpeedMetrics(speed_limit=speed_limit)
        self.los_calc = LevelOfServiceCalculator()
        
        # Traditional RL metrics
        self.waiting_times: List[float] = []
        self.queue_lengths: List[float] = []
        self.throughput: int = 0
        self.phase_changes: int = 0
        
        # Episode tracking
        self.episode_rewards: List[float] = []
        self.current_episode_reward: float = 0.0
        
    def record_vehicle_entry(self, vehicle_id: str, edge_id: str, 
                            timestamp: float, route_length: float = None):
        """
        Record when a vehicle enters the network.
        
        Args:
            vehicle_id: Unique vehicle identifier
            edge_id: Edge/segment identifier
            timestamp: Current simulation time
            route_length: Total route length in meters (optional)
        """
        self.delay_calc.record_vehicle_entry(
            vehicle_id, edge_id, timestamp, route_length
        )
    
    def record_vehicle_exit(self, vehicle_id: str, timestamp: float,
                           travel_time: float, actual_distance: float = None):
        """
        Record when a vehicle exits the network.
        
        Args:
            vehicle_id: Unique vehicle identifier
            timestamp: Current simulation time
            travel_time: Total travel time in seconds
            actual_distance: Actual distance traveled in meters (optional)
        """
        # Record delay
        self.delay_calc.record_vehicle_exit(vehicle_id, timestamp, actual_distance)
        
        # Record travel time
        self.travel_time.add_travel_time(travel_time)
        
        # Increment throughput
        self.throughput += 1
    
    def record_speeds(self, speeds: List[float]):
        """
        Record speed measurements from vehicles in network.
        
        Args:
            speeds: List of speeds in km/h
        """
        self.speed.add_speeds(speeds)
    
    def record_waiting_time(self, waiting_time: float):
        """
        Record total waiting time at current step.
        
        Args:
            waiting_time: Total waiting time across all vehicles
        """
        self.waiting_times.append(waiting_time)
    
    def record_queue_length(self, queue_length: float):
        """
        Record total queue length at current step.
        
        Args:
            queue_length: Total queue length across all lanes
        """
        self.queue_lengths.append(queue_length)
    
    def record_phase_change(self):
        """Record that a phase change occurred."""
        self.phase_changes += 1
    
    def record_reward(self, reward: float):
        """
        Record reward for current step.
        
        Args:
            reward: Reward value
        """
        self.current_episode_reward += reward
    
    def end_episode(self):
        """Mark end of episode and record cumulative reward."""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with all metrics.
        
        Returns:
            Dictionary containing all computed metrics organized by category
        """
        # Calculate average delay for LOS
        avg_delay = self.delay_calc.get_average_delay()
        los_grade = self.los_calc.get_los_from_delay(avg_delay)
        
        report = {
            # Delay metrics
            'delay': {
                **self.delay_calc.get_delay_statistics(),
                'total': self.delay_calc.get_total_delay()
            },
            
            # Travel time metrics
            'travel_time': self.travel_time.get_comprehensive_statistics(),
            
            # Speed metrics
            'speed': self.speed.get_comprehensive_statistics(),
            
            # Level of Service
            'los': {
                'grade': los_grade.value,
                'description': self.los_calc.get_los_description(los_grade),
                'score': self.los_calc.get_los_score(los_grade),
                'acceptable': self.los_calc.is_acceptable_los(los_grade),
                'characteristics': self.los_calc.get_los_characteristics(los_grade)
            },
            
            # Traditional RL metrics
            'waiting_time': {
                'mean': np.mean(self.waiting_times) if self.waiting_times else 0.0,
                'median': np.median(self.waiting_times) if self.waiting_times else 0.0,
                'std': np.std(self.waiting_times) if self.waiting_times else 0.0,
                'total': sum(self.waiting_times),
                'max': max(self.waiting_times) if self.waiting_times else 0.0
            },
            
            'queue_length': {
                'mean': np.mean(self.queue_lengths) if self.queue_lengths else 0.0,
                'median': np.median(self.queue_lengths) if self.queue_lengths else 0.0,
                'std': np.std(self.queue_lengths) if self.queue_lengths else 0.0,
                'max': max(self.queue_lengths) if self.queue_lengths else 0.0
            },
            
            'throughput': {
                'total': self.throughput,
                'vehicles_per_hour': self.throughput * 3600 / max(len(self.waiting_times), 1)
            },
            
            'phase_changes': self.phase_changes,
            
            # Episode performance
            'episode': {
                'total_reward': sum(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                'episodes_completed': len(self.episode_rewards)
            }
        }
        
        return report
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get key summary metrics for quick evaluation.
        
        Returns:
            Dictionary with essential metrics
        """
        avg_delay = self.delay_calc.get_average_delay()
        los_grade = self.los_calc.get_los_from_delay(avg_delay)
        
        return {
            'average_delay': avg_delay,
            'average_travel_time': self.travel_time.get_average_travel_time(),
            'p95_travel_time': self.travel_time.get_percentile_travel_time(95),
            'average_speed': self.speed.get_average_speed(),
            'average_waiting_time': np.mean(self.waiting_times) if self.waiting_times else 0.0,
            'average_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0.0,
            'throughput': self.throughput,
            'los_grade': los_grade.value,
            'los_score': self.los_calc.get_los_score(los_grade)
        }
    
    def compare_with_baseline(self, baseline_metrics: 'EnhancedMetricsTracker',
                             alpha: float = 0.05) -> Dict[str, Any]:
        """
        Statistical comparison with baseline metrics.
        
        Args:
            baseline_metrics: Baseline EnhancedMetricsTracker instance
            alpha: Significance level for statistical tests (default: 0.05)
            
        Returns:
            Dictionary with improvement percentages and statistical significance
        """
        improvements = {}
        significance = {}
        
        # Compare delays
        if self.delay_calc.vehicle_exits and baseline_metrics.delay_calc.vehicle_exits:
            rl_delays = [v['delay'] for v in self.delay_calc.vehicle_exits.values()]
            baseline_delays = [v['delay'] for v in baseline_metrics.delay_calc.vehicle_exits.values()]
            
            rl_mean = np.mean(rl_delays)
            baseline_mean = np.mean(baseline_delays)
            improvements['delay'] = ((baseline_mean - rl_mean) / baseline_mean) * 100
            
            _, p_value = stats.mannwhitneyu(rl_delays, baseline_delays, alternative='less')
            significance['delay'] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'test': 'Mann-Whitney U'
            }
        
        # Compare waiting times
        if self.waiting_times and baseline_metrics.waiting_times:
            rl_mean = np.mean(self.waiting_times)
            baseline_mean = np.mean(baseline_metrics.waiting_times)
            improvements['waiting_time'] = ((baseline_mean - rl_mean) / baseline_mean) * 100
            
            _, p_value = stats.mannwhitneyu(
                self.waiting_times, baseline_metrics.waiting_times, alternative='less'
            )
            significance['waiting_time'] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'test': 'Mann-Whitney U'
            }
        
        # Compare queue lengths
        if self.queue_lengths and baseline_metrics.queue_lengths:
            rl_mean = np.mean(self.queue_lengths)
            baseline_mean = np.mean(baseline_metrics.queue_lengths)
            improvements['queue_length'] = ((baseline_mean - rl_mean) / baseline_mean) * 100
            
            _, p_value = stats.mannwhitneyu(
                self.queue_lengths, baseline_metrics.queue_lengths, alternative='less'
            )
            significance['queue_length'] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'test': 'Mann-Whitney U'
            }
        
        # Compare throughput
        if self.throughput > 0 and baseline_metrics.throughput > 0:
            improvements['throughput'] = (
                (self.throughput - baseline_metrics.throughput) / baseline_metrics.throughput
            ) * 100
        
        # Compare speeds
        if self.speed.speeds and baseline_metrics.speed.speeds:
            rl_mean = np.mean(self.speed.speeds)
            baseline_mean = np.mean(baseline_metrics.speed.speeds)
            improvements['speed'] = ((rl_mean - baseline_mean) / baseline_mean) * 100
            
            _, p_value = stats.mannwhitneyu(
                self.speed.speeds, baseline_metrics.speed.speeds, alternative='greater'
            )
            significance['speed'] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'test': 'Mann-Whitney U'
            }
        
        # LOS comparison
        rl_los = self.los_calc.get_los_from_delay(self.delay_calc.get_average_delay())
        baseline_los = baseline_metrics.los_calc.get_los_from_delay(
            baseline_metrics.delay_calc.get_average_delay()
        )
        
        return {
            'improvements': improvements,
            'significance': significance,
            'los_comparison': {
                'rl': rl_los.value,
                'baseline': baseline_los.value,
                'rl_score': self.los_calc.get_los_score(rl_los),
                'baseline_score': baseline_metrics.los_calc.get_los_score(baseline_los),
                'improvement': (
                    self.los_calc.get_los_score(rl_los) - 
                    baseline_metrics.los_calc.get_los_score(baseline_los)
                )
            }
        }
    
    def reset(self):
        """Reset all metrics for new episode."""
        self.delay_calc.reset()
        self.travel_time.reset()
        self.speed.reset()
        self.waiting_times.clear()
        self.queue_lengths.clear()
        self.throughput = 0
        self.phase_changes = 0
        self.current_episode_reward = 0.0
    
    def __repr__(self) -> str:
        return (f"EnhancedMetricsTracker(vehicles={self.throughput}, "
                f"avg_delay={self.delay_calc.get_average_delay():.2f}s, "
                f"los={self.los_calc.get_los_from_delay(self.delay_calc.get_average_delay()).value})")
