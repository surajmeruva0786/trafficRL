#!/usr/bin/env python3
"""
Green Wave Analyzer
Analyzes signal coordination and green wave patterns along arterial routes.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class SignalChange:
    """Represents a signal phase change event."""
    intersection_id: str
    phase: int
    timestamp: float
    is_green: bool


@dataclass
class CoordinationMetrics:
    """Metrics for arterial coordination quality."""
    coordination_score: float  # 0-1, higher is better
    average_offset: float  # Average time offset between consecutive signals
    offset_variance: float  # Variance in offsets
    green_wave_efficiency: float  # Percentage of vehicles benefiting from coordination


class GreenWaveAnalyzer:
    """
    Analyzes signal coordination and green wave patterns.
    
    Tracks signal timing across intersections and calculates coordination metrics
    to assess green wave quality along arterial routes.
    """
    
    def __init__(self):
        """Initialize the green wave analyzer."""
        self.signal_history: Dict[str, List[SignalChange]] = defaultdict(list)
        self.arterial_routes: Dict[str, List[str]] = {}
        
    def set_arterial_routes(self, routes: Dict[str, List[str]]):
        """
        Set the arterial routes to analyze.
        
        Args:
            routes: Dictionary mapping route ID to list of intersection IDs
        """
        self.arterial_routes = routes
    
    def record_signal_change(self, intersection_id: str, phase: int, 
                            timestamp: float, is_green: bool):
        """
        Record a signal phase change.
        
        Args:
            intersection_id: ID of the intersection
            phase: Signal phase number
            timestamp: Time of the change in seconds
            is_green: Whether this phase is green (True) or red (False)
        """
        change = SignalChange(
            intersection_id=intersection_id,
            phase=phase,
            timestamp=timestamp,
            is_green=is_green
        )
        self.signal_history[intersection_id].append(change)
    
    def calculate_offsets(self, arterial_route: List[str]) -> List[float]:
        """
        Calculate phase offsets between consecutive intersections.
        
        Args:
            arterial_route: List of intersection IDs along the arterial
            
        Returns:
            List of time offsets (in seconds) between consecutive intersections
        """
        offsets = []
        
        for i in range(len(arterial_route) - 1):
            current_id = arterial_route[i]
            next_id = arterial_route[i + 1]
            
            # Get green phase starts for both intersections
            current_greens = [s.timestamp for s in self.signal_history[current_id] 
                            if s.is_green]
            next_greens = [s.timestamp for s in self.signal_history[next_id] 
                          if s.is_green]
            
            if current_greens and next_greens:
                # Calculate average offset between green starts
                offset_samples = []
                for cg in current_greens[:10]:  # Sample first 10 cycles
                    # Find nearest next green
                    nearest = min(next_greens, key=lambda ng: abs(ng - cg))
                    offset_samples.append(nearest - cg)
                
                if offset_samples:
                    avg_offset = np.mean(offset_samples)
                    offsets.append(avg_offset)
        
        return offsets
    
    def calculate_coordination_score(self, arterial_route: List[str]) -> float:
        """
        Calculate coordination score for an arterial route.
        
        A score of 1.0 indicates perfect coordination, 0.0 indicates random signals.
        
        Args:
            arterial_route: List of intersection IDs along the arterial
            
        Returns:
            Coordination score (0-1)
        """
        if len(arterial_route) < 2:
            return 0.0
        
        offsets = self.calculate_offsets(arterial_route)
        
        if not offsets:
            return 0.0
        
        # Check offset consistency
        # Good coordination has consistent offsets
        offset_std = np.std(offsets)
        offset_mean = np.mean(np.abs(offsets))
        
        # Normalize: lower variance = better coordination
        # Ideal offset is positive and consistent
        if offset_mean == 0:
            return 0.0
        
        consistency_score = 1.0 / (1.0 + offset_std / max(offset_mean, 1.0))
        
        # Check if offsets are positive (green wave moving forward)
        positive_offsets = sum(1 for o in offsets if o > 0) / len(offsets)
        
        # Combined score
        coordination_score = consistency_score * positive_offsets
        
        return min(1.0, max(0.0, coordination_score))
    
    def analyze_arterial(self, arterial_route: List[str]) -> CoordinationMetrics:
        """
        Perform comprehensive coordination analysis for an arterial route.
        
        Args:
            arterial_route: List of intersection IDs along the arterial
            
        Returns:
            CoordinationMetrics object with detailed analysis
        """
        offsets = self.calculate_offsets(arterial_route)
        coordination_score = self.calculate_coordination_score(arterial_route)
        
        if offsets:
            avg_offset = np.mean(offsets)
            offset_variance = np.var(offsets)
        else:
            avg_offset = 0.0
            offset_variance = 0.0
        
        # Estimate green wave efficiency (simplified)
        # Higher coordination score = more vehicles benefit
        green_wave_efficiency = coordination_score * 100.0
        
        return CoordinationMetrics(
            coordination_score=coordination_score,
            average_offset=avg_offset,
            offset_variance=offset_variance,
            green_wave_efficiency=green_wave_efficiency
        )
    
    def get_time_space_data(self, arterial_route: List[str], 
                           start_time: float = 0.0, 
                           end_time: float = 3600.0) -> Dict:
        """
        Generate time-space diagram data for visualization.
        
        Args:
            arterial_route: List of intersection IDs along the arterial
            start_time: Start time for analysis
            end_time: End time for analysis
            
        Returns:
            Dictionary with time-space diagram data
        """
        data = {
            'intersections': arterial_route,
            'green_periods': defaultdict(list),
            'red_periods': defaultdict(list)
        }
        
        for intersection_id in arterial_route:
            history = self.signal_history[intersection_id]
            
            # Filter by time range
            relevant_changes = [s for s in history 
                              if start_time <= s.timestamp <= end_time]
            
            # Group into green and red periods
            current_state = None
            period_start = start_time
            
            for change in sorted(relevant_changes, key=lambda x: x.timestamp):
                if current_state is not None:
                    period = (period_start, change.timestamp)
                    if current_state:
                        data['green_periods'][intersection_id].append(period)
                    else:
                        data['red_periods'][intersection_id].append(period)
                
                current_state = change.is_green
                period_start = change.timestamp
            
            # Add final period
            if current_state is not None:
                period = (period_start, end_time)
                if current_state:
                    data['green_periods'][intersection_id].append(period)
                else:
                    data['red_periods'][intersection_id].append(period)
        
        return data
    
    def generate_report(self) -> Dict:
        """
        Generate a comprehensive coordination report for all arterial routes.
        
        Returns:
            Dictionary with coordination analysis for each route
        """
        report = {}
        
        for route_id, route in self.arterial_routes.items():
            metrics = self.analyze_arterial(route)
            report[route_id] = {
                'route': route,
                'coordination_score': metrics.coordination_score,
                'average_offset': metrics.average_offset,
                'offset_variance': metrics.offset_variance,
                'green_wave_efficiency': metrics.green_wave_efficiency,
                'num_intersections': len(route)
            }
        
        return report
    
    def reset(self):
        """Clear all recorded signal history."""
        self.signal_history.clear()


if __name__ == "__main__":
    # Example usage
    analyzer = GreenWaveAnalyzer()
    
    # Set up arterial routes
    routes = {
        'H0': ['I_0_0', 'I_0_1', 'I_0_2'],
        'V0': ['I_0_0', 'I_1_0', 'I_2_0']
    }
    analyzer.set_arterial_routes(routes)
    
    # Simulate coordinated signals on H0
    for i, intersection in enumerate(routes['H0']):
        for cycle in range(5):
            # Green starts with progressive offset
            green_start = cycle * 60 + i * 10  # 10 second offset
            analyzer.record_signal_change(intersection, 0, green_start, True)
            analyzer.record_signal_change(intersection, 0, green_start + 30, False)
    
    # Analyze
    print("Coordination Analysis:")
    report = analyzer.generate_report()
    for route_id, metrics in report.items():
        print(f"\n{route_id}:")
        print(f"  Coordination Score: {metrics['coordination_score']:.3f}")
        print(f"  Average Offset: {metrics['average_offset']:.2f}s")
        print(f"  Green Wave Efficiency: {metrics['green_wave_efficiency']:.1f}%")
