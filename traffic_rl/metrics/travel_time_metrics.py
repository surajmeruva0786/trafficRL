"""
Travel Time Metrics Module

Provides comprehensive travel time analysis including reliability metrics.
Travel time is a key performance indicator that directly impacts user experience.
"""

import numpy as np
from typing import List, Dict, Optional


class TravelTimeMetrics:
    """
    Comprehensive travel time analysis beyond simple averages.
    
    Includes reliability metrics that are crucial for understanding
    the consistency and predictability of travel times, which are
    often more important to users than average travel time alone.
    
    Attributes:
        travel_times: List of all recorded travel times
        free_flow_time: Expected travel time under free-flow conditions
    """
    
    def __init__(self, free_flow_time: Optional[float] = None):
        """
        Initialize travel time metrics tracker.
        
        Args:
            free_flow_time: Expected free-flow travel time in seconds (optional)
        """
        self.travel_times: List[float] = []
        self.free_flow_time = free_flow_time
        
    def add_travel_time(self, travel_time: float):
        """
        Record a vehicle's travel time.
        
        Args:
            travel_time: Travel time in seconds
        """
        self.travel_times.append(travel_time)
    
    def get_average_travel_time(self) -> float:
        """
        Get average travel time.
        
        Returns:
            Average travel time in seconds, or 0 if no data
        """
        return np.mean(self.travel_times) if self.travel_times else 0.0
    
    def get_median_travel_time(self) -> float:
        """
        Get median travel time (50th percentile).
        
        Returns:
            Median travel time in seconds, or 0 if no data
        """
        return np.median(self.travel_times) if self.travel_times else 0.0
    
    def get_percentile_travel_time(self, percentile: float = 95) -> float:
        """
        Get percentile travel time.
        
        95th percentile travel time is particularly important as it represents
        the travel time that 95% of trips will be better than, providing a
        measure of worst-case performance.
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile travel time in seconds, or 0 if no data
        """
        return np.percentile(self.travel_times, percentile) if self.travel_times else 0.0
    
    def get_travel_time_index(self, free_flow_time: Optional[float] = None) -> float:
        """
        Calculate Travel Time Index (TTI).
        
        TTI = Average Travel Time / Free Flow Time
        
        A TTI of 1.0 means travel at free-flow speed.
        A TTI of 1.3 means trips take 30% longer than free-flow.
        
        Args:
            free_flow_time: Free-flow travel time in seconds (uses stored value if not provided)
            
        Returns:
            Travel Time Index, or 0 if insufficient data
        """
        fft = free_flow_time or self.free_flow_time
        if not fft or not self.travel_times:
            return 0.0
        
        avg_travel_time = self.get_average_travel_time()
        return avg_travel_time / fft
    
    def get_planning_time_index(self, free_flow_time: Optional[float] = None) -> float:
        """
        Calculate Planning Time Index (PTI).
        
        PTI = 95th Percentile Travel Time / Free Flow Time
        
        Represents the total time needed to ensure on-time arrival 95% of the time.
        More important than average for trip planning.
        
        Args:
            free_flow_time: Free-flow travel time in seconds (uses stored value if not provided)
            
        Returns:
            Planning Time Index, or 0 if insufficient data
        """
        fft = free_flow_time or self.free_flow_time
        if not fft or not self.travel_times:
            return 0.0
        
        p95_travel_time = self.get_percentile_travel_time(95)
        return p95_travel_time / fft
    
    def get_buffer_time_index(self) -> float:
        """
        Calculate Buffer Time Index (BTI).
        
        BTI = (95th percentile - Average) / Average
        
        Measures the extra time (buffer) needed to ensure on-time arrival.
        Higher values indicate less reliable travel times.
        A BTI of 0.2 means travelers should budget 20% extra time.
        
        Returns:
            Buffer Time Index, or 0 if insufficient data
        """
        if not self.travel_times:
            return 0.0
        
        avg = self.get_average_travel_time()
        if avg == 0:
            return 0.0
        
        p95 = self.get_percentile_travel_time(95)
        return (p95 - avg) / avg
    
    def get_misery_index(self, free_flow_time: Optional[float] = None) -> float:
        """
        Calculate Misery Index.
        
        Misery Index = Average of worst 20% of trips / Free Flow Time
        
        Focuses on the worst travel experiences, which have
        disproportionate impact on user satisfaction.
        
        Args:
            free_flow_time: Free-flow travel time in seconds (uses stored value if not provided)
            
        Returns:
            Misery Index, or 0 if insufficient data
        """
        fft = free_flow_time or self.free_flow_time
        if not fft or not self.travel_times:
            return 0.0
        
        # Get worst 20% of trips
        p80_threshold = np.percentile(self.travel_times, 80)
        worst_trips = [tt for tt in self.travel_times if tt >= p80_threshold]
        
        if not worst_trips:
            return 0.0
        
        return np.mean(worst_trips) / fft
    
    def get_travel_time_variance(self) -> float:
        """
        Get variance of travel times.
        
        Higher variance indicates less predictable travel times.
        
        Returns:
            Variance in seconds squared, or 0 if insufficient data
        """
        return np.var(self.travel_times) if len(self.travel_times) > 1 else 0.0
    
    def get_coefficient_of_variation(self) -> float:
        """
        Get coefficient of variation (CV).
        
        CV = Standard Deviation / Mean
        
        Normalized measure of variability, useful for comparing
        different routes or time periods.
        
        Returns:
            Coefficient of variation, or 0 if insufficient data
        """
        if not self.travel_times:
            return 0.0
        
        mean = self.get_average_travel_time()
        if mean == 0:
            return 0.0
        
        std = np.std(self.travel_times)
        return std / mean
    
    def get_comprehensive_statistics(self, free_flow_time: Optional[float] = None) -> Dict[str, float]:
        """
        Get all travel time statistics in one call.
        
        Args:
            free_flow_time: Free-flow travel time in seconds (uses stored value if not provided)
            
        Returns:
            Dictionary containing all computed metrics
        """
        fft = free_flow_time or self.free_flow_time
        
        return {
            # Basic statistics
            'mean': self.get_average_travel_time(),
            'median': self.get_median_travel_time(),
            'std': np.std(self.travel_times) if self.travel_times else 0.0,
            'min': np.min(self.travel_times) if self.travel_times else 0.0,
            'max': np.max(self.travel_times) if self.travel_times else 0.0,
            
            # Percentiles
            'p50': self.get_percentile_travel_time(50),
            'p75': self.get_percentile_travel_time(75),
            'p85': self.get_percentile_travel_time(85),
            'p90': self.get_percentile_travel_time(90),
            'p95': self.get_percentile_travel_time(95),
            'p99': self.get_percentile_travel_time(99),
            
            # Reliability indices
            'travel_time_index': self.get_travel_time_index(fft),
            'planning_time_index': self.get_planning_time_index(fft),
            'buffer_time_index': self.get_buffer_time_index(),
            'misery_index': self.get_misery_index(fft),
            
            # Variability
            'variance': self.get_travel_time_variance(),
            'coefficient_of_variation': self.get_coefficient_of_variation(),
            
            # Sample size
            'sample_size': len(self.travel_times)
        }
    
    def reset(self):
        """Reset all travel time data."""
        self.travel_times.clear()
    
    def __repr__(self) -> str:
        return (f"TravelTimeMetrics(n={len(self.travel_times)}, "
                f"mean={self.get_average_travel_time():.2f}s, "
                f"p95={self.get_percentile_travel_time(95):.2f}s)")
