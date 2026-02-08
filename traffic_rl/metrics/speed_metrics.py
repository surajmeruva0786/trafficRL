"""
Speed Metrics Module

Provides speed-based performance metrics for traffic flow analysis.
Speed is a fundamental indicator of traffic quality and congestion level.
"""

import numpy as np
from typing import List, Dict


class SpeedMetrics:
    """
    Speed-based performance metrics for traffic analysis.
    
    Speed measurements provide insight into traffic flow quality,
    congestion levels, and the effectiveness of traffic control strategies.
    
    Attributes:
        speed_limit: Posted speed limit in km/h
        speeds: List of recorded speeds in km/h
    """
    
    def __init__(self, speed_limit: float = 50.0):
        """
        Initialize speed metrics tracker.
        
        Args:
            speed_limit: Posted speed limit in km/h (default: 50 km/h)
        """
        self.speed_limit = speed_limit
        self.speeds: List[float] = []
        
    def add_speed(self, speed: float):
        """
        Record a speed measurement.
        
        Args:
            speed: Speed in km/h
        """
        self.speeds.append(speed)
    
    def add_speeds(self, speeds: List[float]):
        """
        Record multiple speed measurements.
        
        Args:
            speeds: List of speeds in km/h
        """
        self.speeds.extend(speeds)
    
    def get_average_speed(self) -> float:
        """
        Get average speed across all measurements.
        
        Returns:
            Average speed in km/h, or 0 if no data
        """
        return np.mean(self.speeds) if self.speeds else 0.0
    
    def get_median_speed(self) -> float:
        """
        Get median speed.
        
        Returns:
            Median speed in km/h, or 0 if no data
        """
        return np.median(self.speeds) if self.speeds else 0.0
    
    def get_speed_variance(self) -> float:
        """
        Get variance of speeds.
        
        Higher variance indicates unstable or inconsistent traffic flow,
        which can lead to safety issues and reduced capacity.
        
        Returns:
            Speed variance in (km/h)^2, or 0 if insufficient data
        """
        return np.var(self.speeds) if len(self.speeds) > 1 else 0.0
    
    def get_speed_standard_deviation(self) -> float:
        """
        Get standard deviation of speeds.
        
        Returns:
            Speed standard deviation in km/h, or 0 if insufficient data
        """
        return np.std(self.speeds) if len(self.speeds) > 1 else 0.0
    
    def get_percent_free_flow(self, threshold: float = 0.8) -> float:
        """
        Get percentage of measurements at or above free-flow speed.
        
        Free-flow is typically defined as 80% or more of the speed limit.
        This metric indicates the quality of traffic flow.
        
        Args:
            threshold: Fraction of speed limit considered free-flow (default: 0.8)
            
        Returns:
            Percentage (0-100) of measurements at free-flow, or 0 if no data
        """
        if not self.speeds:
            return 0.0
        
        free_flow_threshold = threshold * self.speed_limit
        free_flow_count = sum(1 for s in self.speeds if s >= free_flow_threshold)
        return (free_flow_count / len(self.speeds)) * 100
    
    def get_percent_congested(self, threshold: float = 0.5) -> float:
        """
        Get percentage of measurements in congested conditions.
        
        Congestion is typically defined as speeds below 50% of speed limit.
        
        Args:
            threshold: Fraction of speed limit below which is considered congested
            
        Returns:
            Percentage (0-100) of measurements in congestion, or 0 if no data
        """
        if not self.speeds:
            return 0.0
        
        congestion_threshold = threshold * self.speed_limit
        congested_count = sum(1 for s in self.speeds if s < congestion_threshold)
        return (congested_count / len(self.speeds)) * 100
    
    def get_speed_distribution(self, bins: int = 5) -> Dict[str, int]:
        """
        Get distribution of speeds across bins.
        
        Args:
            bins: Number of bins to divide speed range into
            
        Returns:
            Dictionary mapping speed ranges to counts
        """
        if not self.speeds:
            return {}
        
        min_speed = 0
        max_speed = max(self.speed_limit * 1.2, max(self.speeds))
        bin_edges = np.linspace(min_speed, max_speed, bins + 1)
        
        distribution = {}
        for i in range(bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            count = sum(1 for s in self.speeds if lower <= s < upper)
            distribution[f"{lower:.1f}-{upper:.1f} km/h"] = count
        
        return distribution
    
    def get_average_speed_ratio(self) -> float:
        """
        Get ratio of average speed to speed limit.
        
        Returns:
            Ratio (0-1+) of average speed to speed limit, or 0 if no data
        """
        if not self.speeds or self.speed_limit == 0:
            return 0.0
        
        return self.get_average_speed() / self.speed_limit
    
    def get_harmonic_mean_speed(self) -> float:
        """
        Get harmonic mean of speeds.
        
        Harmonic mean is more appropriate for averaging speeds as it
        gives more weight to lower speeds, which have greater impact
        on travel time.
        
        Returns:
            Harmonic mean speed in km/h, or 0 if no data or zero speeds
        """
        if not self.speeds:
            return 0.0
        
        # Filter out zero or negative speeds
        valid_speeds = [s for s in self.speeds if s > 0]
        if not valid_speeds:
            return 0.0
        
        return len(valid_speeds) / sum(1/s for s in valid_speeds)
    
    def get_comprehensive_statistics(self) -> Dict[str, float]:
        """
        Get all speed statistics in one call.
        
        Returns:
            Dictionary containing all computed metrics
        """
        return {
            # Basic statistics
            'mean': self.get_average_speed(),
            'median': self.get_median_speed(),
            'harmonic_mean': self.get_harmonic_mean_speed(),
            'std': self.get_speed_standard_deviation(),
            'variance': self.get_speed_variance(),
            'min': np.min(self.speeds) if self.speeds else 0.0,
            'max': np.max(self.speeds) if self.speeds else 0.0,
            
            # Percentiles
            'p25': np.percentile(self.speeds, 25) if self.speeds else 0.0,
            'p50': np.percentile(self.speeds, 50) if self.speeds else 0.0,
            'p75': np.percentile(self.speeds, 75) if self.speeds else 0.0,
            'p85': np.percentile(self.speeds, 85) if self.speeds else 0.0,
            
            # Flow quality
            'percent_free_flow': self.get_percent_free_flow(),
            'percent_congested': self.get_percent_congested(),
            'speed_ratio': self.get_average_speed_ratio(),
            
            # Reference
            'speed_limit': self.speed_limit,
            'sample_size': len(self.speeds)
        }
    
    def reset(self):
        """Reset all speed data."""
        self.speeds.clear()
    
    def __repr__(self) -> str:
        return (f"SpeedMetrics(n={len(self.speeds)}, "
                f"mean={self.get_average_speed():.2f} km/h, "
                f"limit={self.speed_limit} km/h)")
