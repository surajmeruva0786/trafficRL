"""
Delay Calculator Module

Calculates delay metric: Actual travel time - Free-flow travel time
This is a fundamental metric in transportation engineering for measuring
the impact of congestion and signal control effectiveness.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple, List


class DelayCalculator:
    """
    Calculate delay metric: Actual travel time - Free-flow travel time
    
    Delay is the difference between the actual time a vehicle takes to traverse
    a network segment and the time it would take under free-flow conditions.
    This metric directly measures the impact of congestion and control strategies.
    
    Attributes:
        free_flow_speed: Speed limit in m/s for free-flow calculations
        vehicle_entries: Dict tracking when vehicles enter the network
        vehicle_exits: Dict tracking when vehicles exit and their delays
        segment_delays: Dict tracking delays by network segment
    """
    
    def __init__(self, free_flow_speed: float = 13.89):
        """
        Initialize delay calculator.
        
        Args:
            free_flow_speed: Free-flow speed in m/s (default: 13.89 m/s = 50 km/h)
        """
        self.free_flow_speed = free_flow_speed
        self.vehicle_entries: Dict[str, Dict] = {}
        self.vehicle_exits: Dict[str, Dict] = {}
        self.segment_delays: Dict[str, List[float]] = defaultdict(list)
        
    def calculate_free_flow_time(self, distance: float) -> float:
        """
        Calculate free-flow travel time for a given distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Free-flow travel time in seconds
        """
        return distance / self.free_flow_speed
    
    def record_vehicle_entry(self, vehicle_id: str, edge_id: str, 
                            timestamp: float, route_length: float = None):
        """
        Record when a vehicle enters the network or a segment.
        
        Args:
            vehicle_id: Unique vehicle identifier
            edge_id: Edge/segment identifier
            timestamp: Current simulation time in seconds
            route_length: Total route length in meters (optional)
        """
        self.vehicle_entries[vehicle_id] = {
            'edge_id': edge_id,
            'entry_time': timestamp,
            'route_length': route_length
        }
    
    def record_vehicle_exit(self, vehicle_id: str, timestamp: float,
                           actual_distance: float = None) -> Optional[float]:
        """
        Record when a vehicle exits and calculate its delay.
        
        Args:
            vehicle_id: Unique vehicle identifier
            timestamp: Current simulation time in seconds
            actual_distance: Actual distance traveled in meters (optional)
            
        Returns:
            Calculated delay in seconds, or None if vehicle entry not recorded
        """
        if vehicle_id not in self.vehicle_entries:
            return None
        
        entry_info = self.vehicle_entries[vehicle_id]
        actual_time = timestamp - entry_info['entry_time']
        
        # Use route length from entry or provided distance
        distance = actual_distance or entry_info.get('route_length', 0)
        
        if distance > 0:
            free_flow_time = self.calculate_free_flow_time(distance)
            delay = max(0, actual_time - free_flow_time)
        else:
            # If no distance available, use actual time as delay
            # (conservative estimate)
            delay = actual_time
        
        self.vehicle_exits[vehicle_id] = {
            'actual_time': actual_time,
            'free_flow_time': free_flow_time if distance > 0 else 0,
            'delay': delay,
            'edge_id': entry_info['edge_id'],
            'exit_time': timestamp
        }
        
        # Track by segment
        self.segment_delays[entry_info['edge_id']].append(delay)
        
        return delay
    
    def get_total_delay(self) -> float:
        """
        Get total delay across all vehicles.
        
        Returns:
            Total delay in seconds
        """
        return sum(v['delay'] for v in self.vehicle_exits.values())
    
    def get_average_delay(self) -> float:
        """
        Get average delay per vehicle.
        
        Returns:
            Average delay in seconds, or 0 if no vehicles
        """
        if not self.vehicle_exits:
            return 0.0
        return self.get_total_delay() / len(self.vehicle_exits)
    
    def get_delay_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive delay statistics.
        
        Returns:
            Dictionary containing mean, median, std, min, max, and percentiles
        """
        if not self.vehicle_exits:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p75': 0.0,
                'p90': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        delays = [v['delay'] for v in self.vehicle_exits.values()]
        
        return {
            'mean': np.mean(delays),
            'median': np.median(delays),
            'std': np.std(delays),
            'min': np.min(delays),
            'max': np.max(delays),
            'p50': np.percentile(delays, 50),
            'p75': np.percentile(delays, 75),
            'p90': np.percentile(delays, 90),
            'p95': np.percentile(delays, 95),
            'p99': np.percentile(delays, 99)
        }
    
    def get_delay_by_segment(self) -> Dict[str, Dict[str, float]]:
        """
        Get delay statistics broken down by network segment.
        
        Returns:
            Dictionary mapping segment IDs to their delay statistics
        """
        segment_stats = {}
        
        for segment_id, delays in self.segment_delays.items():
            if delays:
                segment_stats[segment_id] = {
                    'mean': np.mean(delays),
                    'median': np.median(delays),
                    'std': np.std(delays),
                    'count': len(delays)
                }
        
        return segment_stats
    
    def get_delay_by_time_window(self, window_size: int = 300) -> List[Tuple[float, float]]:
        """
        Get average delay over time windows.
        
        Args:
            window_size: Size of time window in seconds (default: 5 minutes)
            
        Returns:
            List of (timestamp, average_delay) tuples
        """
        if not self.vehicle_exits:
            return []
        
        # Get all exit times
        exit_times = [(v['exit_time'], v['delay']) 
                     for v in self.vehicle_exits.values()]
        exit_times.sort()
        
        # Create time windows
        min_time = exit_times[0][0]
        max_time = exit_times[-1][0]
        
        windows = []
        current_time = min_time
        
        while current_time <= max_time:
            window_end = current_time + window_size
            
            # Get delays in this window
            window_delays = [delay for time, delay in exit_times 
                           if current_time <= time < window_end]
            
            if window_delays:
                avg_delay = np.mean(window_delays)
                windows.append((current_time + window_size/2, avg_delay))
            
            current_time = window_end
        
        return windows
    
    def reset(self):
        """Reset all tracking data."""
        self.vehicle_entries.clear()
        self.vehicle_exits.clear()
        self.segment_delays.clear()
    
    def get_vehicle_count(self) -> int:
        """Get total number of vehicles that have exited."""
        return len(self.vehicle_exits)
    
    def __repr__(self) -> str:
        return (f"DelayCalculator(vehicles={len(self.vehicle_exits)}, "
                f"avg_delay={self.get_average_delay():.2f}s)")
