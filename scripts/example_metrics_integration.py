"""
Example integration of enhanced metrics with existing SUMO environment.

This script shows how to integrate the new metrics system into your
existing traffic RL training loop.
"""

import traci
import numpy as np
from typing import Dict, List
from traffic_rl.metrics import EnhancedMetricsTracker


class SUMOMetricsIntegration:
    """
    Integration layer between SUMO and enhanced metrics system.
    
    This class handles the extraction of data from SUMO and feeding it
    to the enhanced metrics tracker.
    """
    
    def __init__(self, tracker: EnhancedMetricsTracker):
        """
        Initialize SUMO metrics integration.
        
        Args:
            tracker: EnhancedMetricsTracker instance
        """
        self.tracker = tracker
        self.active_vehicles: Dict[str, Dict] = {}
        self.completed_vehicles: set = set()
        
    def update_from_sumo(self, tl_id: str = None):
        """
        Update metrics from current SUMO state.
        
        Args:
            tl_id: Traffic light ID (optional, for filtering)
        """
        current_time = traci.simulation.getTime()
        
        # Get all vehicles in simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        # Track new vehicles (entries)
        for veh_id in vehicle_ids:
            if veh_id not in self.active_vehicles and veh_id not in self.completed_vehicles:
                # New vehicle entered
                route_id = traci.vehicle.getRouteID(veh_id)
                route_edges = traci.route.getEdges(route_id)
                
                # Calculate route length
                route_length = sum(
                    traci.lane.getLength(f"{edge}_0") 
                    for edge in route_edges
                )
                
                current_edge = traci.vehicle.getRoadID(veh_id)
                
                self.tracker.record_vehicle_entry(
                    vehicle_id=veh_id,
                    edge_id=current_edge,
                    timestamp=current_time,
                    route_length=route_length
                )
                
                self.active_vehicles[veh_id] = {
                    'entry_time': current_time,
                    'route_length': route_length
                }
        
        # Track vehicle exits (arrivals)
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id in self.active_vehicles:
                entry_info = self.active_vehicles[veh_id]
                travel_time = current_time - entry_info['entry_time']
                
                self.tracker.record_vehicle_exit(
                    vehicle_id=veh_id,
                    timestamp=current_time,
                    travel_time=travel_time,
                    actual_distance=entry_info['route_length']
                )
                
                self.completed_vehicles.add(veh_id)
                del self.active_vehicles[veh_id]
        
        # Collect speeds from active vehicles
        speeds = []
        for veh_id in vehicle_ids:
            speed_ms = traci.vehicle.getSpeed(veh_id)
            speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
            speeds.append(speed_kmh)
        
        if speeds:
            self.tracker.record_speeds(speeds)
        
        # Get waiting times and queue lengths
        if tl_id:
            waiting_time = self._get_total_waiting_time(tl_id)
            queue_length = self._get_total_queue_length(tl_id)
            
            self.tracker.record_waiting_time(waiting_time)
            self.tracker.record_queue_length(queue_length)
    
    def _get_total_waiting_time(self, tl_id: str) -> float:
        """
        Get total waiting time for vehicles near traffic light.
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Total waiting time in seconds
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        total_waiting = 0.0
        
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicles:
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                total_waiting += waiting_time
        
        return total_waiting
    
    def _get_total_queue_length(self, tl_id: str) -> float:
        """
        Get total queue length (number of stopped vehicles).
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Total number of stopped vehicles
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        total_queue = 0
        
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicles:
                speed = traci.vehicle.getSpeed(veh_id)
                if speed < 0.1:  # Stopped vehicle
                    total_queue += 1
        
        return total_queue
    
    def reset(self):
        """Reset tracking for new episode."""
        self.active_vehicles.clear()
        self.completed_vehicles.clear()


# Example usage in training loop
def example_training_integration():
    """
    Example of how to integrate enhanced metrics into training loop.
    
    This is a simplified example showing the key integration points.
    """
    
    print("Example: Enhanced Metrics Integration")
    print("="*60)
    
    # Initialize metrics tracker
    tracker = EnhancedMetricsTracker(
        free_flow_speed=13.89,  # 50 km/h
        speed_limit=50.0,
        free_flow_time=180.0  # Adjust based on your network
    )
    
    # Initialize integration
    integration = SUMOMetricsIntegration(tracker)
    
    print("\nIntegration points in your training loop:")
    print("\n1. At each simulation step:")
    print("   integration.update_from_sumo(tl_id='your_traffic_light_id')")
    
    print("\n2. When phase changes:")
    print("   tracker.record_phase_change()")
    
    print("\n3. When recording rewards:")
    print("   tracker.record_reward(reward)")
    
    print("\n4. At end of episode:")
    print("   tracker.end_episode()")
    print("   report = tracker.get_comprehensive_report()")
    print("   # Log or save report")
    print("   tracker.reset()")
    print("   integration.reset()")
    
    print("\n5. For comparison with baseline:")
    print("   comparison = rl_tracker.compare_with_baseline(baseline_tracker)")
    print("   # Analyze improvements and statistical significance")
    
    print("\n" + "="*60)
    print("Integration Example Complete")
    print("\nKey Benefits:")
    print("  ✓ Industry-standard transportation metrics")
    print("  ✓ Level of Service (LOS) classification")
    print("  ✓ Travel time reliability indices")
    print("  ✓ Statistical comparison with baselines")
    print("  ✓ Comprehensive performance reporting")


if __name__ == "__main__":
    example_training_integration()
