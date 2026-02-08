#!/usr/bin/env python3
"""
Multi-Agent Coordination
Manages coordination between multiple traffic signal agents.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class CoordinationMode(Enum):
    """Coordination modes for multi-agent control."""
    INDEPENDENT = "independent"  # Each agent acts independently
    COORDINATED = "coordinated"  # Agents share observations and coordinate


class MultiAgentCoordination:
    """
    Manages multi-agent coordination for traffic signal control.
    
    Supports both independent and coordinated control modes, with neighbor
    observation sharing and arterial action coordination.
    """
    
    def __init__(self, network_topology: Dict[str, Dict[str, str]]):
        """
        Initialize multi-agent coordination.
        
        Args:
            network_topology: Dictionary mapping intersection_id to neighbors dict
                             e.g., {'I_0_0': {'N': 'I_1_0', 'E': 'I_0_1'}, ...}
        """
        self.network_topology = network_topology
        self.mode = CoordinationMode.INDEPENDENT
        self.arterial_routes: Dict[str, List[str]] = {}
        
        # Cache for observations
        self.local_observations: Dict[str, np.ndarray] = {}
        
    def set_coordination_mode(self, mode: CoordinationMode):
        """
        Set the coordination mode.
        
        Args:
            mode: CoordinationMode (INDEPENDENT or COORDINATED)
        """
        self.mode = mode
        print(f"Coordination mode set to: {mode.value}")
    
    def set_arterial_routes(self, routes: Dict[str, List[str]]):
        """
        Set arterial routes for coordination.
        
        Args:
            routes: Dictionary mapping route_id to list of intersection IDs
        """
        self.arterial_routes = routes
    
    def update_local_observation(self, intersection_id: str, observation: np.ndarray):
        """
        Update the local observation for an intersection.
        
        Args:
            intersection_id: ID of the intersection
            observation: Observation array (state vector)
        """
        self.local_observations[intersection_id] = observation
    
    def get_local_observation(self, intersection_id: str) -> Optional[np.ndarray]:
        """
        Get the local observation for an intersection.
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Observation array or None if not available
        """
        return self.local_observations.get(intersection_id)
    
    def get_neighbor_observations(self, intersection_id: str) -> Dict[str, np.ndarray]:
        """
        Get observations from neighboring intersections.
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Dictionary mapping neighbor direction to observation
        """
        neighbor_obs = {}
        
        if intersection_id in self.network_topology:
            neighbors = self.network_topology[intersection_id]
            
            for direction, neighbor_id in neighbors.items():
                if neighbor_id in self.local_observations:
                    neighbor_obs[direction] = self.local_observations[neighbor_id]
        
        return neighbor_obs
    
    def get_coordinated_observation(self, intersection_id: str) -> np.ndarray:
        """
        Get coordinated observation including neighbor information.
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Extended observation array with neighbor states
        """
        # Get local observation
        local_obs = self.get_local_observation(intersection_id)
        
        if local_obs is None:
            raise ValueError(f"No local observation for {intersection_id}")
        
        if self.mode == CoordinationMode.INDEPENDENT:
            # Return only local observation
            return local_obs
        
        # Get neighbor observations
        neighbor_obs = self.get_neighbor_observations(intersection_id)
        
        # Concatenate observations
        # Order: local, N, S, E, W (use zeros if neighbor doesn't exist)
        obs_parts = [local_obs]
        
        for direction in ['N', 'S', 'E', 'W']:
            if direction in neighbor_obs:
                obs_parts.append(neighbor_obs[direction])
            else:
                # Pad with zeros if neighbor doesn't exist
                obs_parts.append(np.zeros_like(local_obs))
        
        coordinated_obs = np.concatenate(obs_parts)
        return coordinated_obs
    
    def coordinate_arterial_actions(self, arterial_route: List[str], 
                                    actions: Dict[str, int]) -> Dict[str, int]:
        """
        Coordinate actions along an arterial route.
        
        Implements simple coordination strategy: synchronize green phases
        along the arterial to create green waves.
        
        Args:
            arterial_route: List of intersection IDs along the arterial
            actions: Dictionary mapping intersection_id to action
            
        Returns:
            Coordinated actions dictionary
        """
        if self.mode == CoordinationMode.INDEPENDENT:
            # No coordination, return original actions
            return actions
        
        coordinated_actions = actions.copy()
        
        # Simple coordination: if first intersection goes green in arterial direction,
        # encourage downstream intersections to also go green
        if len(arterial_route) < 2:
            return coordinated_actions
        
        # Check if first intersection is going green
        first_intersection = arterial_route[0]
        if first_intersection not in actions:
            return coordinated_actions
        
        first_action = actions[first_intersection]
        
        # If first intersection is activating green (action > 0),
        # encourage downstream intersections to align
        if first_action > 0:
            for i in range(1, len(arterial_route)):
                intersection_id = arterial_route[i]
                if intersection_id in coordinated_actions:
                    # Bias towards same action (simplified coordination)
                    coordinated_actions[intersection_id] = first_action
        
        return coordinated_actions
    
    def get_coordination_reward_bonus(self, intersection_id: str, 
                                     action: int,
                                     neighbor_actions: Dict[str, int]) -> float:
        """
        Calculate reward bonus for coordination.
        
        Rewards agents for coordinating with neighbors.
        
        Args:
            intersection_id: ID of the intersection
            action: Action taken by this agent
            neighbor_actions: Dictionary of neighbor actions
            
        Returns:
            Reward bonus (positive for good coordination)
        """
        if self.mode == CoordinationMode.INDEPENDENT:
            return 0.0
        
        # Check if this intersection is on an arterial
        on_arterial = False
        for route in self.arterial_routes.values():
            if intersection_id in route:
                on_arterial = True
                break
        
        if not on_arterial:
            return 0.0
        
        # Reward for matching actions with neighbors (simplified)
        bonus = 0.0
        matching_neighbors = 0
        
        if intersection_id in self.network_topology:
            neighbors = self.network_topology[intersection_id]
            
            for neighbor_id in neighbors.values():
                if neighbor_id in neighbor_actions:
                    if neighbor_actions[neighbor_id] == action:
                        matching_neighbors += 1
        
        # Bonus proportional to matching neighbors
        if neighbors:
            bonus = (matching_neighbors / len(neighbors)) * 0.1  # Small bonus
        
        return bonus
    
    def get_network_state_summary(self) -> Dict:
        """
        Get summary of network-wide state.
        
        Returns:
            Dictionary with network state statistics
        """
        if not self.local_observations:
            return {}
        
        # Calculate aggregate statistics
        all_obs = np.array(list(self.local_observations.values()))
        
        return {
            'num_intersections': len(self.local_observations),
            'mean_state': np.mean(all_obs, axis=0).tolist(),
            'std_state': np.std(all_obs, axis=0).tolist(),
            'coordination_mode': self.mode.value
        }


if __name__ == "__main__":
    # Example usage
    network_topology = {
        'I_0_0': {'S': 'I_1_0', 'E': 'I_0_1'},
        'I_0_1': {'S': 'I_1_1', 'W': 'I_0_0', 'E': 'I_0_2'},
        'I_0_2': {'S': 'I_1_2', 'W': 'I_0_1'},
        'I_1_0': {'N': 'I_0_0', 'S': 'I_2_0', 'E': 'I_1_1'},
        'I_1_1': {'N': 'I_0_1', 'S': 'I_2_1', 'W': 'I_1_0', 'E': 'I_1_2'},
        'I_1_2': {'N': 'I_0_2', 'S': 'I_2_2', 'W': 'I_1_1'},
        'I_2_0': {'N': 'I_1_0', 'E': 'I_2_1'},
        'I_2_1': {'N': 'I_1_1', 'W': 'I_2_0', 'E': 'I_2_2'},
        'I_2_2': {'N': 'I_1_2', 'W': 'I_2_1'}
    }
    
    coordinator = MultiAgentCoordination(network_topology)
    
    # Set arterial routes
    routes = {
        'H0': ['I_0_0', 'I_0_1', 'I_0_2'],
        'V0': ['I_0_0', 'I_1_0', 'I_2_0']
    }
    coordinator.set_arterial_routes(routes)
    
    # Test independent mode
    print("Testing INDEPENDENT mode:")
    coordinator.set_coordination_mode(CoordinationMode.INDEPENDENT)
    
    # Update observations
    for i_id in network_topology.keys():
        obs = np.random.rand(10)  # Random 10-dim observation
        coordinator.update_local_observation(i_id, obs)
    
    # Get coordinated observation
    coord_obs = coordinator.get_coordinated_observation('I_1_1')
    print(f"Observation shape (independent): {coord_obs.shape}")
    
    # Test coordinated mode
    print("\nTesting COORDINATED mode:")
    coordinator.set_coordination_mode(CoordinationMode.COORDINATED)
    coord_obs = coordinator.get_coordinated_observation('I_1_1')
    print(f"Observation shape (coordinated): {coord_obs.shape}")
    
    # Test action coordination
    actions = {i_id: np.random.randint(0, 4) for i_id in routes['H0']}
    print(f"\nOriginal actions: {actions}")
    coordinated_actions = coordinator.coordinate_arterial_actions(routes['H0'], actions)
    print(f"Coordinated actions: {coordinated_actions}")
