#!/usr/bin/env python3
"""
Multi-Intersection Network Module
Manages grid topology for multi-intersection traffic networks.
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class Intersection:
    """Represents a single intersection in the network."""
    id: str
    row: int
    col: int
    x: float
    y: float
    neighbors: Dict[str, str]  # direction -> neighbor_id
    is_arterial: bool = False


class MultiIntersectionNetwork:
    """
    Manages a grid-based multi-intersection traffic network.
    
    Attributes:
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        spacing (float): Distance between intersections in meters
        intersections (Dict[str, Intersection]): All intersections in the network
    """
    
    def __init__(self, rows: int = 3, cols: int = 3, spacing: float = 500.0):
        """
        Initialize a multi-intersection network.
        
        Args:
            rows: Number of rows in the grid (default: 3)
            cols: Number of columns in the grid (default: 3)
            spacing: Distance between intersections in meters (default: 500.0)
        """
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.intersections: Dict[str, Intersection] = {}
        self.arterial_routes: Dict[str, List[str]] = {}
        
        # Create the grid
        self._create_grid()
        self._identify_arterial_routes()
    
    def _create_grid(self):
        """Create the grid structure with all intersections and neighbor relationships."""
        # Create all intersections
        for row in range(self.rows):
            for col in range(self.cols):
                intersection_id = self._get_intersection_id(row, col)
                x, y = self._calculate_coordinates(row, col)
                
                # Determine neighbors
                neighbors = {}
                if row > 0:  # Has north neighbor
                    neighbors['N'] = self._get_intersection_id(row - 1, col)
                if row < self.rows - 1:  # Has south neighbor
                    neighbors['S'] = self._get_intersection_id(row + 1, col)
                if col > 0:  # Has west neighbor
                    neighbors['W'] = self._get_intersection_id(row, col - 1)
                if col < self.cols - 1:  # Has east neighbor
                    neighbors['E'] = self._get_intersection_id(row, col + 1)
                
                intersection = Intersection(
                    id=intersection_id,
                    row=row,
                    col=col,
                    x=x,
                    y=y,
                    neighbors=neighbors
                )
                
                self.intersections[intersection_id] = intersection
    
    def _identify_arterial_routes(self):
        """Identify arterial routes (main horizontal and vertical roads)."""
        # Horizontal arterials (each row)
        for row in range(self.rows):
            route_id = f"H{row}"
            route = [self._get_intersection_id(row, col) for col in range(self.cols)]
            self.arterial_routes[route_id] = route
            
            # Mark intersections as arterial
            for intersection_id in route:
                self.intersections[intersection_id].is_arterial = True
        
        # Vertical arterials (each column)
        for col in range(self.cols):
            route_id = f"V{col}"
            route = [self._get_intersection_id(row, col) for row in range(self.rows)]
            self.arterial_routes[route_id] = route
            
            # Mark intersections as arterial
            for intersection_id in route:
                self.intersections[intersection_id].is_arterial = True
    
    def _get_intersection_id(self, row: int, col: int) -> str:
        """Generate intersection ID from row and column."""
        return f"I_{row}_{col}"
    
    def _calculate_coordinates(self, row: int, col: int) -> Tuple[float, float]:
        """
        Calculate (x, y) coordinates for an intersection.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        # Center the grid around origin
        x = col * self.spacing
        y = row * self.spacing
        return (x, y)
    
    def get_neighbors(self, intersection_id: str) -> Dict[str, str]:
        """
        Get neighbors of an intersection.
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Dictionary mapping direction to neighbor ID
        """
        if intersection_id not in self.intersections:
            raise ValueError(f"Intersection {intersection_id} not found")
        
        return self.intersections[intersection_id].neighbors
    
    def get_arterial_routes(self) -> Dict[str, List[str]]:
        """
        Get all arterial routes in the network.
        
        Returns:
            Dictionary mapping route ID to list of intersection IDs
        """
        return self.arterial_routes
    
    def get_intersection_coords(self, intersection_id: str) -> Tuple[float, float]:
        """
        Get coordinates of an intersection.
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Tuple of (x, y) coordinates
        """
        if intersection_id not in self.intersections:
            raise ValueError(f"Intersection {intersection_id} not found")
        
        intersection = self.intersections[intersection_id]
        return (intersection.x, intersection.y)
    
    def get_all_intersections(self) -> List[str]:
        """Get list of all intersection IDs."""
        return list(self.intersections.keys())
    
    def get_edge_intersections(self) -> List[str]:
        """Get list of edge intersection IDs (boundary of the grid)."""
        edge_intersections = []
        for intersection_id, intersection in self.intersections.items():
            # Edge if on boundary (row 0, row max, col 0, or col max)
            if (intersection.row == 0 or intersection.row == self.rows - 1 or
                intersection.col == 0 or intersection.col == self.cols - 1):
                edge_intersections.append(intersection_id)
        return edge_intersections
    
    def get_network_info(self) -> Dict:
        """
        Get comprehensive network information.
        
        Returns:
            Dictionary with network statistics and configuration
        """
        return {
            'rows': self.rows,
            'cols': self.cols,
            'spacing': self.spacing,
            'total_intersections': len(self.intersections),
            'total_arterial_routes': len(self.arterial_routes),
            'horizontal_routes': self.rows,
            'vertical_routes': self.cols,
            'edge_intersections': len(self.get_edge_intersections()),
            'network_width': (self.cols - 1) * self.spacing,
            'network_height': (self.rows - 1) * self.spacing
        }
    
    def __repr__(self) -> str:
        """String representation of the network."""
        info = self.get_network_info()
        return (f"MultiIntersectionNetwork({info['rows']}x{info['cols']}, "
                f"{info['total_intersections']} intersections, "
                f"{info['total_arterial_routes']} arterial routes)")


if __name__ == "__main__":
    # Example usage
    network = MultiIntersectionNetwork(rows=3, cols=3, spacing=500.0)
    
    print(network)
    print("\nNetwork Info:")
    for key, value in network.get_network_info().items():
        print(f"  {key}: {value}")
    
    print("\nArterial Routes:")
    for route_id, intersections in network.get_arterial_routes().items():
        print(f"  {route_id}: {' -> '.join(intersections)}")
    
    print("\nSample Intersection (I_1_1):")
    neighbors = network.get_neighbors("I_1_1")
    coords = network.get_intersection_coords("I_1_1")
    print(f"  Coordinates: {coords}")
    print(f"  Neighbors: {neighbors}")
