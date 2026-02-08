#!/usr/bin/env python3
"""
SUMO Grid Route Generator
Generates vehicle routes for multi-intersection grid networks with balanced regime distribution.
"""

import random
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork


class GridRouteGenerator:
    """Generates vehicle routes for grid networks with balanced traffic regimes."""
    
    # Traffic regime definitions (vehicles per hour)
    REGIMES = {
        'low': (50, 100),
        'medium': (150, 250),
        'high': (300, 400)
    }
    
    def __init__(self, network: MultiIntersectionNetwork, output_dir: Path = None):
        """
        Initialize the grid route generator.
        
        Args:
            network: MultiIntersectionNetwork instance
            output_dir: Output directory for route files
        """
        self.network = network
        self.output_dir = output_dir or Path(__file__).parent
        self.edge_intersections = network.get_edge_intersections()
    
    def _get_boundary_edges(self, intersection_id: str) -> List[str]:
        """Get boundary edges for an edge intersection."""
        intersection = self.network.intersections[intersection_id]
        boundary_edges = []
        
        # Check which boundaries this intersection has
        if intersection.row == 0:  # North edge
            boundary_id = f"{intersection_id}_N_boundary"
            boundary_edges.append(f"{boundary_id}_to_{intersection_id}")
        if intersection.row == self.network.rows - 1:  # South edge
            boundary_id = f"{intersection_id}_S_boundary"
            boundary_edges.append(f"{boundary_id}_to_{intersection_id}")
        if intersection.col == 0:  # West edge
            boundary_id = f"{intersection_id}_W_boundary"
            boundary_edges.append(f"{boundary_id}_to_{intersection_id}")
        if intersection.col == self.network.cols - 1:  # East edge
            boundary_id = f"{intersection_id}_E_boundary"
            boundary_edges.append(f"{boundary_id}_to_{intersection_id}")
        
        return boundary_edges
    
    def _get_exit_boundary_edge(self, intersection_id: str) -> str:
        """Get a random exit boundary edge for an edge intersection."""
        intersection = self.network.intersections[intersection_id]
        exit_edges = []
        
        if intersection.row == 0:  # North edge
            boundary_id = f"{intersection_id}_N_boundary"
            exit_edges.append(f"{intersection_id}_to_{boundary_id}")
        if intersection.row == self.network.rows - 1:  # South edge
            boundary_id = f"{intersection_id}_S_boundary"
            exit_edges.append(f"{intersection_id}_to_{boundary_id}")
        if intersection.col == 0:  # West edge
            boundary_id = f"{intersection_id}_W_boundary"
            exit_edges.append(f"{intersection_id}_to_{boundary_id}")
        if intersection.col == self.network.cols - 1:  # East edge
            boundary_id = f"{intersection_id}_E_boundary"
            exit_edges.append(f"{intersection_id}_to_{boundary_id}")
        
        return random.choice(exit_edges) if exit_edges else None
    
    def _generate_route_edges(self, start_intersection: str, end_intersection: str) -> List[str]:
        """
        Generate a simple route from start to end intersection using Manhattan distance.
        
        Args:
            start_intersection: Starting intersection ID
            end_intersection: Ending intersection ID
            
        Returns:
            List of edge IDs forming the route
        """
        start = self.network.intersections[start_intersection]
        end = self.network.intersections[end_intersection]
        
        edges = []
        current_id = start_intersection
        
        # Move horizontally first, then vertically
        while current_id != end_intersection:
            current = self.network.intersections[current_id]
            
            # Move horizontally towards destination
            if current.col < end.col and 'E' in current.neighbors:
                next_id = current.neighbors['E']
                edges.append(f"{current_id}_to_{next_id}")
                current_id = next_id
            elif current.col > end.col and 'W' in current.neighbors:
                next_id = current.neighbors['W']
                edges.append(f"{current_id}_to_{next_id}")
                current_id = next_id
            # Move vertically towards destination
            elif current.row < end.row and 'S' in current.neighbors:
                next_id = current.neighbors['S']
                edges.append(f"{current_id}_to_{next_id}")
                current_id = next_id
            elif current.row > end.row and 'N' in current.neighbors:
                next_id = current.neighbors['N']
                edges.append(f"{current_id}_to_{next_id}")
                current_id = next_id
            else:
                break
        
        return edges
    
    def generate_routes(self, total_vehicles: int = 300, 
                       simulation_time: int = 3600,
                       balanced: bool = True) -> Path:
        """
        Generate routes with balanced regime distribution.
        
        Args:
            total_vehicles: Total number of vehicles to generate
            simulation_time: Simulation duration in seconds
            balanced: If True, use balanced regime distribution (33% each)
            
        Returns:
            Path to generated route file
        """
        routes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        routes_xml += '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        routes_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n'
        
        # Define vehicle type
        routes_xml += '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
        routes_xml += 'length="5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>\n\n'
        
        # Determine regime distribution
        if balanced:
            vehicles_per_regime = total_vehicles // 3
            regime_distribution = {
                'low': vehicles_per_regime,
                'medium': vehicles_per_regime,
                'high': total_vehicles - 2 * vehicles_per_regime  # Remainder goes to high
            }
        else:
            # Random distribution
            regime_distribution = {
                'low': total_vehicles // 3,
                'medium': total_vehicles // 3,
                'high': total_vehicles // 3
            }
        
        print(f"\nGenerating {total_vehicles} vehicles with distribution:")
        for regime, count in regime_distribution.items():
            print(f"  {regime}: {count} vehicles")
        
        vehicle_id = 0
        
        # Generate vehicles for each regime
        for regime, num_vehicles in regime_distribution.items():
            min_vph, max_vph = self.REGIMES[regime]
            
            # Calculate departure times for this regime
            # Spread vehicles evenly across simulation time
            time_interval = simulation_time / num_vehicles if num_vehicles > 0 else 0
            
            for i in range(num_vehicles):
                # Random departure time with some variation
                base_time = i * time_interval
                depart_time = base_time + random.uniform(-time_interval * 0.3, time_interval * 0.3)
                depart_time = max(0, min(depart_time, simulation_time - 1))
                
                # Random origin and destination (edge intersections)
                origin = random.choice(self.edge_intersections)
                destination = random.choice([x for x in self.edge_intersections if x != origin])
                
                # Get entry edge (from boundary)
                entry_edges = self._get_boundary_edges(origin)
                if not entry_edges:
                    continue
                entry_edge = random.choice(entry_edges)
                
                # Generate route through network
                route_edges = self._generate_route_edges(origin, destination)
                
                # Get exit edge (to boundary)
                exit_edge = self._get_exit_boundary_edge(destination)
                if not exit_edge:
                    continue
                
                # Complete route
                full_route = [entry_edge] + route_edges + [exit_edge]
                
                # Create route and vehicle
                route_id = f"route_{vehicle_id}"
                routes_xml += f'    <route id="{route_id}" edges="{" ".join(full_route)}"/>\n'
                routes_xml += f'    <vehicle id="vehicle_{vehicle_id}" type="car" route="{route_id}" '
                routes_xml += f'depart="{depart_time:.2f}"/>\n'
                
                vehicle_id += 1
        
        routes_xml += '\n</routes>\n'
        
        # Write to file
        output_file = self.output_dir / "grid_routes.rou.xml"
        with open(output_file, 'w') as f:
            f.write(routes_xml)
        
        print(f"\n✓ Created {output_file}")
        print(f"✓ Generated {vehicle_id} vehicles")
        
        return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate SUMO grid routes")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns (default: 3)")
    parser.add_argument("--spacing", type=float, default=500.0,
                       help="Spacing between intersections in meters (default: 500.0)")
    parser.add_argument("--vehicles", type=int, default=300,
                       help="Total number of vehicles (default: 300)")
    parser.add_argument("--simulation-time", type=int, default=3600,
                       help="Simulation duration in seconds (default: 3600)")
    parser.add_argument("--no-balanced", action="store_true",
                       help="Disable balanced regime distribution")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: script directory)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SUMO Grid Route Generator")
    print(f"Generating routes for {args.rows}×{args.cols} grid network")
    print("=" * 60)
    
    # Create network
    network = MultiIntersectionNetwork(rows=args.rows, cols=args.cols, spacing=args.spacing)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    
    # Generate routes
    generator = GridRouteGenerator(network, output_dir)
    generator.generate_routes(
        total_vehicles=args.vehicles,
        simulation_time=args.simulation_time,
        balanced=not args.no_balanced
    )
    
    print("\n" + "=" * 60)
    print("✅ Route generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
