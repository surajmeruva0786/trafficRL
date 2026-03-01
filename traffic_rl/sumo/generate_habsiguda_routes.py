#!/usr/bin/env python3
"""
SUMO Route Generator for Habsiguda-Nacharam Corridor.

Generates vehicle routes with realistic traffic patterns for the
Habsiguda-Nacharam corridor. Supports balanced regime distribution
and asymmetric flows on the arterial.
"""

import random
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traffic_rl.network.habsiguda_nacharam import HabsigudaNacharamNetwork


class HabsigudaRouteGenerator:
    """Generates vehicle routes for the Habsiguda-Nacharam corridor."""

    # Traffic regime definitions (vehicles per hour total across network)
    REGIMES = {
        'low': (100, 200),
        'medium': (300, 500),
        'high': (600, 900),
    }

    # Origin-destination flow weights (direction: weight)
    # Higher weight on arterial (E-W) traffic
    FLOW_WEIGHTS = {
        'arterial_ew': 0.45,    # Through-traffic on NH163
        'arterial_we': 0.25,    # Return direction (less in AM peak)
        'local_ns': 0.15,       # Local cross-traffic
        'branch': 0.15,         # Habsiguda Colony branch traffic
    }

    def __init__(self, network: HabsigudaNacharamNetwork, output_dir: Path = None):
        """
        Initialize the route generator.

        Args:
            network: HabsigudaNacharamNetwork instance
            output_dir: Output directory for route files
        """
        self.network = network
        self.output_dir = output_dir or Path(__file__).parent

    def _find_path(self, start_j: str, end_j: str) -> List[str]:
        """
        Find path between two junctions using BFS.

        Returns:
            List of junction IDs forming the path
        """
        if start_j == end_j:
            return [start_j]

        visited = {start_j}
        queue = [[start_j]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            for direction, neighbor in self.network.junctions[current].neighbors.items():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == end_j:
                        return new_path
                    visited.add(neighbor)
                    queue.append(new_path)

        return []  # No path found

    def _path_to_edges(self, path: List[str]) -> List[str]:
        """Convert a junction path to SUMO edge IDs."""
        edges = []
        for i in range(len(path) - 1):
            edge_id = f"{path[i]}_to_{path[i+1]}"
            edges.append(edge_id)
        return edges

    def _get_boundary_entry_edge(self, junction_id: str, direction: str) -> str:
        """Get the entry edge from a boundary node to a junction."""
        boundary_id = f"{junction_id}_{direction}_boundary"
        return f"{boundary_id}_to_{junction_id}"

    def _get_boundary_exit_edge(self, junction_id: str, direction: str) -> str:
        """Get the exit edge from a junction to a boundary node."""
        boundary_id = f"{junction_id}_{direction}_boundary"
        return f"{junction_id}_to_{boundary_id}"

    def _get_available_boundary_directions(self, junction_id: str) -> List[str]:
        """Get directions that have boundary nodes (no neighbor)."""
        junction = self.network.junctions[junction_id]
        return [d for d in junction.approach_directions if d not in junction.neighbors]

    def _generate_arterial_route(self, direction: str = 'ew') -> Tuple[List[str], str, str]:
        """Generate a through-traffic route on the arterial."""
        if direction == 'ew':
            # West entry at J0 → East exit at J5
            start_j, end_j = 'J0', 'J5'
            entry_dir, exit_dir = 'W', self._pick_exit_dir(end_j, prefer='E')
        else:
            # East entry at J5 → West exit at J0
            start_j, end_j = 'J5', 'J0'
            entry_dir, exit_dir = self._pick_entry_dir(start_j), 'W'

        path = self._find_path(start_j, end_j)
        internal_edges = self._path_to_edges(path)

        entry_edge = self._get_boundary_entry_edge(start_j, entry_dir)
        exit_edge = self._get_boundary_exit_edge(end_j, exit_dir)

        full_route = [entry_edge] + internal_edges + [exit_edge]
        return full_route, start_j, end_j

    def _generate_local_route(self) -> Tuple[List[str], str, str]:
        """Generate a local cross-traffic route (N-S movement)."""
        # Pick a junction with N or S boundary
        candidates = []
        for jid in self.network.get_all_junctions():
            boundary_dirs = self._get_available_boundary_directions(jid)
            ns_dirs = [d for d in boundary_dirs if d in ('N', 'S')]
            if ns_dirs:
                candidates.append((jid, ns_dirs))

        if not candidates:
            return self._generate_arterial_route()

        # Pick random origin junction
        start_j, start_dirs = random.choice(candidates)
        entry_dir = random.choice(start_dirs)

        # Pick a different junction as destination (preferably nearby)
        end_candidates = [(jid, dirs) for jid, dirs in candidates if jid != start_j]
        if not end_candidates:
            # Only one junction with N/S — just go through to exit on the other side
            exit_dirs = [d for d in start_dirs if d != entry_dir]
            if exit_dirs:
                exit_dir = exit_dirs[0]
            else:
                # Use E/W exit
                boundary_dirs = self._get_available_boundary_directions(start_j)
                ew_dirs = [d for d in boundary_dirs if d in ('E', 'W')]
                exit_dir = random.choice(ew_dirs) if ew_dirs else entry_dir

            entry_edge = self._get_boundary_entry_edge(start_j, entry_dir)
            exit_edge = self._get_boundary_exit_edge(start_j, exit_dir)
            return [entry_edge, exit_edge], start_j, start_j

        end_j, end_dirs = random.choice(end_candidates)
        exit_dir = random.choice(end_dirs)

        path = self._find_path(start_j, end_j)
        internal_edges = self._path_to_edges(path)

        entry_edge = self._get_boundary_entry_edge(start_j, entry_dir)
        exit_edge = self._get_boundary_exit_edge(end_j, exit_dir)

        full_route = [entry_edge] + internal_edges + [exit_edge]
        return full_route, start_j, end_j

    def _generate_branch_route(self) -> Tuple[List[str], str, str]:
        """Generate a route involving the Habsiguda Colony branch (J1)."""
        # Routes going through J1
        j1_dirs = self._get_available_boundary_directions('J1')

        if random.random() < 0.5:
            # J1 → arterial (colony to main road)
            entry_dir = random.choice(j1_dirs)
            # Pick a random arterial exit
            exit_j = random.choice(['J2', 'J3', 'J4', 'J5'])
            exit_dirs = self._get_available_boundary_directions(exit_j)
            if not exit_dirs:
                exit_j = 'J5'
                exit_dirs = self._get_available_boundary_directions(exit_j)
            exit_dir = random.choice(exit_dirs)

            path = self._find_path('J1', exit_j)
            internal_edges = self._path_to_edges(path)

            entry_edge = self._get_boundary_entry_edge('J1', entry_dir)
            exit_edge = self._get_boundary_exit_edge(exit_j, exit_dir)
            return [entry_edge] + internal_edges + [exit_edge], 'J1', exit_j
        else:
            # Arterial → J1 (main road to colony)
            start_j = random.choice(['J2', 'J3', 'J4', 'J5'])
            start_dirs = self._get_available_boundary_directions(start_j)
            if not start_dirs:
                start_j = 'J5'
                start_dirs = self._get_available_boundary_directions(start_j)
            entry_dir = random.choice(start_dirs)

            exit_dir = random.choice(j1_dirs)

            path = self._find_path(start_j, 'J1')
            internal_edges = self._path_to_edges(path)

            entry_edge = self._get_boundary_entry_edge(start_j, entry_dir)
            exit_edge = self._get_boundary_exit_edge('J1', exit_dir)
            return [entry_edge] + internal_edges + [exit_edge], start_j, 'J1'

    def _pick_entry_dir(self, junction_id: str) -> str:
        """Pick a boundary entry direction for a junction."""
        dirs = self._get_available_boundary_directions(junction_id)
        return random.choice(dirs) if dirs else 'W'

    def _pick_exit_dir(self, junction_id: str, prefer: str = None) -> str:
        """Pick a boundary exit direction, preferring a given direction."""
        dirs = self._get_available_boundary_directions(junction_id)
        if prefer and prefer in dirs:
            return prefer
        return random.choice(dirs) if dirs else 'E'

    def generate_routes(self, total_vehicles: int = 500,
                        simulation_time: int = 3600,
                        balanced: bool = True) -> Path:
        """
        Generate routes with balanced regime distribution.

        Args:
            total_vehicles: Total number of vehicles
            simulation_time: Simulation duration in seconds
            balanced: If True, use balanced regime distribution

        Returns:
            Path to generated route file
        """
        routes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        routes_xml += '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        routes_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n'

        # Vehicle types
        routes_xml += '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
        routes_xml += 'length="5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>\n'
        routes_xml += '    <vType id="auto" accel="2.0" decel="3.5" sigma="0.7" '
        routes_xml += 'length="4.5" minGap="2.0" maxSpeed="11.11" guiShape="passenger/sedan"/>\n\n'

        # Distribute vehicles across regimes
        if balanced:
            vehicles_per_regime = total_vehicles // 3
            regime_distribution = {
                'low': vehicles_per_regime,
                'medium': vehicles_per_regime,
                'high': total_vehicles - 2 * vehicles_per_regime,
            }
        else:
            regime_distribution = {
                'low': total_vehicles // 3,
                'medium': total_vehicles // 3,
                'high': total_vehicles // 3,
            }

        print(f"\nGenerating {total_vehicles} vehicles with distribution:")
        for regime, count in regime_distribution.items():
            print(f"  {regime}: {count} vehicles")

        vehicle_id = 0
        all_vehicles = []  # Collect all (depart_time, xml) pairs

        for regime, num_vehicles in regime_distribution.items():
            time_interval = simulation_time / num_vehicles if num_vehicles > 0 else 0

            for i in range(num_vehicles):
                # Random departure time
                base_time = i * time_interval
                depart_time = base_time + random.uniform(-time_interval * 0.3, time_interval * 0.3)
                depart_time = max(0, min(depart_time, simulation_time - 1))

                # Choose route type based on flow weights
                r = random.random()
                cumulative = 0
                route_edges = None

                for flow_type, weight in self.FLOW_WEIGHTS.items():
                    cumulative += weight
                    if r < cumulative:
                        if flow_type == 'arterial_ew':
                            route_edges, _, _ = self._generate_arterial_route('ew')
                        elif flow_type == 'arterial_we':
                            route_edges, _, _ = self._generate_arterial_route('we')
                        elif flow_type == 'local_ns':
                            route_edges, _, _ = self._generate_local_route()
                        elif flow_type == 'branch':
                            route_edges, _, _ = self._generate_branch_route()
                        break

                if route_edges is None:
                    route_edges, _, _ = self._generate_arterial_route('ew')

                if len(route_edges) < 2:
                    continue

                # Vehicle type
                vtype = random.choice(['car', 'auto'])

                route_id = f"route_{vehicle_id}"
                vehicle_xml = f'    <route id="{route_id}" edges="{" ".join(route_edges)}"/>\n'
                vehicle_xml += f'    <vehicle id="vehicle_{vehicle_id}" type="{vtype}" '
                vehicle_xml += f'route="{route_id}" depart="{depart_time:.2f}"/>\n'

                all_vehicles.append((depart_time, vehicle_xml))
                vehicle_id += 1

        # Sort by departure time (SUMO requires this)
        all_vehicles.sort(key=lambda x: x[0])

        for _, xml in all_vehicles:
            routes_xml += xml

        routes_xml += '\n</routes>\n'

        # Write to file
        output_file = self.output_dir / "habsiguda_routes.rou.xml"
        with open(output_file, 'w') as f:
            f.write(routes_xml)

        print(f"\n✓ Created {output_file}")
        print(f"✓ Generated {vehicle_id} vehicles")

        return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate SUMO routes for Habsiguda-Nacharam corridor"
    )
    parser.add_argument("--vehicles", type=int, default=500,
                        help="Total number of vehicles (default: 500)")
    parser.add_argument("--simulation-time", type=int, default=3600,
                        help="Simulation duration in seconds (default: 3600)")
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable balanced regime distribution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: script directory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Habsiguda-Nacharam Route Generator")
    print("=" * 60)

    network = HabsigudaNacharamNetwork()
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    generator = HabsigudaRouteGenerator(network, output_dir)
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
