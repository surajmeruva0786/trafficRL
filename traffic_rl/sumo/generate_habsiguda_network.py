#!/usr/bin/env python3
"""
SUMO Network Generator for Habsiguda-Nacharam Corridor.

Generates SUMO network files (.nod.xml, .edg.xml, .net.xml) for the
real-world Habsiguda-Nacharam traffic corridor in Hyderabad.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traffic_rl.network.habsiguda_nacharam import HabsigudaNacharamNetwork


class HabsigudaNetworkGenerator:
    """Generates SUMO network files for the Habsiguda-Nacharam corridor."""

    def __init__(self, network: HabsigudaNacharamNetwork, output_dir: Optional[Path] = None):
        """
        Initialize the generator.

        Args:
            network: HabsigudaNacharamNetwork instance
            output_dir: Output directory for generated files
        """
        self.network = network
        self.output_dir = output_dir or Path(__file__).parent

    def create_node_file(self) -> Path:
        """Create the node XML file for all junctions and boundary nodes."""
        nodes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        nodes_xml += '<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        nodes_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n'

        # Add junction nodes (traffic light controlled)
        for jid, junction in self.network.junctions.items():
            nodes_xml += f'    <node id="{jid}" type="traffic_light" '
            nodes_xml += f'x="{junction.x}" y="{junction.y}" />\n'

        # Add boundary nodes for each approach that has no neighbor
        for jid, junction in self.network.junctions.items():
            for direction in junction.approach_directions:
                if direction not in junction.neighbors:
                    # This approach needs a boundary node
                    bx, by = self._get_boundary_coords(junction, direction)
                    boundary_id = f"{jid}_{direction}_boundary"
                    nodes_xml += f'    <node id="{boundary_id}" type="dead_end" '
                    nodes_xml += f'x="{bx}" y="{by}" />\n'

        nodes_xml += '</nodes>\n'

        node_file = self.output_dir / "habsiguda_network.nod.xml"
        with open(node_file, 'w') as f:
            f.write(nodes_xml)

        print(f"✓ Created {node_file}")
        return node_file

    def create_edge_file(self) -> Path:
        """Create the edge XML file for all road segments."""
        edges_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        edges_xml += '<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        edges_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n'

        # Internal edges (between junctions) — bidirectional
        for seg in self.network.road_segments:
            edge_id = seg.id
            edges_xml += f'    <edge id="{edge_id}" from="{seg.from_junction}" '
            edges_xml += f'to="{seg.to_junction}" '
            edges_xml += f'priority="{3 if seg.road_type == "arterial" else 2}" '
            edges_xml += f'numLanes="{seg.num_lanes}" '
            edges_xml += f'speed="{seg.speed_limit}" />\n'

        # Boundary edges (from boundary dead-end nodes to junctions)
        for jid, junction in self.network.junctions.items():
            for direction in junction.approach_directions:
                if direction not in junction.neighbors:
                    boundary_id = f"{jid}_{direction}_boundary"

                    # Determine road properties for this boundary approach
                    lanes, speed = self._get_boundary_road_props(jid, direction)

                    # Incoming edge (boundary → junction)
                    in_edge = f"{boundary_id}_to_{jid}"
                    edges_xml += f'    <edge id="{in_edge}" from="{boundary_id}" '
                    edges_xml += f'to="{jid}" priority="2" '
                    edges_xml += f'numLanes="{lanes}" speed="{speed}" />\n'

                    # Outgoing edge (junction → boundary)
                    out_edge = f"{jid}_to_{boundary_id}"
                    edges_xml += f'    <edge id="{out_edge}" from="{jid}" '
                    edges_xml += f'to="{boundary_id}" priority="2" '
                    edges_xml += f'numLanes="{lanes}" speed="{speed}" />\n'

        edges_xml += '</edges>\n'

        edge_file = self.output_dir / "habsiguda_network.edg.xml"
        with open(edge_file, 'w') as f:
            f.write(edges_xml)

        print(f"✓ Created {edge_file}")
        return edge_file

    def generate_network(self, node_file: Path, edge_file: Path) -> Path:
        """Use netconvert to generate the network file."""
        output_file = self.output_dir / "habsiguda_network.net.xml"

        cmd = [
            "netconvert",
            "--node-files", str(node_file),
            "--edge-files", str(edge_file),
            "--output-file", str(output_file),
            "--no-turnarounds", "true",
            "--junctions.corner-detail", "5",
            "--junctions.limit-turn-speed", "5.5",
            "--default.lanewidth", "3.2",
            "--default.junctions.radius", "4",
            "--tls.default-type", "static",
        ]

        print(f"\n🔧 Running netconvert...")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                check=True
            )

            print("✓ Network generation successful!")
            print(f"✓ Created {output_file}")

            if result.stdout:
                print(f"\nnetconvert output:\n{result.stdout}")

            return output_file

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running netconvert:")
            print(f"Return code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ Error: netconvert not found in PATH")
            print("Please ensure SUMO is installed and netconvert is in your PATH")
            sys.exit(1)

    def generate(self) -> Path:
        """Generate the complete Habsiguda-Nacharam network."""
        print("=" * 60)
        print("Habsiguda-Nacharam SUMO Network Generator")
        print("=" * 60)
        print()

        info = self.network.get_network_info()
        print(f"Corridor: {info['name']}")
        print(f"Location: {info['location']}")
        print(f"Junctions: {info['total_junctions']}")
        print()

        for jid, jname in info['junction_names'].items():
            jtype = info['junction_types'][jid]
            print(f"  {jid}: {jname} ({jtype})")
        print()

        # Create XML files
        node_file = self.create_node_file()
        edge_file = self.create_edge_file()

        # Generate network
        network_file = self.generate_network(node_file, edge_file)

        print("\n" + "=" * 60)
        print("✅ Habsiguda-Nacharam network generation complete!")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  - {node_file}")
        print(f"  - {edge_file}")
        print(f"  - {network_file}")

        return network_file

    def _get_boundary_coords(self, junction, direction: str, offset: float = 500.0):
        """Calculate boundary node coordinates."""
        if direction == 'N':
            return junction.x, junction.y + offset
        elif direction == 'S':
            return junction.x, junction.y - offset
        elif direction == 'E':
            return junction.x + offset, junction.y
        elif direction == 'W':
            return junction.x - offset, junction.y
        return junction.x, junction.y

    def _get_boundary_road_props(self, junction_id: str, direction: str):
        """Get road properties for a boundary approach."""
        # Arterial boundaries (E/W on main corridor junctions)
        arterial_junctions = {'J0', 'J2', 'J3', 'J4', 'J5'}
        if junction_id in arterial_junctions and direction in ('E', 'W'):
            return 2, 13.89  # 2 lanes, 50 km/h

        # Default: local road
        return 1, 8.33  # 1 lane, 30 km/h


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate SUMO network for Habsiguda-Nacharam corridor"
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: script directory)")

    args = parser.parse_args()

    # Create network
    network = HabsigudaNacharamNetwork()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    # Generate network files
    generator = HabsigudaNetworkGenerator(network, output_dir)
    generator.generate()


if __name__ == "__main__":
    main()
