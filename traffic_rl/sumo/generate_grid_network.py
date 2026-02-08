#!/usr/bin/env python3
"""
SUMO Grid Network Generator
Generates SUMO network files for multi-intersection grid topology.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork


class GridNetworkGenerator:
    """Generates SUMO network files for grid topology."""
    
    def __init__(self, network: MultiIntersectionNetwork, output_dir: Optional[Path] = None):
        """
        Initialize the grid network generator.
        
        Args:
            network: MultiIntersectionNetwork instance
            output_dir: Output directory for generated files (default: script directory)
        """
        self.network = network
        self.output_dir = output_dir or Path(__file__).parent
        
    def create_node_file(self) -> Path:
        """Create the node XML file for all intersections and boundary nodes."""
        nodes_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        nodes_xml += '<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        nodes_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n'
        
        # Add intersection nodes (traffic light controlled)
        for intersection_id, intersection in self.network.intersections.items():
            nodes_xml += f'    <node id="{intersection_id}" type="traffic_light" '
            nodes_xml += f'x="{intersection.x}" y="{intersection.y}" />\n'
        
        # Add boundary nodes (dead ends) for each edge intersection
        # North boundary
        for col in range(self.network.cols):
            intersection_id = self.network._get_intersection_id(0, col)
            intersection = self.network.intersections[intersection_id]
            boundary_id = f"{intersection_id}_N_boundary"
            nodes_xml += f'    <node id="{boundary_id}" type="dead_end" '
            nodes_xml += f'x="{intersection.x}" y="{intersection.y - self.network.spacing}" />\n'
        
        # South boundary
        for col in range(self.network.cols):
            intersection_id = self.network._get_intersection_id(self.network.rows - 1, col)
            intersection = self.network.intersections[intersection_id]
            boundary_id = f"{intersection_id}_S_boundary"
            nodes_xml += f'    <node id="{boundary_id}" type="dead_end" '
            nodes_xml += f'x="{intersection.x}" y="{intersection.y + self.network.spacing}" />\n'
        
        # West boundary
        for row in range(self.network.rows):
            intersection_id = self.network._get_intersection_id(row, 0)
            intersection = self.network.intersections[intersection_id]
            boundary_id = f"{intersection_id}_W_boundary"
            nodes_xml += f'    <node id="{boundary_id}" type="dead_end" '
            nodes_xml += f'x="{intersection.x - self.network.spacing}" y="{intersection.y}" />\n'
        
        # East boundary
        for row in range(self.network.rows):
            intersection_id = self.network._get_intersection_id(row, self.network.cols - 1)
            intersection = self.network.intersections[intersection_id]
            boundary_id = f"{intersection_id}_E_boundary"
            nodes_xml += f'    <node id="{boundary_id}" type="dead_end" '
            nodes_xml += f'x="{intersection.x + self.network.spacing}" y="{intersection.y}" />\n'
        
        nodes_xml += '</nodes>\n'
        
        node_file = self.output_dir / "grid_network.nod.xml"
        with open(node_file, 'w') as f:
            f.write(nodes_xml)
        
        print(f"✓ Created {node_file}")
        return node_file
    
    def create_edge_file(self) -> Path:
        """Create the edge XML file for all road segments."""
        edges_xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        edges_xml += '<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        edges_xml += 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n'
        
        # Internal edges (between intersections)
        for intersection_id, intersection in self.network.intersections.items():
            for direction, neighbor_id in intersection.neighbors.items():
                edge_id = f"{intersection_id}_to_{neighbor_id}"
                edges_xml += f'    <edge id="{edge_id}" from="{intersection_id}" to="{neighbor_id}" '
                edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
        
        # Boundary edges (from boundary nodes to edge intersections)
        # North boundary
        for col in range(self.network.cols):
            intersection_id = self.network._get_intersection_id(0, col)
            boundary_id = f"{intersection_id}_N_boundary"
            # Incoming edge
            edge_id = f"{boundary_id}_to_{intersection_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{boundary_id}" to="{intersection_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
            # Outgoing edge
            edge_id = f"{intersection_id}_to_{boundary_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{intersection_id}" to="{boundary_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
        
        # South boundary
        for col in range(self.network.cols):
            intersection_id = self.network._get_intersection_id(self.network.rows - 1, col)
            boundary_id = f"{intersection_id}_S_boundary"
            # Incoming edge
            edge_id = f"{boundary_id}_to_{intersection_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{boundary_id}" to="{intersection_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
            # Outgoing edge
            edge_id = f"{intersection_id}_to_{boundary_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{intersection_id}" to="{boundary_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
        
        # West boundary
        for row in range(self.network.rows):
            intersection_id = self.network._get_intersection_id(row, 0)
            boundary_id = f"{intersection_id}_W_boundary"
            # Incoming edge
            edge_id = f"{boundary_id}_to_{intersection_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{boundary_id}" to="{intersection_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
            # Outgoing edge
            edge_id = f"{intersection_id}_to_{boundary_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{intersection_id}" to="{boundary_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
        
        # East boundary
        for row in range(self.network.rows):
            intersection_id = self.network._get_intersection_id(row, self.network.cols - 1)
            boundary_id = f"{intersection_id}_E_boundary"
            # Incoming edge
            edge_id = f"{boundary_id}_to_{intersection_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{boundary_id}" to="{intersection_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
            # Outgoing edge
            edge_id = f"{intersection_id}_to_{boundary_id}"
            edges_xml += f'    <edge id="{edge_id}" from="{intersection_id}" to="{boundary_id}" '
            edges_xml += 'priority="2" numLanes="1" speed="13.89" />\n'
        
        edges_xml += '</edges>\n'
        
        edge_file = self.output_dir / "grid_network.edg.xml"
        with open(edge_file, 'w') as f:
            f.write(edges_xml)
        
        print(f"✓ Created {edge_file}")
        return edge_file
    
    def generate_network(self, node_file: Path, edge_file: Path) -> Path:
        """Use netconvert to generate the network file."""
        output_file = self.output_dir / "grid_network.net.xml"
        
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
        """Generate the complete grid network."""
        print("=" * 60)
        print("SUMO Grid Network Generator")
        print(f"Generating {self.network.rows}×{self.network.cols} grid network")
        print("=" * 60)
        print()
        
        # Create XML files
        node_file = self.create_node_file()
        edge_file = self.create_edge_file()
        
        # Generate network
        network_file = self.generate_network(node_file, edge_file)
        
        print("\n" + "=" * 60)
        print("✅ Grid network generation complete!")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  - {node_file}")
        print(f"  - {edge_file}")
        print(f"  - {network_file}")
        print(f"\nNetwork info:")
        for key, value in self.network.get_network_info().items():
            print(f"  {key}: {value}")
        
        return network_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate SUMO grid network")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns (default: 3)")
    parser.add_argument("--spacing", type=float, default=500.0, 
                       help="Spacing between intersections in meters (default: 500.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: script directory)")
    
    args = parser.parse_args()
    
    # Create network
    network = MultiIntersectionNetwork(rows=args.rows, cols=args.cols, spacing=args.spacing)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    
    # Generate network files
    generator = GridNetworkGenerator(network, output_dir)
    generator.generate()


if __name__ == "__main__":
    main()
