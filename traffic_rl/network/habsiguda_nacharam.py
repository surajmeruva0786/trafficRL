#!/usr/bin/env python3
"""
Habsiguda-Nacharam Traffic Network Module.

Models the real-world traffic corridor between Habsiguda and Nacharam/Mallapur
in Hyderabad, India. This is NOT a grid — it's a corridor with a branch.

Layout:
         J0 -------- J2 -------- J3 -------- J4 -------- J5
         |
         J1

Junctions:
  J0: Habsiguda Junction (T-junction)
  J1: Habsiguda Colony Junction (4-way)
  J2: Nagendra Nagar Junction (4-way)
  J3: ECIL X Roads (4-way)
  J4: Nacharam X Roads (4-way)
  J5: Mallapur Junction (T-junction)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Junction:
    """Represents a single junction in the Habsiguda-Nacharam corridor."""
    id: str
    name: str
    x: float  # meters
    y: float  # meters
    junction_type: str  # 'T' or '4way'
    neighbors: Dict[str, str] = field(default_factory=dict)  # direction -> neighbor_id
    approach_directions: List[str] = field(default_factory=list)  # available approach directions


@dataclass
class RoadSegment:
    """Represents a road segment connecting two junctions."""
    id: str
    name: str
    from_junction: str
    to_junction: str
    num_lanes: int
    speed_limit: float  # m/s
    length: float  # meters (computed from coordinates)
    road_type: str  # 'arterial', 'collector', 'local'


class HabsigudaNacharamNetwork:
    """
    Models the Habsiguda-Nacharam traffic corridor in Hyderabad.

    This is a real-world corridor layout with 6 signalized junctions
    connected by the Nacharam-Mallapur Road (NH163) with a branch
    to Habsiguda Colony.

    Attributes:
        junctions: Dictionary of all junctions
        road_segments: List of all road segments
        arterial_route: Main corridor junction IDs
    """

    # Junction definitions with approximate real-world coordinates
    JUNCTION_DEFS = {
        'J0': {
            'name': 'Habsiguda Junction',
            'x': 0.0, 'y': 0.0,
            'junction_type': 'T',
            'approaches': ['W', 'E', 'S'],  # Tarnaka Rd (W), Nacharam Rd (E), Main Rd (S)
        },
        'J1': {
            'name': 'Habsiguda Colony Junction',
            'x': 0.0, 'y': -800.0,
            'junction_type': '4way',
            'approaches': ['N', 'S', 'E', 'W'],
        },
        'J2': {
            'name': 'Nagendra Nagar Junction',
            'x': 1200.0, 'y': 0.0,
            'junction_type': '4way',
            'approaches': ['N', 'S', 'E', 'W'],
        },
        'J3': {
            'name': 'ECIL X Roads',
            'x': 2400.0, 'y': 0.0,
            'junction_type': '4way',
            'approaches': ['N', 'S', 'E', 'W'],
        },
        'J4': {
            'name': 'Nacharam X Roads',
            'x': 3600.0, 'y': 0.0,
            'junction_type': '4way',
            'approaches': ['N', 'S', 'E', 'W'],
        },
        'J5': {
            'name': 'Mallapur Junction',
            'x': 4800.0, 'y': 0.0,
            'junction_type': 'T',
            'approaches': ['W', 'N', 'S'],  # Nacharam Rd (W), Mallapur Rd (N/S)
        },
    }

    # Road connections: (from, to, name, num_lanes, speed_m_s, road_type)
    ROAD_DEFS = [
        # Main arterial corridor (NH163: Habsiguda → Mallapur) — bidirectional
        ('J0', 'J2', 'Nacharam-Mallapur Rd', 2, 13.89, 'arterial'),
        ('J2', 'J3', 'Nacharam-Mallapur Rd', 2, 13.89, 'arterial'),
        ('J3', 'J4', 'Nacharam-Mallapur Rd', 2, 13.89, 'arterial'),
        ('J4', 'J5', 'Nacharam-Mallapur Rd', 2, 13.89, 'arterial'),
        # Branch road (Habsiguda Main Road)
        ('J0', 'J1', 'Habsiguda Main Rd', 1, 8.33, 'collector'),
    ]

    def __init__(self):
        """Initialize the Habsiguda-Nacharam network."""
        self.junctions: Dict[str, Junction] = {}
        self.road_segments: List[RoadSegment] = []
        self.arterial_route: List[str] = ['J0', 'J2', 'J3', 'J4', 'J5']

        self._create_junctions()
        self._create_road_segments()

    def _create_junctions(self):
        """Create all junctions with neighbor relationships."""
        # First pass: create junctions
        for jid, jdef in self.JUNCTION_DEFS.items():
            self.junctions[jid] = Junction(
                id=jid,
                name=jdef['name'],
                x=jdef['x'],
                y=jdef['y'],
                junction_type=jdef['junction_type'],
                approach_directions=jdef['approaches'],
            )

        # Second pass: establish neighbor relationships from road definitions
        neighbor_map = {
            # (from_junction, to_junction): (direction_from_to, direction_to_from)
            ('J0', 'J2'): ('E', 'W'),
            ('J2', 'J3'): ('E', 'W'),
            ('J3', 'J4'): ('E', 'W'),
            ('J4', 'J5'): ('E', 'W'),
            ('J0', 'J1'): ('S', 'N'),
        }

        for (fj, tj), (d_ft, d_tf) in neighbor_map.items():
            self.junctions[fj].neighbors[d_ft] = tj
            self.junctions[tj].neighbors[d_tf] = fj

    def _create_road_segments(self):
        """Create all road segments."""
        for from_j, to_j, name, lanes, speed, rtype in self.ROAD_DEFS:
            fj = self.junctions[from_j]
            tj = self.junctions[to_j]
            length = ((fj.x - tj.x) ** 2 + (fj.y - tj.y) ** 2) ** 0.5

            # Forward direction
            self.road_segments.append(RoadSegment(
                id=f"{from_j}_to_{to_j}",
                name=name,
                from_junction=from_j,
                to_junction=to_j,
                num_lanes=lanes,
                speed_limit=speed,
                length=length,
                road_type=rtype,
            ))
            # Reverse direction
            self.road_segments.append(RoadSegment(
                id=f"{to_j}_to_{from_j}",
                name=name,
                from_junction=to_j,
                to_junction=from_j,
                num_lanes=lanes,
                speed_limit=speed,
                length=length,
                road_type=rtype,
            ))

    def get_all_junctions(self) -> List[str]:
        """Get list of all junction IDs."""
        return list(self.junctions.keys())

    def get_junction(self, junction_id: str) -> Junction:
        """Get a junction by its ID."""
        return self.junctions[junction_id]

    def get_neighbors(self, junction_id: str) -> Dict[str, str]:
        """Get neighbor junctions."""
        return self.junctions[junction_id].neighbors

    def get_boundary_junctions(self) -> List[str]:
        """Get junctions at the boundary (endpoints of the corridor)."""
        boundary = []
        for jid, junction in self.junctions.items():
            # Boundary junctions have fewer than 3 neighbors for 4-way,
            # or fewer than 2 for T-junction
            if len(junction.neighbors) < len(junction.approach_directions):
                boundary.append(jid)
        return boundary

    def get_road_segments_for_junction(self, junction_id: str) -> List[RoadSegment]:
        """Get all road segments connected to a junction."""
        return [seg for seg in self.road_segments
                if seg.from_junction == junction_id or seg.to_junction == junction_id]

    def get_arterial_route(self) -> List[str]:
        """Get the main arterial corridor junction sequence."""
        return self.arterial_route

    def get_road_segment(self, from_j: str, to_j: str) -> Optional[RoadSegment]:
        """Get a specific road segment between two junctions."""
        for seg in self.road_segments:
            if seg.from_junction == from_j and seg.to_junction == to_j:
                return seg
        return None

    def get_network_info(self) -> Dict:
        """Get comprehensive network information."""
        return {
            'name': 'Habsiguda-Nacharam Corridor',
            'location': 'Hyderabad, India',
            'total_junctions': len(self.junctions),
            'total_road_segments': len(self.road_segments),
            'arterial_length_m': sum(
                seg.length for seg in self.road_segments
                if seg.road_type == 'arterial' and seg.from_junction < seg.to_junction
            ),
            'junction_types': {
                jid: j.junction_type for jid, j in self.junctions.items()
            },
            'junction_names': {
                jid: j.name for jid, j in self.junctions.items()
            },
        }

    def __repr__(self) -> str:
        info = self.get_network_info()
        return (f"HabsigudaNacharamNetwork("
                f"{info['total_junctions']} junctions, "
                f"{info['total_road_segments']} road segments, "
                f"arterial={info['arterial_length_m']:.0f}m)")


if __name__ == "__main__":
    network = HabsigudaNacharamNetwork()
    print(network)
    print()

    info = network.get_network_info()
    print("Network Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nJunction Details:")
    for jid in network.get_all_junctions():
        j = network.get_junction(jid)
        print(f"  {jid} ({j.name}): type={j.junction_type}, "
              f"pos=({j.x}, {j.y}), "
              f"neighbors={j.neighbors}, "
              f"approaches={j.approach_directions}")

    print("\nRoad Segments:")
    for seg in network.road_segments:
        if seg.from_junction < seg.to_junction:  # Show each once
            print(f"  {seg.id}: {seg.name} ({seg.road_type}), "
                  f"lanes={seg.num_lanes}, speed={seg.speed_limit:.1f}m/s, "
                  f"length={seg.length:.0f}m")

    print(f"\nArterial Route: {' → '.join(network.get_arterial_route())}")
    print(f"Boundary Junctions: {network.get_boundary_junctions()}")
