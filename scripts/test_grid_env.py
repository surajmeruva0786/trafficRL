"""Quick test to verify GridSUMOEnv fixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork
from traffic_rl.env.grid_sumo_env import GridSUMOEnv

print("Testing GridSUMOEnv fixes...")
print("=" * 60)

# Create network
network = MultiIntersectionNetwork(rows=3, cols=3, spacing=500.0)
print(f"✓ Created network with {len(network.intersections)} intersections")

# Test traffic light ID conversion
env = GridSUMOEnv(
    network=network,
    net_file='traffic_rl/sumo/grid_network.net.xml',
    route_file='traffic_rl/sumo/grid_routes.rou.xml',
    use_gui=False
)

# Test TL ID conversion
test_id = "I_0_0"
tl_id = env._get_tl_id(test_id)
print(f"✓ TL ID conversion: {test_id} -> {tl_id}")
assert tl_id == "I_0_0", f"Expected I_0_0, got {tl_id}"

# Test lane naming
lanes = env._get_lanes_for_intersection("I_1_1")
print(f"✓ Lanes for I_1_1:")
for direction, lane in lanes.items():
    print(f"  {direction}: {lane}")

expected_lanes = {
    'N': 'I_0_1_to_I_1_1_0',
    'S': 'I_2_1_to_I_1_1_0',
    'E': 'I_1_2_to_I_1_1_0',
    'W': 'I_1_0_to_I_1_1_0'
}

for direction, expected in expected_lanes.items():
    actual = lanes.get(direction)
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"  ✓ {direction} lane correct")

print("=" * 60)
print("✓ All tests passed! GridSUMOEnv is ready.")
