#!/usr/bin/env python3
"""
Simple script to visualize the grid network training environment with SUMO GUI.
This lets you see what the 3x3 grid looks like during simulation.
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork
from traffic_rl.env.grid_sumo_env import GridSUMOEnv


def view_grid_environment(duration_seconds=300):
    """
    Open SUMO GUI to visualize the grid network.
    
    Args:
        duration_seconds: How long to run the simulation (default: 5 minutes)
    """
    print("=" * 60)
    print("Grid Network Environment Viewer")
    print("=" * 60)
    
    # Create network
    network = MultiIntersectionNetwork(rows=3, cols=3, spacing=500.0)
    print(f"✓ Created {len(network.intersections)} intersections")
    
    # Setup paths
    net_file = 'traffic_rl/sumo/grid_network.net.xml'
    route_file = 'traffic_rl/sumo/grid_routes.rou.xml'
    
    # Verify files exist
    if not os.path.exists(net_file):
        print(f"❌ Network file not found: {net_file}")
        print("Run: python traffic_rl/sumo/generate_grid_network.py")
        return
    
    if not os.path.exists(route_file):
        print(f"❌ Route file not found: {route_file}")
        print("Run: python traffic_rl/sumo/generate_grid_routes.py --vehicles 2500")
        return
    
    print(f"✓ Network file: {net_file}")
    print(f"✓ Route file: {route_file}")
    
    # Initialize Grid SUMO Environment with GUI
    print("\n🎨 Opening SUMO GUI...")
    print("=" * 60)
    print("CONTROLS:")
    print("  - Click the ▶️ Play button to start simulation")
    print("  - Use the speed slider to adjust simulation speed")
    print("  - Right-click intersections to see traffic light status")
    print("  - Click on vehicles to see their routes")
    print("=" * 60)
    
    env = GridSUMOEnv(
        network=network,
        net_file=net_file,
        route_file=route_file,
        use_gui=True,  # ← Enable GUI!
        step_length=1.0,
        yellow_time=3,
        min_green_time=10,
        max_steps=duration_seconds
    )
    
    # Reset environment
    states = env.reset(seed=42)
    print(f"\n✓ Environment initialized")
    print(f"✓ Managing {len(states)} intersections")
    print(f"\nRunning simulation for {duration_seconds} seconds...")
    print("Watch the SUMO GUI window!\n")
    
    # Get intersection IDs
    intersection_ids = list(states.keys())
    
    # Run simulation with random actions (just for visualization)
    step = 0
    done = False
    
    while not done and step < duration_seconds:
        # Random actions for each intersection (just to see movement)
        import random
        actions = {i_id: random.choice([0, 1]) for i_id in intersection_ids}
        
        # Execute step
        next_states, rewards, done, info = env.step(actions)
        
        # Print progress every 30 seconds
        if step % 30 == 0 and step > 0:
            metrics = env.get_metrics()
            print(f"[{step}s] Vehicles arrived: {metrics['total_arrived']}, "
                  f"Throughput: {metrics['throughput']:.1f} veh/hr")
        
        states = next_states
        step += 1
    
    # Final metrics
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    metrics = env.get_metrics()
    print(f"Total vehicles arrived: {metrics['total_arrived']}")
    print(f"Average throughput: {metrics['throughput']:.2f} vehicles/hour")
    print(f"Total phase changes: {metrics['total_phase_changes']}")
    print("=" * 60)
    
    # Close environment
    env.close()
    print("\n✓ Environment closed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View grid network in SUMO GUI")
    parser.add_argument("--duration", type=int, default=300,
                       help="Simulation duration in seconds (default: 300)")
    
    args = parser.parse_args()
    
    view_grid_environment(duration_seconds=args.duration)
