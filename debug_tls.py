
import os
import sys
import traci
import sumolib

# Setup paths
sumo_binary = "sumo"
config_file = "traffic_rl/sumo/simulation.sumocfg"

# Start SUMO
traci.start([sumo_binary, "-c", config_file, "--no-step-log", "true"])

try:
    # Get TLS ID list
    tls_list = traci.trafficlight.getIDList()
    print(f"Traffic Light IDs found: {tls_list}")
    
    if not tls_list:
        print("ERROR: No traffic lights found in simulation!")
        sys.exit(1)
        
    tl_id = tls_list[0]
    print(f"Inspecting TLS: {tl_id}")
    
    # Get controlled links
    # formatted as list of list of links: [[(from, to, via)], [], ...]
    controlled_links = traci.trafficlight.getControlledLinks(tl_id)
    
    print(f"Number of indices (signals): {len(controlled_links)}")
    
    print("\nLink Index Mapping:")
    for i, links in enumerate(controlled_links):
        # links is a list of connections controlled by index i
        if not links:
            print(f"Index {i}: No links")
            continue
            
        print(f"Index {i}:")
        for link in links:
            # link is (ingress_lane, egress_lane, via_lane)
            print(f"  From: {link[0]} -> To: {link[1]}")
            
    # Check current state
    current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
    print(f"\nCurrent State String: {current_state}")
    print(f"State Length: {len(current_state)}")
    
finally:
    traci.close()
