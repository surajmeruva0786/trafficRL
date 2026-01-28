"""
Simple network validation test using SUMO directly.
This validates that the new network file is properly formatted and can be loaded.
"""

import subprocess
import sys
from pathlib import Path

SUMO_DIR = Path(__file__).parent.absolute()

def test_network_validation():
    """Test that the network file is valid using netconvert."""
    print("=" * 60)
    print("SUMO Network Validation Test")
    print("=" * 60)
    print()
    
    network_file = SUMO_DIR / "network.net.xml"
    
    if not network_file.exists():
        print(f"❌ Network file not found: {network_file}")
        return False
    
    print(f"✓ Network file exists: {network_file}")
    print(f"\n🔧 Validating network with netconvert...")
    
    # Use netconvert to validate the network
    cmd = [
        "netconvert",
        "--sumo-net-file", str(network_file),
        "--output-file", str(SUMO_DIR / "test_output.net.xml"),
        "--plain-output-prefix", str(SUMO_DIR / "test_plain")
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        print("✅ Network validation PASSED!")
        print(f"\nnetconvert successfully processed the network file.")
        print(f"The network has proper structure and valid geometry.")
        
        # Clean up test files
        import os
        for test_file in [
            SUMO_DIR / "test_output.net.xml",
            SUMO_DIR / "test_plain.nod.xml",
            SUMO_DIR / "test_plain.edg.xml",
            SUMO_DIR / "test_plain.con.xml",
            SUMO_DIR / "test_plain.tll.xml"
        ]:
            if test_file.exists():
                os.remove(test_file)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Network validation FAILED!")
        print(f"\nnetconvert error:")
        print(e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ Network validation TIMEOUT!")
        return False
    except FileNotFoundError:
        print(f"❌ netconvert not found in PATH")
        print(f"Please ensure SUMO is installed")
        return False


def test_simulation_config():
    """Test that the simulation config is valid."""
    print(f"\n{'=' * 60}")
    print("Testing Simulation Configuration")
    print("=" * 60)
    print()
    
    config_file = SUMO_DIR / "simulation.sumocfg"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print(f"✓ Config file exists: {config_file}")
    
    # Check that referenced files exist
    network_file = SUMO_DIR / "network.net.xml"
    routes_file = SUMO_DIR / "routes.rou.xml"
    
    if not network_file.exists():
        print(f"❌ Network file missing: {network_file}")
        return False
    
    if not routes_file.exists():
        print(f"❌ Routes file missing: {routes_file}")
        return False
    
    print(f"✓ Network file exists: {network_file}")
    print(f"✓ Routes file exists: {routes_file}")
    
    print(f"\n✅ Simulation configuration is complete!")
    return True


def main():
    """Run all validation tests."""
    print("\n")
    
    # Test 1: Network validation
    network_valid = test_network_validation()
    
    # Test 2: Configuration check
    config_valid = test_simulation_config()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Network Validation: {'✅ PASS' if network_valid else '❌ FAIL'}")
    print(f"Config Check: {'✅ PASS' if config_valid else '❌ FAIL'}")
    print(f"{'=' * 60}")
    
    if network_valid and config_valid:
        print(f"\n✅ All validation tests PASSED!")
        print(f"\nThe new network is ready to use.")
        print(f"You can now:")
        print(f"  1. Visualize with: sumo-gui -c {SUMO_DIR / 'simulation.sumocfg'}")
        print(f"  2. Run training with your existing scripts")
        return 0
    else:
        print(f"\n❌ Some validation tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
